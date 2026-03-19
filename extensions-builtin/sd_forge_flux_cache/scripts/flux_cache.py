##  First Block Cache + TeaCache for FastForge webui
import torch
import numpy as np
from torch import Tensor
import gradio as gr
import gc
from modules import scripts
from modules.ui_components import InputAccordion
from modules import shared
from backend.nn.flux_optimized import (
    IntegratedFluxTransformer2DModel, 
    timestep_embedding,
    _vram_monitor
)
from einops import rearrange


#GLOBAL STATE

class CacheState:
    """Global state for both methods"""
    # Who is currently active: None, "fbc", "teacache"
    active_method = None
    
    # Saved Originals
    original_forward = None
    
    # Saved tiling mode (for restoration)
    original_tiling_mode = None
    
    # First Block Cache state
    fbc_threshold = 0.01
    fbc_nocache_steps = 1
    fbc_max_cached = 0
    fbc_always_last = True
    fbc_steps = 20
    fbc_cnt = 0
    fbc_previous_residual = None
    fbc_first_block_output = None
    fbc_accumulated_distance = 0.0
    fbc_printed = False
    
    # TeaCache state
    tc_rel_l1_thresh = 0.50
    tc_steps = 20
    tc_cnt = 0
    tc_accumulated_rel_l1_distance = 0.0
    tc_previous_modulated_input = None
    tc_previous_residual = None
    tc_printed = False


def force_normal_mode():
    """Forcefully enable normal mode"""
    try:
        current_mode = getattr(shared.opts, 'flux_tiling_mode', 'normal')
        if current_mode != 'normal':
            # Save only if you haven't saved it yet
            if CacheState.original_tiling_mode is None:
                CacheState.original_tiling_mode = current_mode
                print(f"[Cache] Saved original tiling mode: {current_mode}")
            shared.opts.flux_tiling_mode = 'normal'
            print(f"[Cache] Forced NORMAL mode")
            return True
        else:
            # It's already normal, but we still save it if necessary
            if CacheState.original_tiling_mode is None:
                CacheState.original_tiling_mode = 'normal'
                print(f"[Cache] Tiling already NORMAL, saved")
    except Exception as e:
        print(f"[Cache] Warning: could not set tiling mode: {e}")
    return False


def restore_tiling_mode():
    """Restoring the original tiling mode"""
    if CacheState.original_tiling_mode is not None:
        try:
            # Restore only if not normal (so as not to lose the settings)
            if CacheState.original_tiling_mode != 'normal':
                shared.opts.flux_tiling_mode = CacheState.original_tiling_mode
                print(f"[Cache] Restored tiling mode: {CacheState.original_tiling_mode}")
            else:
                print(f"[Cache] Original was NORMAL, no restore needed")
        except Exception as e:
            print(f"[Cache] Error restoring tiling mode: {e}")
        CacheState.original_tiling_mode = None


def patch_forward(new_forward):
    """Patch forward while preserving the original"""
    if CacheState.original_forward is None:
        CacheState.original_forward = IntegratedFluxTransformer2DModel.forward
        print(f"[Cache] Saved original forward")
    IntegratedFluxTransformer2DModel.forward = new_forward


def unpatch_forward():
    """Restoring the original forward"""
    if CacheState.original_forward is not None:
        IntegratedFluxTransformer2DModel.forward = CacheState.original_forward
        print(f"[Cache] Restored original forward")
        CacheState.original_forward = None


def clear_all_cache():
    """Clearing the entire cache"""
    # FBC
    CacheState.fbc_previous_residual = None
    CacheState.fbc_first_block_output = None
    CacheState.fbc_accumulated_distance = 0.0
    CacheState.fbc_cnt = 0
    CacheState.fbc_printed = False
    
    # TC
    CacheState.tc_previous_modulated_input = None
    CacheState.tc_previous_residual = None
    CacheState.tc_accumulated_rel_l1_distance = 0.0
    CacheState.tc_cnt = 0
    CacheState.tc_printed = False
    
    CacheState.active_method = None
    torch.cuda.empty_cache()
    print("[Cache] All cache cleared")


# FIRST BLOCK CACHE

class FirstBlockCache(scripts.Script):
    def __init__(self):
        super().__init__()
        self.last_input_shape = None

    def title(self):
        return "First Block Cache"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with InputAccordion(value=False, label=self.title()) as enabled:
            with gr.Row():
                nocache_steps = gr.Number(
                    label="Uncached starting steps", scale=0,
                    minimum=0, maximum=12, value=1, step=1,
                )
                threshold = gr.Slider(
                    label="Threshold (max 0.50, step 0.001)", 
                    minimum=0.0, maximum=0.50, value=0.43, step=0.001,
                )
            with gr.Row():
                max_cached = gr.Number(
                    label="Max consecutive cached (0=unlimited)", scale=0,
                    minimum=0, maximum=99, value=0, step=1,
                )
                always_last = gr.Checkbox(
                    label="Never cache last step", value=True
                )
                
        for comp in [enabled, threshold, nocache_steps, max_cached, always_last]:
            comp.do_not_save_to_config = True

        self.infotext_fields = [
            (enabled, "FirstBlockCache Enabled"),
            (threshold, "FirstBlockCache Threshold"),
            (nocache_steps, "FirstBlockCache NoCache Steps"),
            (max_cached, "FirstBlockCache Max Cached"),
            (always_last, "FirstBlockCache Always Last"),
        ]

        return [enabled, threshold, nocache_steps, max_cached, always_last]

    def process(self, p, enabled, threshold, nocache_steps, max_cached, always_last):
        # Checking the resolution change
        current_input_shape = (p.width, p.height)
        if self.last_input_shape is not None and current_input_shape != self.last_input_shape:
            clear_all_cache()
        self.last_input_shape = current_input_shape

        if not enabled:
            # Shutdown
            if CacheState.active_method == "fbc":
                unpatch_forward()
                restore_tiling_mode()
                clear_all_cache()
                print("[FirstBlockCache] Disabled")
            return

        # TURN ON
        # Check if TeaCache is already enabled
        if CacheState.active_method == "teacache":
            print("[ERROR] TeaCache is already active! Disable it first.")
            return
        
        # Set itself as the active method
        CacheState.active_method = "fbc"

        CacheState.original_tiling_mode = None
        
        # Force normal mode
        force_normal_mode()

        # Save the parameters
        CacheState.fbc_threshold = float(threshold)
        CacheState.fbc_nocache_steps = int(nocache_steps)
        CacheState.fbc_max_cached = int(max_cached)
        CacheState.fbc_always_last = always_last
        CacheState.fbc_steps = p.steps
        CacheState.fbc_cnt = 0
        CacheState.fbc_printed = False

        print(f"[FirstBlockCache] Enabled: threshold={threshold}, nocache={nocache_steps}")
        
        # Patch forward
        patch_forward(fbc_forward)
        
        p.extra_generation_params.update({
            "fbc_enabled": True,
            "fbc_threshold": threshold,
            "fbc_nocache_steps": nocache_steps,
        })


# TEA CACHE

class TeaCache(scripts.Script):
    def __init__(self):
        super().__init__()
        self.last_input_shape = None

    def title(self):
        return "TeaCache"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with InputAccordion(value=False, label=self.title()) as enable:
            with gr.Group(elem_classes="teacache"):
                rel_l1_thresh_slider = gr.Slider(
                    label="Relative L1 Threshold",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    value=0.5,
                    tooltip="Threshold for caching intermediate results. Lower values cache more aggressively."
                )
                steps_slider = gr.Slider(
                    label="Steps",
                    minimum=1,
                    maximum=100,
                    step=1,
                    value=20,
                    tooltip="Number of steps to cache intermediate results."
                )

        self.paste_field_names = []
        self.infotext_fields = [
            (enable, "TeaCache Enabled"),
            (rel_l1_thresh_slider, "TeaCache Relative L1 Threshold"),
            (steps_slider, "TeaCache Steps"),
        ]

        for comp, name in self.infotext_fields:
            comp.do_not_save_to_config = True
            self.paste_field_names.append(name)

        return [enable, rel_l1_thresh_slider, steps_slider]

    def process(self, p, enable, rel_l1_thresh_slider, steps_slider):
        # Checking the resolution change
        current_input_shape = (p.width, p.height)
        if self.last_input_shape is not None and current_input_shape != self.last_input_shape:
            clear_all_cache()
        self.last_input_shape = current_input_shape

        if not enable:
            # Shutdown
            if CacheState.active_method == "teacache":
                unpatch_forward()
                restore_tiling_mode()
                clear_all_cache()
                print("[TeaCache] Disabled")
            return

        # Check if FirstBlockCache is already enabled
        if CacheState.active_method == "fbc":
            print("[ERROR] FirstBlockCache is already active! Disable it first.")
            return
        
        # Set yourself as the active method
        CacheState.active_method = "teacache"

        CacheState.original_tiling_mode = None
        
        # Force normal mode
        force_normal_mode()

        # Save the parameters
        CacheState.tc_rel_l1_thresh = float(rel_l1_thresh_slider)
        CacheState.tc_steps = int(steps_slider)
        CacheState.tc_cnt = 0
        CacheState.tc_accumulated_rel_l1_distance = 0.0
        CacheState.tc_previous_modulated_input = None
        CacheState.tc_previous_residual = None
        CacheState.tc_printed = False

        print(f"[TeaCache] Enabled: threshold={rel_l1_thresh_slider}, steps={steps_slider}")
        
        # Patch forward
        patch_forward(tc_forward)
        
        p.extra_generation_params.update({
            "teacache_enabled": True,
            "teacache_threshold": rel_l1_thresh_slider,
            "teacache_steps": steps_slider,
        })


# FIRST BLOCK CACHE FORWARD

def fbc_forward(self, x, timestep, context, y, guidance=None,
                tiling_mode=None, image_height=None, image_width=None, **kwargs):
    """First Block Cache"""
    
    # Checking that we are active
    if CacheState.active_method != "fbc":
        if CacheState.original_forward:
            return CacheState.original_forward(self, x, timestep, context, y, guidance, 
                                               tiling_mode, image_height, image_width, **kwargs)
        raise RuntimeError("No original forward available")

    if not CacheState.fbc_printed:
        print(f"[FirstBlockCache] Active: thresh={CacheState.fbc_threshold}")
        CacheState.fbc_printed = True

    # Force normal mode
    tiling_mode = 'normal'

    # Parameters
    threshold = CacheState.fbc_threshold
    nocache_steps = CacheState.fbc_nocache_steps
    always_last = CacheState.fbc_always_last
    max_cached = CacheState.fbc_max_cached
    steps = CacheState.fbc_steps
    cnt = CacheState.fbc_cnt

    # Preparation
    if next(self.parameters()).device.type != 'cuda':
        self.cuda()
        torch.cuda.synchronize()

    bs, c, h, w = x.shape
    input_dtype = x.dtype
    input_device = x.device

    patch_size = 2
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size

    if pad_h or pad_w:
        x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="circular")

    img = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size)

    h_len = (h + pad_h) // patch_size
    w_len = (w + pad_w) // patch_size

    img_ids = torch.zeros((h_len, w_len, 3), device=input_device, dtype=input_dtype)
    img_ids[..., 1] = torch.linspace(0, h_len - 1, steps=h_len, device=input_device, dtype=input_dtype)[:, None]
    img_ids[..., 2] = torch.linspace(0, w_len - 1, steps=w_len, device=input_device, dtype=input_dtype)[None, :]
    img_ids = img_ids.expand(bs, -1, -1, -1).reshape(bs, h_len * w_len, 3)

    txt_ids = torch.zeros((bs, context.shape[1], 3), device=input_device, dtype=input_dtype)

    # Embeddings
    img = self.img_in(img)
    vec = self.time_in(timestep_embedding(timestep, 256).to(img.dtype))

    if self.guidance_embed and guidance is not None:
        vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))

    vec = vec + self.vector_in(y)
    txt = self.txt_in(context)
    ids = torch.cat((txt_ids, img_ids), dim=1)
    pe = self.pe_embedder(ids)

    ori_img = img.clone()

    # First block ALWAYS
    first_block = self.double_blocks[0]
    img, txt = first_block(img=img, txt=txt, vec=vec, pe=pe)

    # First Block Cache Check
    use_early_exit = False
    
    if cnt >= nocache_steps and not (always_last and cnt >= steps - 1):
        if CacheState.fbc_first_block_output is not None and CacheState.fbc_previous_residual is not None:
            # We count the change
            current_change = ((img - CacheState.fbc_first_block_output).abs().mean() / 
                            (CacheState.fbc_first_block_output.abs().mean() + 1e-6)).cpu().item()
            
            CacheState.fbc_accumulated_distance += current_change
            
            if CacheState.fbc_accumulated_distance < threshold:
                use_early_exit = True
            else:
                CacheState.fbc_accumulated_distance = 0.0

    # Save the output of the first block
    CacheState.fbc_first_block_output = img.clone()
    CacheState.fbc_cnt = (cnt + 1) % steps

    if use_early_exit:
        # Early exit!
        img = ori_img + CacheState.fbc_previous_residual.to(img.device)
        print(f"  [FBC] Step {cnt}: EARLY EXIT (dist={CacheState.fbc_accumulated_distance:.4f})")
    else:
        # Continue with the remaining blocks
        for block in self.double_blocks[1:]:
            img, txt = block(img=img, txt=txt, vec=vec, pe=pe)

        # Single blocks
        img = torch.cat((txt, img), dim=1)
        for block in self.single_blocks:
            img = block(img, vec=vec, pe=pe)
        
        img = img[:, txt.shape[1]:, ...]
        
        # Save the full residual
        CacheState.fbc_previous_residual = (img - ori_img).detach()
        print(f"  [FBC] Step {cnt}: full compute (dist={CacheState.fbc_accumulated_distance:.4f})")

    # Final layer
    img = self.final_layer(img, vec)

    # Back to spatial
    out = rearrange(img, "b (h w) (c ph pw) -> b c (h ph) (w pw)",
                   h=h_len, w=w_len, ph=patch_size, pw=patch_size)
    return out[:, :, :h, :w].to(input_dtype)


# TEA CACHE FORWARD

def tc_forward(self, x, timestep, context, y, guidance=None,
               tiling_mode=None, image_height=None, image_width=None, **kwargs):
    """TeaCache"""
    
    # Checking that we are active
    if CacheState.active_method != "teacache":
        if CacheState.original_forward:
            return CacheState.original_forward(self, x, timestep, context, y, guidance,
                                               tiling_mode, image_height, image_width, **kwargs)
        raise RuntimeError("No original forward available")

    if not CacheState.tc_printed:
        print(f"[TeaCache] Active: thresh={CacheState.tc_rel_l1_thresh}")
        CacheState.tc_printed = True

    # Force normal mode
    tiling_mode = 'normal'

    # Parameters
    rel_l1_thresh = CacheState.tc_rel_l1_thresh
    steps = CacheState.tc_steps
    cnt = CacheState.tc_cnt

    # Preparation
    if next(self.parameters()).device.type != 'cuda':
        print("[TeaCache] Moving model to GPU for NORMAL mode...")
        self.cuda()
        torch.cuda.synchronize()

    bs, c, h, w = x.shape
    input_dtype = x.dtype
    input_device = x.device

    patch_size = 2
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size

    if pad_h or pad_w:
        x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="circular")

    img = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size)

    h_len = (h + pad_h) // patch_size
    w_len = (w + pad_w) // patch_size

    img_ids = torch.zeros((h_len, w_len, 3), device=input_device, dtype=input_dtype)
    img_ids[..., 1] = torch.linspace(0, h_len - 1, steps=h_len, device=input_device, dtype=input_dtype)[:, None]
    img_ids[..., 2] = torch.linspace(0, w_len - 1, steps=w_len, device=input_device, dtype=input_dtype)[None, :]
    img_ids = img_ids.expand(bs, -1, -1, -1).reshape(bs, h_len * w_len, 3)

    txt_ids = torch.zeros((bs, context.shape[1], 3), device=input_device, dtype=input_dtype)

    # Embeddings
    img = self.img_in(img)
    vec = self.time_in(timestep_embedding(timestep, 256).to(img.dtype))

    if self.guidance_embed and guidance is not None:
        vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))

    vec = vec + self.vector_in(y)
    txt = self.txt_in(context)
    ids = torch.cat((txt_ids, img_ids), dim=1)
    pe = self.pe_embedder(ids)

    # TeaCache logic
    modulated_inp = img.clone()

    # Checking form match
    if CacheState.tc_previous_modulated_input is not None:
        if CacheState.tc_previous_modulated_input.shape != modulated_inp.shape:
            CacheState.tc_previous_modulated_input = None
            CacheState.tc_previous_residual = None
            CacheState.tc_accumulated_rel_l1_distance = 0.0
            print("  [TeaCache] Shape mismatch, cache reset")

    # Decision: Count or Cache
    should_calc = True
    
    if cnt == 0 or cnt == steps - 1:
        should_calc = True
        CacheState.tc_accumulated_rel_l1_distance = 0.0
    else:
        if CacheState.tc_previous_modulated_input is not None:
            # Polynomial correction
            coefficients = [4.98651651e+02, -2.83781631e+02, 5.58554382e+01, -3.82021401e+00, 2.64230861e-01]
            rescale_func = np.poly1d(coefficients)
            
            rel_l1 = ((modulated_inp - CacheState.tc_previous_modulated_input).abs().mean() / 
                     (CacheState.tc_previous_modulated_input.abs().mean() + 1e-6)).cpu().item()
            
            CacheState.tc_accumulated_rel_l1_distance += rescale_func(rel_l1)
            
            if CacheState.tc_accumulated_rel_l1_distance < rel_l1_thresh:
                should_calc = False
            else:
                should_calc = True
                CacheState.tc_accumulated_rel_l1_distance = 0.0

    CacheState.tc_previous_modulated_input = modulated_inp.clone()
    CacheState.tc_cnt = (cnt + 1) % steps

    # Execution
    can_use_cache = (
        not should_calc and 
        CacheState.tc_previous_residual is not None and
        CacheState.tc_previous_residual.shape == img.shape
    )
    
    if can_use_cache:
        # USING CACHE
        img = img + CacheState.tc_previous_residual.to(img.device)
        print(f"  [TeaCache] Step {cnt}: CACHE (acc={CacheState.tc_accumulated_rel_l1_distance:.4f})")
    else:
        # FULL RUN
        ori_img = img.clone()
        
        # Double blocks
        for block in self.double_blocks:
            img, txt = block(img=img, txt=txt, vec=vec, pe=pe)

        # Single blocks
        img = torch.cat((txt, img), dim=1)
        for block in self.single_blocks:
            img = block(img, vec=vec, pe=pe)
        
        img = img[:, txt.shape[1]:, ...]
        
        # Save the residual
        CacheState.tc_previous_residual = (img - ori_img).detach()
        print(f"  [TeaCache] Step {cnt}: compute (acc={CacheState.tc_accumulated_rel_l1_distance:.4f})")

    # Final layer
    img = self.final_layer(img, vec)

    # Back to spatial
    out = rearrange(img, "b (h w) (c ph pw) -> b c (h ph) (w pw)",
                   h=h_len, w=w_len, ph=patch_size, pw=patch_size)
    return out[:, :, :h, :w].to(input_dtype)