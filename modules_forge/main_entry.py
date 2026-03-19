import os
import sys
import math
import psutil
import gc
import torch
import gradio as gr

from gradio.context import Context
from modules import shared_items, shared, ui_common, sd_models, processing, infotext_utils, paths, ui_loadsave
from backend import memory_management, stream
from backend.args import dynamic_args
from modules.shared import cmd_opts
from modules.sd_models import model_data


total_vram = int(memory_management.total_vram)

ui_forge_preset: gr.Radio = None

ui_checkpoint: gr.Dropdown = None
ui_vae: gr.Dropdown = None
ui_clip_skip: gr.Slider = None

ui_forge_unet_storage_dtype_options: gr.Radio = None
ui_forge_async_loading: gr.Radio = None
ui_forge_pin_shared_memory: gr.Radio = None
ui_forge_inference_memory: gr.Slider = None


forge_unet_storage_dtype_options = {
    'Automatic': (None, False),
    'Automatic (fp16 LoRA)': (None, True),
    'bnb-nf4': ('nf4', False),
    'bnb-nf4 (fp16 LoRA)': ('nf4', True),
    'float8-e4m3fn': (torch.float8_e4m3fn, False),
    'float8-e4m3fn (fp16 LoRA)': (torch.float8_e4m3fn, True),
    'bnb-fp4': ('fp4', False),
    'bnb-fp4 (fp16 LoRA)': ('fp4', True),
    'float8-e5m2': (torch.float8_e5m2, False),
    'float8-e5m2 (fp16 LoRA)': (torch.float8_e5m2, True),
}

module_list = {}


def bind_to_opts(comp, k, save=False, callback=None):
    def on_change(v):
        shared.opts.set(k, v)
        if save:
            shared.opts.save(shared.config_filename)
        if callback is not None:
            callback()
        return

    comp.change(on_change, inputs=[comp], queue=False, show_progress=False)
    return


import os, sys


def restart_forge():
        os.environ["FORGE_NO_BROWSER"] = "1"
        os.execv(sys.executable, [sys.executable] + sys.argv)


# Global Variables for UI Tilting

ui_flux_tiling_mode = None
ui_flux_tile_size = None
ui_flux_tile_overlap = None

def make_flux_tiling_ui():
    """Creates a UI for managing Flux tiling"""
    global ui_flux_tiling_mode, ui_flux_tile_size, ui_flux_tile_overlap

    with gr.Accordion("FLUX Tiling", open=False):
        ui_flux_tiling_mode = gr.Dropdown(
            label="Tiling Mode",
            choices=["auto", "normal", "tiled"],
            value=getattr(shared.opts, "flux_tiling_mode", "normal"),
            info="Auto: VRAM-based| Normal: no tiling | Tiled: Forced tilingо"
        )
        
        ui_flux_tile_size = gr.Slider(
            label="Tile Size (px)",
            minimum=256,
            maximum=1024,
            step=128,
            value=getattr(shared.opts, "flux_tile_size", 384),
        )
        
        ui_flux_tile_overlap = gr.Slider(
            label="Tile Overlap (px)",
            minimum=32,
            maximum=256,
            step=32,
            value=getattr(shared.opts, "flux_tile_overlap", 64),
        )
    
    # Link to settings
    bind_to_opts(ui_flux_tiling_mode, 'flux_tiling_mode', save=True)
    bind_to_opts(ui_flux_tile_size, 'flux_tile_size', save=True)
    bind_to_opts(ui_flux_tile_overlap, 'flux_tile_overlap', save=True)
    
    return ui_flux_tiling_mode, ui_flux_tile_size, ui_flux_tile_overlap


def make_checkpoint_manager_ui():
    global ui_checkpoint, ui_vae, ui_clip_skip, ui_forge_unet_storage_dtype_options
    global ui_forge_async_loading, ui_forge_pin_shared_memory
    global ui_forge_inference_memory, ui_forge_preset

    if shared.opts.sd_model_checkpoint in [None, 'None', 'none', '']:
        if len(sd_models.checkpoints_list) == 0:
            sd_models.list_models()
        if len(sd_models.checkpoints_list) > 0:
            shared.opts.set('sd_model_checkpoint', next(iter(sd_models.checkpoints_list.values())).name)

    ckpt_list, vae_list = refresh_models()

    if not ckpt_list:
        ckpt_list = ["No models found"]
        default_ckpt = "No models found"
    else:
        if shared.opts.sd_model_checkpoint in ckpt_list:
            default_ckpt = shared.opts.sd_model_checkpoint
        else:
            default_ckpt = ckpt_list[0]

    restart_button = ui_common.ToolButton(
        value="🔴",
        elem_id="forge_restart_engine",
        tooltip="Restart Forge"
    )

    safe_switch_btn = ui_common.ToolButton(
        value="🛡️",
        elem_id="forge_safe_switch",
        tooltip="Safe Model Switch (guaranteed model unload first)"
    )

    ui_forge_preset = gr.Dropdown(
        label="UI",
        value=shared.opts.forge_preset,
        choices=[
            'LowVRAM', 'SD Dynamic', 'SD ForgeSD', 'SD -1Gb', 'SD -2Gb', 'SD -3Gb',
            'SDXL Dynamic', 'SDXL ForgeSD', 'SDXL -2Gb', 'SDXL -3Gb',
            'FLUX Dynamic', 'FLUX ForgeSD', 'FLUX -2Gb', 'FLUX -3Gb', 'All'
        ],
        interactive=True,
        filterable=True,
        elem_id="forge_ui_preset"
    )

    ui_checkpoint = gr.Dropdown(
        value=default_ckpt,
        label="Checkpoint",
        elem_classes=['model_selection'],
        choices=ckpt_list,
        interactive=(default_ckpt != "No models found")
    )

    refresh_button = ui_common.ToolButton(
        value=ui_common.refresh_symbol,
        elem_id="forge_refresh_checkpoint",
        tooltip="Refresh"
    )

    clear_memory_button = ui_common.ToolButton(
        value="🧹",
        elem_id="forge_clear_memory",
        tooltip="Clear RAM/VRAM"
    )

    ui_vae = gr.Dropdown(
        value=lambda: [os.path.basename(x) for x in shared.opts.forge_additional_modules],
        multiselect=True,
        label="VAE / Text Encoder",
        render=False,
        choices=vae_list,
        elem_classes=['model_selection']
    )

    def safe_switch_handler(ckpt_name):
        """Wrap with additional protection"""
        try:
            # Double cleaning
            clear_memory()
            deep_memory_purge()
            
            result = checkpoint_change(ckpt_name, save=True, refresh=True)
            
            # Final cleanup if something went wrong
            if not result:
                deep_memory_purge()
                
            return result
        except Exception as e:
            print(f"[SafeSwitch] Failed: {e}")
            deep_memory_purge()  # Recovery
            raise

    def gr_refresh_models():
        a, b = refresh_models()
        if not a:
            a = ["No models found"]
        return gr.update(choices=a), gr.update(choices=b)

    restart_button.click(
        fn=restart_forge,
        inputs=[],
        outputs=[],
        _js="""() => { setTimeout(() => window.location.reload(), 38000); }"""
    )

    refresh_button.click(
        fn=gr_refresh_models,
        inputs=[],
        outputs=[ui_checkpoint, ui_vae],
        show_progress=False,
        queue=False
    )

    clear_memory_button.click(
        fn=clear_memory,
        inputs=[],
        outputs=[],
        show_progress=False,
        queue=False
    )

    Context.root_block.load(
        fn=gr_refresh_models,
        inputs=[],
        outputs=[ui_checkpoint, ui_vae],
        show_progress=False,
        queue=False
    )

    ui_vae.render()

    ui_forge_unet_storage_dtype_options = gr.Dropdown(
        label="Diffusion in Low Bits",
        value=lambda: shared.opts.forge_unet_storage_dtype,
        choices=list(forge_unet_storage_dtype_options.keys())
    )
    bind_to_opts(ui_forge_unet_storage_dtype_options, 'forge_unet_storage_dtype', save=False, callback=refresh_model_loading_parameters)

    ckpt = (shared.opts.sd_model_checkpoint or "").lower()
    default_value = 'Queue' if ckpt.endswith('.gguf') else shared.opts.forge_async_loading

    ui_forge_async_loading = gr.Radio(
        label="Swap Method",
        choices=['Queue', 'Async'],
        value=default_value,
        interactive=not ckpt.endswith('.gguf')
    )

    ui_forge_pin_shared_memory = gr.Radio(
        label="Swap Location",
        value=lambda: shared.opts.forge_pin_shared_memory,
        choices=['CPU', 'Shared']
    )

    ui_forge_inference_memory = gr.Slider(
        label="GPU Weights (MB)",
        value=lambda: total_vram - shared.opts.forge_inference_memory,
        minimum=0,
        maximum=int(memory_management.total_vram),
        step=1
    )

    ui_memory_cleanup_strategy = gr.Dropdown(
        label="Memory Cleanup Strategy",
        value=getattr(shared.opts, "memory_cleanup_strategy", "Smart Purge"),
        choices=["Smart Purge", "Smart", "Always", "Full", "Soft (Cache)", "GPU Cache", "None (Upscale)"],
        interactive=True
    )

    bind_to_opts(
        ui_memory_cleanup_strategy,
        'memory_cleanup_strategy',
        save=True,
        callback=lambda: memory_management.free_memory_strategy(shared.opts.memory_cleanup_strategy)
    )

    ui_lora_computation_strategy = gr.Dropdown(
        label="LoRa computation cast",
        value=getattr(shared.opts, "lora_computation_strategy", "Auto HW"),
        choices=["Auto HW", "Autocast", "float16", "bfloat16", "float32"],
        interactive=True
    )

    ui_flux_tiling_mode, ui_flux_tile_size, ui_flux_tile_overlap = make_flux_tiling_ui()

    bind_to_opts(
        ui_lora_computation_strategy,
        'lora_computation_strategy',
        save=True
    )

    mem_comps = [ui_forge_inference_memory, ui_forge_async_loading, ui_forge_pin_shared_memory]

    ui_forge_inference_memory.change(ui_refresh_memory_management_settings, inputs=mem_comps, queue=False, show_progress=False)
    ui_forge_async_loading.change(ui_refresh_memory_management_settings, inputs=mem_comps, queue=False, show_progress=False)
    ui_forge_pin_shared_memory.change(ui_refresh_memory_management_settings, inputs=mem_comps, queue=False, show_progress=False)

    Context.root_block.load(ui_refresh_memory_management_settings, inputs=mem_comps, queue=False, show_progress=False)

    ui_clip_skip = gr.Slider(
        label="Clip skip",
        value=lambda: shared.opts.CLIP_stop_at_last_layers,
        minimum=1,
        maximum=12,
        step=1
    )
    bind_to_opts(ui_clip_skip, 'CLIP_stop_at_last_layers', save=True)

    def safe_checkpoint_change(ckpt_name):
        """Guaranteed download with notifications"""
        if not ckpt_name or ckpt_name == "No models found":
            return gr.update()
        
        print(f"\n{'='*60}")
        print(f"[SafeSwitch] Target: {ckpt_name}")
        print(f"{'='*60}")
        
        # Memory check
        is_gguf = ckpt_name.lower().endswith('.gguf')
        current_infer = shared.opts.forge_inference_memory
        current_model = total_vram - current_infer
        
        print(f"[SafeSwitch] Memory: model={current_model}MB, inference={current_infer}MB")
        
        # 🟡 YELLOW warning
        if current_model < 2000:
            msg = f"Model memory low ({current_model}MB). Auto-correcting to 5GB..."
            print(f"[WARNING] {msg}")
            gr.Warning(msg)
            
            # Correction...
            target_model = min(5120, int(total_vram * 0.6))
            target_infer = total_vram - target_model
            shared.opts.set('forge_inference_memory', target_infer)
            shared.opts.save(shared.config_filename)
            memory_management.inference_memory = target_infer
            memory_management.current_inference_memory = target_infer * 1024 * 1024
            
            # 🟢 GREEN confirmation
            gr.Info(f"Memory corrected: {target_model}MB for model")
        
        # 🔵 Blue info for GGUF
        if is_gguf:
            gr.Info("GGUF mode: Using quantized model with safe loading")
        
        # === BASIC LOGIC ===
        try:
            # 🔵 Phase 1
            gr.Info("Phase 1: Cleaning memory...")
            deep_memory_purge()
            
            # 🔵 Phase 2
            gr.Info("Phase 2: Loading model...")
            result = checkpoint_change(ckpt_name, save=True, refresh=True)
            
            # Checking the result
            final_infer = shared.opts.forge_inference_memory
            final_model = total_vram - final_infer
            print(f"[SafeSwitch] Final: model={final_model}MB, inference={final_infer}MB")
            
            # 🟢 Success
            gr.Info("✅ Model loaded successfully!")
            
            # 🟡 Warning if memory is still low
            if final_model < 2000:
                gr.Warning(f"Low VRAM: only {final_model}MB for model weights")
            
            async_update = force_queue_if_special_model(ckpt_name)
            
            print(f"{'='*60}")
            print(f"[SafeSwitch] SUCCESS")
            print(f"{'='*60}\n")
            
            # Return ONLY update for async_loading
            return async_update
            
        except Exception as e:
            print(f"\n[SafeSwitch] FAILED: {e}")
            import traceback
            traceback.print_exc()
            
            # 🔴 Red error
            gr.Error(f"Failed to load model: {str(e)[:100]}")
            
            # Emergency cleaning
            try:
                if hasattr(shared, 'sd_model'):
                    shared.sd_model = None
                model_data.loaded_sd_models.clear()
                gc.collect()
                torch.cuda.empty_cache()
                gr.Info("Emergency cleanup completed")  # 🟢
            except:
                pass
            
            return gr.update(value='Queue', interactive=False)

    ui_checkpoint.change(
        fn=safe_checkpoint_change,
        inputs=[ui_checkpoint],
        outputs=[ui_forge_async_loading],
        show_progress=True
    )
    ui_vae.change(modules_change, inputs=[ui_vae], queue=False, show_progress=False)

    return


def clear_memory():
    import gc
    torch.cuda.empty_cache()
    gc.collect()
    print("[Memory] Cleared VRAM and RAM manually.")
	

def find_files_with_extensions(base_path, extensions):
    found_files = {}
    for root, _, files in os.walk(base_path):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                full_path = os.path.join(root, file)
                found_files[file] = full_path
    return found_files


def refresh_models():
    global module_list

    shared_items.refresh_checkpoints()
    ckpt_list = shared_items.list_checkpoint_tiles(shared.opts.sd_checkpoint_dropdown_use_short)

    file_extensions = ['ckpt', 'pt', 'bin', 'safetensors', 'gguf', "pth", "sft"]

    module_list.clear()
    
    module_paths = [
        os.path.abspath(os.path.join(paths.models_path, "VAE")),
        os.path.abspath(os.path.join(paths.models_path, "text_encoder")),
    ]

    if isinstance(shared.cmd_opts.vae_dir, str):
        module_paths.append(os.path.abspath(shared.cmd_opts.vae_dir))
    if isinstance(shared.cmd_opts.text_encoder_dir, str):
        module_paths.append(os.path.abspath(shared.cmd_opts.text_encoder_dir))

    for vae_path in module_paths:
        vae_files = find_files_with_extensions(vae_path, file_extensions)
        module_list.update(vae_files)

    return ckpt_list, module_list.keys()


def ui_refresh_memory_management_settings(model_memory, async_loading, pin_shared_memory):
    """ Passes precalculated 'model_memory' from "GPU Weights" UI slider (skip redundant calculation) """
    refresh_memory_management_settings(
        async_loading=async_loading,
        pin_shared_memory=pin_shared_memory,
        model_memory=model_memory  # Use model_memory directly from UI slider value
    )

def refresh_memory_management_settings(async_loading=None, inference_memory=None, pin_shared_memory=None, model_memory=None):
    # Fallback to defaults if values are not passed
    async_loading = async_loading if async_loading is not None else shared.opts.forge_async_loading
    inference_memory = inference_memory if inference_memory is not None else shared.opts.forge_inference_memory
    pin_shared_memory = pin_shared_memory if pin_shared_memory is not None else shared.opts.forge_pin_shared_memory

    # If model_memory is provided, calculate inference memory accordingly, otherwise use inference_memory directly
    if model_memory is None:
        model_memory = total_vram - inference_memory
    else:
        inference_memory = total_vram - model_memory

    shared.opts.set('forge_async_loading', async_loading)
    shared.opts.set('forge_inference_memory', inference_memory)
    shared.opts.set('forge_pin_shared_memory', pin_shared_memory)

    # Updating internal variables
    stream.async_loading = async_loading
    memory_management.inference_memory = inference_memory
    memory_management.pin_shared_memory = pin_shared_memory

    # print(f"[Forge][Memory] Updated settings: async_loading={async_loading}, inference_memory={inference_memory}, pin_shared_memory={pin_shared_memory}")

    # 💾 Saving to config.json
    shared.opts.save(shared.config_filename)

    stream.stream_activated = async_loading == 'Async'
    memory_management.current_inference_memory = inference_memory * 1024 * 1024  # Convert MB to bytes
    memory_management.PIN_SHARED_MEMORY = pin_shared_memory == 'Shared'

    log_dict = dict(
        stream=stream.should_use_stream(),
        inference_memory=memory_management.minimum_inference_memory() / (1024 * 1024),
        pin_shared_memory=memory_management.PIN_SHARED_MEMORY
    )

    # print(f'Environment vars changed: {log_dict}')

    if inference_memory < min(512, total_vram * 0.05):
        print('------------------')
        print(f'[Low VRAM Warning] You just set Forge to use 100% GPU memory ({model_memory:.2f} MB) to load model weights.')
        print('[Low VRAM Warning] This means you will have 0% GPU memory (0.00 MB) to do matrix computation. Computations may fallback to CPU or go Out of Memory.')
        print('[Low VRAM Warning] In many cases, image generation will be 10x slower.')
        print("[Low VRAM Warning] To solve the problem, you can set the 'GPU Weights' (on the top of page) to a lower value.")
        print("[Low VRAM Warning] If you cannot find 'GPU Weights', you can click the 'all' option in the 'UI' area on the left-top corner of the webpage.")
        print('[Low VRAM Warning] Make sure that you know what you are testing.')
        print('------------------')
    else:
        compute_percentage = (inference_memory / total_vram) * 100.0
        print(f'[GPU Setting] You will use {(100 - compute_percentage):.2f}% GPU memory ({model_memory:.2f} MB) to load weights, and use {compute_percentage:.2f}% GPU memory ({inference_memory:.2f} MB) to do matrix computation.')

    processing.need_global_unload = True
    return


def force_queue_if_special_model(ckpt_name):
    # Safe lowercase conversion
    ckpt_name = ckpt_name.lower() if isinstance(ckpt_name, str) else ""

    # Keywords indicating special models
    special_keywords = ['.gguf', 'kontext', 'fill', 'canny', 'depth']

    # Check for keywords
    is_special = any(keyword in ckpt_name for keyword in special_keywords)

    # Update state and UI
    if is_special:
        shared.opts.set('forge_async_loading', 'Queue')
        stream.async_loading = 'Queue'
        return gr.update(value='Queue', interactive=False)
    else:
        current_value = shared.opts.forge_async_loading or 'Async'
        return gr.update(value=current_value, interactive=True)


def on_prompt_fields_change(prompt_text, negative_text):
    text = f"{prompt_text or ''} {negative_text or ''}".lower()
    if '<lora:' in text:
        # switch and block the radio
        shared.opts.set('forge_async_loading', 'Queue')
        stream.async_loading = 'Queue'
        return gr.update(value='Queue', interactive=False)
    # otherwise - we return the standard logic for special models
    return force_queue_if_special_model(shared.opts.sd_model_checkpoint)


def refresh_model_loading_parameters():
    """With protection against overwriting of GGUF settings by models"""
    from modules.sd_models import select_checkpoint, model_data

    checkpoint_info = select_checkpoint()
    
    # Save the current settings BEFORE making any changes
    saved_model_mem = total_vram - shared.opts.forge_inference_memory
    saved_async = shared.opts.forge_async_loading
    saved_pin = shared.opts.forge_pin_shared_memory
    
    print(f"[RefreshParams] Current settings: model={saved_model_mem}MB, async={saved_async}")

    unet_storage_dtype, lora_fp16 = forge_unet_storage_dtype_options.get(
        shared.opts.forge_unet_storage_dtype, (None, False)
    )

    dynamic_args['online_lora'] = lora_fp16

    model_data.forge_loading_parameters = dict(
        checkpoint_info=checkpoint_info,
        additional_modules=shared.opts.forge_additional_modules,
        unet_storage_dtype=unet_storage_dtype
    )

    # === PROTECTION: Checking if the settings have been reset ===
    if checkpoint_info and any(k in checkpoint_info.filename.lower() for k in ['.gguf', 'kontext', 'fill', 'canny', 'depth']):
        print("[RefreshParams] Special model detected - checking memory settings...")
        
        # Check if the settings have changed
        current_infer = shared.opts.forge_inference_memory
        current_model = total_vram - current_infer
        
        if abs(current_model - saved_model_mem) > 100:  # Difference > 100 MB
            print(f"[RefreshParams] WARNING: Memory changed! Was {saved_model_mem}MB, now {current_model}MB")
            print(f"[RefreshParams] Restoring correct values...")
            
            # Restore the correct values
            shared.opts.set('forge_inference_memory', total_vram - saved_model_mem)
            shared.opts.set('forge_async_loading', saved_async)
            shared.opts.set('forge_pin_shared_memory', saved_pin)
            
            # Updating the backend
            memory_management.inference_memory = total_vram - saved_model_mem
            stream.async_loading = saved_async
            
            shared.opts.save(shared.config_filename)
            print("[RefreshParams] Settings restored")

    print(f'Model selected: {model_data.forge_loading_parameters}')
    print(f'Using online LoRAs in FP16: {lora_fp16}')

    # CPU Fallback with Critical VRAM
    try:
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        max_reserved = torch.cuda.get_device_properties(0).total_memory

        if reserved > max_reserved * 0.95:
            print('[Fallback] VRAM critical. Switching to CPU.')
            shared.opts.set('forge_pin_shared_memory', 'CPU')
            memory_management.PIN_SHARED_MEMORY = True
    except Exception as e:
        print(f'[Fallback] VRAM check error: {e}')

    processing.need_global_unload = True
    return


def deep_memory_purge():
    """Guaranteed cleaning with settings verification"""
    print("[DeepPurge] Starting...")
    
    # Checking current settings
    try:
        infer = shared.opts.forge_inference_memory
        model = total_vram - infer
        print(f"[DeepPurge] Memory config: model={model}MB, inference={infer}MB")
        
        # Warning if something is wrong
        if model < 1000:
            print(f"[DeepPurge] WARNING: Model memory only {model}MB! Check GPU Weights slider.")
    except:
        pass
    
    # 1. Python garbage collector - all generations
    gc.set_threshold(700, 10, 10)  # Aggressive settings
    
    count0 = gc.collect(generation=0)
    count1 = gc.collect(generation=1)
    count2 = gc.collect(generation=2)
    
    print(f"[DeepPurge]   GC gen 0: collected {count0} objects")
    print(f"[DeepPurge]   GC gen 1: collected {count1} objects")
    print(f"[DeepPurge]   GC gen 2: collected {count2} objects")
    
    # 2. PyTorch CUDA cleanup
    if torch.cuda.is_available():
        # Syncing
        torch.cuda.synchronize()
        
        # Cleaning caches
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Reset memory stats
        try:
            torch.cuda.reset_peak_memory_stats()
        except:
            pass
            
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"[DeepPurge]   CUDA: Allocated={allocated:.2f}MB, Reserved={reserved:.2f}MB")
    
    # 3. Windows working set (Windows only)
    if os.name == 'nt':
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            # -1 = current process
            result = kernel32.SetProcessWorkingSetSize(-1, ctypes.c_size_t(-1), ctypes.c_size_t(-1))
            if result:
                print("[DeepPurge]   Windows working set trimmed")
            else:
                print("[DeepPurge]   Windows trim failed (non-critical)")
        except Exception as e:
            print(f"[DeepPurge]   Windows trim error: {e}")
    
    # 4. RAM info
    try:
        ram = psutil.virtual_memory()
        print(f"[DeepPurge]   RAM: Available={ram.available/1024**2:.2f}MB")
    except:
        pass
    
    print("[DeepPurge] Completed")


def checkpoint_change(ckpt_name: str, save=True, refresh=True):
    """Fixed version with correct memory calculation for GGUF"""
    new_ckpt_info = sd_models.get_closet_checkpoint_match(ckpt_name)
    current_ckpt_info = sd_models.get_closet_checkpoint_match(shared.opts.data.get('sd_model_checkpoint', ''))

    if new_ckpt_info == current_ckpt_info:
        return False

    print(f"[Checkpoint] Switching: {current_ckpt_info.name if current_ckpt_info else 'None'} -> {new_ckpt_info.name if new_ckpt_info else 'None'}")

    shared.opts.set('sd_model_checkpoint', ckpt_name)
    
    if ckpt_name and isinstance(ckpt_name, str) and ckpt_name.lower().endswith('.gguf'):
        print("[Checkpoint] GGUF detected - forcing safe mode")
        
        # Force Queue mode
        shared.opts.set('forge_async_loading', 'Queue')
        stream.async_loading = 'Queue'

        # Get the REAL model_memory from the current settings
        # forge_inference_memory = compute memory
        # model_memory = total_vram - forge_inference_memory
        
        current_inference_mem = shared.opts.forge_inference_memory
        correct_model_mem = total_vram - current_inference_mem
        
        print(f"[Checkpoint] GGUF memory setup:")
        print(f"[Checkpoint]   Total VRAM: {total_vram} MB")
        print(f"[Checkpoint]   Inference (computations): {current_inference_mem} MB")
        print(f"[Checkpoint]   Model (weights): {correct_model_mem} MB")
        
        # Pass the CORRECT value
        ui_refresh_memory_management_settings(
            model_memory=correct_model_mem,
            async_loading='Queue',
            pin_shared_memory=shared.opts.forge_pin_shared_memory
        )
        
        print("[Checkpoint] GGUF settings applied")

    if save:
        shared.opts.save(shared.config_filename)

    if processing.need_global_unload:
        import gc
        torch.cuda.empty_cache()
        gc.collect()

    if refresh:
        refresh_model_loading_parameters()

    return force_queue_if_special_model(ckpt_name)


def auto_switch_async_ui(ckpt_name):
    return force_queue_if_special_model(ckpt_name)


def modules_change(module_values:list, save=True, refresh=True) -> bool:
    """ module values may be provided as file paths, or just the module names. Returns True if modules changed. """
    modules = []
    for v in module_values:
        module_name = os.path.basename(v) # If the input is a filepath, extract the file name
        if module_name in module_list:
            modules.append(module_list[module_name])
    
    # skip further processing if value unchanged
    if sorted(modules) == sorted(shared.opts.data.get('forge_additional_modules', [])):
        return False

    shared.opts.set('forge_additional_modules', modules)

    if save:
        shared.opts.save(shared.config_filename)
    if refresh:
        refresh_model_loading_parameters()
    return True


def get_a1111_ui_component(tab, label):
    fields = infotext_utils.paste_fields[tab]['fields']
    for f in fields:
        if f.label == label or f.api == label:
            return f.component


def forge_main_entry():
    ui_txt2img_width = get_a1111_ui_component('txt2img', 'Size-1')
    ui_txt2img_height = get_a1111_ui_component('txt2img', 'Size-2')
    ui_txt2img_cfg = get_a1111_ui_component('txt2img', 'CFG scale')
    ui_txt2img_distilled_cfg = get_a1111_ui_component('txt2img', 'Distilled CFG Scale')
    ui_txt2img_sampler = get_a1111_ui_component('txt2img', 'sampler_name')
    ui_txt2img_scheduler = get_a1111_ui_component('txt2img', 'scheduler')

    ui_img2img_width = get_a1111_ui_component('img2img', 'Size-1')
    ui_img2img_height = get_a1111_ui_component('img2img', 'Size-2')
    ui_img2img_cfg = get_a1111_ui_component('img2img', 'CFG scale')
    ui_img2img_distilled_cfg = get_a1111_ui_component('img2img', 'Distilled CFG Scale')
    ui_img2img_sampler = get_a1111_ui_component('img2img', 'sampler_name')
    ui_img2img_scheduler = get_a1111_ui_component('img2img', 'scheduler')

    ui_txt2img_hr_cfg = get_a1111_ui_component('txt2img', 'Hires CFG Scale')
    ui_txt2img_hr_distilled_cfg = get_a1111_ui_component('txt2img', 'Hires Distilled CFG Scale')
    
    ui_txt2img_steps = get_a1111_ui_component('txt2img', 'Steps')
    ui_img2img_steps = get_a1111_ui_component('img2img', 'Steps')

    ui_prompt = get_a1111_ui_component('txt2img', 'Prompt')
    ui_negative = get_a1111_ui_component('txt2img', 'Negative prompt')

    ui_txt2img_prompt = get_a1111_ui_component('txt2img', 'Prompt')
    ui_txt2img_negative = get_a1111_ui_component('txt2img', 'Negative prompt')
    ui_img2img_prompt = get_a1111_ui_component('img2img', 'Prompt')
    ui_img2img_negative = get_a1111_ui_component('img2img', 'Negative prompt')

    ui_img2img_seed = get_a1111_ui_component('img2img', 'Seed')
    ui_img2img_denoise = get_a1111_ui_component('img2img', 'Denoising strength')
    ui_img2img_image_cfg = get_a1111_ui_component('img2img', 'Image CFG scale')
    
    ui_img2img_0_controlnet_enable_checkbox = get_a1111_ui_component('img2img', 'ControlNet 0 enable')   
    ui_img2img_0_controlnet_enable_accordion = get_a1111_ui_component('img2img', 'ControlNet 0 accordion')   
    ui_img2img_resize_mode = get_a1111_ui_component('img2img', 'ControlNet 0 resize_mode')
    ui_controlnet_weight = get_a1111_ui_component('img2img', 'ControlNet 0 weight')
    ui_cn0_module = get_a1111_ui_component('img2img', 'ControlNet 0 module')
    ui_cn0_model = get_a1111_ui_component('img2img', 'ControlNet 0 model')
    ui_cn0_control_mode = get_a1111_ui_component('img2img', 'ControlNet 0 control_mode')

    ui_freeu_enabled = get_a1111_ui_component('img2img', 'freeu_enabled')
    ui_freeu_b1 = get_a1111_ui_component('img2img', 'freeu_b1')
    ui_freeu_b2 = get_a1111_ui_component('img2img', 'freeu_b2')
    ui_freeu_s1 = get_a1111_ui_component('img2img', 'freeu_s1')
    ui_freeu_s2 = get_a1111_ui_component('img2img', 'freeu_s2')
    ui_freeu_start = get_a1111_ui_component('img2img', 'freeu_start')
    ui_freeu_end = get_a1111_ui_component('img2img', 'freeu_end')

    ui_pagi_enabled = get_a1111_ui_component('img2img', 'pagi_enabled')
    ui_pagi_scale = get_a1111_ui_component('img2img', 'pagi_scale')
    ui_pagi_attenuation = get_a1111_ui_component('img2img', 'pagi_attenuation')
    ui_pagi_start_step = get_a1111_ui_component('img2img', 'pagi_start_step')
    ui_pagi_end_step = get_a1111_ui_component('img2img', 'pagi_end_step')
    
    ui_dynthres_enabled = get_a1111_ui_component('img2img', 'dynthres_enabled')
    ui_dynthres_mimic_scale = get_a1111_ui_component('img2img', 'dynthres_mimic_scale')
    ui_dynthres_threshold_percentile = get_a1111_ui_component('img2img', 'dynthres_threshold_percentile')
    ui_dynthres_mimic_mode = get_a1111_ui_component('img2img', 'dynthres_mimic_mode')
    ui_dynthres_mimic_scale_min = get_a1111_ui_component('img2img', 'dynthres_mimic_scale_min')
    ui_dynthres_cfg_mode = get_a1111_ui_component('img2img', 'dynthres_cfg_mode')
    ui_dynthres_cfg_scale_min = get_a1111_ui_component('img2img', 'dynthres_cfg_scale_min')
    ui_dynthres_sched_val = get_a1111_ui_component('img2img', 'dynthres_sched_val')
    ui_dynthres_separate_feature_channels = get_a1111_ui_component('img2img', 'dynthres_separate_feature_channels')
    ui_dynthres_scaling_startpoint = get_a1111_ui_component('img2img', 'dynthres_scaling_startpoint')
    ui_dynthres_variability_measure = get_a1111_ui_component('img2img', 'dynthres_variability_measure')
    ui_dynthres_interpolate_phi = get_a1111_ui_component('img2img', 'dynthres_interpolate_phi')
    

    # ClarityHD hidden profiles buttons
    clarity_buttons = []
    for i in range(6):
        btn = gr.Button(
            f"Clarity Profile {i+1}",
            elem_id=f"clarity_profile_{i+1}_btn",
            visible=False
        )
        clarity_buttons.append(btn)

    # preset button option
    def clarity_preset_1():
        return [
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
            "masterpiece,best quality,highres,<lora:SD_FX_MoreDetails:0.5>,<lora:SDXLrender_v2.0:0.25>,", #prompt
            "worst quality, low quality, normal quality:2,JuggernautNegative-neg,", #negative
            10, #steps
            "DPM++ 2M", "Align Your Steps", #sampler+scheduler
            8, #CFG
            768, 768, #resolution
            5398475983, #seed
            0.5, #global denoise
            "Just Resize", #mode for img2img
            1.0, #Image CFG scale
            0.75, "tile_resample", "control_v11f1e_sd15_tile [a371b31b]", "Balanced", True, True, #controlnet
            True, 1.05, 1.08, 0.95, 0.8, 0.0, 1.0, #freeU
            True, 1.0, 0.0, 0.0, 1.0, #PAGI
            True, 1, 1, "Power Down", 1, "Power Down", 1, 2, "enable", "MEAN", "AD", 1.0
        ]

    def clarity_preset_2():
        return [
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
            "masterpiece,best quality,highres,<lora:SD_FX_MoreDetails:0.5>,<lora:SDXLrender_v2.0:0.25>,", #prompt
            "worst quality, low quality, normal quality:2,JuggernautNegative-neg,", #negative
            10, #steps
            "DPM++ 2M", "Align Your Steps", #sampler+scheduler
            8, #CFG
            768, 768, #resolution
            5398475983, #seed
            0.5, #global denoise
            "Just Resize", #mode for img2img
            1.0, #Image CFG scale
            0.75, "tile_resample", "control_v11f1e_sd15_tile [a371b31b]", "ControlNet is more important", True, True, #controlnet
            True, 1.05, 1.08, 0.95, 0.8, 0.0, 1.0, #freeU
            True, 1.0, 0.0, 0.0, 1.0, #PAGI
            True, 1, 1, "Power Down", 1, "Power Down", 1, 2, "enable", "MEAN", "AD", 1.0
        ]

    def clarity_preset_3():
        return [
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
            "masterpiece,best quality,highres,<lora:SD_FX_MoreDetails:0.5>,<lora:SDXLrender_v2.0:0.25>,", #prompt
            "worst quality, low quality, normal quality:2,JuggernautNegative-neg,", #negative
            10, #steps
            "DPM++ 2M", "Align Your Steps", #sampler+scheduler
            8, #CFG
            1024, 1024, #resolution
            5398475983, #seed
            0.5, #global denoise
            "Just Resize", #mode for img2img
            1.0, #Image CFG scale
            0.75, "tile_resample", "control_v11f1e_sd15_tile [a371b31b]", "Balanced", True, True, #controlnet
            True, 1.05, 1.08, 0.95, 0.8, 0.0, 1.0, #freeU
            True, 1.0, 0.0, 0.0, 1.0, #PAGI
            True, 1, 1, "Power Down", 1, "Power Down", 1, 2, "enable", "MEAN", "AD", 1.0
        ]

    def clarity_preset_4():
        return [
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
            "masterpiece,best quality,highres,<lora:SD_FX_MoreDetails:0.5>,<lora:SDXLrender_v2.0:0.25>,", #prompt
            "worst quality, low quality, normal quality:2,JuggernautNegative-neg,", #negative
            10, #steps
            "DPM++ 2M", "Align Your Steps", #sampler+scheduler
            8, #CFG
            1024, 1024, #resolution
            5398475983, #seed
            0.5, #global denoise
            "Just Resize", #mode for img2img
            1.0, #Image CFG scale
            0.75, "tile_resample", "control_v11f1e_sd15_tile [a371b31b]", "ControlNet is more important", True, True, #controlnet
            True, 1.05, 1.08, 0.95, 0.8, 0.0, 1.0, #freeU
            True, 1.0, 0.0, 0.0, 1.0, #PAGI
            True, 1, 1, "Power Down", 1, "Power Down", 1, 2, "enable", "MEAN", "AD", 1.0
        ]

    def clarity_preset_5():
        return [
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
            "masterpiece,best quality,highres,<lora:SD_FX_MoreDetails:0.5>,<lora:SDXLrender_v2.0:0.25>,", #prompt
            "worst quality, low quality, normal quality:2,JuggernautNegative-neg,", #negative
            10, #steps
            "DPM++ 2M", "Align Your Steps", #sampler+scheduler
            8, #CFG
            1152, 1152, #resolution
            5398475983, #seed
            0.5, #global denoise
            "Just Resize", #mode for img2img
            1.0, #Image CFG scale
            0.75, "tile_resample", "control_v11f1e_sd15_tile [a371b31b]", "Balanced", True, True, #controlnet
            True, 1.05, 1.08, 0.95, 0.8, 0.0, 1.0, #freeU
            True, 1.0, 0.0, 0.0, 1.0, #PAGI
            True, 1, 1, "Power Down", 1, "Power Down", 1, 2, "enable", "MEAN", "AD", 1.0
        ]

    def clarity_preset_6():
        return [
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
            "masterpiece,best quality,highres,<lora:SD_FX_MoreDetails:0.5>,<lora:SDXLrender_v2.0:0.25>,", #prompt
            "worst quality, low quality, normal quality:2,JuggernautNegative-neg,", #negative
            10, #steps
            "DPM++ 2M", "Align Your Steps", #sampler+scheduler
            8, #CFG
            1152, 1152, #resolution
            5398475983, #seed
            0.5, #global denoise
            "Just Resize", #mode for img2img
            1.0, #Image CFG scale
            0.75, "tile_resample", "control_v11f1e_sd15_tile [a371b31b]", "ControlNet is more important", True, True, #controlnet
            True, 1.05, 1.08, 0.95, 0.8, 0.0, 1.0, #freeU
            True, 1.0, 0.0, 0.0, 1.0, #PAGI
            True, 1, 1, "Power Down", 1, "Power Down", 1, 2, "enable", "MEAN", "AD", 1.0
        ]

    
    # making outputs list
    clarity_outputs = [
        # txt2img
        ui_txt2img_prompt,
        ui_txt2img_negative,
        ui_txt2img_steps,
        ui_txt2img_sampler,
        ui_txt2img_scheduler,
        ui_txt2img_cfg,
        ui_txt2img_width,
        ui_txt2img_height,

        # img2img
        ui_img2img_prompt,
        ui_img2img_negative,
        ui_img2img_steps,
        ui_img2img_sampler,
        ui_img2img_scheduler,
        ui_img2img_cfg,
        ui_img2img_width,
        ui_img2img_height,
    ]

    if ui_img2img_seed:
        clarity_outputs.append(ui_img2img_seed)

    if ui_img2img_denoise:
        clarity_outputs.append(ui_img2img_denoise)

    if ui_img2img_resize_mode:
        clarity_outputs.append(ui_img2img_resize_mode)

    if ui_img2img_image_cfg:
        clarity_outputs.append(ui_img2img_image_cfg)

    # ControlNet Unit 0
    if ui_controlnet_weight:
        clarity_outputs.append(ui_controlnet_weight)

    if ui_cn0_module:
        clarity_outputs.append(ui_cn0_module)

    if ui_cn0_model:
        clarity_outputs.append(ui_cn0_model)

    if ui_cn0_control_mode:
        clarity_outputs.append(ui_cn0_control_mode)
        
    if ui_img2img_0_controlnet_enable_checkbox:
        clarity_outputs.append(ui_img2img_0_controlnet_enable_checkbox)
        
    if ui_img2img_0_controlnet_enable_accordion:
        clarity_outputs.append(ui_img2img_0_controlnet_enable_accordion)           

    #FreeU
    if ui_freeu_enabled:
        clarity_outputs.append(ui_freeu_enabled)

    if ui_freeu_b1:
        clarity_outputs.append(ui_freeu_b1)

    if ui_freeu_b2:
        clarity_outputs.append(ui_freeu_b2)

    if ui_freeu_s1:
        clarity_outputs.append(ui_freeu_s1)

    if ui_freeu_s2:
        clarity_outputs.append(ui_freeu_s2)

    if ui_freeu_start:
        clarity_outputs.append(ui_freeu_start)

    if ui_freeu_end:
        clarity_outputs.append(ui_freeu_end)

    # PAGI
    if ui_pagi_enabled:
        clarity_outputs.append(ui_pagi_enabled)
        
    if ui_pagi_scale:
        clarity_outputs.append(ui_pagi_scale)

    if ui_pagi_attenuation:
        clarity_outputs.append(ui_pagi_attenuation)

    if ui_pagi_start_step:
        clarity_outputs.append(ui_pagi_start_step)

    if ui_pagi_end_step:
        clarity_outputs.append(ui_pagi_end_step)
    
    #Dynamic CFG
    if ui_dynthres_enabled:
        clarity_outputs.append(ui_dynthres_enabled)
    
    if ui_dynthres_mimic_scale:
        clarity_outputs.append(ui_dynthres_mimic_scale)
    
    if ui_dynthres_threshold_percentile:
        clarity_outputs.append(ui_dynthres_threshold_percentile)
    
    if ui_dynthres_mimic_mode:
        clarity_outputs.append(ui_dynthres_mimic_mode)
           
    if ui_dynthres_mimic_scale_min:
        clarity_outputs.append(ui_dynthres_mimic_scale_min)
           
    if ui_dynthres_cfg_mode:
        clarity_outputs.append(ui_dynthres_cfg_mode)
           
    if ui_dynthres_cfg_scale_min:
        clarity_outputs.append(ui_dynthres_cfg_scale_min)
           
    if ui_dynthres_sched_val:
        clarity_outputs.append(ui_dynthres_sched_val)
           
    if ui_dynthres_separate_feature_channels:
        clarity_outputs.append(ui_dynthres_separate_feature_channels)
           
    if ui_dynthres_scaling_startpoint:
        clarity_outputs.append(ui_dynthres_scaling_startpoint)
           
    if ui_dynthres_variability_measure:
        clarity_outputs.append(ui_dynthres_variability_measure)
           
    if ui_dynthres_interpolate_phi:
        clarity_outputs.append(ui_dynthres_interpolate_phi)
    

    #ClarityHD profile buttons
    clarity_buttons[0].click(fn=clarity_preset_1, inputs=[], outputs=clarity_outputs)
    clarity_buttons[1].click(fn=clarity_preset_2, inputs=[], outputs=clarity_outputs)
    clarity_buttons[2].click(fn=clarity_preset_3, inputs=[], outputs=clarity_outputs)
    clarity_buttons[3].click(fn=clarity_preset_4, inputs=[], outputs=clarity_outputs)
    clarity_buttons[4].click(fn=clarity_preset_5, inputs=[], outputs=clarity_outputs)
    clarity_buttons[5].click(fn=clarity_preset_6, inputs=[], outputs=clarity_outputs)


    output_targets = [
        ui_vae,
        ui_clip_skip,
        ui_forge_unet_storage_dtype_options,
        ui_forge_async_loading,
        ui_forge_pin_shared_memory,
        ui_forge_inference_memory,
        ui_txt2img_width,
        ui_img2img_width,
        ui_txt2img_height,
        ui_img2img_height,
        ui_txt2img_cfg,
        ui_img2img_cfg,
        ui_txt2img_distilled_cfg,
        ui_img2img_distilled_cfg,
        ui_txt2img_sampler,
        ui_img2img_sampler,
        ui_txt2img_scheduler,
        ui_img2img_scheduler,
        ui_txt2img_hr_cfg,
        ui_txt2img_hr_distilled_cfg,
        ui_txt2img_steps,
        ui_img2img_steps,
    ]

    ui_forge_preset.change(on_preset_change, inputs=[ui_forge_preset], outputs=output_targets, queue=False, show_progress=False)
    ui_forge_preset.change(js="clickLoraRefresh", fn=None, queue=False, show_progress=False)
    Context.root_block.load(on_preset_change, inputs=None, outputs=output_targets, queue=False, show_progress=False)

    for comp in (ui_prompt, ui_negative):
        comp.change(
            fn=on_prompt_fields_change,
            inputs=[ui_prompt, ui_negative],
            outputs=[ui_forge_async_loading],
            queue=False,
            show_progress=False
        )

    Context.root_block.load(
        fn=on_prompt_fields_change,
        inputs=[ui_prompt, ui_negative],
        outputs=[ui_forge_async_loading],
        queue=False,
        show_progress=False
    )

    refresh_model_loading_parameters()
    return


def is_special_model(ckpt_name):
    ckpt_name = ckpt_name.lower() if isinstance(ckpt_name, str) else ""
    return any(k in ckpt_name for k in ['.gguf', 'kontext', 'fill', 'canny', 'depth'])


def on_preset_change(preset=None):
    if preset is not None:
        shared.opts.set('forge_preset', preset)
        shared.opts.save(shared.config_filename)

    if preset == 'LowVRAM':
        ckpt = shared.opts.sd_model_checkpoint

        return [
            gr.update(visible=True),  # ui_vae
            gr.update(visible=False), # ui_clip_skip
            gr.update(visible=True, value='bnb-nf4'),  # low-bit
            gr.update(visible=True, value='Queue'),
            gr.update(visible=False, value='CPU'),
            gr.update(visible=True, value=int(total_vram * 0.5)),
            gr.update(value=getattr(shared.opts, "flux_t2i_width", 672)),               # ui_txt2img_width
            gr.update(value=getattr(shared.opts, "flux_i2i_width", 672)),              # ui_img2img_width
            gr.update(value=getattr(shared.opts, "flux_t2i_height", 1200)),             # ui_txt2img_height
            gr.update(value=getattr(shared.opts, "flux_i2i_height", 1200)),             # ui_img2img_height
            gr.update(value=getattr(shared.opts, "flux_t2i_cfg", 1)),                   # ui_txt2img_cfg
            gr.update(value=getattr(shared.opts, "flux_i2i_cfg", 1)),                   # ui_img2img_cfg
            gr.update(visible=True, value=getattr(shared.opts, "flux_t2i_d_cfg", 3.5)), # ui_txt2img_distilled_cfg
            gr.update(visible=True, value=getattr(shared.opts, "flux_i2i_d_cfg", 3.5)), # ui_img2img_distilled_cfg
            gr.update(value=getattr(shared.opts, "flux_t2i_sampler", 'Euler')),         # ui_txt2img_sampler
            gr.update(value=getattr(shared.opts, "flux_i2i_sampler", 'Euler')),         # ui_img2img_sampler
            gr.update(value=getattr(shared.opts, "flux_t2i_scheduler", 'Simple')),      # ui_txt2img_scheduler
            gr.update(value=getattr(shared.opts, "flux_i2i_scheduler", 'Simple')),      # ui_img2img_scheduler
            gr.update(visible=True, value=getattr(shared.opts, "flux_t2i_hr_cfg", 1.0)),    # ui_txt2img_hr_cfg
            gr.update(visible=True, value=getattr(shared.opts, "flux_t2i_hr_d_cfg", 3.5)),  # ui_txt2img_hr_distilled_cfg
            gr.update(value=getattr(shared.opts, "flux_t2i_steps", 20)),  # ui_txt2img_steps
            gr.update(value=getattr(shared.opts, "flux_i2i_steps", 20)),  # ui_img2img_steps
    ]

    if shared.opts.forge_preset == 'SD Dynamic':
        ckpt = shared.opts.sd_model_checkpoint

        return [
            gr.update(visible=True),                                                    # ui_vae
            gr.update(visible=True, value=1),                                           # ui_clip_skip
            gr.update(visible=False, value='Automatic'),                                # ui_forge_unet_storage_dtype_options
            gr.update(visible=True, value='Queue'),                      # ui_forge_async_loading
            gr.update(visible=False, value='CPU'),                                      # ui_forge_pin_shared_memory
            gr.update(visible=True, value=max(int(total_vram * 0.6), 512)),            # ui_forge_inference_memory
            gr.update(value=getattr(shared.opts, "sd_t2i_width", 672)),                 # ui_txt2img_width
            gr.update(value=getattr(shared.opts, "sd_i2i_width", 672)),                 # ui_img2img_width
            gr.update(value=getattr(shared.opts, "sd_t2i_height", 1200)),                # ui_txt2img_height
            gr.update(value=getattr(shared.opts, "sd_i2i_height", 1200)),                # ui_img2img_height
            gr.update(value=getattr(shared.opts, "sd_t2i_cfg", 7)),                     # ui_txt2img_cfg
            gr.update(value=getattr(shared.opts, "sd_i2i_cfg", 7)),                     # ui_img2img_cfg
            gr.update(visible=False, value=3.5),                                        # ui_txt2img_distilled_cfg
            gr.update(visible=False, value=3.5),                                        # ui_img2img_distilled_cfg
            gr.update(value=getattr(shared.opts, "sd_t2i_sampler", 'Euler_Dy')),    # ui_txt2img_sampler
            gr.update(value=getattr(shared.opts, "sd_i2i_sampler", 'DPM++ 2M')),    # ui_img2img_sampler
            gr.update(value=getattr(shared.opts, "sd_t2i_scheduler", 'Exponential')),     # ui_txt2img_scheduler
            gr.update(value=getattr(shared.opts, "sd_i2i_scheduler", 'Align Your Steps')),     # ui_img2img_scheduler
            gr.update(visible=True, value=getattr(shared.opts, "sd_t2i_hr_cfg", 7.0)),  # ui_txt2img_hr_cfg
            gr.update(visible=False, value=3.5),                                        # ui_txt2img_hr_distilled_cfg
            gr.update(value=getattr(shared.opts, "sd_t2i_steps", 30)),  # ui_txt2img_steps
            gr.update(value=getattr(shared.opts, "sd_i2i_steps", 20)),  # ui_img2img_steps
        ]

    if shared.opts.forge_preset == 'SD ForgeSD':
        ckpt = shared.opts.sd_model_checkpoint

        return [
            gr.update(visible=True),                                                    # ui_vae
            gr.update(visible=True, value=1),                                           # ui_clip_skip
            gr.update(visible=False, value='Automatic'),                                # ui_forge_unet_storage_dtype_options
            gr.update(visible=True, value='Queue'),                                     # ui_forge_async_loading
            gr.update(visible=False, value='CPU'),                                      # ui_forge_pin_shared_memory
            gr.update(visible=False, value=total_vram - 1024),                          # ui_forge_inference_memory
            gr.update(value=getattr(shared.opts, "sd_t2i_width", 512)),                 # ui_txt2img_width
            gr.update(value=getattr(shared.opts, "sd_i2i_width", 512)),                 # ui_img2img_width
            gr.update(value=getattr(shared.opts, "sd_t2i_height", 640)),                # ui_txt2img_height
            gr.update(value=getattr(shared.opts, "sd_i2i_height", 512)),                # ui_img2img_height
            gr.update(value=getattr(shared.opts, "sd_t2i_cfg", 7)),                     # ui_txt2img_cfg
            gr.update(value=getattr(shared.opts, "sd_i2i_cfg", 7)),                     # ui_img2img_cfg
            gr.update(visible=False, value=3.5),                                        # ui_txt2img_distilled_cfg
            gr.update(visible=False, value=3.5),                                        # ui_img2img_distilled_cfg
            gr.update(value=getattr(shared.opts, "sd_t2i_sampler", 'Euler a')),         # ui_txt2img_sampler
            gr.update(value=getattr(shared.opts, "sd_i2i_sampler", 'Euler a')),         # ui_img2img_sampler
            gr.update(value=getattr(shared.opts, "sd_t2i_scheduler", 'Automatic')),     # ui_txt2img_scheduler
            gr.update(value=getattr(shared.opts, "sd_i2i_scheduler", 'Automatic')),     # ui_img2img_scheduler
            gr.update(visible=True, value=getattr(shared.opts, "sd_t2i_hr_cfg", 7.0)),  # ui_txt2img_hr_cfg
            gr.update(visible=False, value=3.5),                                        # ui_txt2img_hr_distilled_cfg
            gr.update(value=getattr(shared.opts, "sd_t2i_steps", 20)),  # ui_txt2img_steps
            gr.update(value=getattr(shared.opts, "sd_i2i_steps", 20)),  # ui_img2img_steps
        ]

    if shared.opts.forge_preset == 'SD -1Gb':
        ckpt = shared.opts.sd_model_checkpoint

        return [
            gr.update(visible=True),                                                    # ui_vae
            gr.update(visible=True, value=1),                                           # ui_clip_skip
            gr.update(visible=False, value='Automatic'),                                # ui_forge_unet_storage_dtype_options
            gr.update(visible=True, value='Queue'),                                     # ui_forge_async_loading
            gr.update(visible=False, value='CPU'),                                      # ui_forge_pin_shared_memory
            gr.update(visible=True, value=total_vram - 1024),                          # ui_forge_inference_memory
            gr.update(value=getattr(shared.opts, "sd_t2i_width", 672)),                 # ui_txt2img_width
            gr.update(value=getattr(shared.opts, "sd_i2i_width", 672)),                 # ui_img2img_width
            gr.update(value=getattr(shared.opts, "sd_t2i_height", 1200)),                # ui_txt2img_height
            gr.update(value=getattr(shared.opts, "sd_i2i_height", 1200)),                # ui_img2img_height
            gr.update(value=getattr(shared.opts, "sd_t2i_cfg", 7)),                     # ui_txt2img_cfg
            gr.update(value=getattr(shared.opts, "sd_i2i_cfg", 7)),                     # ui_img2img_cfg
            gr.update(visible=False, value=3.5),                                        # ui_txt2img_distilled_cfg
            gr.update(visible=False, value=3.5),                                        # ui_img2img_distilled_cfg
            gr.update(value=getattr(shared.opts, "sd_t2i_sampler", 'Euler_Dy')),    # ui_txt2img_sampler
            gr.update(value=getattr(shared.opts, "sd_i2i_sampler", 'DPM++ 2M')),    # ui_img2img_sampler
            gr.update(value=getattr(shared.opts, "sd_t2i_scheduler", 'Exponential')),     # ui_txt2img_scheduler
            gr.update(value=getattr(shared.opts, "sd_i2i_scheduler", 'Align Your Steps')),     # ui_img2img_scheduler
            gr.update(visible=True, value=getattr(shared.opts, "sd_t2i_hr_cfg", 7.0)),  # ui_txt2img_hr_cfg
            gr.update(visible=False, value=3.5),                                        # ui_txt2img_hr_distilled_cfg
            gr.update(value=getattr(shared.opts, "sd_t2i_steps", 30)),  # ui_txt2img_steps
            gr.update(value=getattr(shared.opts, "sd_i2i_steps", 20)),  # ui_img2img_steps
        ]

    if shared.opts.forge_preset == 'SD -2Gb':
        ckpt = shared.opts.sd_model_checkpoint

        return [
            gr.update(visible=True),                                                    # ui_vae
            gr.update(visible=True, value=1),                                           # ui_clip_skip
            gr.update(visible=False, value='Automatic'),                                # ui_forge_unet_storage_dtype_options
            gr.update(visible=True, value='Queue'),                                     # ui_forge_async_loading
            gr.update(visible=False, value='CPU'),                                      # ui_forge_pin_shared_memory
            gr.update(visible=True, value=total_vram - 2048),                           # ui_forge_inference_memory
            gr.update(value=getattr(shared.opts, "sd_t2i_width", 672)),                 # ui_txt2img_width
            gr.update(value=getattr(shared.opts, "sd_i2i_width", 672)),                 # ui_img2img_width
            gr.update(value=getattr(shared.opts, "sd_t2i_height", 1200)),                # ui_txt2img_height
            gr.update(value=getattr(shared.opts, "sd_i2i_height", 1200)),                # ui_img2img_height
            gr.update(value=getattr(shared.opts, "sd_t2i_cfg", 7)),                     # ui_txt2img_cfg
            gr.update(value=getattr(shared.opts, "sd_i2i_cfg", 7)),                     # ui_img2img_cfg
            gr.update(visible=False, value=3.5),                                        # ui_txt2img_distilled_cfg
            gr.update(visible=False, value=3.5),                                        # ui_img2img_distilled_cfg
            gr.update(value=getattr(shared.opts, "sd_t2i_sampler", 'Euler_Dy')),    # ui_txt2img_sampler
            gr.update(value=getattr(shared.opts, "sd_i2i_sampler", 'DPM++ 2M')),    # ui_img2img_sampler
            gr.update(value=getattr(shared.opts, "sd_t2i_scheduler", 'Exponential')),     # ui_txt2img_scheduler
            gr.update(value=getattr(shared.opts, "sd_i2i_scheduler", 'Align Your Steps')),     # ui_img2img_scheduler
            gr.update(visible=True, value=getattr(shared.opts, "sd_t2i_hr_cfg", 7.0)),  # ui_txt2img_hr_cfg
            gr.update(visible=False, value=3.5),                                        # ui_txt2img_hr_distilled_cfg
            gr.update(value=getattr(shared.opts, "sd_t2i_steps", 30)),  # ui_txt2img_steps
            gr.update(value=getattr(shared.opts, "sd_i2i_steps", 20)),  # ui_img2img_steps
        ]

    if shared.opts.forge_preset == 'SD -3Gb':
        ckpt = shared.opts.sd_model_checkpoint

        return [
            gr.update(visible=True),                                                    # ui_vae
            gr.update(visible=True, value=1),                                           # ui_clip_skip
            gr.update(visible=False, value='Automatic'),                                # ui_forge_unet_storage_dtype_options
            gr.update(visible=True, value='Queue'),                                     # ui_forge_async_loading
            gr.update(visible=False, value='CPU'),                                      # ui_forge_pin_shared_memory
            gr.update(visible=True, value=total_vram - 2704),                            # ui_forge_inference_memory
            gr.update(value=getattr(shared.opts, "sd_t2i_width", 672)),                 # ui_txt2img_width
            gr.update(value=getattr(shared.opts, "sd_i2i_width", 672)),                 # ui_img2img_width
            gr.update(value=getattr(shared.opts, "sd_t2i_height", 1200)),                # ui_txt2img_height
            gr.update(value=getattr(shared.opts, "sd_i2i_height", 1200)),                # ui_img2img_height
            gr.update(value=getattr(shared.opts, "sd_t2i_cfg", 7)),                     # ui_txt2img_cfg
            gr.update(value=getattr(shared.opts, "sd_i2i_cfg", 7)),                     # ui_img2img_cfg
            gr.update(visible=False, value=3.5),                                        # ui_txt2img_distilled_cfg
            gr.update(visible=False, value=3.5),                                        # ui_img2img_distilled_cfg
            gr.update(value=getattr(shared.opts, "sd_t2i_sampler", 'Euler_Dy')),    # ui_txt2img_sampler
            gr.update(value=getattr(shared.opts, "sd_i2i_sampler", 'DPM++ 2M')),    # ui_img2img_sampler
            gr.update(value=getattr(shared.opts, "sd_t2i_scheduler", 'Exponential')),     # ui_txt2img_scheduler
            gr.update(value=getattr(shared.opts, "sd_i2i_scheduler", 'Align Your Steps')),     # ui_img2img_scheduler
            gr.update(visible=True, value=getattr(shared.opts, "sd_t2i_hr_cfg", 7.0)),  # ui_txt2img_hr_cfg
            gr.update(visible=False, value=3.5),                                        # ui_txt2img_hr_distilled_cfg
            gr.update(value=getattr(shared.opts, "sd_t2i_steps", 30)),  # ui_txt2img_steps
            gr.update(value=getattr(shared.opts, "sd_i2i_steps", 20)),  # ui_img2img_steps
        ]

    if shared.opts.forge_preset == 'SDXL Dynamic':
        ckpt = shared.opts.sd_model_checkpoint

        return [
            gr.update(visible=True),                                                    # ui_vae
            gr.update(visible=True, value=1),                                          # ui_clip_skip
            gr.update(visible=True, value='Automatic'),                                 # ui_forge_unet_storage_dtype_options
            gr.update(visible=True, value='Queue'),                                     # ui_forge_async_loading
            gr.update(visible=False, value='CPU'),                                      # ui_forge_pin_shared_memory
            gr.update(visible=True, value=max(int(total_vram * 0.6), 512)),            # ui_forge_inference_memory
            gr.update(value=getattr(shared.opts, "xl_t2i_width", 672)),                 # ui_txt2img_width
            gr.update(value=getattr(shared.opts, "xl_i2i_width", 672)),                # ui_img2img_width
            gr.update(value=getattr(shared.opts, "xl_t2i_height", 1200)),               # ui_txt2img_height
            gr.update(value=getattr(shared.opts, "xl_i2i_height", 1200)),               # ui_img2img_height
            gr.update(value=getattr(shared.opts, "xl_t2i_cfg", 5)),                     # ui_txt2img_cfg
            gr.update(value=getattr(shared.opts, "xl_i2i_cfg", 5)),                     # ui_img2img_cfg
            gr.update(visible=False, value=3.5),                                        # ui_txt2img_distilled_cfg
            gr.update(visible=False, value=3.5),                                        # ui_img2img_distilled_cfg
            gr.update(value=getattr(shared.opts, "xl_t2i_sampler", 'Euler_Dy')),         # ui_txt2img_sampler
            gr.update(value=getattr(shared.opts, "xl_i2i_sampler", 'Euler')),         # ui_img2img_sampler
            gr.update(value=getattr(shared.opts, "xl_t2i_scheduler", 'Exponential')),     # ui_txt2img_scheduler
            gr.update(value=getattr(shared.opts, "xl_i2i_scheduler", 'Karras')),     # ui_img2img_scheduler
            gr.update(visible=True, value=getattr(shared.opts, "xl_t2i_hr_cfg", 5.0)),  # ui_txt2img_hr_cfg
            gr.update(visible=False, value=3.5),                                        # ui_txt2img_hr_distilled_cfg
            gr.update(value=getattr(shared.opts, "xl_t2i_steps", 30)),  # ui_txt2img_steps
            gr.update(value=getattr(shared.opts, "xl_i2i_steps", 20)),  # ui_img2img_steps
        ]

    if shared.opts.forge_preset == 'SDXL ForgeSD':
        ckpt = shared.opts.sd_model_checkpoint
        model_mem = getattr(shared.opts, "xl_GPU_MB", total_vram - 1024)
        if model_mem < 0 or model_mem > total_vram:
            model_mem = total_vram - 1024
        return [
            gr.update(visible=True),                                                    # ui_vae
            gr.update(visible=False, value=1),                                          # ui_clip_skip
            gr.update(visible=True, value='Automatic'),                                 # ui_forge_unet_storage_dtype_options
            gr.update(visible=True, value='Queue'),                                     # ui_forge_async_loading
            gr.update(visible=False, value='CPU'),                                      # ui_forge_pin_shared_memory
            gr.update(visible=True, value=model_mem),                                   # ui_forge_inference_memory
            gr.update(value=getattr(shared.opts, "xl_t2i_width", 896)),                 # ui_txt2img_width
            gr.update(value=getattr(shared.opts, "xl_i2i_width", 1024)),                # ui_img2img_width
            gr.update(value=getattr(shared.opts, "xl_t2i_height", 1152)),               # ui_txt2img_height
            gr.update(value=getattr(shared.opts, "xl_i2i_height", 1024)),               # ui_img2img_height
            gr.update(value=getattr(shared.opts, "xl_t2i_cfg", 5)),                     # ui_txt2img_cfg
            gr.update(value=getattr(shared.opts, "xl_i2i_cfg", 5)),                     # ui_img2img_cfg
            gr.update(visible=False, value=3.5),                                        # ui_txt2img_distilled_cfg
            gr.update(visible=False, value=3.5),                                        # ui_img2img_distilled_cfg
            gr.update(value=getattr(shared.opts, "xl_t2i_sampler", 'Euler a')),         # ui_txt2img_sampler
            gr.update(value=getattr(shared.opts, "xl_i2i_sampler", 'Euler')),         # ui_img2img_sampler
            gr.update(value=getattr(shared.opts, "xl_t2i_scheduler", 'Automatic')),     # ui_txt2img_scheduler
            gr.update(value=getattr(shared.opts, "xl_i2i_scheduler", 'Karras')),     # ui_img2img_scheduler
            gr.update(visible=True, value=getattr(shared.opts, "xl_t2i_hr_cfg", 5.0)),  # ui_txt2img_hr_cfg
            gr.update(visible=False, value=3.5),                                        # ui_txt2img_hr_distilled_cfg
            gr.update(value=getattr(shared.opts, "xl_t2i_steps", 20)),  # ui_txt2img_steps
            gr.update(value=getattr(shared.opts, "xl_i2i_steps", 20)),  # ui_img2img_steps
        ]

    if shared.opts.forge_preset == 'SDXL -2Gb':
        ckpt = shared.opts.sd_model_checkpoint

        return [
            gr.update(visible=True),                                                    # ui_vae
            gr.update(visible=True, value=1),                                          # ui_clip_skip
            gr.update(visible=True, value='Automatic'),                                 # ui_forge_unet_storage_dtype_options
            gr.update(visible=True, value='Queue'),                                     # ui_forge_async_loading
            gr.update(visible=False, value='CPU'),                                      # ui_forge_pin_shared_memory
            gr.update(visible=True, value=total_vram - 2048),                                   # ui_forge_inference_memory
            gr.update(value=getattr(shared.opts, "xl_t2i_width", 672)),                 # ui_txt2img_width
            gr.update(value=getattr(shared.opts, "xl_i2i_width", 672)),                # ui_img2img_width
            gr.update(value=getattr(shared.opts, "xl_t2i_height", 1200)),               # ui_txt2img_height
            gr.update(value=getattr(shared.opts, "xl_i2i_height", 1200)),               # ui_img2img_height
            gr.update(value=getattr(shared.opts, "xl_t2i_cfg", 5)),                     # ui_txt2img_cfg
            gr.update(value=getattr(shared.opts, "xl_i2i_cfg", 5)),                     # ui_img2img_cfg
            gr.update(visible=False, value=3.5),                                        # ui_txt2img_distilled_cfg
            gr.update(visible=False, value=3.5),                                        # ui_img2img_distilled_cfg
            gr.update(value=getattr(shared.opts, "xl_t2i_sampler", 'Euler_Dy')),         # ui_txt2img_sampler
            gr.update(value=getattr(shared.opts, "xl_i2i_sampler", 'Euler')),         # ui_img2img_sampler
            gr.update(value=getattr(shared.opts, "xl_t2i_scheduler", 'Exponential')),     # ui_txt2img_scheduler
            gr.update(value=getattr(shared.opts, "xl_i2i_scheduler", 'Karras')),     # ui_img2img_scheduler
            gr.update(visible=True, value=getattr(shared.opts, "xl_t2i_hr_cfg", 5.0)),  # ui_txt2img_hr_cfg
            gr.update(visible=False, value=3.5),                                        # ui_txt2img_hr_distilled_cfg
            gr.update(value=getattr(shared.opts, "xl_t2i_steps", 30)),  # ui_txt2img_steps
            gr.update(value=getattr(shared.opts, "xl_i2i_steps", 20)),  # ui_img2img_steps
        ]

    if shared.opts.forge_preset == 'SDXL -3Gb':
        ckpt = shared.opts.sd_model_checkpoint

        return [
            gr.update(visible=True),                                                    # ui_vae
            gr.update(visible=True, value=1),                                          # ui_clip_skip
            gr.update(visible=True, value='Automatic'),                                 # ui_forge_unet_storage_dtype_options
            gr.update(visible=True, value='Queue'),                                     # ui_forge_async_loading
            gr.update(visible=False, value='CPU'),                                      # ui_forge_pin_shared_memory
            gr.update(visible=True, value=total_vram - 2704),                                   # ui_forge_inference_memory
            gr.update(value=getattr(shared.opts, "xl_t2i_width", 672)),                 # ui_txt2img_width
            gr.update(value=getattr(shared.opts, "xl_i2i_width", 672)),                # ui_img2img_width
            gr.update(value=getattr(shared.opts, "xl_t2i_height", 1200)),               # ui_txt2img_height
            gr.update(value=getattr(shared.opts, "xl_i2i_height", 1200)),               # ui_img2img_height
            gr.update(value=getattr(shared.opts, "xl_t2i_cfg", 5)),                     # ui_txt2img_cfg
            gr.update(value=getattr(shared.opts, "xl_i2i_cfg", 5)),                     # ui_img2img_cfg
            gr.update(visible=False, value=3.5),                                        # ui_txt2img_distilled_cfg
            gr.update(visible=False, value=3.5),                                        # ui_img2img_distilled_cfg
            gr.update(value=getattr(shared.opts, "xl_t2i_sampler", 'Euler_Dy')),         # ui_txt2img_sampler
            gr.update(value=getattr(shared.opts, "xl_i2i_sampler", 'Euler')),         # ui_img2img_sampler
            gr.update(value=getattr(shared.opts, "xl_t2i_scheduler", 'Exponential')),     # ui_txt2img_scheduler
            gr.update(value=getattr(shared.opts, "xl_i2i_scheduler", 'Karras')),     # ui_img2img_scheduler
            gr.update(visible=True, value=getattr(shared.opts, "xl_t2i_hr_cfg", 5.0)),  # ui_txt2img_hr_cfg
            gr.update(visible=False, value=3.5),                                        # ui_txt2img_hr_distilled_cfg
            gr.update(value=getattr(shared.opts, "xl_t2i_steps", 30)),  # ui_txt2img_steps
            gr.update(value=getattr(shared.opts, "xl_i2i_steps", 20)),  # ui_img2img_steps
        ]

    if shared.opts.forge_preset == 'FLUX Dynamic':
        ckpt = shared.opts.sd_model_checkpoint

        return [
            gr.update(visible=True),                                                    # ui_vae
            gr.update(visible=False, value=1),                                          # ui_clip_skip
            gr.update(visible=True, value='Automatic'),                                 # ui_forge_unet_storage_dtype_options
            gr.update(visible=True, value='Queue'),                                     # ui_forge_async_loading
            gr.update(visible=False, value='CPU'),                                       # ui_forge_pin_shared_memory
            gr.update(visible=True, value=max(int(total_vram * 0.6), 512)),             # ui_forge_inference_memory
            gr.update(value=getattr(shared.opts, "flux_t2i_width", 672)),               # ui_txt2img_width
            gr.update(value=getattr(shared.opts, "flux_i2i_width", 672)),              # ui_img2img_width
            gr.update(value=getattr(shared.opts, "flux_t2i_height", 1200)),             # ui_txt2img_height
            gr.update(value=getattr(shared.opts, "flux_i2i_height", 1200)),             # ui_img2img_height
            gr.update(value=getattr(shared.opts, "flux_t2i_cfg", 1)),                   # ui_txt2img_cfg
            gr.update(value=getattr(shared.opts, "flux_i2i_cfg", 1)),                   # ui_img2img_cfg
            gr.update(visible=True, value=getattr(shared.opts, "flux_t2i_d_cfg", 3.5)), # ui_txt2img_distilled_cfg
            gr.update(visible=True, value=getattr(shared.opts, "flux_i2i_d_cfg", 3.5)), # ui_img2img_distilled_cfg
            gr.update(value=getattr(shared.opts, "flux_t2i_sampler", 'Euler')),         # ui_txt2img_sampler
            gr.update(value=getattr(shared.opts, "flux_i2i_sampler", 'Euler')),         # ui_img2img_sampler
            gr.update(value=getattr(shared.opts, "flux_t2i_scheduler", 'Simple')),      # ui_txt2img_scheduler
            gr.update(value=getattr(shared.opts, "flux_i2i_scheduler", 'Simple')),      # ui_img2img_scheduler
            gr.update(visible=True, value=getattr(shared.opts, "flux_t2i_hr_cfg", 1.0)),    # ui_txt2img_hr_cfg
            gr.update(visible=True, value=getattr(shared.opts, "flux_t2i_hr_d_cfg", 3.5)),  # ui_txt2img_hr_distilled_cfg
            gr.update(value=getattr(shared.opts, "flux_t2i_steps", 20)),  # ui_txt2img_steps
            gr.update(value=getattr(shared.opts, "flux_i2i_steps", 20)),  # ui_img2img_steps
        ]

    if shared.opts.forge_preset == 'FLUX ForgeSD':
        ckpt = shared.opts.sd_model_checkpoint
        model_mem = getattr(shared.opts, "flux_GPU_MB", total_vram - 1024)
        if model_mem < 0 or model_mem > total_vram:
            model_mem = total_vram - 1024
        return [
            gr.update(visible=True),                                                    # ui_vae
            gr.update(visible=False, value=1),                                          # ui_clip_skip
            gr.update(visible=True, value='Automatic'),                                 # ui_forge_unet_storage_dtype_options
            gr.update(visible=True, value='Queue'),                                     # ui_forge_async_loading
            gr.update(visible=False, value='CPU'),                                       # ui_forge_pin_shared_memory
            gr.update(visible=True, value=model_mem),                                   # ui_forge_inference_memory
            gr.update(value=getattr(shared.opts, "flux_t2i_width", 896)),               # ui_txt2img_width
            gr.update(value=getattr(shared.opts, "flux_i2i_width", 1024)),              # ui_img2img_width
            gr.update(value=getattr(shared.opts, "flux_t2i_height", 1152)),             # ui_txt2img_height
            gr.update(value=getattr(shared.opts, "flux_i2i_height", 1024)),             # ui_img2img_height
            gr.update(value=getattr(shared.opts, "flux_t2i_cfg", 1)),                   # ui_txt2img_cfg
            gr.update(value=getattr(shared.opts, "flux_i2i_cfg", 1)),                   # ui_img2img_cfg
            gr.update(visible=True, value=getattr(shared.opts, "flux_t2i_d_cfg", 3.5)), # ui_txt2img_distilled_cfg
            gr.update(visible=True, value=getattr(shared.opts, "flux_i2i_d_cfg", 3.5)), # ui_img2img_distilled_cfg
            gr.update(value=getattr(shared.opts, "flux_t2i_sampler", 'Euler')),         # ui_txt2img_sampler
            gr.update(value=getattr(shared.opts, "flux_i2i_sampler", 'Euler')),         # ui_img2img_sampler
            gr.update(value=getattr(shared.opts, "flux_t2i_scheduler", 'Simple')),      # ui_txt2img_scheduler
            gr.update(value=getattr(shared.opts, "flux_i2i_scheduler", 'Simple')),      # ui_img2img_scheduler
            gr.update(visible=True, value=getattr(shared.opts, "flux_t2i_hr_cfg", 1.0)),    # ui_txt2img_hr_cfg
            gr.update(visible=True, value=getattr(shared.opts, "flux_t2i_hr_d_cfg", 3.5)),  # ui_txt2img_hr_distilled_cfg
            gr.update(value=getattr(shared.opts, "flux_t2i_steps", 20)),  # ui_txt2img_steps
            gr.update(value=getattr(shared.opts, "flux_i2i_steps", 20)),  # ui_img2img_steps
        ]

    if shared.opts.forge_preset == 'FLUX -2Gb':
        ckpt = shared.opts.sd_model_checkpoint

        return [
            gr.update(visible=True),                                                    # ui_vae
            gr.update(visible=False, value=1),                                          # ui_clip_skip
            gr.update(visible=True, value='Automatic'),                                 # ui_forge_unet_storage_dtype_options
            gr.update(visible=True, value='Queue'),                                     # ui_forge_async_loading
            gr.update(visible=False, value='CPU'),                                       # ui_forge_pin_shared_memory
            gr.update(visible=True, value=total_vram - 2048),                                   # ui_forge_inference_memory
            gr.update(value=getattr(shared.opts, "flux_t2i_width", 672)),               # ui_txt2img_width
            gr.update(value=getattr(shared.opts, "flux_i2i_width", 672)),              # ui_img2img_width
            gr.update(value=getattr(shared.opts, "flux_t2i_height", 1200)),             # ui_txt2img_height
            gr.update(value=getattr(shared.opts, "flux_i2i_height", 1200)),             # ui_img2img_height
            gr.update(value=getattr(shared.opts, "flux_t2i_cfg", 1)),                   # ui_txt2img_cfg
            gr.update(value=getattr(shared.opts, "flux_i2i_cfg", 1)),                   # ui_img2img_cfg
            gr.update(visible=True, value=getattr(shared.opts, "flux_t2i_d_cfg", 3.5)), # ui_txt2img_distilled_cfg
            gr.update(visible=True, value=getattr(shared.opts, "flux_i2i_d_cfg", 3.5)), # ui_img2img_distilled_cfg
            gr.update(value=getattr(shared.opts, "flux_t2i_sampler", 'Euler')),         # ui_txt2img_sampler
            gr.update(value=getattr(shared.opts, "flux_i2i_sampler", 'Euler')),         # ui_img2img_sampler
            gr.update(value=getattr(shared.opts, "flux_t2i_scheduler", 'Simple')),      # ui_txt2img_scheduler
            gr.update(value=getattr(shared.opts, "flux_i2i_scheduler", 'Simple')),      # ui_img2img_scheduler
            gr.update(visible=True, value=getattr(shared.opts, "flux_t2i_hr_cfg", 1.0)),    # ui_txt2img_hr_cfg
            gr.update(visible=True, value=getattr(shared.opts, "flux_t2i_hr_d_cfg", 3.5)),  # ui_txt2img_hr_distilled_cfg
            gr.update(value=getattr(shared.opts, "flux_t2i_steps", 20)),  # ui_txt2img_steps
            gr.update(value=getattr(shared.opts, "flux_i2i_steps", 20)),  # ui_img2img_steps
        ]

    if shared.opts.forge_preset == 'FLUX -3Gb':
        ckpt = shared.opts.sd_model_checkpoint

        return [
            gr.update(visible=True),                                                    # ui_vae
            gr.update(visible=False, value=1),                                          # ui_clip_skip
            gr.update(visible=True, value='Automatic'),                                 # ui_forge_unet_storage_dtype_options
            gr.update(visible=True, value='Queue'),                                     # ui_forge_async_loading
            gr.update(visible=False, value='CPU'),                                       # ui_forge_pin_shared_memory
            gr.update(visible=True, value=total_vram - 2704),                                   # ui_forge_inference_memory
            gr.update(value=getattr(shared.opts, "flux_t2i_width", 672)),               # ui_txt2img_width
            gr.update(value=getattr(shared.opts, "flux_i2i_width", 672)),              # ui_img2img_width
            gr.update(value=getattr(shared.opts, "flux_t2i_height", 1200)),             # ui_txt2img_height
            gr.update(value=getattr(shared.opts, "flux_i2i_height", 1200)),             # ui_img2img_height
            gr.update(value=getattr(shared.opts, "flux_t2i_cfg", 1)),                   # ui_txt2img_cfg
            gr.update(value=getattr(shared.opts, "flux_i2i_cfg", 1)),                   # ui_img2img_cfg
            gr.update(visible=True, value=getattr(shared.opts, "flux_t2i_d_cfg", 3.5)), # ui_txt2img_distilled_cfg
            gr.update(visible=True, value=getattr(shared.opts, "flux_i2i_d_cfg", 3.5)), # ui_img2img_distilled_cfg
            gr.update(value=getattr(shared.opts, "flux_t2i_sampler", 'Euler')),         # ui_txt2img_sampler
            gr.update(value=getattr(shared.opts, "flux_i2i_sampler", 'Euler')),         # ui_img2img_sampler
            gr.update(value=getattr(shared.opts, "flux_t2i_scheduler", 'Simple')),      # ui_txt2img_scheduler
            gr.update(value=getattr(shared.opts, "flux_i2i_scheduler", 'Simple')),      # ui_img2img_scheduler
            gr.update(visible=True, value=getattr(shared.opts, "flux_t2i_hr_cfg", 1.0)),    # ui_txt2img_hr_cfg
            gr.update(visible=True, value=getattr(shared.opts, "flux_t2i_hr_d_cfg", 3.5)),  # ui_txt2img_hr_distilled_cfg
            gr.update(value=getattr(shared.opts, "flux_t2i_steps", 20)),  # ui_txt2img_steps
            gr.update(value=getattr(shared.opts, "flux_i2i_steps", 20)),  # ui_img2img_steps
        ]

    loadsave = ui_loadsave.UiLoadsave(cmd_opts.ui_config_file)
    ui_settings_from_file = loadsave.ui_settings.copy()

    ckpt = shared.opts.sd_model_checkpoint

    return [
            gr.update(visible=True),                                                    # ui_vae
            gr.update(visible=False, value=1),                                          # ui_clip_skip
            gr.update(visible=True, value='Automatic'),                                 # ui_forge_unet_storage_dtype_options
            gr.update(visible=True, value='Queue'),                                     # ui_forge_async_loading
            gr.update(visible=False, value='CPU'),                                       # ui_forge_pin_shared_memory
            gr.update(visible=True, value=max(int(total_vram * 0.6), 512)),             # ui_forge_inference_memory
            gr.update(value=getattr(shared.opts, "flux_t2i_width", 672)),               # ui_txt2img_width
            gr.update(value=getattr(shared.opts, "flux_i2i_width", 672)),              # ui_img2img_width
            gr.update(value=getattr(shared.opts, "flux_t2i_height", 1200)),             # ui_txt2img_height
            gr.update(value=getattr(shared.opts, "flux_i2i_height", 1200)),             # ui_img2img_height
            gr.update(value=getattr(shared.opts, "flux_t2i_cfg", 1)),                   # ui_txt2img_cfg
            gr.update(value=getattr(shared.opts, "flux_i2i_cfg", 1)),                   # ui_img2img_cfg
            gr.update(visible=True, value=getattr(shared.opts, "flux_t2i_d_cfg", 3.5)), # ui_txt2img_distilled_cfg
            gr.update(visible=True, value=getattr(shared.opts, "flux_i2i_d_cfg", 3.5)), # ui_img2img_distilled_cfg
            gr.update(value=getattr(shared.opts, "flux_t2i_sampler", 'Euler')),         # ui_txt2img_sampler
            gr.update(value=getattr(shared.opts, "flux_i2i_sampler", 'Euler')),         # ui_img2img_sampler
            gr.update(value=getattr(shared.opts, "flux_t2i_scheduler", 'Simple')),      # ui_txt2img_scheduler
            gr.update(value=getattr(shared.opts, "flux_i2i_scheduler", 'Simple')),      # ui_img2img_scheduler
            gr.update(visible=True, value=getattr(shared.opts, "flux_t2i_hr_cfg", 1.0)),    # ui_txt2img_hr_cfg
            gr.update(visible=True, value=getattr(shared.opts, "flux_t2i_hr_d_cfg", 3.5)),  # ui_txt2img_hr_distilled_cfg
            gr.update(value=getattr(shared.opts, "flux_t2i_steps", 20)),  # ui_txt2img_steps
            gr.update(value=getattr(shared.opts, "flux_i2i_steps", 20)),  # ui_img2img_steps
    ]
