import torch
import hashlib
import os

import packages_3rdparty.webui_lora_collection.lora as lora_utils_webui
import packages_3rdparty.comfyui_lora_collection.lora as lora_utils_comfyui

from backend import memory_management, utils
from backend.operations import bnb_avaliable

import torch._dynamo
torch._dynamo.config.suppress_errors = True

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

extra_weight_calculators = {}
lora_collection_priority = [lora_utils_webui, lora_utils_comfyui]

import importlib

def _default_autocast_dtype(strategy_override=None):
    try:
        shared = importlib.import_module("modules.shared")
        strategy = strategy_override or getattr(shared.opts, "lora_computation_strategy", "Auto HW")
    except Exception:
        strategy = strategy_override or "Auto HW"

    if strategy == "Auto HW":
        if torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability()
            return torch.bfloat16 if major >= 8 else torch.float16
        else:
            return torch.float32
    elif strategy == "Autocast":
        return torch.bfloat16 if torch.cuda.is_available() else torch.float32
    elif strategy == "float16":
        return torch.float16
    elif strategy == "bfloat16":
        return torch.bfloat16
    elif strategy == "float32":
        return torch.float32
    else:
        return torch.bfloat16 if torch.cuda.is_available() else torch.float32


def get_function(function_name: str):
    for lora_collection in lora_collection_priority:
        if hasattr(lora_collection, function_name):
            return getattr(lora_collection, function_name)
    raise AttributeError(f"{function_name} not found in lora collections")

def load_lora(lora, to_load):
    patch_dict, remaining_dict = get_function('load_lora')(lora, to_load)
    return patch_dict, remaining_dict

def inner_str(k, prefix="", suffix=""):
    return k[len(prefix):-len(suffix)]

def model_lora_keys_clip(model, key_map={}):
    model_keys, key_maps = get_function('model_lora_keys_clip')(model, key_map)

    for model_key in model_keys:
        if model_key.endswith(".weight"):
            if model_key.startswith("t5xxl.transformer."):
                for prefix in ['te1', 'te2', 'te3']:
                    formatted = inner_str(model_key, "t5xxl.transformer.", ".weight")
                    formatted = formatted.replace(".", "_")
                    formatted = f"lora_{prefix}_{formatted}"
                    key_map[formatted] = model_key

    return key_maps

def model_lora_keys_unet(model, key_map={}):
    model_keys, key_maps = get_function('model_lora_keys_unet')(model, key_map)
    return key_maps

# -----------------------------------------------------------------------------
# Core math helpers
# -----------------------------------------------------------------------------

@torch.inference_mode()
def weight_decompose(dora_scale, weight, lora_diff, alpha, strength, computation_dtype, function):
    dora_scale = memory_management.cast_to_device(dora_scale, weight.device, computation_dtype)
    # scale with alpha in-place
    lora_diff.mul_(alpha)

    # compute additive result in weight dtype
    weight_calc = weight + function(lora_diff).to(weight.dtype)

    wd_on_output_axis = dora_scale.shape[0] == weight_calc.shape[0]
    if wd_on_output_axis:
        # norm along output axis
        weight_norm = (
            weight.reshape(weight.shape[0], -1)
            .norm(dim=1, keepdim=True)
            .reshape(weight.shape[0], *[1] * (weight.dim() - 1))
        )
    else:
        # norm along input axis
        weight_norm = (
            weight_calc.transpose(0, 1)
            .reshape(weight_calc.shape[1], -1)
            .norm(dim=1, keepdim=True)
            .reshape(weight_calc.shape[1], *[1] * (weight_calc.dim() - 1))
            .transpose(0, 1)
        )
    # add epsilon to avoid division by zero
    weight_norm = weight_norm + torch.finfo(weight.dtype).eps

    # apply dora rescaling
    scale = (dora_scale / weight_norm).to(weight.dtype)
    weight_calc.mul_(scale)

    if strength != 1.0:
        # apply strength in-place
        diff = weight_calc - weight
        weight.add_(diff.mul(strength))
    else:
        # full overwrite
        weight.copy_(weight_calc)

    return weight

# -----------------------------------------------------------------------------
# merge_lora_to_weight optimized
# -----------------------------------------------------------------------------

# Use torch.compile where supported to speed up compute-heavy paths
try:
    compile_decorator = torch.compile
except AttributeError:
    # Fallback for older versions
    def compile_decorator(fn):
        if os.name == "nt":  # Windows
            return fn
        try:
            return torch.compile(fn)
        except Exception:
            return fn

@torch.inference_mode()
#@compile_decorator
def merge_lora_to_weight(patches, weight, key="online_lora", computation_dtype=_default_autocast_dtype()):
    """
    Optimized merge:
      - Minimal dtype conversions
      - Inplace ops (add_, mul_, copy_) where safe
      - AMP autocast around heavy matmul/einsum
      - Reduced cloning, padding only when necessary
    """

    weight_dtype_backup = None
    if computation_dtype == weight.dtype:
        # avoid clone unless necessary—most ops below are inplace/additive
        pass
    else:
        weight_dtype_backup = weight.dtype
        weight = weight.to(dtype=computation_dtype)

    for p in patches:
        strength, v, strength_model, offset, function = p
        function = function or (lambda a: a)

        # When slicing to a sub-tensor, remember parent to restore
        old_weight = None
        if offset is not None:
            old_weight = weight
            weight = weight.narrow(offset[0], offset[1], offset[2])

        # Model strength scaling inplace
        if strength_model != 1.0:
            weight.mul_(strength_model)

        # Nested list patch form
        if isinstance(v, list):
            v = (merge_lora_to_weight(v[1:], v[0].clone(), key, computation_dtype=computation_dtype),)

        # Determine patch type
        if len(v) == 1:
            patch_type = "diff"
        else:
            patch_type = v[0]
            v = v[1]

        # ----------------- Patch implementations -----------------

        if patch_type == "diff":
            w1 = v[0]
            if strength != 0.0:
                if w1.shape == weight.shape:
                    diff = memory_management.cast_to_device(w1, weight.device, weight.dtype)
                    weight.add_(diff.mul(strength))
                elif w1.ndim == weight.ndim == 4:
                    # Expand to larger channel shape with minimal copies
                    new_shape = [max(n, m) for n, m in zip(weight.shape, w1.shape)]
                    new_weight = torch.zeros(size=new_shape, device=weight.device, dtype=weight.dtype)
                    # copy existing
                    new_weight[:weight.shape[0], :weight.shape[1], :weight.shape[2], :weight.shape[3]].copy_(weight)
                    # add diff
                    new_diff = memory_management.cast_to_device(w1, weight.device, weight.dtype).mul(strength)
                    new_weight[:new_diff.shape[0], :new_diff.shape[1], :new_diff.shape[2], :new_diff.shape[3]].add_(new_diff)
                    weight = new_weight
                else:
                    # shape mismatch that cannot be merged
                    # keep as lightweight log to avoid sync stalls
                    # print(f"WARNING SHAPE MISMATCH {key}: {w1.shape} != {weight.shape}")
                    pass

        elif patch_type == "set":
            weight.copy_(v[0])

        elif patch_type == "lora":
            mat1 = memory_management.cast_to_device(v[0], weight.device, computation_dtype)
            mat2 = memory_management.cast_to_device(v[1], weight.device, computation_dtype)
            dora_scale = v[4]
            alpha = (v[2] / mat2.shape[0]) if v[2] is not None else 1.0

            if v[3] is not None:
                mat3 = memory_management.cast_to_device(v[3], weight.device, computation_dtype)
                # Build mat2 via mm on flattened dims
                with torch.autocast(device_type="cuda", dtype=computation_dtype, enabled=torch.cuda.is_available()):
                    mat2 = torch.mm(mat2.transpose(0, 1).flatten(start_dim=1),
                                    mat3.transpose(0, 1).flatten(start_dim=1)).reshape(
                        [mat2.shape[1], mat2.shape[0], mat3.shape[2], mat3.shape[3]]
                    ).transpose(0, 1)

            with torch.autocast(device_type="cuda", dtype=computation_dtype, enabled=torch.cuda.is_available()):
                lora_diff = torch.mm(mat1.flatten(start_dim=1), mat2.flatten(start_dim=1))

            # Try reshape; if mismatch on last dim, pad minimal side
            try:
                lora_diff = lora_diff.reshape(weight.shape)
            except RuntimeError:
                # Only adjust last dim; avoid expensive global pads
                last = -1
                need = weight.shape[last] - lora_diff.shape[last]
                if need > 0:
                    pad = [0, 0] * (lora_diff.dim() - 1) + [0, need]
                    lora_diff = torch.nn.functional.pad(lora_diff, pad, mode='constant', value=0)
                elif need < 0:
                    pad = [0, 0] * (weight.dim() - 1) + [0, -need]
                    weight = torch.nn.functional.pad(weight, pad, mode='constant', value=0)
                # assume new shapes match now
                lora_diff = lora_diff.reshape(weight.shape)

            if dora_scale is not None:
                weight = weight_decompose(dora_scale, weight, lora_diff, alpha, strength, computation_dtype, function)
            else:
                upd = function(lora_diff.mul(alpha * strength)).to(weight.dtype)
                weight.add_(upd)

        elif patch_type == "lokr":
            w1, w2 = v[0], v[1]
            w1_a, w1_b = v[3], v[4]
            w2_a, w2_b = v[5], v[6]
            t2 = v[7]
            dora_scale = v[8]
            dim = None

            if w1 is None:
                dim = w1_b.shape[0]
                with torch.autocast(device_type="cuda", dtype=computation_dtype, enabled=torch.cuda.is_available()):
                    w1 = torch.mm(memory_management.cast_to_device(w1_a, weight.device, computation_dtype),
                                  memory_management.cast_to_device(w1_b, weight.device, computation_dtype))
            else:
                w1 = memory_management.cast_to_device(w1, weight.device, computation_dtype)

            if w2 is None:
                dim = w2_b.shape[0]
                if t2 is None:
                    with torch.autocast(device_type="cuda", dtype=computation_dtype, enabled=torch.cuda.is_available()):
                        w2 = torch.mm(memory_management.cast_to_device(w2_a, weight.device, computation_dtype),
                                      memory_management.cast_to_device(w2_b, weight.device, computation_dtype))
                else:
                    with torch.autocast(device_type="cuda", dtype=computation_dtype, enabled=torch.cuda.is_available()):
                        w2 = torch.einsum('i j k l, j r, i p -> p r k l',
                                          memory_management.cast_to_device(t2, weight.device, computation_dtype),
                                          memory_management.cast_to_device(w2_b, weight.device, computation_dtype),
                                          memory_management.cast_to_device(w2_a, weight.device, computation_dtype))
            else:
                w2 = memory_management.cast_to_device(w2, weight.device, computation_dtype)

            if w2.dim() == 4:
                w1 = w1.unsqueeze(2).unsqueeze(2)

            alpha = (v[2] / dim) if (v[2] is not None and dim is not None) else 1.0

            with torch.autocast(device_type="cuda", dtype=computation_dtype, enabled=torch.cuda.is_available()):
                lora_diff = torch.kron(w1, w2).reshape(weight.shape)

            if dora_scale is not None:
                weight = weight_decompose(dora_scale, weight, lora_diff, alpha, strength, computation_dtype, function)
            else:
                weight.add_(function(lora_diff.mul(alpha * strength)).to(weight.dtype))

        elif patch_type == "loha":
            w1a, w1b = v[0], v[1]
            alpha = (v[2] / w1b.shape[0]) if v[2] is not None else 1.0
            w2a, w2b = v[3], v[4]
            dora_scale = v[7]

            if v[5] is not None:
                t1, t2 = v[5], v[6]
                with torch.autocast(device_type="cuda", dtype=computation_dtype, enabled=torch.cuda.is_available()):
                    m1 = torch.einsum('i j k l, j r, i p -> p r k l',
                                      memory_management.cast_to_device(t1, weight.device, computation_dtype),
                                      memory_management.cast_to_device(w1b, weight.device, computation_dtype),
                                      memory_management.cast_to_device(w1a, weight.device, computation_dtype))
                    m2 = torch.einsum('i j k l, j r, i p -> p r k l',
                                      memory_management.cast_to_device(t2, weight.device, computation_dtype),
                                      memory_management.cast_to_device(w2b, weight.device, computation_dtype),
                                      memory_management.cast_to_device(w2a, weight.device, computation_dtype))
            else:
                with torch.autocast(device_type="cuda", dtype=computation_dtype, enabled=torch.cuda.is_available()):
                    m1 = torch.mm(memory_management.cast_to_device(w1a, weight.device, computation_dtype),
                                  memory_management.cast_to_device(w1b, weight.device, computation_dtype))
                    m2 = torch.mm(memory_management.cast_to_device(w2a, weight.device, computation_dtype),
                                  memory_management.cast_to_device(w2b, weight.device, computation_dtype))

            lora_diff = (m1 * m2).reshape(weight.shape)
            if dora_scale is not None:
                weight = weight_decompose(dora_scale, weight, lora_diff, alpha, strength, computation_dtype, function)
            else:
                weight.add_(function(lora_diff.mul(alpha * strength)).to(weight.dtype))

        elif patch_type == "glora":
            dora_scale = v[5]

            old_glora = False
            if v[3].shape[1] == v[2].shape[0] == v[0].shape[0] == v[1].shape[1]:
                old_glora = True
            if v[3].shape[0] == v[2].shape[1] == v[0].shape[1] == v[1].shape[0]:
                if not (old_glora and v[1].shape[0] == weight.shape[0] and weight.shape[0] == weight.shape[1]):
                    old_glora = False

            a1 = memory_management.cast_to_device(v[0].flatten(start_dim=1), weight.device, computation_dtype)
            a2 = memory_management.cast_to_device(v[1].flatten(start_dim=1), weight.device, computation_dtype)
            b1 = memory_management.cast_to_device(v[2].flatten(start_dim=1), weight.device, computation_dtype)
            b2 = memory_management.cast_to_device(v[3].flatten(start_dim=1), weight.device, computation_dtype)

            alpha = 1.0 if v[4] is None else ((v[4] / v[0].shape[0]) if old_glora else (v[4] / v[1].shape[0]))

            if old_glora:
                with torch.autocast(device_type="cuda", dtype=computation_dtype, enabled=torch.cuda.is_available()):
                    part = torch.mm(torch.mm(weight.flatten(start_dim=1).to(dtype=computation_dtype), a2), a1)
                    lora_diff = (torch.mm(b2, b1) + part).reshape(weight.shape)
            else:
                with torch.autocast(device_type="cuda", dtype=computation_dtype, enabled=torch.cuda.is_available()):
                    if weight.dim() > 2:
                        tmp = torch.einsum("o i ..., i j -> o j ...", weight.to(dtype=computation_dtype), a1)
                        tmp = torch.einsum("o i ..., i j -> o j ...", tmp, a2)
                        lora_diff = tmp.reshape(weight.shape)
                    else:
                        lora_diff = torch.mm(torch.mm(weight.to(dtype=computation_dtype), a1), a2).reshape(weight.shape)
                    lora_diff.add_(torch.mm(b1, b2).reshape(weight.shape))

            if dora_scale is not None:
                weight = weight_decompose(dora_scale, weight, lora_diff, alpha, strength, computation_dtype, function)
            else:
                weight.add_(function(lora_diff.mul(alpha * strength)).to(weight.dtype))

        elif patch_type in extra_weight_calculators:
            weight = extra_weight_calculators[patch_type](weight, strength, v)

        # Restore parent weight reference if we narrowed
        if old_weight is not None:
            weight = old_weight

    if weight_dtype_backup is not None:
        weight = weight.to(dtype=weight_dtype_backup)

    return weight

# -----------------------------------------------------------------------------
# Device helpers
# -----------------------------------------------------------------------------

def get_parameter_devices(model):
    return {key: p.device for key, p in model.named_parameters()}

def set_parameter_devices(model, parameter_devices):
    for key, device in parameter_devices.items():
        p = utils.get_attr(model, key)
        if p.device != device:
            p = utils.tensor2parameter(p.to(device=device))
            utils.set_attr_raw(model, key, p)
    return model

# -----------------------------------------------------------------------------
# LoraLoader with optimized refresh
# -----------------------------------------------------------------------------

class LoraLoader:
    def __init__(self, model):
        self.model = model
        self.backup = {}
        self.online_backup = []
        self.loaded_hash = str([])

    @torch.inference_mode()
    def refresh(self, lora_patches, offload_device=torch.device('cpu'), force_refresh=False):
        # stable and fast hash
        hashes = hashlib.md5(str(sorted(lora_patches.keys())).encode()).hexdigest()
        if hashes == self.loaded_hash and not force_refresh:
            return

        # Flatten patches by (key, online_mode)
        all_patches = {}
        for (_, _, _, online_mode), patches in lora_patches.items():
            for key, current_patches in patches.items():
                all_patches.setdefault((key, online_mode), []).extend(current_patches)

        memory_management.signal_empty_cache = True
        parameter_devices = get_parameter_devices(self.model)

        # Restore backup inplace
        self.online_backup = []
        for k, w in self.backup.items():
            if not isinstance(w, torch.nn.Parameter):
                w = torch.nn.Parameter(w, requires_grad=False)
            utils.set_attr_raw(self.model, k, w)
        self.backup.clear()

        set_parameter_devices(self.model, parameter_devices=parameter_devices)

        autocast_dtype = _default_autocast_dtype()

        for (key, online_mode), current_patches in all_patches.items():
            try:
                parent_layer, child_key, weight = utils.get_attr_with_parent(self.model, key)
                assert isinstance(weight, torch.nn.Parameter)
            except Exception:
                raise ValueError(f"Wrong LoRA Key: {key}")

            if online_mode:
                if not hasattr(parent_layer, 'forge_online_loras'):
                    parent_layer.forge_online_loras = {}
                parent_layer.forge_online_loras[child_key] = current_patches
                self.online_backup.append(parent_layer)
                continue

            # Cache backup on offload device to reduce GPU memory pressure
            if key not in self.backup:
                self.backup[key] = weight.to(device=offload_device)

            # Dequantize if needed (bnb/gguf paths)
            bnb_layer = None
            if hasattr(weight, 'bnb_quantized') and bnb_avaliable:
                bnb_layer = parent_layer
                from backend.operations_bnb import functional_dequantize_4bit
                weight = functional_dequantize_4bit(weight)

            gguf_cls = getattr(weight, 'gguf_cls', None)
            gguf_parameter = None
            if gguf_cls is not None:
                gguf_parameter = weight
                from backend.operations_gguf import dequantize_tensor
                weight = dequantize_tensor(weight)

            if not current_patches:
                continue

            # Apply patches with AMP for speed; fallback to CPU FP32 on OOM
            try:
                with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=torch.cuda.is_available()):
                    merged = merge_lora_to_weight(current_patches, weight, key, computation_dtype=autocast_dtype)
            except RuntimeError:
                set_parameter_devices(self.model, parameter_devices={k: offload_device for k in parameter_devices.keys()})
                memory_management.soft_empty_cache()
                merged = merge_lora_to_weight(current_patches, weight, key, computation_dtype=torch.float32)

            # Re-quantize or reload as required
            if bnb_layer is not None:
                bnb_layer.reload_weight(merged)
                continue

            if gguf_cls is not None:
                gguf_cls.quantize_pytorch(merged, gguf_parameter)
                continue

            # Inplace set parameter
            utils.set_attr_raw(self.model, key, torch.nn.Parameter(merged, requires_grad=False))

        set_parameter_devices(self.model, parameter_devices=parameter_devices)
        self.loaded_hash = hashes

        # Clean large temporaries once, only if they exist
        try:
            if 'all_patches' in locals():
                del all_patches
            if 'current_patches' in locals():
                del current_patches
            if 'weight' in locals():
                del weight
        finally:
            torch.cuda.empty_cache()
            memory_management.soft_empty_cache()

        torch.cuda.empty_cache()
        memory_management.soft_empty_cache()
