import gradio as gr

from modules import scripts, infotext_utils
from modules.script_callbacks import on_cfg_denoiser, remove_current_script_callbacks
from backend.patcher.base import set_model_options_patch_replace
from backend.sampling.sampling_function import calc_cond_uncond_batch

class PerturbedAttentionGuidanceForForge(scripts.Script):
    sorting_priority = 13

    # runtime state
    current_scale = 1.0
    doPAG = True

    def title(self):
        return "PerturbedAttentionGuidance Integrated"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label="PerturbedAttentionGuidance", elem_id="perturbed-attention-integrated"):
            pagi_enabled = gr.Checkbox(label="Enable PerturbedAttentionGuidance", value=False)
            with gr.Row():
                scale = gr.Slider(label='Scale', minimum=0.0, maximum=100.0, step=0.1, value=1.0)
                attenuation = gr.Slider(label='Attenuation (linear, % of scale)', minimum=0.0, maximum=100.0, step=0.1, value=0.0)
            with gr.Row():
                start_step = gr.Slider(label='Start step', minimum=0.0, maximum=1.0, step=0.01, value=0.0)
                end_step = gr.Slider(label='End step', minimum=0.0, maximum=1.0, step=0.01, value=1.0)

        self.infotext_fields = [
            (pagi_enabled,  lambda d: d.get("pagi_enabled", False)),
            (pagi_enabled,  "pagi_enabled"),
            (scale,         "pagi_scale"),
            (attenuation,   "pagi_attenuation"),
            (start_step,    "pagi_start_step"),
            (end_step,      "pagi_end_step"),
        ]

        for tab in ['txt2img', 'img2img']:
            infotext_utils.paste_fields.setdefault(tab, {})
            fields = infotext_utils.paste_fields[tab].setdefault('fields', [])
            existing = {key for _, key in fields}
            for comp, key in self.infotext_fields:
                if isinstance(key, str):
                    comp.api = key
                    if key not in existing:
                        fields.append((comp, key))

        return pagi_enabled, scale, attenuation, start_step, end_step

    def denoiser_callback(self, params):
        this_step = params.sampling_step / max(1, (params.total_sampling_steps - 1))
        self.doPAG = (this_step >= self.PAG_start) and (this_step <= self.PAG_end)

    def process_before_every_sampling(self, p, *script_args, **kwargs):
        pagi_enabled_ui, scale, attenuation, start_step, end_step = script_args

        pagi_enabled = bool(pagi_enabled_ui or p.extra_generation_params.get("pagi_enabled", False))
        if not pagi_enabled:
            return

        self.current_scale = float(scale)
        self.PAG_start = float(start_step)
        self.PAG_end = float(end_step)

        on_cfg_denoiser(self.denoiser_callback)

        unet = p.sd_model.forge_objects.unet.clone()

        def attn_proc(q, k, v, to):
            return v

        def post_cfg_function(args):
            denoised = args["denoised"]

            if self.current_scale <= 0.0:
                return denoised
            if not self.doPAG:
                return denoised

            model, cond_denoised, cond, sigma, x, options = \
                args["model"], args["cond_denoised"], args["cond"], args["sigma"], args["input"], args["model_options"].copy()

            new_options = set_model_options_patch_replace(options, attn_proc, "attn1", "middle", 0)

            degraded, _ = calc_cond_uncond_batch(model, cond, None, x, sigma, new_options)

            result = denoised + (cond_denoised - degraded) * self.current_scale

            self.current_scale -= float(scale) * float(attenuation) / 100.0

            return result

        unet.set_model_sampler_post_cfg_function(post_cfg_function)
        p.sd_model.forge_objects.unet = unet

        p.extra_generation_params.update(dict(
            pagi_enabled     = True,
            pagi_scale       = float(scale),
            pagi_attenuation = float(attenuation),
            pagi_start_step  = float(start_step),
            pagi_end_step    = float(end_step),
        ))
        return

    def postprocess(self, params, processed, *args):
        remove_current_script_callbacks()
        return
