import math

import modules.scripts as scripts
import gradio as gr
from PIL import Image

from modules import processing, shared, images, devices
from modules.processing import Processed
from modules.shared import opts, state


class Script(scripts.Script):
    def title(self):
        return "ClarityHD Upscale"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):
        info = gr.HTML(
            "<p style=\"margin-bottom:0.75em\">Will upscale the image by the selected scale factor; "
            "use Profile buttons to select desired quiality.</p>"
        )

        tile_w = gr.Slider(minimum=64, maximum=2048, step=16, label='Tile width', value=1152)
        tile_h = gr.Slider(minimum=64, maximum=2048, step=16, label='Tile height', value=1152)
        overlap = gr.Slider(minimum=0, maximum=256, step=16, label='Tile overlap', value=128, elem_id=self.elem_id("overlap"))
        scale_factor = gr.Slider(minimum=1.0, maximum=10.0, step=0.05, label='Scale Factor', value=4.0, elem_id=self.elem_id("scale_factor"))

        with gr.Row():
            reset_btn = gr.Button("Reset Tile")

        upscaler_index = gr.Dropdown(
            label='Upscaler',
            choices=[x.name for x in shared.sd_upscalers],
            value=shared.sd_upscalers[0].name,
            type="value",
            elem_id=self.elem_id("upscaler_index")
        )

        profile_label = gr.HTML("<h4>Presets:</h4>")

        with gr.Row():
            with gr.Column():
                ssdir_label = gr.HTML("<h4>4xSSDIRDAT</h4>")
                clarity_left_buttons = []
                left_names = [
                    "768 Balanced",
                    "768 Preffered",
                    "1024 Balanced",
                    "1024 Preffered",
                    "1152 Balanced",
                    "1152 Preffered"
                ]
                # 4xSSDIRDAT tile sizes + overlap
                tile_settings_ssdir = [
                    (768, 768, 64),
                    (768, 768, 64),
                    (1024, 1024, 128),
                    (1024, 1024, 128),
                    (1152, 1152, 128),
                    (1152, 1152, 128)
                ]
                for i, (name, (tw, th, ov)) in enumerate(zip(left_names, tile_settings_ssdir), start=1):
                    btn = gr.Button(name, elem_id=self.elem_id(f"clarity_profile_{i}_btn"))
                    clarity_left_buttons.append((btn, tw, th, ov))

            with gr.Column():
                ultra_label = gr.HTML("<h4>4x-UltraSharp</h4>")
                clarity_right_buttons = []
                right_names = [
                    "768 Balanced",
                    "768 Preffered",
                    "1024 Balanced",
                    "1024 Preffered",
                    "1152 Balanced",
                    "1152 Preffered"
                ]
                # 4x-UltraSharp tile size + overlap
                tile_settings_ultra = [
                    (768, 768, 64),
                    (768, 768, 64),
                    (1024, 1024, 128),
                    (1024, 1024, 128),
                    (1152, 1152, 128),
                    (1152, 1152, 128)
                ]
                for i, (name, (tw, th, ov)) in enumerate(zip(right_names, tile_settings_ultra), start=1):
                    btn = gr.Button(name, elem_id=self.elem_id(f"clarity_profile_{i}_btn"))
                    clarity_right_buttons.append((btn, tw, th, ov))

        description_block = gr.HTML(
            "<div style='margin-top:1em; padding:1em; border:1px solid #ccc; border-radius:6px;'>"
            "<h4>Information</h4>"
            "<p>ClarityHD Upscaler is ComfyUI' Tiled Clarity Upscale enhanced port.</p>"
            "<p>To use it, select 'ClarityHD Upscale' script from scripts list:</p>"
            "<ul style='list-style:none; padding-left:0;'>"
            "<li>📌 Go to IMG2IMG -> Resise By -> 1x</li>"
            "<li>📌 Choose JuggernautReborn SD1.5 model and SD1.5 VAE from StableDiffusion models list.</li>"
            "<li>📌 You need to have SD1.5 MoreDetails and SDXLrender LoRas.</li>"
            "<li>📌 Select desired Profile in UI to get desired upscale quiality. It will enable d and setup all needed extensions and values into them.</li>"
            "<li>📌 Select desired upscale factor and tap 'Generate'.</li>"
            "</ul>"
            "<div style='margin-top:1em; padding:0.75em; border:1px solid #e0c200; border-radius:4px;'>"
            "<strong>⚠️ Warning!</strong><br>"
            "Higher Tile size - better quality, but higher VRAM usage.<br>"
            "Denoise 0.5 force more details; Denoise 0.34 saves original details.<br>"
            "CFG 1.0 saves original detailes, CFG 8.0 force more detailes. CFG Detailer method better to Denoise.<br>"
            "Balanced try to save original details, while Preffered use ControlNet preffered mode to get more new details.<br>"
            "If you're got OOM, enable Tiled VAE.<br>"
            "You can add Detail Daemon extension to add even more details."
            "</div>"
            "</div>"
        )


        def reset_sizes():
            return 1024, 1024, 128  # reset size + overlap

        reset_btn.click(fn=reset_sizes, inputs=[], outputs=[tile_w, tile_h, overlap])

        # 4xSSDIRDAT
        for i, (btn, tw, th, ov) in enumerate(clarity_left_buttons, start=1):
            btn.click(fn=None, inputs=[], outputs=[], 
                      _js=f"() => document.getElementById('clarity_profile_{i}_btn').click()")
            btn.click(fn=lambda tw=tw, th=th, ov=ov: ("4xSSDIRDAT", tw, th, ov),
                      inputs=[], outputs=[upscaler_index, tile_w, tile_h, overlap])

        # 4x-UltraSharp
        for i, (btn, tw, th, ov) in enumerate(clarity_right_buttons, start=1):
            btn.click(fn=None, inputs=[], outputs=[], 
                      _js=f"() => document.getElementById('clarity_profile_{i}_btn').click()")
            btn.click(fn=lambda tw=tw, th=th, ov=ov: ("4x-UltraSharp", tw, th, ov),
                      inputs=[], outputs=[upscaler_index, tile_w, tile_h, overlap])

        return [info, overlap, upscaler_index, scale_factor, tile_w, tile_h, reset_btn]


    def run(self, p, _, overlap, upscaler_index, scale_factor, tile_w, tile_h, reset_btn):
        if isinstance(upscaler_index, str):
            upscaler_index = [x.name.lower() for x in shared.sd_upscalers].index(upscaler_index.lower())
        processing.fix_seed(p)
        upscaler = shared.sd_upscalers[upscaler_index]

        p.extra_generation_params["Overlap"] = overlap
        p.extra_generation_params["Upscaler"] = upscaler.name

        initial_info = None
        seed = p.seed

        init_img = p.init_images[0]
        init_img = images.flatten(init_img, opts.img2img_background_color)

        if upscaler.name != "None":
            img = upscaler.scaler.upscale(init_img, scale_factor, upscaler.data_path)
        else:
            img = init_img

        devices.torch_gc()

        grid = images.split_grid(img, tile_w=tile_w, tile_h=tile_h, overlap=overlap)
        orig_w, orig_h = p.width, p.height
        p.width = tile_w
        p.height = tile_h


        batch_size = p.batch_size
        upscale_count = p.n_iter
        p.n_iter = 1
        p.do_not_save_grid = True
        p.do_not_save_samples = True

        work = []

        for _y, _h, row in grid.tiles:
            for tiledata in row:
                work.append(tiledata[2])

        batch_count = math.ceil(len(work) / batch_size)
        state.job_count = batch_count * upscale_count

        print(f"ClarityHD Upscaling will process a total of {len(work)} images tiled as {len(grid.tiles[0][2])}x{len(grid.tiles)} per upscale in a total of {state.job_count} batches.")

        result_images = []
        for n in range(upscale_count):
            start_seed = seed + n
            p.seed = start_seed

            work_results = []
            for i in range(batch_count):
                p.batch_size = batch_size
                p.init_images = work[i * batch_size:(i + 1) * batch_size]

                state.job = f"Batch {i + 1 + n * batch_count} out of {state.job_count}"
                processed = processing.process_images(p)

                if initial_info is None:
                    initial_info = processed.info

                p.seed = processed.seed + 1
                work_results += processed.images

            image_index = 0
            for _y, _h, row in grid.tiles:
                for tiledata in row:
                    tiledata[2] = work_results[image_index] if image_index < len(work_results) else Image.new("RGB", (p.width, p.height))
                    image_index += 1

            combined_image = images.combine_grid(grid)
            result_images.append(combined_image)

            if opts.samples_save:
                images.save_image(combined_image, p.outpath_samples, "", start_seed, p.prompt, opts.samples_format, info=initial_info, p=p)

        processed = Processed(p, result_images, seed, initial_info)

        p.n_iter = upscale_count
        
        p.width = orig_w
        p.height = orig_h

        return processed
