import os
import gradio as gr

from modules import scripts
from scripts.anydoor import run_webui

ref_dir = os.path.join(scripts.basedir(), "examples/foreground")
ref_mask_dir = os.path.join(scripts.basedir(), "examples/foreground_mask")
image_dir = os.path.join(scripts.basedir(), "examples/background")
ref_list = [os.path.join(ref_dir, file) for file in os.listdir(ref_dir) if
            ".jpg" in file or ".png" in file or ".jpeg" in file]
ref_list.sort()
ref_mask_list = [os.path.join(ref_mask_dir, file) for file in os.listdir(ref_mask_dir) if
                 ".jpg" in file or ".png" in file or ".jpeg" in file]
ref_mask_list.sort()
image_list = [os.path.join(image_dir, file) for file in os.listdir(image_dir) if
              ".jpg" in file or ".png" in file or ".jpeg" in file]
image_list.sort()


def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as ui_component:
        with gr.Column():
            base = gr.Image(label="Background", source="upload", tool="sketch", type="pil", height=512,
                            brush_color='#FFFFFF', mask_opacity=0.5)
            with gr.Row():
                with gr.Column():
                    ref = gr.Image(label="Reference", source="upload", tool="sketch", type="pil", height=256, brush_color='#FFFFFF', mask_opacity=0.5)
                    ref_mask = gr.Image(label="Reference Mask", source="upload", type="pil", height=256)
                with gr.Accordion("Advanced Option", open=True):
                    num_samples = gr.Slider(label="number", minimum=1, maximum=20, value=3, step=1)
                    strength = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
                    ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=30, step=1)
                    scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=4.5, step=0.1)
                    reference_mask_refine = gr.Checkbox(label='Reference Mask Refine', value=False, interactive=True)
                    enable_shape_control = gr.Checkbox(label='Enable Shape Control', value=False, interactive=True)

            run_local_button = gr.Button(label="Generate", value="Run")

            with gr.Row():
                with gr.Column():
                    gr.Examples(image_list, inputs=[base], label="Examples - Background Image", examples_per_page=50)
                with gr.Column():
                    gr.Examples(ref_list, inputs=[ref], label="Examples - Reference Object", examples_per_page=50)
                with gr.Column():
                    gr.Examples(ref_mask_list, inputs=[ref_mask], label="Examples - Reference Mask", examples_per_page=50)

            baseline_gallery = gr.Gallery(label='Output', show_label=True, elem_id="gallery", columns=1, height=512,
                                          rows=9, preview=True, selected_index=0)

        run_local_button.click(fn=run_webui,
                               inputs=[base,
                                       ref,
                                       ref_mask,
                                       reference_mask_refine,
                                       num_samples,
                                       strength,
                                       ddim_steps,
                                       scale,
                                       enable_shape_control,
                                       ],
                               outputs=[baseline_gallery]
                               )

        return [(ui_component, "Anydoor", "anydoor")]
