# sd-webui-anydoor

A Stable Diffusion WebUI Extension for [`AnyDoor: Zero-shot Object-level Image Customization`](https://github.com/ali-vilab/AnyDoor?tab=readme-ov-file) 

## Installation

1. Open `Extentions` Tab and go to `Install from URL` Tab
2. Paste `https://github.com/davidlightmysterion/sd-webui-anydoor.git` and Click `Install`
3. Restart WebUI and you will see `Anydoor` Extension

## Model Preparation

Download checkpoints from [repo](https://github.com/ali-vilab/AnyDoor?tab=readme-ov-file) , and put checkpoints to dir `stable-diffusion-webui/extensions/sd-webui-anydoor/models`

- epoch=1-step=8687.ckpt (16.84GB)
- dinov2_vitg14_pretrain.pth (4.55GB)

## Snapshots

You can use a prepared reference mask or sketch on your own reference image. Remember to draw the mask on the background image.

![1](./assets/1.png)

