# sd-webui-anydoor

Stable Diffusion WebUI extension for [_AnyDoor: Zero-shot Object-level Image Customization_](https://github.com/ali-vilab/AnyDoor?tab=readme-ov-file) 

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

## API Interface

A simple **POST** API `/anydoor/predict`, using a rectangle bounding box as the input mask for convenience.

You can also check the parameters on `localhost:7860/docs`.

### a. input parameters

|    parameter name    |                                                 description                                                  | must |
|:--------------------:|:------------------------------------------------------------------------------------------------------------:|:----:|
|     input_image      |                                 base64 string of background image, type: str                                 |  Y   |
|      ref_image       |                                 base64 string of reference image, type: str                                  |  Y   |
|       ref_mask       |                                  base64 string of reference mask, type: str                                  |  Y   |
|      bg_mask_x1      |                          top left corner x for background mask region, type: float                           |  Y   |
|      bg_mask_y1      |                          top left corner y for background mask region, type: float                           |  Y   |
|      bg_mask_x2      |                            bottom right x for background mask region, type: float                            |  Y   |
|      bg_mask_y2      |                            bottom right y for background mask region, type: float                            |  Y   |
|     num_samples      |                                            type: int, default: 1                                             |  N   |
|       strength       |                                          type: float, default: 1.0                                           |  N   |
|      ddim_steps      |                                            type: int, default: 30                                            |  N   |
|        scale         |                                     cfg scale, type: float, default: 4.5                                     |  N   |
| enable_shape_control |                                          type: bool, default: False                                          |  N   |
| reference_mask_refine | whether use `coarse_mask_refine.pth` to refine the user annotated reference mask, type: bool, default: False |  N   |

### b. output parameters

| parameter_name |                                      description                                      |
|:--------------:|:-------------------------------------------------------------------------------------:|
|  status_code   |                     200 indicates success, 400 indicates failure                      |
|      msg       |                                        message                                        |
|     images     | generated images, type: list of base64 string, same with internal `txt2img` interface |

### c. example

```python
import requests
import base64

payload = {
    "input_image": "base64str",
    "ref_image": "base64str",
    "ref_mask": "base64str",
    "bg_mask_x1": 0.34,
    "bg_mask_y1": 0.34,
    "bg_mask_x2": 0.44,
    "bg_mask_y2": 0.44,
    "num_samples": 3,
    "strength": 1,
    "ddim_steps": 30,
    "scale": 4.5,
    "enable_shape_control": False,
    "reference_mask_refine": False,
}
response = requests.post("http://localhost:7860/anydoor/predict", json=payload)
base64_str = response.json()["images"][0]
with open("image.jpg", "wb") as file:
    file.write(base64.b64decode(base64_str))
```

