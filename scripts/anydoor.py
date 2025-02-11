import gc
import os
import sys
import cv2
import torch
import einops
import numpy as np
import gradio as gr
from PIL import Image, ImageDraw
from omegaconf import OmegaConf
from collections import OrderedDict
from modules import scripts, shared, script_callbacks
from modules.devices import device, torch_gc, cpu
from modules.safe import unsafe_torch_load, load
from scripts.data_utils import *
from scripts.anydoor_modules.cldm.model import create_model, load_state_dict
from scripts.anydoor_modules.cldm.ddim_hacked import DDIMSampler
from scripts.anydoor_modules.cldm.hack import disable_verbosity, enable_sliced_attention
from scripts.coarse_mask_refine_util import BaselineModel

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

model_cache = OrderedDict()
model_dir = os.path.join(scripts.basedir(), "models")
config_dir = os.path.join(scripts.basedir(), "configs")
example_dir = os.path.join(scripts.basedir(), "examples")

def process_image_mask(image_np, mask_np):
    iseg_model = init_iseg_model()
    img = torch.from_numpy(image_np.transpose((2, 0, 1)))
    img = img.float().div(255).unsqueeze(0)
    mask = torch.from_numpy(mask_np).float().unsqueeze(0).unsqueeze(0)
    pred = iseg_model(img, mask)['instances'][0, 0].detach().numpy() > 0.5
    return pred.astype(np.uint8)

def load_anydoor_model():
    save_memory = False
    disable_verbosity()
    if save_memory:
        enable_sliced_attention()

    config = OmegaConf.load(os.path.join(config_dir, "demo.yaml"))
    model_ckpt = os.path.join(model_dir, config.pretrained_model)
    model_config = os.path.join(config_dir, config.config_file)

    torch.load = unsafe_torch_load
    model = create_model(model_config).cpu()
    model.load_state_dict(load_state_dict(model_ckpt, location=device))
    model.to(device=device)
    ddim_sampler = DDIMSampler(model)
    torch.load = load
    return model, ddim_sampler

def clear_cache():
    # model_cache.clear()
    gc.collect()
    torch_gc()

def init_anydoor_model(model_name):
    print(f"Initializing Anydoor to {device}")
    if model_name in model_cache:
        model, ddim_sampler = model_cache[model_name]
        if shared.cmd_opts.lowvram or (str(device) not in str(model.device)):
            model.to(device=device)
        return model, ddim_sampler
    else:
        clear_cache()
        model_cache[model_name] = load_anydoor_model()
        return model_cache[model_name]

def init_iseg_model():
    model_name = "coarse_mask_refine.pth"
    print(f"Initializing {model_name} to cpu")
    if model_name in model_cache:
        return model_cache[model_name]
    else:
        iseg_model = BaselineModel().eval()
        weights = torch.load(os.path.join(model_dir, model_name), map_location='cpu')['state_dict']
        iseg_model.load_state_dict(weights, strict=True)
        model_cache[model_name] = iseg_model
        return iseg_model

def process_pairs(ref_image, ref_mask, tar_image, tar_mask, max_ratio=0.8, enable_shape_control=False):
    # ========= Reference ===========
    # ref expand
    ref_box_yyxx = get_bbox_from_mask(ref_mask)

    # ref filter mask
    ref_mask_3 = np.stack([ref_mask, ref_mask, ref_mask], -1)
    masked_ref_image = ref_image * ref_mask_3 + np.ones_like(ref_image) * 255 * (1 - ref_mask_3)

    y1, y2, x1, x2 = ref_box_yyxx
    masked_ref_image = masked_ref_image[y1:y2, x1:x2, :]
    ref_mask = ref_mask[y1:y2, x1:x2]

    ratio = np.random.randint(11, 15) / 10  # 11,13
    masked_ref_image, ref_mask = expand_image_mask(masked_ref_image, ref_mask, ratio=ratio)
    ref_mask_3 = np.stack([ref_mask, ref_mask, ref_mask], -1)

    # to square and resize
    masked_ref_image = pad_to_square(masked_ref_image, pad_value=255, random=False)
    masked_ref_image = cv2.resize(masked_ref_image.astype(np.uint8), (224, 224)).astype(np.uint8)

    ref_mask_3 = pad_to_square(ref_mask_3 * 255, pad_value=0, random=False)
    ref_mask_3 = cv2.resize(ref_mask_3.astype(np.uint8), (224, 224)).astype(np.uint8)
    ref_mask = ref_mask_3[:, :, 0]

    # collage aug
    masked_ref_image_compose, ref_mask_compose = masked_ref_image, ref_mask
    ref_mask_3 = np.stack([ref_mask_compose, ref_mask_compose, ref_mask_compose], -1)
    ref_image_collage = sobel(masked_ref_image_compose, ref_mask_compose / 255)

    # ========= Target ===========
    tar_box_yyxx = get_bbox_from_mask(tar_mask)
    tar_box_yyxx = expand_bbox(tar_mask, tar_box_yyxx, ratio=[1.1, 1.2])  # 1.1  1.3
    tar_box_yyxx_full = tar_box_yyxx

    # crop
    tar_box_yyxx_crop = expand_bbox(tar_image, tar_box_yyxx, ratio=[1.3, 3.0])
    tar_box_yyxx_crop = box2squre(tar_image, tar_box_yyxx_crop)  # crop box
    y1, y2, x1, x2 = tar_box_yyxx_crop

    cropped_target_image = tar_image[y1:y2, x1:x2, :]
    cropped_tar_mask = tar_mask[y1:y2, x1:x2]

    tar_box_yyxx = box_in_box(tar_box_yyxx, tar_box_yyxx_crop)
    y1, y2, x1, x2 = tar_box_yyxx

    # collage
    ref_image_collage = cv2.resize(ref_image_collage.astype(np.uint8), (x2 - x1, y2 - y1))
    ref_mask_compose = cv2.resize(ref_mask_compose.astype(np.uint8), (x2 - x1, y2 - y1))
    ref_mask_compose = (ref_mask_compose > 128).astype(np.uint8)

    collage = cropped_target_image.copy()
    collage[y1:y2, x1:x2, :] = ref_image_collage

    collage_mask = cropped_target_image.copy() * 0.0
    collage_mask[y1:y2, x1:x2, :] = 1.0
    if enable_shape_control:
        collage_mask = np.stack([cropped_tar_mask, cropped_tar_mask, cropped_tar_mask], -1)

    # the size before pad
    H1, W1 = collage.shape[0], collage.shape[1]

    cropped_target_image = pad_to_square(cropped_target_image, pad_value=0, random=False).astype(np.uint8)
    collage = pad_to_square(collage, pad_value=0, random=False).astype(np.uint8)
    collage_mask = pad_to_square(collage_mask, pad_value=2, random=False).astype(np.uint8)

    # the size after pad
    H2, W2 = collage.shape[0], collage.shape[1]

    cropped_target_image = cv2.resize(cropped_target_image.astype(np.uint8), (512, 512)).astype(np.float32)
    collage = cv2.resize(collage.astype(np.uint8), (512, 512)).astype(np.float32)
    collage_mask = cv2.resize(collage_mask.astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST).astype(
        np.float32)
    collage_mask[collage_mask == 2] = -1

    masked_ref_image = masked_ref_image / 255
    cropped_target_image = cropped_target_image / 127.5 - 1.0
    collage = collage / 127.5 - 1.0
    collage = np.concatenate([collage, collage_mask[:, :, :1]], -1)

    item = dict(ref=masked_ref_image.copy(), jpg=cropped_target_image.copy(), hint=collage.copy(),
                extra_sizes=np.array([H1, W1, H2, W2]),
                tar_box_yyxx_crop=np.array(tar_box_yyxx_crop),
                tar_box_yyxx=np.array(tar_box_yyxx_full),
                )
    return item

def crop_back(pred, tar_image,  extra_sizes, tar_box_yyxx_crop):
    H1, W1, H2, W2 = extra_sizes
    y1, y2, x1, x2 = tar_box_yyxx_crop
    pred = cv2.resize(pred, (W2, H2))
    m = 3  # maigin_pixel

    if W1 == H1:
        tar_image[y1+m:y2-m, x1+m:x2-m, :] = pred[m:-m, m:-m]
        return tar_image

    if W1 < W2:
        pad1 = int((W2 - W1) / 2)
        pad2 = W2 - W1 - pad1
        pred = pred[:, pad1: -pad2, :]
    else:
        pad1 = int((H2 - H1) / 2)
        pad2 = H2 - H1 - pad1
        pred = pred[pad1: -pad2, :, :]
    tar_image[y1+m:y2-m, x1+m:x2-m, :] = pred[m:-m, m:-m]
    return tar_image

def inference_single_image(
    ref_image,
    ref_mask,
    tar_image,
    tar_mask,
    strength,
    ddim_steps,
    scale,
    enable_shape_control,
):
    model, ddim_sampler = init_anydoor_model("anydoor")
    save_memory = False
    raw_background = tar_image.copy()
    item = process_pairs(ref_image, ref_mask, tar_image, tar_mask, enable_shape_control = enable_shape_control)

    ref = item['ref']
    hint = item['hint']
    num_samples = 1

    control = torch.from_numpy(hint.copy()).float().to(device)
    control = torch.stack([control for _ in range(num_samples)], dim=0)
    control = einops.rearrange(control, 'b h w c -> b c h w').clone()

    clip_input = torch.from_numpy(ref.copy()).float().to(device)
    clip_input = torch.stack([clip_input for _ in range(num_samples)], dim=0)
    clip_input = einops.rearrange(clip_input, 'b h w c -> b c h w').clone()

    H, W = 512, 512

    cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning( clip_input )]}
    un_cond = {"c_concat": [control],
               "c_crossattn": [model.get_learned_conditioning([torch.zeros((1,3,224,224))] * num_samples)]}
    shape = (4, H // 8, W // 8)

    if save_memory:
        model.low_vram_shift(is_diffusing=True)

    model.control_scales = ([strength] * 13)
    samples, _ = ddim_sampler.sample(ddim_steps, num_samples,
                                     shape, cond, verbose=False, eta=0,
                                     unconditional_guidance_scale=scale,
                                     unconditional_conditioning=un_cond)

    if save_memory:
        model.low_vram_shift(is_diffusing=False)

    x_samples = model.decode_first_stage(samples)
    x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy()

    result = x_samples[0][:, :, ::-1]
    result = np.clip(result, 0, 255)

    pred = x_samples[0]
    pred = np.clip(pred,0,255)[1:,:,:]
    sizes = item['extra_sizes']
    tar_box_yyxx_crop = item['tar_box_yyxx_crop']
    tar_image = crop_back(pred, tar_image, sizes, tar_box_yyxx_crop)

    # keep background unchanged
    y1, y2, x1, x2 = item['tar_box_yyxx']
    raw_background[y1:y2, x1:x2, :] = tar_image[y1:y2, x1:x2, :]
    return raw_background


def run_webui(base, ref, ref_mask, *args):
    image = base["image"].convert("RGB")
    mask = base["mask"].convert("L")
    ref_image = ref["image"].convert("RGB")
    ref_mask = ref["mask"].convert("L") if not ref_mask else ref_mask.convert("L")
    return run_local(image, mask, ref_image, ref_mask, *args)

def run_local(image, mask, ref_image, ref_mask, reference_mask_refine, num_samples, *args):
    image = np.asarray(image)
    mask = np.asarray(mask)
    mask = np.where(mask > 128, 1, 0).astype(np.uint8)
    ref_image = np.asarray(ref_image)
    ref_mask = np.asarray(ref_mask)
    ref_mask = np.where(ref_mask > 128, 1, 0).astype(np.uint8)

    if ref_mask.sum() == 0:
        raise gr.Error('No mask for the reference image.')

    if mask.sum() == 0:
        raise gr.Error('No mask for the background image.')

    if reference_mask_refine:
        ref_mask = process_image_mask(ref_image, ref_mask)

    results = []
    for _ in range(num_samples):
        synthesis = inference_single_image(ref_image.copy(), ref_mask.copy(), image.copy(), mask.copy(), *args)
        synthesis = torch.from_numpy(synthesis).permute(2, 0, 1)
        synthesis = synthesis.permute(1, 2, 0).numpy()
        results.append(synthesis)
    return results

def run_api(bg_image, ref_image, ref_mask, bg_mask_x1, bg_mask_y1, bg_mask_x2, bg_mask_y2, num_samples, *args):
    bg_mask = Image.new("L", bg_image.size, 0)

    bg_image_w, bg_image_h = bg_image.size
    bg_mask_x1, bg_mask_x2 = bg_mask_x1 * bg_image_w, bg_mask_x2 * bg_image_w
    bg_mask_y1, bg_mask_y2 = bg_mask_y1 * bg_image_h, bg_mask_y2 * bg_image_h
    draw = ImageDraw.Draw(bg_mask)
    draw.rectangle((bg_mask_x1, bg_mask_y1, bg_mask_x2, bg_mask_y2), fill=255)

    return run_local(bg_image, bg_mask, ref_image, ref_mask, num_samples, *args)
