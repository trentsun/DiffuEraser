import gc
import copy
import cv2
import os
import numpy as np
import torch
import torchvision
import traceback  # 添加 traceback 导入
from einops import repeat
from PIL import Image, ImageFilter
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UniPCMultistepScheduler,
    LCMScheduler,
)
from diffusers.schedulers import TCDScheduler
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.utils.torch_utils import randn_tensor
from transformers import AutoTokenizer, PretrainedConfig

from libs.unet_motion_model import MotionAdapter, UNetMotionModel
from libs.brushnet_CA import BrushNetModel
from libs.unet_2d_condition import UNet2DConditionModel
from diffueraser.pipeline_diffueraser import StableDiffusionDiffuEraserPipeline


checkpoints = {
    "2-Step": ["pcm_{}_smallcfg_2step_converted.safetensors", 2, 0.0],
    "4-Step": ["pcm_{}_smallcfg_4step_converted.safetensors", 4, 0.0],
    "8-Step": ["pcm_{}_smallcfg_8step_converted.safetensors", 8, 0.0],
    "16-Step": ["pcm_{}_smallcfg_16step_converted.safetensors", 16, 0.0],
    "Normal CFG 4-Step": ["pcm_{}_normalcfg_4step_converted.safetensors", 4, 7.5],
    "Normal CFG 8-Step": ["pcm_{}_normalcfg_8step_converted.safetensors", 8, 7.5],
    "Normal CFG 16-Step": ["pcm_{}_normalcfg_16step_converted.safetensors", 16, 7.5],
    "LCM-Like LoRA": [
        "pcm_{}_lcmlike_lora_converted.safetensors",
        2,
        0.0,
    ],
}

def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")

def resize_frames(frames, size=None):    
    if size is not None:
        out_size = size
        process_size = (out_size[0]-out_size[0]%8, out_size[1]-out_size[1]%8)
        frames = [f.resize(process_size) for f in frames]
    else:
        out_size = frames[0].size
        process_size = (out_size[0]-out_size[0]%8, out_size[1]-out_size[1]%8)
        if not out_size == process_size:
            frames = [f.resize(process_size) for f in frames]
        
    return frames

def read_mask(validation_mask, fps, n_total_frames, img_size, mask_dilation_iter, frames):
    cap = cv2.VideoCapture(validation_mask)
    if not cap.isOpened():
        print("Error: Could not open mask video.")
        exit()
    mask_fps = cap.get(cv2.CAP_PROP_FPS)
    if mask_fps != fps:
        cap.release()
        raise ValueError("The frame rate of all input videos needs to be consistent.")

    masks = []
    masked_images = []
    valid_frame_indices = []  # 新增:记录有效掩码的帧索引
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:  
            break
        if(idx >= n_total_frames):
            break
        mask = Image.fromarray(frame[...,::-1]).convert('L')
        if mask.size != img_size:
            mask = mask.resize(img_size, Image.NEAREST)
        mask = np.asarray(mask)
        
        # 判断当前帧是否有掩码
        if not np.any(mask):  # 如果掩码全为0,跳过这一帧
            idx += 1
            continue
            
        m = np.array(mask > 0).astype(np.uint8)
        m = cv2.erode(m,
                    cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                    iterations=1)
        m = cv2.dilate(m,
                    cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                    iterations=mask_dilation_iter)

        mask = Image.fromarray(m * 255)
        masks.append(mask)

        masked_image = np.array(frames[idx])*(1-(np.array(mask)[:,:,np.newaxis].astype(np.float32)/255))
        masked_image = Image.fromarray(masked_image.astype(np.uint8))
        masked_images.append(masked_image)
        
        valid_frame_indices.append(idx)  # 记录有效帧的索引
        idx += 1
    cap.release()

    return masks, masked_images, valid_frame_indices  # 返回有效帧索引

def read_priori(priori, fps, n_total_frames, img_size):
    cap = cv2.VideoCapture(priori)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()
    priori_fps = cap.get(cv2.CAP_PROP_FPS)
    if priori_fps != fps:
        cap.release()
        raise ValueError("The frame rate of all input videos needs to be consistent.")

    prioris=[]
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: 
            break
        if(idx >= n_total_frames):
            break
        img = Image.fromarray(frame[...,::-1])
        if img.size != img_size:
            img = img.resize(img_size)
        prioris.append(img)
        idx += 1
    cap.release()

    os.remove(priori) # remove priori 

    return prioris

def read_video(validation_image, video_length, nframes, max_img_size):
    vframes, aframes, info = torchvision.io.read_video(filename=validation_image, pts_unit='sec', end_pts=video_length) # RGB
    fps = info['video_fps']
    n_total_frames = int(video_length * fps)
    n_clip = int(np.ceil(n_total_frames/nframes))

    frames = list(vframes.numpy())[:n_total_frames]
    frames = [Image.fromarray(f) for f in frames]
    max_size = max(frames[0].size)
    if(max_size<256):
        raise ValueError("The resolution of the uploaded video must be larger than 256x256.")
    if(max_size>4096):
        raise ValueError("The resolution of the uploaded video must be smaller than 4096x4096.")
    if max_size>max_img_size:
        ratio = max_size/max_img_size
        ratio_size = (int(frames[0].size[0]/ratio),int(frames[0].size[1]/ratio))
        img_size = (ratio_size[0]-ratio_size[0]%8, ratio_size[1]-ratio_size[1]%8)
        resize_flag=True
    elif (frames[0].size[0]%8==0) and (frames[0].size[1]%8==0):
        img_size = frames[0].size
        resize_flag=False
    else:
        ratio_size = frames[0].size
        img_size = (ratio_size[0]-ratio_size[0]%8, ratio_size[1]-ratio_size[1]%8)
        resize_flag=True
    if resize_flag:
        frames = resize_frames(frames, img_size)
        img_size = frames[0].size

    return frames, fps, img_size, n_clip, n_total_frames


class DiffuEraser:
    def __init__(
            self, device, base_model_path, vae_path, diffueraser_path, revision=None,
            ckpt="Normal CFG 4-Step", mode="sd15", loaded=None):
        self.device = device

        ## load model
        self.vae = AutoencoderKL.from_pretrained(vae_path).half().to(device)
        self.noise_scheduler = DDPMScheduler.from_pretrained(base_model_path, 
                subfolder="scheduler",
                prediction_type="v_prediction",
                timestep_spacing="trailing",
                rescale_betas_zero_snr=True
            )
        self.tokenizer = AutoTokenizer.from_pretrained(
                    base_model_path,
                    subfolder="tokenizer",
                    use_fast=False,
                )
        text_encoder_cls = import_model_class_from_model_name_or_path(base_model_path,revision)
        self.text_encoder = text_encoder_cls.from_pretrained(
                base_model_path, subfolder="text_encoder"
            ).half().to(device)
        self.brushnet = BrushNetModel.from_pretrained(diffueraser_path, subfolder="brushnet").half().to(device)
        self.unet_main = UNetMotionModel.from_pretrained(
            diffueraser_path, subfolder="unet_main",
        ).half().to(device)

        ## set pipeline
        self.pipeline = StableDiffusionDiffuEraserPipeline.from_pretrained(
            base_model_path,
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet_main,
            brushnet=self.brushnet
        ).to(self.device, torch.float16)
        
        self.pipeline.scheduler = UniPCMultistepScheduler.from_config(self.pipeline.scheduler.config)
        self.pipeline.set_progress_bar_config(disable=True)
        self.noise_scheduler = UniPCMultistepScheduler.from_config(self.pipeline.scheduler.config)
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True)

        ## use PCM
        self.ckpt = ckpt
        PCM_ckpts = checkpoints[ckpt][0].format(mode)
        self.guidance_scale = checkpoints[ckpt][2]
        if loaded != (ckpt + mode):
            self.pipeline.load_lora_weights(
                "weights/PCM_Weights", weight_name=PCM_ckpts, subfolder=mode
            )
            loaded = ckpt + mode

            if ckpt == "LCM-Like LoRA":
                self.pipeline.scheduler = LCMScheduler()
            else:
                self.pipeline.scheduler = TCDScheduler(
                    num_train_timesteps=1000,
                    beta_start=0.00085,
                    beta_end=0.012,
                    beta_schedule="scaled_linear",
                    timestep_spacing="trailing",
                )
        self.num_inference_steps = checkpoints[ckpt][1]
        print(f"ckpt: {ckpt}")
        print(f"num_inference_steps: {self.num_inference_steps}")
        self.guidance_scale = 0

    def forward(self, frames, masks, priori, output_path,
                max_img_size = 1280, video_length=2, mask_dilation_iter=4,
                nframes=32, seed=None, revision = None, guidance_scale=None, blended=True):
        """
        主要的处理函数，完成视频修复的整个流程
        
        参数:
        - frames: 输入视频帧列表
        - masks: 输入掩码列表
        - priori: 先验视频路径
        - output_path: 输出视频路径
        - max_img_size: 最大图像尺寸
        - video_length: 处理的视频长度
        - mask_dilation_iter: 掩码扩张程度
        - nframes: 每次处理的帧数
        - seed: 随机种子
        - revision: 模型版本
        - guidance_scale: 引导尺度
        - blended: 是否使用混合
        """
        import time
        start_time = time.time()
        
        print("\n=== DiffuEraser 处理开始 ===")
        
        # 记录有掩码的帧(仅用于显示信息)
        valid_frame_indices = []
        for i, mask in enumerate(masks):
            if np.any(np.array(mask)):  # 检查掩码是否包含非零值
                valid_frame_indices.append(i)
                
        print(f"处理总帧数: {len(frames)}, 其中有掩码的帧数: {len(valid_frame_indices)}")
        
        validation_prompt = ""  # 
        guidance_scale_final = self.guidance_scale if guidance_scale==None else guidance_scale

        if (max_img_size<256 or max_img_size>1920):
            raise ValueError("The max_img_size must be larger than 256, smaller than 1920.")

        ################ 处理输入帧 ################ 
        t0 = time.time()
        # 计算目标尺寸
        img_size = frames[0].size
        max_size = max(img_size)
        
        if(max_size<256):
            raise ValueError("The resolution of the input frames must be larger than 256x256.")
        if(max_size>4096):
            raise ValueError("The resolution of the input frames must be smaller than 4096x4096.")
            
        if max_size>max_img_size:
            ratio = max_size/max_img_size
            ratio_size = (int(img_size[0]/ratio),int(img_size[1]/ratio))
            img_size = (ratio_size[0]-ratio_size[0]%8, ratio_size[1]-ratio_size[1]%8)
            resize_flag=True
        elif (img_size[0]%8==0) and (img_size[1]%8==0):
            resize_flag=False
        else:
            ratio_size = img_size
            img_size = (ratio_size[0]-ratio_size[0]%8, ratio_size[1]-ratio_size[1]%8)
            resize_flag=True
            
        if resize_flag:
            frames = resize_frames(frames, img_size)
            masks = resize_frames(masks, img_size)
        print(f"帧处理完成，耗时: {time.time() - t0:.2f}秒")
        print(f"处理分辨率: {img_size}")

        ################ 读取先验视频 ################
        t0 = time.time()
        cap = cv2.VideoCapture(priori)
        if not cap.isOpened():
            print("Error: Could not open priori video.")
            return frames
            
        prioris = []
        frame_count = 0
        while frame_count < len(frames):
            ret, frame = cap.read()
            if not ret:
                break
            img = Image.fromarray(frame[...,::-1])
            if img.size != img_size:
                img = img.resize(img_size)
            prioris.append(img)
            frame_count += 1
        cap.release()
        os.remove(priori)
        print(f"先验视频处理完成，耗时: {time.time() - t0:.2f}秒")

        ################ 准备数据 ################
        t0 = time.time()
        validation_images_input = []
        process_frames = []
        process_masks = []
        
        for idx in range(len(frames)):
            frame = frames[idx]
            mask = masks[idx]
            process_frames.append(frame)
            process_masks.append(mask)
            masked_image = np.array(frame)*(1-(np.array(mask)[:,:,np.newaxis].astype(np.float32)/255))
            masked_image = Image.fromarray(masked_image.astype(np.uint8))
            validation_images_input.append(masked_image)
        print(f"数据准备完成，耗时: {time.time() - t0:.2f}秒")

        ################ 模型推理 ################
        print("\n开始模型推理...")
        inference_start = time.time()
        
        if seed is None:
            print("使用随机种子")
            generator = None
        else:
            print(f"使用固定种子: {seed}")
            generator = torch.Generator(device=self.device).manual_seed(seed)

        # 准备噪声
        t0 = time.time()
        tar_width, tar_height = frames[0].size
        shape = (
            nframes,
            4,
            tar_height//8,
            tar_width//8
        )
        if self.text_encoder is not None:
            prompt_embeds_dtype = self.text_encoder.dtype
        elif self.unet_main is not None:
            prompt_embeds_dtype = self.unet_main.dtype
        else:
            prompt_embeds_dtype = torch.float16
            
        noise_pre = randn_tensor(shape, device=torch.device(self.device), dtype=prompt_embeds_dtype, generator=generator)
        noise = repeat(noise_pre, "t c h w->(repeat t) c h w", repeat=int(np.ceil(len(frames)/nframes)))[:len(frames),...]
        print(f"噪声准备完成，耗时: {time.time() - t0:.2f}秒")

        # 处理先验
        t0 = time.time()
        images_preprocessed = []
        for image in prioris:
            image = self.image_processor.preprocess(image, height=tar_height, width=tar_width).to(dtype=torch.float32)
            image = image.to(device=torch.device(self.device), dtype=torch.float16)
            images_preprocessed.append(image)
        pixel_values = torch.cat(images_preprocessed)
        print(f"先验预处理完成，耗时: {time.time() - t0:.2f}秒")

        t0 = time.time()
        with torch.no_grad():
            pixel_values = pixel_values.to(dtype=torch.float16)
            latents = []
            num=4
            for i in range(0, pixel_values.shape[0], num):
                latents.append(self.vae.encode(pixel_values[i : i + num]).latent_dist.sample())
            latents = torch.cat(latents, dim=0)
        latents = latents * self.vae.config.scaling_factor
        print(f"潜在编码完成，耗时: {time.time() - t0:.2f}秒")

        timesteps = torch.tensor([0], device=self.device)
        timesteps = timesteps.long()

        validation_masks_input_ori = copy.deepcopy(process_masks)
        resized_frames_ori = copy.deepcopy(process_frames)

        ################ 预推理阶段 ################
        t0 = time.time()
        if len(frames) > nframes*2:  # 仅当帧数超过nframes*2时进行预推理
            print("\n执行预推理阶段...")
            ## sample
            step = len(frames) / nframes
            sample_index = [int(i * step) for i in range(nframes)]
            sample_index = sample_index[:32]  # 限制最大采样数为22
            
            print(f"预推理采样帧数: {len(sample_index)}")
            validation_masks_input_pre = [process_masks[i] for i in sample_index]
            validation_images_input_pre = [validation_images_input[i] for i in sample_index]
            latents_pre = torch.stack([latents[i] for i in sample_index])

            ## add priori
            noisy_latents_pre = self.noise_scheduler.add_noise(latents_pre, noise_pre[:len(sample_index)], timesteps) 
            latents_pre = noisy_latents_pre

            with torch.no_grad():
                latents_pre_out = self.pipeline(
                    num_frames=len(sample_index),  # 使用实际的采样帧数
                    prompt=validation_prompt, 
                    images=validation_images_input_pre, 
                    masks=validation_masks_input_pre, 
                    num_inference_steps=self.num_inference_steps, 
                    generator=generator,
                    guidance_scale=guidance_scale_final,
                    latents=latents_pre,
                ).latents
            torch.cuda.empty_cache()  

            def decode_latents(latents, weight_dtype):
                latents = 1 / self.vae.config.scaling_factor * latents
                video = []
                for t in range(latents.shape[0]):
                    video.append(self.vae.decode(latents[t:t+1, ...].to(weight_dtype)).sample)
                video = torch.concat(video, dim=0)
                video = video.float()
                return video
                
            with torch.no_grad():
                video_tensor_temp = decode_latents(latents_pre_out, weight_dtype=torch.float16)
                images_pre_out = self.image_processor.postprocess(video_tensor_temp, output_type="pil")
            torch.cuda.empty_cache()  

            ## replace input frames with updated frames
            black_image = Image.new('L', masks[0].size, color=0)
            for i,index in enumerate(sample_index):
                latents[index] = latents_pre_out[i]
                process_masks[index] = black_image
                validation_images_input[index] = images_pre_out[i]
                process_frames[index] = images_pre_out[i]
            print(f"预推理完成，耗时: {time.time() - t0:.2f}秒")
        else:
            print("跳过预推理阶段(总帧数不足nframes*2)")
        gc.collect()
        torch.cuda.empty_cache()

        ################ 主推理阶段 ################
        print("\n开始主推理...")
        t0 = time.time()
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        latents = noisy_latents
        
        with torch.no_grad():
            processed_images = self.pipeline(
                num_frames=nframes,
                prompt=validation_prompt,
                images=validation_images_input,
                masks=process_masks,
                num_inference_steps=self.num_inference_steps,
                generator=generator,
                guidance_scale=guidance_scale_final,
                latents=latents,
            ).frames
        print(f"主推理完成，耗时: {time.time() - t0:.2f}秒")
        print(f"总推理耗时: {time.time() - inference_start:.2f}秒")

        ################ 合成视频 ################
        t0 = time.time()
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"),
                            24, frames[0].size)
                            
        if blended:
            # 使用高斯模糊进行混合
            mask_blurreds = []
            for mask in validation_masks_input_ori:
                mask_blurred = cv2.GaussianBlur(np.array(mask), (21, 21), 0)/255.
                binary_mask = 1-(1-np.array(mask)/255.) * (1-mask_blurred)
                mask_blurreds.append(Image.fromarray((binary_mask*255).astype(np.uint8)))
            binary_masks = mask_blurreds
        else:
            binary_masks = validation_masks_input_ori
        
        # 将处理后的帧插回原始帧列表
        result_frames = frames.copy()
        for idx in range(len(frames)):
            mask = np.expand_dims(np.array(binary_masks[idx]),2).repeat(3, axis=2).astype(np.float32)/255.
            img = (np.array(processed_images[idx]).astype(np.uint8) * mask \
                + np.array(resized_frames_ori[idx]).astype(np.uint8) * (1 - mask)).astype(np.uint8)
            if resize_flag:
                img = cv2.resize(img, frames[0].size)
            result_frames[idx] = Image.fromarray(img)
                            
        # 写入所有帧
        for frame in result_frames:
            writer.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
                
        writer.release()
        print(f"视频合成完成，耗时: {time.time() - t0:.2f}秒")
        
        total_time = time.time() - start_time
        print("\n=== 处理完成 ===")
        print(f"总耗时: {total_time:.2f}秒")

        return result_frames
            



