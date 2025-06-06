import torch
import os 
import time
import argparse
import cv2
import numpy as np
from diffueraser.diffueraser import DiffuEraser
from propainter.inference import Propainter, get_device
import torchvision
from PIL import Image
import gc


def merge_video_segments(segment_frames_list, output_path, fps):
    """合并多个视频片段
    Args:
        segment_frames_list: 视频片段帧列表的列表
        output_path: 最终输出路径
        fps: 视频帧率
    """
    if not segment_frames_list:
        return
        
    # 获取第一帧的尺寸信息
    first_frame = segment_frames_list[0][0]
    height, width = np.array(first_frame).shape[:2]
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    final_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 依次写入每个片段的帧
    for frames in segment_frames_list:
        for frame in frames:
            final_video.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
            
    final_video.release()

def main():
    ## input params
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_video', type=str, default="examples/example3/video.mp4", help='Path to the input video')
    parser.add_argument('--input_mask', type=str, default="examples/example3/mask.mp4" , help='Path to the input mask')
    parser.add_argument('--video_length', type=int, default=10, help='The maximum length of output video')
    parser.add_argument('--mask_dilation_iter', type=int, default=8, help='Adjust it to change the degree of mask expansion')
    parser.add_argument('--max_img_size', type=int, default=960, help='The maximum length of output width and height')
    parser.add_argument('--save_path', type=str, default="results" , help='Path to the output')
    parser.add_argument('--ref_stride', type=int, default=10, help='Propainter params')
    parser.add_argument('--neighbor_length', type=int, default=10, help='Propainter params')
    parser.add_argument('--subvideo_length', type=int, default=50, help='Propainter params')
    parser.add_argument('--base_model_path', type=str, default="weights/stable-diffusion-v1-5" , help='Path to sd1.5 base model')
    parser.add_argument('--vae_path', type=str, default="weights/sd-vae-ft-mse" , help='Path to vae')
    parser.add_argument('--diffueraser_path', type=str, default="weights/diffuEraser" , help='Path to DiffuEraser')
    parser.add_argument('--propainter_model_dir', type=str, default="weights/propainter" , help='Path to priori model')
    parser.add_argument('--sampling_scale', type=float, default=0.5, help='Scale factor for down/up sampling')
    parser.add_argument('--enable_sampling', default=False, action='store_true', help='Enable video sampling processing')
    parser.add_argument('--segment_frames', type=int, default=250, help='Number of frames per segment')
    args = parser.parse_args()
                  
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        
    # 获取视频信息
    cap = cv2.VideoCapture(args.input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise ValueError("无法获取有效的视频帧率，请检查视频文件。")
    total_frames = int(fps * args.video_length)
    cap.release()
    
    # 计算需要处理的片段数
    num_segments = (total_frames + args.segment_frames - 1) // args.segment_frames
    segment_results = []
    
    ## model initialization
    device = get_device()
    # PCM params
    ckpt = "LCM-Like LoRA"
    video_inpainting_sd = DiffuEraser(device, args.base_model_path, args.vae_path, args.diffueraser_path, ckpt=ckpt)
    propainter = Propainter(args.propainter_model_dir, device=device)
    
    start_time = time.time()
    
    # 打开视频和掩码文件流
    video_stream = cv2.VideoCapture(args.input_video)
    mask_stream = cv2.VideoCapture(args.input_mask)
    
    try:
        # 处理每个片段
        for i in range(num_segments):
            print(f"处理第 {i+1}/{num_segments} 个视频片段")
            start_frame = i * args.segment_frames
            num_frames = min(args.segment_frames, total_frames - start_frame)
            
            # 读取当前片段的视频帧
            frames = []
            for _ in range(num_frames):
                ret, frame = video_stream.read()
                if not ret:
                    break
                # 将BGR转换为RGB并创建PIL Image
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                frames.append(frame_pil)
            
            # 读取当前片段的掩码帧
            masks = []
            for _ in range(num_frames):
                ret, frame = mask_stream.read()
                if not ret:
                    break
                mask = Image.fromarray(frame[...,::-1]).convert('L')
                masks.append(mask)
            
            if len(frames) != len(masks):
                raise ValueError(f"视频帧数({len(frames)})与掩码帧数({len(masks)})不匹配")
            
            # 创建临时文件路径
            temp_priori = os.path.join(args.save_path, f"temp_priori_{i}.mp4")
            temp_output = os.path.join(args.save_path, f"temp_output_{i}.mp4")
            
            # 处理当前片段
            propainter.forward(frames, masks, temp_priori,
                            ref_stride=args.ref_stride, neighbor_length=args.neighbor_length, 
                            subvideo_length=args.subvideo_length, mask_dilation=args.mask_dilation_iter)
            
            segment_result = video_inpainting_sd.forward(frames, masks, temp_priori, temp_output,
                                    max_img_size=args.max_img_size, video_length=num_frames/fps,
                                    mask_dilation_iter=args.mask_dilation_iter)
            
            # 保存当前片段结果
            segment_results.append(segment_result)
            end_time = time.time()  
            inference_time = end_time - start_time  
            print(f"处理第 {i+1}/{num_segments} 个视频片段 inference time: {inference_time:.4f} s")
            # 清理临时文件和内存
            # if os.path.exists(temp_priori):
            #     os.remove(temp_priori)
            
            # 释放当前批次的内存
            del frames
            del masks
            torch.cuda.empty_cache()
            gc.collect()
    
    finally:
        # 确保文件流被正确关闭
        video_stream.release()
        mask_stream.release()
    
    # 合并所有片段
    final_output = os.path.join(args.save_path, "diffueraser_result.mp4")
    merge_video_segments(segment_results, final_output, fps)
    
    end_time = time.time()  
    inference_time = end_time - start_time  
    print(f"DiffuEraser inference time: {inference_time:.4f} s")
    
    torch.cuda.empty_cache()

if __name__ == '__main__':
    main()


   