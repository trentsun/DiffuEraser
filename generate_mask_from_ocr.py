import cv2
import numpy as np
import json
from PIL import Image

def load_ocr_results(json_path):
    """
    加载OCR结果文件
    Args:
        json_path: OCR结果JSON文件路径
    Returns:
        list: 字幕轨道信息列表
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        ocr_data = json.load(f)
    print(f"加载OCR数据: 共{len(ocr_data)}个字幕轨道")
    return ocr_data

def merge_overlapping_intervals(intervals):
    """
    合并重叠的时间区间
    Args:
        intervals: 时间区间列表，每个元素为 (start_frame, end_frame)
    Returns:
        list: 合并后的时间区间列表
    """
    if not intervals:
        return []
    
    # 按开始时间排序
    intervals.sort(key=lambda x: x[0])
    
    merged = [intervals[0]]
    for current in intervals[1:]:
        previous = merged[-1]
        # 如果当前区间的开始时间小于等于前一个区间的结束时间，则合并
        if current[0] <= previous[1]:
            merged[-1] = (previous[0], max(previous[1], current[1]))
        else:
            merged.append(current)
    
    return merged

def generate_mask_from_ocr(video_path, ocr_json_path, output_path, mask_dilation=4):
    """
    根据OCR结果生成掩码视频
    Args:
        video_path: 输入视频路径
        ocr_json_path: OCR结果JSON文件路径
        output_path: 输出掩码视频路径
        mask_dilation: 掩码膨胀迭代次数
    """
    print(f"\n开始处理视频: {video_path}")
    print(f"OCR数据文件: {ocr_json_path}")
    
    # 读取视频信息
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("无法打开视频文件")
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    print(f"视频信息: {width}x{height}, {fps}fps, 总帧数: {total_frames}")
    
    # 加载OCR结果
    ocr_data = load_ocr_results(ocr_json_path)
    
    height_threshold = height // 2
    print(f"高度阈值设置为: {height_threshold} (视频高度的一半)")
    
    # 创建一个字典来存储每一帧的字幕区域
    frame_bboxes = {}
    
    # 收集每一帧的字幕框信息
    for track in ocr_data:
        for bbox_info in track['dynamic_bboxes']:
            frame_num = int(bbox_info[0])
            x1, y1, x2, y2 = map(int, bbox_info[1])
            
            # 只处理在阈值以下的字幕
            if y1 >= height_threshold:
                if frame_num not in frame_bboxes:
                    frame_bboxes[frame_num] = []
                frame_bboxes[frame_num].append((x1, y1, x2, y2))
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)
    
    mask_frame_count = 0
    
    # 为每一帧生成掩码
    print("\n开始生成掩码视频...")
    for frame_idx in range(total_frames):
        actual_frame = frame_idx + 1
        if actual_frame % 100 == 0:
            print(f"处理进度: {actual_frame}/{total_frames}")
        
        # 创建空白掩码
        mask = np.zeros((height, width), dtype=np.uint8)
        frame_has_mask = False
        
        # 如果当前帧有字幕框，则在掩码上绘制
        if actual_frame in frame_bboxes:
            for bbox in frame_bboxes[actual_frame]:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
                frame_has_mask = True
        
        if frame_has_mask:
            mask_frame_count += 1
            
            # 对掩码进行膨胀处理
            if mask_dilation > 0:
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                mask = cv2.dilate(mask, kernel, iterations=mask_dilation)
        
        # 写入掩码帧
        out.write(mask)
    
    # 释放资源
    out.release()
    
    print(f"\n处理完成:")
    print(f"总帧数: {total_frames}")
    print(f"生成掩码的帧数: {mask_frame_count}")
    print(f"掩码视频已保存到: {output_path}")
    
    return output_path

if __name__ == "__main__":
    # 示例用法
    video_path = "2.mp4"
    ocr_json_path = "ocr_result.json"
    output_path = "2_mask.mp4"
    generate_mask_from_ocr(video_path, ocr_json_path, output_path) 