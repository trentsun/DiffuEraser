import cv2
import numpy as np

class RegionSelector:
    def __init__(self, window_name):
        self.window_name = window_name
        self.start_point = None
        self.end_point = None
        self.is_drawing = False
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start_point = (x, y)
            self.is_drawing = True
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.is_drawing:
                self.end_point = (x, y)
                
        elif event == cv2.EVENT_LBUTTONUP:
            self.end_point = (x, y)
            self.is_drawing = False

def select_region_from_video(video_path):
    """
    从视频中选择区域
    返回: (x1, x2, y1, y2) 坐标
    """
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("无法打开视频文件")
    
    # 读取第一帧
    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise ValueError("无法读取视频帧")
    
    # 创建窗口和选择器
    window_name = "选择区域 (按Enter确认，ESC取消)"
    cv2.namedWindow(window_name)
    selector = RegionSelector(window_name)
    cv2.setMouseCallback(window_name, selector.mouse_callback)
    
    drawing_frame = frame.copy()
    while True:
        if selector.start_point and selector.end_point:
            drawing_frame = frame.copy()
            cv2.rectangle(drawing_frame, selector.start_point, selector.end_point, (0, 255, 0), 2)
        
        cv2.imshow(window_name, drawing_frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == 13:  # Enter键
            break
        elif key == 27:  # ESC键
            selector.start_point = None
            selector.end_point = None
            break
    
    cv2.destroyAllWindows()
    cap.release()
    
    if selector.start_point and selector.end_point:
        x1 = min(selector.start_point[0], selector.end_point[0])
        x2 = max(selector.start_point[0], selector.end_point[0])
        y1 = min(selector.start_point[1], selector.end_point[1])
        y2 = max(selector.start_point[1], selector.end_point[1])
        return (x1, x2, y1, y2)
    return None

def generate_mask_video(output_path, width=1080, height=1920, 
                       mask_area=(0, 1080, 1200, 1700), 
                       duration_seconds=10, fps=30):
    """
    生成mask视频
    参数:
        output_path: 输出视频路径
        width: 视频宽度 (1080)
        height: 视频高度 (1920)
        mask_area: mask区域 (x1, x2, y1, y2)
        duration_seconds: 视频时长(秒)
        fps: 帧率
    """
    # 创建VideoWriter对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)
    
    # 创建单帧mask
    mask = np.zeros((height, width), dtype=np.uint8)
    x1, x2, y1, y2 = mask_area
    mask[y1:y2, x1:x2] = 255
    
    # 写入视频帧 - 将浮点数转换为整数
    total_frames = int(duration_seconds * fps)
    for _ in range(total_frames):
        out.write(mask)
    
    # 释放资源
    out.release()

if __name__ == "__main__":
    # 从视频中选择区域
    input_video = "/Users/sundongzhe/workspace/aigc/DiffuEraser/output_clip_no_blackscreen.mp4"  # 替换为您的输入视频路径
    
    try:
        # 获取视频信息
        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            raise ValueError("无法打开视频文件")
            
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_seconds = frame_count / fps
        cap.release()
        
        # 选择区域
        print("请在视频帧上拖动鼠标选择要遮罩的区域...")
        mask_area = select_region_from_video(input_video)
        
        if mask_area:
            # 生成mask视频
            output_path = "mask.mp4"
            generate_mask_video(
                output_path=output_path,
                width=width,
                height=height,
                mask_area=mask_area,
                duration_seconds=duration_seconds,
                fps=fps
            )
            print(f"Mask视频已生成: {output_path}")
            print(f"选择的区域坐标: {mask_area}")
        else:
            print("未选择区域，操作已取消")
            
    except Exception as e:
        print(f"发生错误: {str(e)}") 