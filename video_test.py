from moviepy.editor import *
# pip install moviepy
input_video = "/home/frank/Documents/output_fig5_bucket.001_ply.mp4"
output_video = "/home/frank/Documents/output_video.mp4"
import cv2

def crop_video(input_video_path, output_video_path, x, y, width, height):
    # 读取输入视频
    input_video = cv2.VideoCapture(input_video_path)

    # 获取视频的一些属性
    fps = int(input_video.get(cv2.CAP_PROP_FPS))
    frame_count = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))

    # 使用XVID编码器
    codec = cv2.VideoWriter_fourcc(*'XVID')

    # 创建输出视频
    output_video = cv2.VideoWriter(output_video_path, codec, fps, (width, height))

    # 对视频逐帧进行裁剪
    for _ in range(frame_count):
        ret, frame = input_video.read()

        if not ret:
            break

        cropped_frame = frame[y:y + height, x:x + width]
        output_video.write(cropped_frame)

    # 释放资源
    input_video.release()
    output_video.release()

# 示例用法
h = 900
w = 900
crop_video(input_video, output_video, int(960-w/2), int(540 - h/2), w, h)