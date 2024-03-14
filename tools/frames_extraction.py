import math
import os
import decord
import cv2
from tqdm import tqdm
import numpy as np
import multiprocessing
import os
import socket

video_info = {}


def extract_frames(input_folder,input_video, output_folder, fps, new_height=None, new_width=None, resume=False, node_num=1, rank=0, convert_to_rgb=False, single_thread=False):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    video_files = []
    assert os.path.exists(input_folder), f'Input folder {input_folder} does not exist'
    assert os.path.isdir(input_folder), f'Input folder {input_folder} is not a folder'
    if input_video is not None:
        assert input_video.endswith(('.avi', '.mp4', '.mpg')), f'Input video {input_video} is not a video file'
        assert os.path.exists(os.path.join(input_folder, input_video)), f'Input video {input_video} does not exist'
    for filename in os.listdir(input_folder):
        if input_video is not None:
            if filename != input_video:
                continue
        filepath = os.path.join(input_folder, filename)
        if os.path.isfile(filepath) and filename.lower().endswith(('.avi', '.mp4', '.mpg')):
            # 检查是否已经提取过帧
            file_prefix = os.path.splitext(os.path.basename(filepath))[0]
            output_filepath = os.path.join(output_folder, file_prefix)
            if os.path.exists(output_filepath):
                # actual frames extracted
                extracted_frame_num = os.listdir(output_filepath)
                extracted_frame_num = len(extracted_frame_num)
                # expected frames
                # read info from video
                if file_prefix not in video_info:
                    video = decord.VideoReader(filepath)
                    video_info[file_prefix] = {}
                    video_info[file_prefix]['frame_count'] = len(video)
                    video_info[file_prefix]['fps'] = video.get_avg_fps()
                ori_frame_num = video_info[file_prefix]['frame_count']
                ori_fps = video_info[file_prefix]['fps']
                expected_frame_num = math.floor(ori_frame_num / ori_fps * 24)
                if extracted_frame_num == expected_frame_num and resume:
                    continue
            video_files.append(filepath)

    if node_num != 0:
        # split video_files
        video_files = np.array(video_files)
        video_files = np.array_split(video_files, node_num)[rank].tolist()
    print(f'Start to process {len(video_files)} video file(s)')

    if single_thread:
        for video_file in video_files:
            process_video(video_file, output_folder, fps, resume, new_height, new_width,convert_to_rgb)
        return
    # 使用多进程处理视频文件
    num_processes = multiprocessing.cpu_count() - 4 if len(video_files) > 4 else len(video_files)
    pool = multiprocessing.Pool(processes=num_processes)
    for video_file in video_files:
        pool.apply_async(process_video, args=(video_file, output_folder, fps, resume, new_height, new_width,convert_to_rgb))
    pool.close()
    pool.join()
    print("done!")


def process_video(video_file, output_folder, fps, resume, new_height, new_width,convert_to_rgb):
    # 检查是否已经提取过帧
    file_prefix = os.path.splitext(os.path.basename(video_file))[0]
    output_filepath = os.path.join(output_folder, file_prefix)
    if not os.path.exists(output_filepath):
        os.makedirs(output_filepath)

    # 打开视频文件
    video = decord.VideoReader(video_file)

    # 获取视频的基本信息
    num_frames = len(video)
    frame_rate = video.get_avg_fps()

    # 确定需要的视频帧数和对应的索引
    num_extracted_frames = math.floor(num_frames * fps / frame_rate)
    extracted_num_in_disk = len(os.listdir(output_filepath))
    start=0
    if extracted_num_in_disk == num_extracted_frames and resume:
        print(f'Video {video_file} has already been processed')
        return
    elif resume and extracted_num_in_disk > 0:
        start = extracted_num_in_disk - 1
    indexes = np.linspace(0, num_frames - 1, num=num_extracted_frames).astype(int)
    indexes = indexes[start:]
    print(f'Total extracted frames of video {video_file}: {num_extracted_frames}')

    # 提取并保存视频帧
    frame_count = extracted_num_in_disk
    for idx in tqdm(indexes):
        frame = video[idx].asnumpy()
        frame_filename = f"{file_prefix}_{frame_count}.jpg"
        frame_filepath = os.path.join(output_filepath, frame_filename)
        if new_height is not None and new_width is not None:
            frame = cv2.resize(frame, (new_width, new_height))
        # change to RGB
        if convert_to_rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imwrite(frame_filepath, frame)
        frame_count += 1

    if num_extracted_frames != frame_count:
        print(f'Warning: expected {num_extracted_frames} frames, but extracted {frame_count} frames')
    print(f'Extracted {frame_count} frames from video {video_file}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str, default='videos')
    parser.add_argument("--input_video", type=str, default=None, help="input video file")
    parser.add_argument('--output_folder', type=str, default='extracted_frames')
    parser.add_argument('--fps', type=int, default=24)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--node_num', type=int, default=1)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument("--new_height", type=int, default=None)
    parser.add_argument("--new_width", type=int, default=None)
    parser.add_argument("--convert_to_rgb", action='store_true')
    args = parser.parse_args()
    resume = args.resume
    print(f'Working on {socket.gethostname()}')
    extract_frames(args.input_folder, args.input_video, args.output_folder, args.fps,
                    args.new_height, args.new_width,
                    resume, args.node_num, args.rank, args.convert_to_rgb)
