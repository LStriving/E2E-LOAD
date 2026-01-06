import os
import subprocess
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import json
from tqdm import tqdm  # 进度条，需要安装：pip install tqdm

class VideoFrameExtractor:
    def __init__(self, output_dir='frames', width=456, height=256, fps=24, quality=2):
        self.output_dir = output_dir
        self.width = width
        self.height = height
        self.fps = fps
        self.quality = quality  # JPEG质量，1-31，越小质量越好
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 检查ffmpeg是否可用
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        except:
            print("错误: 请先安装ffmpeg并添加到PATH")
            exit(1)
    
    def extract_frames(self, video_path):
        """
        提取单个视频的帧
        """
        try:
            video_path = str(video_path)
            video_name = Path(video_path).stem
            
            # 清理视频名中的特殊字符
            video_name = "".join(c for c in video_name if c.isalnum() or c in (' ', '_', '-')).rstrip()
            
            # 创建帧目录
            frame_dir = os.path.join(self.output_dir, video_name)
            os.makedirs(frame_dir, exist_ok=True)
            
            # 检查是否已处理过（避免重复处理）
            existing_frames = len(list(Path(frame_dir).glob('*.jpg')))
            # calculate expected frames 
            # get video info first
            # info_cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=duration', '-of', 'json', video_path]
            # result = subprocess.run(info_cmd, capture_output=True, text=True)
            # info = json.loads(result.stdout)
            # duration = float(info['streams'][0]['duration'])
            # expected_frames = int(duration * self.fps)

            if existing_frames > 0:
                return {
                    'video': video_path,
                    'frame_dir': frame_dir,
                    'frames_extracted': existing_frames,
                    'status': 'already_exists',
                    'error': None
                }
            
            # 构建ffmpeg命令
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-r', str(self.fps),  # 输出帧率
                '-vf', f'scale={self.width}:{self.height}:force_original_aspect_ratio=decrease,pad={self.width}:{self.height}:(ow-iw)/2:(oh-ih)/2',  # 保持宽高比并填充
                '-q:v', str(self.quality),  # 质量参数
                '-start_number', '0',
                '-loglevel', 'error',  # 减少日志输出
                os.path.join(frame_dir, 'img_%05d.jpg')
            ]
            
            # 执行命令
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # 计算提取的帧数
                frames = list(Path(frame_dir).glob('*.jpg'))
                frames_count = len(frames)
                
                # 如果需要重命名以保持连续编号
                self._rename_frames_sequentially(frame_dir, frames)
                
                return {
                    'video': video_path,
                    'frame_dir': frame_dir,
                    'frames_extracted': frames_count,
                    'processing_time': time.time() - start_time,
                    'status': 'success',
                    'error': None
                }
            else:
                return {
                    'video': video_path,
                    'frame_dir': frame_dir,
                    'frames_extracted': 0,
                    'status': 'failed',
                    'error': result.stderr
                }
                
        except Exception as e:
            return {
                'video': video_path,
                'frame_dir': None,
                'frames_extracted': 0,
                'status': 'error',
                'error': str(e)
            }
    
    def _rename_frames_sequentially(self, frame_dir, frames):
        """确保帧按顺序编号"""
        frames.sort()
        for idx, frame_path in enumerate(frames):
            new_name = f"img_{idx:05d}.jpg"
            if frame_path.name != new_name:
                frame_path.rename(Path(frame_dir) / new_name)
    
    def batch_extract(self, video_paths, max_workers=None, batch_size=10):
        """
        批量提取帧（多进程）
        
        Args:
            video_paths: 视频路径列表
            max_workers: 最大进程数（默认CPU核心数）
            batch_size: 每个进程一次处理的数量
        """
        if max_workers is None:
            max_workers = os.cpu_count() or 4
        
        print(f"开始批量提取，使用 {max_workers} 个进程...")
        print(f"总计 {len(video_paths)} 个视频")
        
        results = []
        failed_videos = []
        
        # 分批处理，避免内存问题
        for i in range(0, len(video_paths), batch_size):
            batch = video_paths[i:i+batch_size]
            
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # 提交任务
                future_to_video = {
                    executor.submit(self.extract_frames, video): video 
                    for video in batch
                }
                
                # 使用进度条
                with tqdm(total=len(batch), desc=f"批次 {i//batch_size + 1}") as pbar:
                    for future in as_completed(future_to_video):
                        video = future_to_video[future]
                        try:
                            result = future.result()
                            results.append(result)
                            
                            if result['status'] == 'success':
                                pbar.set_postfix({
                                    '提取': f"{result['frames_extracted']}帧",
                                    '时间': f"{result.get('processing_time', 0):.1f}s"
                                })
                            elif result['status'] == 'already_exists':
                                pbar.set_postfix({'状态': '已存在'})
                            else:
                                failed_videos.append((video, result['error']))
                                pbar.set_postfix({'状态': '失败'})
                                
                        except Exception as e:
                            failed_videos.append((video, str(e)))
                            pbar.set_postfix({'状态': '异常'})
                        
                        pbar.update(1)
        
        # 输出统计信息
        self._print_statistics(results, failed_videos)
        
        return results, failed_videos
    
    def _print_statistics(self, results, failed_videos):
        """打印统计信息"""
        successful = [r for r in results if r['status'] in ['success', 'already_exists']]
        total_frames = sum(r['frames_extracted'] for r in successful)
        
        print("\n" + "="*50)
        print("处理完成！")
        print(f"成功处理视频: {len(successful)}/{len(results)}")
        print(f"失败视频: {len(failed_videos)}")
        print(f"总计提取帧数: {total_frames}")
        print(f"输出目录: {self.output_dir}")
        
        # 保存失败记录
        if failed_videos:
            fail_log = os.path.join(self.output_dir, 'failed_videos.log')
            with open(fail_log, 'w') as f:
                for video, error in failed_videos:
                    f.write(f"{video}: {error}\n")
            print(f"失败记录已保存至: {fail_log}")
        
        # 保存处理摘要
        summary = {
            'total_videos': len(results),
            'successful': len(successful),
            'failed': len(failed_videos),
            'total_frames': total_frames,
            'output_dir': self.output_dir,
            'settings': {
                'width': self.width,
                'height': self.height,
                'fps': self.fps,
                'quality': self.quality
            }
        }
        
        summary_file = os.path.join(self.output_dir, 'extraction_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"处理摘要已保存至: {summary_file}")

def find_video_files(input_dir, extensions=['.mp4', '.mpg', '.mpeg', '.avi', '.mov']):
    """
    递归查找视频文件
    """
    video_files = []
    
    for root, _, files in os.walk(input_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                video_files.append(os.path.join(root, file))
    
    return video_files

def main():
    parser = argparse.ArgumentParser(description='多进程视频帧提取器')
    parser.add_argument('--input', '-i', type=str, required=True, help='输入目录')
    parser.add_argument('--output', '-o', type=str, default='frames', help='输出目录')
    parser.add_argument('--width', type=int, default=456, help='帧宽度')
    parser.add_argument('--height', type=int, default=256, help='帧高度')
    parser.add_argument('--fps', type=int, default=24, help='提取帧率')
    parser.add_argument('--workers', type=int, default=None, help='进程数（默认CPU核心数）')
    parser.add_argument('--quality', type=int, default=2, help='JPEG质量（1-31，越小质量越好）')
    parser.add_argument('--batch_size', type=int, default=5, help='每批处理视频数')
    parser.add_argument('--skip_existing', action='store_true', help='跳过已存在的帧目录')
    
    args = parser.parse_args()
    
    # 查找视频文件
    print(f"正在搜索视频文件...")
    video_files = find_video_files(args.input)
    
    if not video_files:
        print("未找到视频文件")
        return
    
    # 初始化提取器
    extractor = VideoFrameExtractor(
        output_dir=args.output,
        width=args.width,
        height=args.height,
        fps=args.fps,
        quality=args.quality
    )
    
    # 批量提取
    extractor.batch_extract(video_files, max_workers=args.workers, batch_size=args.batch_size)

if __name__ == "__main__":
    main()
