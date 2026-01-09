class VideoSampleInfo:
    def __init__(self, video_path, label, work_indices, 
                 long_memory_indices=None):
        self.video_path = video_path
        self.label = label
        self.work_indices = work_indices  # 存储转换后的原始视频帧下标
        self.long_memory_indices = long_memory_indices # (indices, mask)