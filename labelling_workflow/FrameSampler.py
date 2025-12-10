import math

class FrameSampler:
    """Calculate target frames for labelling using dynamic touchstone sampling."""
    
    def __init__(self, total_frames: int, valid_start: int, valid_end: int, target_labels: int):
        """
        Initialize frame sampler.
        
        Args:
            total_frames: Total frames in video
            valid_start: Start frame of valid range (inclusive)
            valid_end: End frame of valid range (inclusive)
            target_labels: Target number of labels
        
        Raises:
            ValueError: If parameters are invalid
        """
        if valid_start < 0 or valid_end >= total_frames:
            raise ValueError("Valid range must be within video bounds")
        if valid_start >= valid_end:
            raise ValueError("valid_start must be less than valid_end")
        if target_labels <= 0:
            raise ValueError("target_labels must be positive")
        
        self.total_frames = total_frames
        self.valid_start = valid_start
        self.valid_end = valid_end
        self.target_labels = target_labels
        self.valid_frame_count = valid_end - valid_start + 1
    
    def get_target_frames(self) -> tuple[list[int], int, bool]:
        """
        Calculate target frames to label.
        
        Returns:
            tuple: (target_frames list, interval m, warning_flag)
                - target_frames: List of frame indices to label
                - m: Calculated interval (every m-th frame)
                - warning_flag: True if fewer labels than target (video too short)
        """
        interval_between_labels = math.floor(self.valid_frame_count / self.target_labels)
        
        warning = False
        if interval_between_labels < 1:
            # Video is too short
            warning = True
            interval_between_labels = 1
        
        target_frames = []
        current_frame = self.valid_start
        
        while current_frame <= self.valid_end and len(target_frames) < self.target_labels:
            target_frames.append(current_frame)
            current_frame += interval_between_labels
        
        return target_frames, interval_between_labels, warning
