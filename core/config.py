CNN_WIDTH = 192
CNN_HEIGHT = 128
IMAGE_SIZE = (CNN_WIDTH, CNN_HEIGHT)

NUM_DIGIT_CLASSES = 10
NUM_DIGIT_SLOTS = 4
NUM_DIVIDERS = 4

# CLAHE parameters used in dataset transforms & UI previews
CLAHE_CLIP_LIMIT = 2.0
CLAHE_GRID_SIZE = (8, 8)

# Default filter thresholds
FILTER_CONF_THRESH = 0.5      # Minimum confidence to accept prediction
FILTER_ENT_THRESH = 1.5       # Maximum entropy to accept prediction
FILTER_JUMP_THRESH = 0.1      # Maximum weight change (kg) between frames

# Strict mode thresholds
STRICT_CONF_THRESH = 0.85
STRICT_ENT_THRESH = 0.3
STRICT_JUMP_THRESH = 0.05

class VideoParams:
    _DEFAULTS = {
        'brightness' : 0,
        'saturation' : 100,
        'contrast' : 100,
        'gamma': 100,
        'temporal_avg_enabled': False,
        'temporal_avg_window': 5,
    }

    def __init__(
        self,
        trim_start: int = 0,
        trim_end: int | None = None,
        crop_coords: tuple[int, int, int, int] | None = None, # x, y, w, h
        warp_quad: list[tuple[int, int]] | None = None,
        warp_enabled: bool = False,
        temporal_avg_enabled: bool | None = None,
        temporal_avg_window: int | None = None,
        # validate() uses ranges:
        # brightness: -255 - 255
        # saturation: 0 - 300
        # contrast: 0 - 200
        # gamma: 50 - 200 (100 = no change)
        brightness: int | None = None, 
        saturation: int | None = None,
        contrast: int | None = None,
        gamma: int | None = None
    ):
        self.trim_start = trim_start
        self.trim_end = trim_end
        self.crop_coords = crop_coords
        self.warp_quad = warp_quad
        self.warp_enabled = warp_enabled
        self.temporal_avg_enabled = self._DEFAULTS['temporal_avg_enabled'] if temporal_avg_enabled is None else temporal_avg_enabled
        self.temporal_avg_window = self._DEFAULTS['temporal_avg_window'] if temporal_avg_window is None else temporal_avg_window
        self.brightness = self._DEFAULTS['brightness'] if brightness is None else brightness
        self.saturation = self._DEFAULTS['saturation'] if saturation is None else saturation
        self.contrast = self._DEFAULTS['contrast'] if contrast is None else contrast
        self.gamma = self._DEFAULTS['gamma'] if gamma is None else gamma
        self.validate()

    def __eq__(self, other):
        if not isinstance(other, VideoParams):
            return NotImplemented
        return (self.trim_start == other.trim_start and self.trim_end == other.trim_end and self.crop_coords == other.crop_coords
            and self.warp_quad == other.warp_quad and self.warp_enabled == other.warp_enabled
                and self.temporal_avg_enabled == other.temporal_avg_enabled and self.temporal_avg_window == other.temporal_avg_window
                and self.brightness == other.brightness and self.saturation == other.saturation
                and self.contrast == other.contrast and self.gamma == other.gamma)

    def validate(self):
        if self.trim_start < 0:
            raise ValueError("trim_start must be non-negative")
        if self.trim_end is not None and self.trim_end <= self.trim_start:
            raise ValueError("trim_end must be greater than trim_start")
        if self.crop_coords is not None:
            if not isinstance(self.crop_coords, tuple) or len(self.crop_coords) != 4:
                raise ValueError("crop_coords must be a tuple of four integers")
            if not all(isinstance(x, int) for x in self.crop_coords):
                raise ValueError("all crop_coords values must be non-negative integers")
        if self.warp_quad is not None:
            if not isinstance(self.warp_quad, list) or len(self.warp_quad) != 4:
                raise ValueError("warp_quad must be a list of four (x, y) points")
            for point in self.warp_quad:
                if not isinstance(point, tuple) or len(point) != 2:
                    raise ValueError("each warp_quad point must be a tuple (x, y)")
                if not all(isinstance(coord, int) for coord in point):
                    raise ValueError("warp_quad coordinates must be integers")
        if not isinstance(self.warp_enabled, bool):
            raise ValueError("warp_enabled must be a boolean")
        if not isinstance(self.temporal_avg_enabled, bool):
            raise ValueError("temporal_avg_enabled must be a boolean")
        if not isinstance(self.temporal_avg_window, int) or self.temporal_avg_window < 1 or self.temporal_avg_window % 2 == 0:
            raise ValueError("temporal_avg_window must be an odd integer >= 1")
        if not (-255 <= self.brightness <= 255):
            raise ValueError("brightness must be between -255 and 255")
        if not (0 <= self.saturation <= 300):
            raise ValueError("saturation must be between 0 and 300")
        if not (0 <= self.contrast <= 200):
            raise ValueError("contrast must be between 0 and 200")
        if not (50 <= self.gamma <= 200):
            raise ValueError("gamma must be between 50 and 200")

    @classmethod
    def get_default_value(cls, param_name: str):
        return cls._DEFAULTS.get(param_name.lower())