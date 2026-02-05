# Character map for CTC decoding (digits 0-9 and decimal point)
# Index 11 is reserved for CTC blank token
CHAR_MAP = {i: str(i) for i in range(10)}
CHAR_MAP[10] = '.'

# Minimum non-blank token count for reliable confidence
MIN_NONBLANK_COUNT = 3

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
        'contrast' : 100
    }

    def __init__(
        self,
        trim_start: int = 0,
        trim_end: int | None = None,
        crop_coords: tuple[int, int, int, int] | None = None, # x, y, w, h
        # validate() uses ranges:
        # brightness: -255 - 255
        # saturation: 0 - 300
        # contrast: 0 - 200
        brightness: int | None = None, 
        saturation: int | None = None,
        contrast: int | None = None
    ):
        self.trim_start = trim_start
        self.trim_end = trim_end
        self.crop_coords = crop_coords
        self.brightness = self._DEFAULTS['brightness'] if brightness is None else brightness
        self.saturation = self._DEFAULTS['saturation'] if saturation is None else saturation
        self.contrast = self._DEFAULTS['contrast'] if contrast is None else contrast
        self.validate()

    def __eq__(self, other):
        if not isinstance(other, VideoParams):
            return NotImplemented
        return (self.trim_start == other.trim_start and self.trim_end == other.trim_end and self.crop_coords == other.crop_coords
                and self.brightness == other.brightness and self.saturation == other.saturation and self.contrast == other.contrast)            

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
        if not (-255 <= self.brightness <= 255):
            raise ValueError("brightness must be between -255 and 255")
        if not (0 <= self.saturation <= 300):
            raise ValueError("saturation must be between 0 and 300")
        if not (0 <= self.contrast <= 200):
            raise ValueError("contrast must be between 0 and 200")

    @classmethod
    def get_default_value(cls, param_name: str):
        return cls._DEFAULTS.get(param_name.lower())