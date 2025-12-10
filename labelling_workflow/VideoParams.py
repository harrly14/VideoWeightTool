class VideoParams:
    """Parameters for video display adjustments during labelling (brightness, saturation, contrast)."""
    
    _DEFAULTS = {
        'brightness': 0,
        'saturation': 100,
        'contrast': 100
    }

    def __init__(
        self,
        brightness: int | None = None,
        saturation: int | None = None,
        contrast: int | None = None
    ):
        """
        Initialize video parameters.
        
        Args:
            brightness: Brightness adjustment (-255 to 255, default 0)
            saturation: Saturation adjustment (0 to 300, default 100 is normal)
            contrast: Contrast adjustment (0 to 200, default 100 is normal)
        """
        self.brightness = self._DEFAULTS['brightness'] if brightness is None else brightness
        self.saturation = self._DEFAULTS['saturation'] if saturation is None else saturation
        self.contrast = self._DEFAULTS['contrast'] if contrast is None else contrast
        self.validate()

    def __eq__(self, other):
        if not isinstance(other, VideoParams):
            return NotImplemented
        return (
            self.brightness == other.brightness
            and self.saturation == other.saturation
            and self.contrast == other.contrast
        )

    def validate(self):
        """Validate parameter ranges."""
        if not (-255 <= self.brightness <= 255):
            raise ValueError("brightness must be between -255 and 255")
        if not (0 <= self.saturation <= 300):
            raise ValueError("saturation must be between 0 and 300")
        if not (0 <= self.contrast <= 200):
            raise ValueError("contrast must be between 0 and 200")

    @classmethod
    def get_default_value(cls, param_name: str):
        """Get default value for a parameter."""
        return cls._DEFAULTS.get(param_name.lower())