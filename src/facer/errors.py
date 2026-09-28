"""Library exceptions, independent of HTTP frameworks."""


class InvalidImageError(ValueError):
    """An image cannot be read or is not a nonempty uint8 BGR array."""


class NoFaceError(ValueError):
    """No face was detected in a verification input."""


class CropOutOfBoundsError(ValueError):
    """A requested face crop does not overlap the image."""


class ModelNotLoadedError(RuntimeError):
    """A required ONNX model is missing or cannot be loaded."""
