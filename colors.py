from enum import Enum
# enum for basic color values (RGB, 0-1 range)
class BasicColor(Enum):
    RED    = (1.0, 0.0, 0.0)
    GREEN  = (0.0, 1.0, 0.0)
    BLUE   = (0.0, 0.0, 1.0)
    YELLOW = (1.0, 1.0, 0.0)
    CYAN   = (0.0, 1.0, 1.0)
    MAGENTA= (1.0, 0.0, 1.0)
    BLACK  = (0.0, 0.0, 0.0)
    WHITE  = (1.0, 1.0, 1.0)
    ORANGE = (1.0, 0.5, 0.0)
    PURPLE = (0.5, 0.0, 0.5)
    BROWN  = (0.6, 0.3, 0.1)
    PINK   = (1.0, 0.4, 0.7)
    GRAY   = (0.5, 0.5, 0.5)
    LIGHT_BLUE = (0.6, 0.6, 1.0)

# Mapping from color name (lowercase) to the BasicColor enum value
COLOR_MAP = {
    "red": BasicColor.RED,
    "green": BasicColor.GREEN,
    "blue": BasicColor.BLUE,
    "yellow": BasicColor.YELLOW,
    "cyan": BasicColor.CYAN,
    "magenta": BasicColor.MAGENTA,
    "black": BasicColor.BLACK,
    "white": BasicColor.WHITE,
    "orange": BasicColor.ORANGE,
    "purple": BasicColor.PURPLE,
    "brown": BasicColor.BROWN,
    "pink": BasicColor.PINK,
    "gray": BasicColor.GRAY,
    "light_blue": BasicColor.LIGHT_BLUE,
    # additional color names (aliases)
    "violet": BasicColor.PURPLE,
}

def known_color_names() -> list[str]:
    return [color_name for color_name in COLOR_MAP.keys()]


def color_to_rgb(color_name: str):
    """Convert a color name to an (r, g, b) numpy array. Returns mid-gray for unknown names."""
    import numpy as np

    entry = COLOR_MAP.get(color_name.lower())
    if entry is None:
        return np.array([0.5, 0.5, 0.5])
    return np.array(entry.value)