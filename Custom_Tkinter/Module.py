from pathlib import Path
import sys

def center_window(window, width, height):
    # Calculate the x and y coordinates to center the window
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()

    x = (screen_width // 2) - (width // 2)
    y = (screen_height // 2) - (height // 2)

    window.geometry(f"{width}x{height}+{x}+{y}")

def get_base_path():
    if hasattr(sys, '_MEIPASS'):
        return Path(sys._MEIPASS) / 'assets' / 'frame0'
    return Path(__file__).parent / 'assets' / 'frame0'

ASSETS_PATH = get_base_path()

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)