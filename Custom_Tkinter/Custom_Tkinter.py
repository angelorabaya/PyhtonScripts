import customtkinter as ctk
from pathlib import Path
import sys
from PIL import Image

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

ctk.set_appearance_mode("dark")
root = ctk.CTk()
root.title("Authentication")
root.wm_attributes('-toolwindow',True)
center_window(root,205,290)

image_path = relative_to_assets("image_1.png")
image_image_1 = ctk.CTkImage(dark_image=Image.open(image_path),size=(102,72))

image_pathb = relative_to_assets("image_2.png")
image_image_2 = ctk.CTkImage(dark_image=Image.open(image_pathb),size=(25,25))

image_pathc = relative_to_assets("image_3.png")
image_image_3 = ctk.CTkImage(dark_image=Image.open(image_pathc),size=(20,20))

labela = ctk.CTkLabel(root, image=image_image_1, text="")
labela.pack(padx=0, pady=20)

# Create a frame for username
frameun = ctk.CTkFrame(root)
frameun.pack(padx=0, pady=10)

image_labelun = ctk.CTkLabel(frameun, image=image_image_2, text="")
image_labelun.pack(side='left')

entryuser = ctk.CTkEntry(frameun, placeholder_text="", width=130, height=27, border_width=1, corner_radius=7, font=("Helvetica",14),takefocus=True)
entryuser.pack(padx=0, pady=3)

# Create a frame for password
framepw = ctk.CTkFrame(root)
framepw.pack(padx=0, pady=2)

image_labelpw = ctk.CTkLabel(framepw, image=image_image_3, text="")
image_labelpw.pack(side='left', padx=(3, 0), pady=5)

entrypass = ctk.CTkEntry(framepw, placeholder_text="", width=130, height=27, border_width=1, corner_radius=7, show="*", font=("Helvetica",14),takefocus=True)
entrypass.pack(side='left', padx=(2, 0))

btna = ctk.CTkButton(master=root, width=120, height=30, border_width=0, corner_radius=8, text="OK", command=lambda: print("Button"))
btna.pack(padx=0, pady=20)

entryuser.focus_set()

root.mainloop()
