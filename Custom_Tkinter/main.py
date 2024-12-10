import customtkinter as ctk
from tkinter import messagebox
from PIL import Image
from Module import center_window, relative_to_assets
from DB import fetch_data
from main_form import MainForm

class LoginForm:
    def __init__(self, master):
        self.master = master

        self.master.title("Authentication")
        self.master.wm_attributes('-toolwindow', True)
        center_window(root, 205, 290)
        root_bg_color = self.master.cget("fg_color")

        def button_click():
            result = fetch_data("Users",{'username': entryuser.get(), 'password': int(entrypass.get())})
            if result is not None:
                #root.withdraw()
                #root.quit()
                root.destroy()

                main_root = ctk.CTk()
                main_app = MainForm(main_root)
                #main_root.protocol("WM_DELETE_WINDOW", lambda: [root.destroy(), main_root.destroy()])
                main_root.mainloop()
            else:
                messagebox.showinfo("Information", "Invalid username or password!")


        image_path = relative_to_assets("image_1.png")
        image_image_1 = ctk.CTkImage(dark_image=Image.open(image_path),size=(102,72))

        image_pathb = relative_to_assets("image_2.png")
        image_image_2 = ctk.CTkImage(dark_image=Image.open(image_pathb),size=(25,25))

        image_pathc = relative_to_assets("image_3.png")
        image_image_3 = ctk.CTkImage(dark_image=Image.open(image_pathc),size=(20,20))

        labela = ctk.CTkLabel(self.master, image=image_image_1, text="")
        labela.pack(padx=0, pady=20)

        # Create a frame for username
        frameun = ctk.CTkFrame(self.master, fg_color=root_bg_color)
        frameun.pack(padx=0, pady=10)

        image_labelun = ctk.CTkLabel(frameun, image=image_image_2, text="")
        image_labelun.pack(side='left')

        entryuser = ctk.CTkEntry(frameun, placeholder_text="", width=130, height=27, border_width=1, corner_radius=7, font=("Helvetica",14),takefocus=True)
        entryuser.pack(padx=0, pady=3)

        # Create a frame for password
        framepw = ctk.CTkFrame(self.master, fg_color=root_bg_color)
        framepw.pack(padx=0, pady=2)

        image_labelpw = ctk.CTkLabel(framepw, image=image_image_3, text="")
        image_labelpw.pack(side='left', padx=(3, 0), pady=5)

        entrypass = ctk.CTkEntry(framepw, placeholder_text="", width=130, height=27, border_width=1, corner_radius=7, show="*", font=("Helvetica",14),takefocus=True)
        entrypass.pack(side='left', padx=(2, 0))

        btna = ctk.CTkButton(master=self.master, width=120, height=30, border_width=0, corner_radius=8, text="OK", command=lambda: button_click())
        btna.pack(padx=0, pady=20)

        self.master.after(100, entryuser.focus_set)

if __name__ == "__main__":
    root = ctk.CTk()
    ctk.set_appearance_mode("dark")
    app = LoginForm(root)
    root.mainloop()
