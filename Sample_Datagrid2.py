import tkinter as tk
import customtkinter as ctk

# Sample data
data = [
    ["Name", "Age", "City"],
    ["Alice", 30, "New York"],
    ["Bob", 25, "Los Angeles"],
    ["Charlie", 35, "Chicago"],
]

class DataGrid(ctk.CTkFrame):
    def __init__(self, master, data):
        super().__init__(master)

        for i, row in enumerate(data):
            for j, item in enumerate(row):
                cell = ctk.CTkLabel(self, text=item)
                cell.grid(row=i, column=j, padx=5, pady=5)

def main():
    ctk.set_appearance_mode("dark")  # Modes: "system", "light", "dark"
    ctk.set_default_color_theme("blue")  # Customize theme

    root = ctk.CTk()
    root.title("CustomTkinter DataGrid")

    grid_frame = DataGrid(root, data)
    grid_frame.pack(padx=20, pady=20)

    root.mainloop()

if __name__ == "__main__":
    main()