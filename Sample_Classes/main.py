import customtkinter as ctk
from secondform import SecondForm

def main():
    # Initialize the main application
    ctk.set_appearance_mode("Dark")  # Options: "System" (default), "Dark", "Light"
    ctk.set_default_color_theme("green")  # Options: "blue" (default), "green", "dark-blue"

    root = ctk.CTk()  # Create the main window
    root.title("Main Application")
    root.destroy()
    root.mainloop()

    rootb = ctk.CTk()  # Create the main window
    rootb.title("Second Window")
    SecondForm(rootb)  # Create an instance of SecondForm
    rootb.mainloop()

    #root.mainloop()

    # Setting a default value for the Entry variable from main.py
    #second_form_instance.entry_var.set("Hello, World!")

if __name__ == "__main__":
    main()