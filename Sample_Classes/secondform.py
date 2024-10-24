import customtkinter as ctk

class SecondForm:
    def __init__(self, master):
        self.master = master  # Reference to the parent window
        self.frame = ctk.CTkFrame(master)
        self.frame.pack(padx=10, pady=10)

        self.entry_var = ctk.StringVar()  # Entry variable
        self.entry = ctk.CTkEntry(self.frame, textvariable=self.entry_var)
        self.entry.pack(pady=10)

        self.submit_button = ctk.CTkButton(self.frame, text="Submit", command=self.submit)
        self.submit_button.pack(padx=5, pady=5)

    def submit(self):
        #print(f"Submitted value: {self.entry_var.get()}")
        self.entry_var.set("Test")