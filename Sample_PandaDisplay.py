import pandas as pd
import customtkinter as ctk

ctk.set_appearance_mode("dark")

# Create a sample DataFrame
data = {
    'Column1': [1, 2, 3],
    'Column2': ['A', 'B', 'C'],
}
df = pd.DataFrame(data)

# Initialize CustomTkinter
app = ctk.CTk()

# Convert the DataFrame to string
df_string = df.to_string(index=False)

# Create a label to display the DataFrame
label = ctk.CTkLabel(app, text=df_string, anchor="w")  # Use anchor="w" for left alignment
label.pack(padx=20, pady=20)

# Start the application
app.mainloop()