import tkinter as tk
from tkinter import ttk


def create_data_grid():
    root = tk.Tk()
    root.title("Data Grid Example")

    # Create a Treeview widget
    tree = ttk.Treeview(root, columns=("Name", "Age", "City"), show="headings")

    # Define the column headings
    tree.heading("Name", text="Name")
    tree.heading("Age", text="Age")
    tree.heading("City", text="City")

    # Example data to display
    data = [
        ("Alice", 30, "New York"),
        ("Bob", 25, "Los Angeles"),
        ("Charlie", 35, "Chicago"),
        ("A1", 35, "Location1"),
        ("A2", 35, "Location2"),
        ("A3", 35, "Location3"),
        ("A4", 35, "Location4"),
        ("A5", 35, "Location5"),
        ("A6", 35, "Location6"),
        ("A7", 35, "Location7"),
        ("A8", 35, "Location8"),
        ("A9", 35, "Location9"),
    ]

    # Insert data into the Treeview
    for item in data:
        tree.insert("", tk.END, values=item)

    # Add a scrollbar
    scrollbar = ttk.Scrollbar(root, orient="vertical", command=tree.yview)
    tree.configure(yscroll=scrollbar.set)
    scrollbar.pack(side='right', fill='y')

    # Pack the Treeview widget
    tree.pack(expand=True, fill=tk.BOTH)

    root.mainloop()


create_data_grid()