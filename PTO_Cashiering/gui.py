from pathlib import Path
import sys
from tkinter import Tk, Canvas, PhotoImage, Entry, Button, messagebox
import pymongo

def connect_to_mongo(uri, db_name):
    client = pymongo.MongoClient(uri)
    return client[db_name]

mongo_uri = "mongodb+srv://jonathancruz2011:iamgod3875@clusterenro.qb6n0yq.mongodb.net/?retryWrites=true&w=majority&appName=ClusterENRO"
db_name = "ENRO"
collection_name = "Users"

# Establish a connection to the database
db = connect_to_mongo(mongo_uri, db_name)
collection = db[collection_name]

def get_base_path():
    if hasattr(sys, '_MEIPASS'):
        return Path(sys._MEIPASS) / 'assets' / 'frame0'
    return Path(__file__).parent / 'assets' / 'frame0'

ASSETS_PATH = get_base_path()

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

def center_window(window, width, height):
    # Calculate the x and y coordinates to center the window
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()

    x = (screen_width // 2) - (width // 2)
    y = (screen_height // 2) - (height // 2)

    window.geometry(f"{width}x{height}+{x}+{y}")

def authenticate_entry(username, password):
    # Query the database for the specific aop_control value
    query = {'username': username, 'password': password}
    #result = collection.find_one(query)
    result = collection.find_one(query)

    if result:
        print("user existed")
        window.destroy()
    else:
        #print("Invalid username or password!")
        messagebox.showinfo("Information", "Invalid username or password!")

window = Tk()
window.title("Authentication")

#window.geometry("205x335")
window.configure(bg = "#130101")
center_window(window,205,335)
#window.overrideredirect(True)
window.wm_attributes('-toolwindow',True)

canvas = Canvas(
    window,
    bg = "#130101",
    height = 305,
    width = 205,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge"
)

canvas.place(x = 0, y = 0)
entry_image_1 = PhotoImage(
    file=relative_to_assets("entry_1.png"))
entry_bg_1 = canvas.create_image(
    119.0,
    166.0,
    image=entry_image_1
)
entry_1 = Entry(
    bd=0,
    bg="#514D4D",
    fg="#c2c6d1",
    highlightthickness=0,
    font=("Helvetica",12),
    takefocus=True
)
entry_1.place(
    x=58.0,
    y=150.0,
    width=122.0,
    height=30.0
)

entry_image_2 = PhotoImage(
    file=relative_to_assets("entry_2.png"))
entry_bg_2 = canvas.create_image(
    119.0,
    212.0,
    image=entry_image_2
)
entry_2 = Entry(
    bd=0,
    bg="#514D4D",
    fg="#c2c6d1",
    highlightthickness=0,
    show="*",
    font=("Helvetica",12),
    takefocus=True
)
entry_2.place(
    x=58.0,
    y=196.0,
    width=122.0,
    height=30.0
)

button_image_1 = PhotoImage(
    file=relative_to_assets("button_1.png"))
button_1 = Button(
    image=button_image_1,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: authenticate_entry(entry_1.get(),int(entry_2.get())),
    relief="flat",
    takefocus=True
)
button_1.place(
    x=28.0,
    y=248.0,
    width=148.0,
    height=46.0
)

image_image_1 = PhotoImage(
    file=relative_to_assets("image_1.png"))
image_1 = canvas.create_image(
    102.0,
    72.0,
    image=image_image_1
)

image_image_2 = PhotoImage(
    file=relative_to_assets("image_2.png"))
image_2 = canvas.create_image(
    28.0,
    166.0,
    image=image_image_2
)

image_image_3 = PhotoImage(
    file=relative_to_assets("image_3.png"))
image_3 = canvas.create_image(
    28.0,
    210.0,
    image=image_image_3
)
window.resizable(False, False)
entry_1.focus_set()

# Optional: Customize tab order
#entry_1.bind("<Tab>", lambda event: entry_2.focus_set())
#entry_2.bind("<Tab>", lambda event: button_1.focus_set())

window.mainloop()
