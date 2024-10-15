port tkinter as tk
from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient("mongodb+srv://jonathancruz2011:iamgod3875@clusterenro.qb6n0yq.mongodb.net/?retryWrites=true&w=majority&appName=ClusterENRO")
db = client["ENRO"]
collection = db["Users"]

def login():
    username = entry_username.get()
    password = entry_password.get()

    user = collection.find_one({"username": username, "password": password})

    if user:
        result_label.config(text="Login Successful", fg="green")
    else:
        result_label.config(text="Invalid Username or Password", fg="red")

# Set up the main application window
root = tk.Tk()
root.title("Login Screen")

# Create UI elements
label_username = tk.Label(root, text="Username:")
label_username.pack()

entry_username = tk.Entry(root)
entry_username.pack()

label_password = tk.Label(root, text="Password:")
label_password.pack()

entry_password = tk.Entry(root, show="*")
entry_password.pack()

login_button = tk.Button(root, text="Login", command=login)
login_button.pack()

result_label = tk.Label(root, text="")
result_label.pack()

# Run the application
root.mainloop()