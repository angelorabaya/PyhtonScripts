import customtkinter as ctk
from Module import center_window
from DB import fetch_data, fetch_data_many
from tkinter import messagebox, ttk


class MainForm:
    def __init__(self, master):
        self.master = master
        master.title("PTO Point-of-Sale Module")
        master.wm_attributes('-toolwindow', True)
        center_window(self.master, 500, 500)
        root_bg_color = self.master.cget("fg_color")

        entry_ctrl_value = ctk.StringVar()
        entry_name_value = ctk.StringVar()
        entry_date_value = ctk.StringVar()
        entry_nature_value = ctk.StringVar()
        entry_total_value = ctk.StringVar()

        def get_details(*args):
            result = fetch_data_many("Assessment_dtl", {'aop_control': 'AOP' + entry_ctrl.get()})
            if result is not None:
                for document in result:
                    tree.insert('', 'end', values=[document['aop_item'],document['aop_volume'],document['aop_charge'],document['aop_total']])
            else:
                messagebox.showinfo("Information", "No record found!")

        def get_info(*args):
            result = fetch_data("Assessment_hdr", {'aop_control': 'AOP' + entry_ctrl.get()})
            if result is not None:
                entry_name_value.set(result.get('ph_cname'))
                entry_date_value.set(result.get('aop_date'))
                entry_nature_value.set(result.get('aop_nature'))
                format_number = f"{result.get('aop_total'):,}"
                entry_total_value.set(format_number)
                get_details()
            else:
                messagebox.showinfo("Information", "No record found!")

        def clear_fields(*args):
            entry_name_value.set("")
            entry_date_value.set("")
            entry_nature_value.set("")
            entry_total_value.set("")
            for item in tree.get_children():
                tree.delete(item)

        #create control no
        frame_ctrl = ctk.CTkFrame(self.master, fg_color=root_bg_color)
        frame_ctrl.pack(fill='x', padx=20, pady=(15,5))
        label_ctrl = ctk.CTkLabel(frame_ctrl, text="Control No.: AOP", anchor='w')
        label_ctrl.pack(side='left')
        entry_ctrl = ctk.CTkEntry(frame_ctrl, textvariable=entry_ctrl_value, border_width=1, font=("Helvetica",16, "bold"))
        entry_ctrl.pack(side='left', padx=(5, 0))
        entry_ctrl.bind("<Return>", get_info)
        entry_ctrl_value.trace("w", clear_fields)

        #create name
        frame_name = ctk.CTkFrame(self.master, fg_color=root_bg_color)
        frame_name.pack(fill='x', padx=20, pady=5)
        label_name = ctk.CTkLabel(frame_name, text="Name", anchor='w')
        label_name.pack(side='left')
        entry_name = ctk.CTkEntry(frame_name, textvariable=entry_name_value, width=360, height=54, border_width=1, state="readonly")
        entry_name.pack(side='left', padx=(64, 0))

        #create date
        frame_date = ctk.CTkFrame(self.master, fg_color=root_bg_color)
        frame_date.pack(fill='x', padx=20, pady=5)
        label_date = ctk.CTkLabel(frame_date, text="Date", anchor='w')
        label_date.pack(side='left')
        entry_date = ctk.CTkEntry(frame_date, textvariable=entry_date_value, border_width=1, state="readonly")
        entry_date.pack(side='left', padx=(71, 0))

        #create nature
        frame_nature = ctk.CTkFrame(self.master, fg_color=root_bg_color)
        frame_nature.pack(fill='x', padx=20, pady=5)
        label_nature = ctk.CTkLabel(frame_nature, text="Nature", anchor='w')
        label_nature.pack(side='left')
        entry_nature = ctk.CTkEntry(frame_nature, textvariable=entry_nature_value, width=360, height=27, border_width=1, state="readonly")
        entry_nature.pack(side='left', padx=(60, 0))

        #create total
        frame_total = ctk.CTkFrame(self.master, fg_color=root_bg_color)
        frame_total.pack(fill='x', padx=20, pady=5)
        label_total = ctk.CTkLabel(frame_total, text="Total", anchor='w')
        label_total.pack(side='left')
        entry_total = ctk.CTkEntry(frame_total, textvariable=entry_total_value, border_width=1, state="readonly")
        entry_total.pack(side='left', padx=(69, 0))

        style = ttk.Style()
        style.configure("Treeview", background="lightblue", foreground="black")
        style.map("Treeview",
                  background=[("selected", "blue")],  # Change selected row color
                  foreground=[("selected", "white")])  # Change text color of selected row

        #create datagrid
        frame_datagrid = ctk.CTkFrame(self.master)
        frame_datagrid.pack(fill='x', padx=20, pady=10)
        columns = ['aop_item', 'aop_volume', 'aop_charge', 'aop_total']
        tree = ttk.Treeview(frame_datagrid, columns=columns, show='headings', height=4)
        tree.heading("aop_item", text="Item", anchor='w')
        tree.column("aop_item", anchor='w', width=200)
        tree.heading("aop_volume", text="Unit", anchor='w')
        tree.column("aop_volume", anchor='w', width=80)
        tree.heading("aop_charge", text="Charge", anchor='w')
        tree.column("aop_charge", anchor='w', width=100)
        tree.heading("aop_total", text="Total", anchor='w')
        tree.column("aop_total", anchor='w', width=100)
        tree.pack(expand=True, fill='both')

        self.master.after(100, entry_ctrl.focus_set)

if __name__ == "__main__":
    root = ctk.CTk()
    ctk.set_appearance_mode("dark")
    app = MainForm(root)
    root.mainloop()