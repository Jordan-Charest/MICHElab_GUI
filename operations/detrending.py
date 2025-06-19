import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
import sys
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from utils.hdf5 import get_data_from_dataset, save_data_to_dataset
from utils.detrending import compute_dff_using_minfilter
from utils.parsing import parse_key_value_args

def detrend_data(filename, dataset_path, window=60, sigma1=1, sigma2=10, offset=0):

    data, attributes = get_data_from_dataset(filename, dataset_path)

    if data.ndim == 1:
        data = compute_dff_using_minfilter(data, window=window, sigma1=sigma1, sigma2=sigma2, offset=offset)
    elif data.ndim == 3:
        for row in range(data.shape[1]):
            for col in range(data.shape[2]):
                data[:,row,col] = compute_dff_using_minfilter(data[:,row,col], window=window, sigma1=sigma1, sigma2=sigma2, offset=offset)
    else:
        raise ValueError("Cannot detrend data that is not 1D or 3D.")
    
    return data, attributes


def apply_detrending_to_data(filename, dataset_paths, window=60, sigma1=1, sigma2=10, offset=0, copy_name=None):

    for dataset_path in dataset_paths:
    
        data, attributes = detrend_data(filename, dataset_path, window=window, sigma1=sigma1, sigma2=sigma2, offset=offset)

        attributes.update({"detrend_window": window, "detrend_sigma1": sigma1, "detrend_sigma2": sigma2, "detrend_offset": offset})

        if copy_name is None:
            save_data_to_dataset(filename, dataset_path, data, attributes)
        else:
            save_data_to_dataset(filename, dataset_path+"_"+copy_name, data, attributes)
class DetrendPreviewApp:
    def __init__(self, root, filename, dataset_list):
        self.root = root
        self.root.title("Detrending Preview")
        self.filename = filename
        self.dataset_list = dataset_list
        self.preview_path = dataset_list[0]
        self.window = 60
        self.sigma1 = 1.0
        self.sigma2 = 10.0
        self.offset = 10.0
        self.copy_name = None

        tk.Label(root, text="Select Dataset").grid(row=0, column=0, padx=10, pady=5)
        self.dataset_combobox = ttk.Combobox(root, values=dataset_list, state="readonly")
        self.dataset_combobox.current(0)
        self.dataset_combobox.grid(row=0, column=1, padx=10, pady=5)
        self.dataset_combobox.bind("<<ComboboxSelected>>", self.update_preview)

        self.create_input("Window", 1, self.window)
        self.create_input("Sigma1", 2, self.sigma1)
        self.create_input("Sigma2", 3, self.sigma2)
        self.create_input("Offset", 4, self.offset)

        self.make_copy_var = tk.BooleanVar()
        self.make_copy_check = tk.Checkbutton(root, text="Make Copy", variable=self.make_copy_var, command=self.toggle_copy_name)
        self.make_copy_check.grid(row=5, column=0, columnspan=2, pady=5)

        tk.Label(root, text="Copy Name").grid(row=6, column=0, padx=10, pady=5)
        self.copy_name_entry = tk.Entry(root, state="disabled")
        self.copy_name_entry.grid(row=6, column=1, padx=10, pady=5)

        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(5, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().grid(row=0, column=2, rowspan=7, padx=10, pady=5)

        tk.Button(root, text="Preview", command=self.update_preview).grid(row=7, column=0, columnspan=2, padx=10, pady=10)
        tk.Button(root, text="Compute", command=self.compute).grid(row=8, column=0, columnspan=2, padx=10, pady=10)
        
        self.update_preview()

    def create_input(self, label, row, default):
        tk.Label(self.root, text=label).grid(row=row, column=0, padx=10, pady=5)
        entry = tk.Entry(self.root)
        entry.insert(tk.END, str(default))
        entry.grid(row=row, column=1, padx=10, pady=5)
        setattr(self, f"{label.lower()}_entry", entry)

    def update_preview(self, event=None):
        try:
            self.preview_path = self.dataset_combobox.get()
            self.window = int(self.window_entry.get())
            self.sigma1 = float(self.sigma1_entry.get())
            self.sigma2 = float(self.sigma2_entry.get())
            self.offset = float(self.offset_entry.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Enter valid numeric values.")
            return
        
        original_data, _ = get_data_from_dataset(self.filename, self.preview_path)
        original_data = np.nanmean(original_data, axis=(1,2))

        preview_data, _ = detrend_data(self.filename, self.preview_path, self.window, self.sigma1, self.sigma2, self.offset)
        preview_data = np.nanmean(preview_data, axis=(1,2))
        
        self.ax1.clear()
        self.ax1.plot(original_data, label="Original Data")
        self.ax1.set_title(f"Original Data (Min: {original_data.min():.2f}, Max: {original_data.max():.2f})")
        self.ax1.legend()
        
        self.ax2.clear()
        self.ax2.plot(preview_data, label="Detrended Data", color="orange")
        self.ax2.legend()
        
        self.canvas.draw()

    def toggle_copy_name(self):
        if self.make_copy_var.get():
            self.copy_name_entry.config(state="normal")
        else:
            self.copy_name_entry.config(state="disabled")
            self.copy_name_entry.delete(0, tk.END)

    def compute(self):
        try:
            self.window = int(self.window_entry.get())
            self.sigma1 = float(self.sigma1_entry.get())
            self.sigma2 = float(self.sigma2_entry.get())
            self.offset = float(self.offset_entry.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Enter valid numeric values.")
            return

        self.copy_name = self.copy_name_entry.get().strip() if self.make_copy_var.get() else None
        self.root.quit()
        self.result = (self.preview_path, self.window, self.sigma1, self.sigma2, self.offset, self.copy_name)

def main():
    if "--batch" in sys.argv: # batch mode
        
        filename = sys.argv[2]
        datasets = sys.argv[3].split(',')
        args = parse_key_value_args(sys.argv[4:])
        
        apply_detrending_to_data(filename, datasets, window=args["window"], sigma1=args["sigma1"], sigma2=args["sigma2"], offset=args["offset"], copy_name=args["copy_name"])

    else: # GUI mode

        if len(sys.argv) < 3:
            print("Usage: python script.py <HDF5 file> <dataset1> [<dataset2> ...]")

        else:
            hdf5_file = sys.argv[1]
            datasets = sys.argv[2:]

            root = tk.Tk()
            app = DetrendPreviewApp(root, hdf5_file, datasets)
            root.mainloop()

            preview_path, window, sigma1, sigma2, offset, copy_name = app.result
            
            apply_detrending_to_data(hdf5_file, datasets, window=window, sigma1=sigma1, sigma2=sigma2, offset=offset, copy_name=copy_name)
        
if __name__ == "__main__":
    main()