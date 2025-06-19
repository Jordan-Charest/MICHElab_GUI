import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
import sys
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from utils.hdf5 import get_data_from_dataset, save_data_to_dataset
from scipy.signal import butter, filtfilt
from utils.parsing import parse_key_value_args

def bandpass_filter(filename, dataset_path, lowcut, highcut, method="butter", order=3):
    data, attributes = get_data_from_dataset(filename, dataset_path)
    fps = attributes["fps"]

    if method == "butter":
        nyq = fps / 2
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')

        def butterworth(data, b, a):
            max_value = np.max(np.abs(data))
            filtered_data = filtfilt(b, a, data)
            max_value_filtered = np.max(np.abs(filtered_data))
            if max_value_filtered != 0:
                filtered_data *= (max_value / max_value_filtered)
            return filtered_data

        if data.ndim == 1:
            filtered_data = butterworth(data, b, a)
        elif data.ndim == 3:
            filtered_data = np.zeros_like(data)
            for row in range(data.shape[1]):
                for col in range(data.shape[2]):
                    filtered_data[:, row, col] = butterworth(data[:, row, col], b, a)
        else:
            raise ValueError("Cannot filter data that is not in 1D or 3D.")

    elif method == "fourier":
        def fourier(data, fps, band):
            spectrum = np.fft.fft(data)
            filtered_spectrum = np.copy(spectrum)
            f = np.fft.fftfreq(len(data), 1 / fps)
            filtered_spectrum[np.abs(f) < band[0]] = 0
            filtered_spectrum[np.abs(f) > band[1]] = 0
            return np.fft.ifft(filtered_spectrum)

        if data.ndim == 1:
            filtered_data = fourier(data, fps, [lowcut, highcut])
        elif data.ndim == 3:
            filtered_data = np.zeros_like(data)
            for row in range(data.shape[1]):
                for col in range(data.shape[2]):
                    filtered_data[:, row, col] = fourier(data[:, row, col], fps, [lowcut, highcut])
        else:
            raise ValueError("Cannot filter data that is not in 1D or 3D.")

    return filtered_data, attributes

def apply_filter_to_data(filename, dataset_paths, lowcut, highcut, method="butter", order=3, copy_name=None):
    
    for dataset_path in dataset_paths:
    
        data, attributes = bandpass_filter(filename, dataset_path, lowcut, highcut, method=method, order=order)

        attributes.update({"bandpass_lowcut": lowcut, "bandpass_highcut": highcut, "method": method})
        if method == "butter": attributes.update({"order": order})

        if copy_name is None:
            save_data_to_dataset(filename, dataset_path, data, attributes)
        else:
            save_data_to_dataset(filename, dataset_path+"_"+copy_name, data, attributes)

class FilterPreviewApp:
    def __init__(self, root, filename, dataset_list):
        self.root = root
        self.root.title("Filtering Preview")
        self.filename = filename
        self.dataset_list = dataset_list
        self.lowcut = 0.1
        self.highcut = 1.0
        self.method = "butter"
        self.order = 3
        self.copy_name = None

        tk.Label(root, text="Select Dataset").grid(row=0, column=0, padx=10, pady=5)
        self.dataset_combobox = ttk.Combobox(root, values=dataset_list, state="readonly")
        self.dataset_combobox.current(0)
        self.dataset_combobox.grid(row=0, column=1, padx=10, pady=5)
        self.dataset_combobox.bind("<<ComboboxSelected>>", self.update_preview)

        self.create_input("Lowcut", 1, self.lowcut)
        self.create_input("Highcut", 2, self.highcut)
        self.create_input("Order", 3, self.order)

        tk.Label(root, text="Method").grid(row=4, column=0, padx=10, pady=5)
        self.method_combobox = ttk.Combobox(root, values=["butter", "fourier"], state="readonly")
        self.method_combobox.current(0)
        self.method_combobox.grid(row=4, column=1, padx=10, pady=5)
        self.method_combobox.bind("<<ComboboxSelected>>", self.toggle_order)

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

    def toggle_order(self, event=None):
        if self.method_combobox.get() == "fourier":
            self.order_entry.config(state="disabled")
        else:
            self.order_entry.config(state="normal")

    def toggle_copy_name(self):
        if self.make_copy_var.get():
            self.copy_name_entry.config(state="normal")
        else:
            self.copy_name_entry.config(state="disabled")
            self.copy_name_entry.delete(0, tk.END)

    def update_preview(self, event=None):
        try:
            self.preview_path = self.dataset_combobox.get()
            self.method = self.method_combobox.get()
            self.lowcut = float(self.lowcut_entry.get())
            self.highcut = float(self.highcut_entry.get())
            self.order = float(self.order_entry.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Enter valid numeric values.")
            return
        
        original_data, attr = get_data_from_dataset(self.filename, self.preview_path)
        nyquist_freq = attr["fps"] / 2
        if (self.lowcut < 0) or (self.highcut < 0) or (self.lowcut > nyquist_freq) or (self.highcut > nyquist_freq):
            messagebox.showerror("Invalid Input", f"Lowcut and Highcut values should be between 0 and the Nyquist frequency ({nyquist_freq:.3f}).")
            return
        if self.lowcut >= self.highcut:
            messagebox.showerror("Invalid Input", "Lowcut value cannot be equal or higher than Highcut.")
            return
        
        original_data = np.nanmean(original_data, axis=(1,2)) # TODO: adapt for 1d as well

        preview_data, _ = bandpass_filter(self.filename, self.preview_path, self.lowcut, self.highcut, self.method, self.order)
        preview_data = np.nanmean(preview_data, axis=(1,2))
        
        self.ax1.clear()
        self.ax1.plot(original_data, label="Original Data")
        self.ax1.set_title(f"Original Data (Min: {original_data.min():.2f}, Max: {original_data.max():.2f})")
        self.ax1.legend()
        
        self.ax2.clear()
        self.ax2.plot(preview_data, label="Filtered Data", color="orange")
        self.ax2.legend()
        
        self.canvas.draw()

    def compute(self):
        try:
            self.preview_path = self.dataset_combobox.get()
            self.method = self.method_combobox.get()
            self.lowcut = float(self.lowcut_entry.get())
            self.highcut = float(self.highcut_entry.get())
            self.order = float(self.order_entry.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Enter valid numeric values.")
            return

        self.copy_name = self.copy_name_entry.get().strip() if self.make_copy_var.get() else None
        self.root.quit()
        self.result = (self.method, self.lowcut, self.highcut, self.order, self.copy_name)


def main():
    if "--batch" in sys.argv:  # Batch mode
        
        filename = sys.argv[2]
        datasets = sys.argv[3].split(',')
        args = parse_key_value_args(sys.argv[4:])

        apply_filter_to_data(filename, datasets, method=args["method"], lowcut=args["lowcut"], highcut=args["highcut"], order=args["order"], copy_name=args["copy_name"])

    else:  # GUI mode
        if len(sys.argv) < 3:
            print("Usage: python script.py <HDF5 file> <dataset1> [<dataset2> ...]")
            return

        hdf5_file = sys.argv[1]
        datasets = sys.argv[2:]

        root = tk.Tk()
        app = FilterPreviewApp(root, hdf5_file, datasets)
        root.mainloop()

        method, lowcut, highcut, order, copy_name = app.result

        apply_filter_to_data(hdf5_file, datasets, method=method, lowcut=lowcut, highcut=highcut, order=order, copy_name=copy_name)

if __name__ == "__main__":
    main()