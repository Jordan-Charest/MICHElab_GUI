import ast
import os
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import shutil
import subprocess

import h5py
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numba as nb
import numpy as np
import tifffile

from utils.hdf5 import get_data_from_dataset, delete_hdf5_dataset, add_data_to_hdf5

VALID_EXTENSIONS = [("Numpy or TIFF", "*.npy *.tif *.tiff")] # For adding datasets to existing HDF5 file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_DIRECTORY = os.path.join(BASE_DIR, "../operations") # Where the operations scripts are situated

@nb.njit(parallel=True)
def compute_seedbased(data, x, y):
    """
    Computes the correlation between the time series of pixel (y, x) and all other pixels,
    while preserving NaN values, using Numba for speed.

    Parameters:
        data (numpy.ndarray): 3D array of shape (Time, Height, Width).
        y (int): Seed pixel row index.
        x (int): Seed pixel column index.

    Returns:
        numpy.ndarray: 2D correlation map of shape (Height, Width).
    """
    T, H, W = data.shape
    corr_map = np.full((H, W), np.nan)  # Initialize with NaNs

    # Extract seed time series
    seed_ts = data[:, y, x]

    # Compute mean of seed time series, ignoring NaNs
    valid_seed = ~np.isnan(seed_ts)
    if np.sum(valid_seed) == 0:
        return corr_map  # If seed is all NaNs, return all NaNs

    mean_seed = np.nanmean(seed_ts)

    # Precompute global mean and std for all pixels
    mean_data = np.full((H, W), np.nan)
    std_data = np.full((H, W), np.nan)
    numerator = np.zeros((H, W))
    
    # Compute mean and std for each pixel in parallel
    for row in nb.prange(H):
        for col in range(W):
            pixel_ts = data[:, row, col]
            valid_mask = ~np.isnan(pixel_ts)
            if np.sum(valid_mask) == 0:
                continue  # Leave as NaN if all values are NaN
            
            mean_px = np.mean(pixel_ts[valid_mask])
            std_px = np.sqrt(np.sum((pixel_ts[valid_mask] - mean_px) ** 2))
            mean_data[row, col] = mean_px
            std_data[row, col] = std_px

            # Compute numerator for covariance
            numerator[row, col] = np.sum((pixel_ts[valid_mask] - mean_px) * (seed_ts[valid_mask] - mean_seed))

    # Compute std of seed
    std_seed = np.sqrt(np.nansum((seed_ts - mean_seed) ** 2))
    
    # Compute correlation while avoiding division by zero
    for row in nb.prange(H):
        for col in range(W):
            if std_seed > 0 and std_data[row, col] > 0:
                corr_map[row, col] = numerator[row, col] / (std_seed * std_data[row, col])

    return corr_map

# ===================================================================================================================


class processingGUI(ttk.Frame):

    def __init__(self, parent):
        super().__init__(parent)
        # self.title("HDF5 Data Editor")

        self.selected_script_order = []

        self.create_widgets()

    def create_widgets(self):
        
        # FILE SELECTOR
        tk.Label(self, text="Input File:").pack()
        input_frame = tk.Frame(self)
        input_frame.pack()
        self.input_entry = tk.Entry(input_frame, width=50)
        self.input_entry.pack(side=tk.LEFT)
        tk.Button(input_frame, text="Browse", command=lambda: self.browse_file()).pack(side=tk.RIGHT)
        
        # PROCESS WITHOUT COPY CHECKBOX
        self.process_without_copy_var = tk.BooleanVar()
        process_without_copy_checkbox = tk.Checkbutton(self, text="Process without copy", variable=self.process_without_copy_var, command=self.toggle_output_entry)
        process_without_copy_checkbox.pack()
        
        # OUTPUT FILE FIELD
        tk.Label(self, text="Output File:").pack()
        self.output_entry = tk.Entry(self, width=50)
        self.output_entry.pack()
        
        # DATASET EXPLORATION TABLE
        dataset_frame = tk.Frame(self)
        dataset_frame.pack()
        self.dataset_tree = ttk.Treeview(dataset_frame, columns=("Shape", "Type"), show="tree headings", height=5)
        self.dataset_tree.heading("#0", text="Dataset Path")
        self.dataset_tree.heading("Shape", text="Shape")
        self.dataset_tree.heading("Type", text="Type")
        self.dataset_tree.pack(side=tk.LEFT)
        self.dataset_tree.bind("<<TreeviewSelect>>", self.update_attributes_display)
        
        # ATTRIBUTES DISPLAY
        self.attributes_text = tk.Text(dataset_frame, width=40, height=10, state=tk.DISABLED)
        self.attributes_text.pack(side=tk.RIGHT)

        # ADD OR DELETE DATASETS BUTTONS
        button_frame = tk.Frame(self)
        button_frame.pack(pady=5)
        tk.Button(button_frame, text="Add Dataset", command=self.open_add_dataset_dialog).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Delete Dataset", command=self.delete_selected_dataset).pack(side=tk.LEFT, padx=5)
        
        # PROCESSING SCRIPTS DISPLAY
        tk.Label(self, text="Processing Scripts:").pack()
        script_frame = tk.Frame(self)
        script_frame.pack()
        self.order_listbox = tk.Listbox(script_frame, width=70, height=5)
        self.order_listbox.pack(side=tk.RIGHT, padx=10)
        self.script_listbox = tk.Listbox(script_frame, width=50, height=5, selectmode=tk.MULTIPLE)
        self.script_listbox.pack(side=tk.LEFT)
        self.script_listbox.bind("<<ListboxSelect>>", self.on_script_select)
        self.load_scripts()
        
        # PREVIEW BUTTON
        preview_button_frame = tk.Frame(self)
        preview_button_frame.pack()
        preview_button = tk.Button(preview_button_frame, text="Preview", command=self.on_preview_button_click)
        preview_button.pack(side=tk.LEFT)

        # SEED BASED CORRELATION BUTTON
        seedbased_btn = tk.Button(preview_button_frame, text="Seed Based Correlation", command=self.on_seedbased)
        seedbased_btn.pack(side=tk.LEFT)
        
        # COLORMAP CHECKBOXES
        self.colormap_var = tk.IntVar(value=1)
        colormap_frame = tk.Frame(self)
        colormap_frame.pack()
        tk.Checkbutton(colormap_frame, text="Viridis", variable=self.colormap_var, onvalue=1, offvalue=0).pack(side=tk.LEFT)
        tk.Checkbutton(colormap_frame, text="Coolwarm", variable=self.colormap_var, onvalue=0, offvalue=1).pack(side=tk.LEFT)
        
        # TIMECOURSE AND POWER SPECTRUM DENSITY CHECKBUTTONS
        self.timecourse_var = tk.BooleanVar(value=False)  # Default to unchecked
        self.psd_var = tk.BooleanVar(value=False)  # Default to unchecked

        # Create the frame for the timecourse and PSD checkbuttons
        tc_psd_frame = tk.Frame(self)
        tc_psd_frame.pack()

        # Timecourse checkbutton
        timecourse_cb = tk.Checkbutton(tc_psd_frame, text="1D Timecourse from 3D data", variable=self.timecourse_var, command=lambda: handle_check(self.timecourse_var, self.psd_var))
        timecourse_cb.pack(side=tk.LEFT)

        # Power Spectrum Density checkbutton
        psd_cb = tk.Checkbutton(tc_psd_frame, text="Power Spectrum Density", variable=self.psd_var, command=lambda: handle_check(self.psd_var, self.timecourse_var))
        psd_cb.pack(side=tk.LEFT)

        # Custom function to enforce mutual exclusivity
        def handle_check(checked_var, other_var):
            if checked_var.get():
                other_var.set(False)  # Uncheck the other button if one is checked
            else:
                # If both are unchecked, do nothing or apply other logic as needed
                pass
        
        self.log_text = tk.Text(self, width=80, height=5, state=tk.NORMAL)
        self.log_text.pack()
        
        tk.Button(self, text="Run Processing", command=self.run_processing).pack()


    def browse_file(self, filename=None):
        """Browse files to select a HDF5 file to explore / edit.

        Args:
            entry (tk.Entry): tkinter Entry object to input filepath
        """

        if filename is None:
            self.filename = filedialog.askopenfilename(filetypes=[("HDF5 files", "*.h5;*.hdf5")])
        else:
            self.filename = filename

        if self.filename:
            self.input_entry.delete(0, tk.END) # Update input filename field
            self.input_entry.insert(0, self.filename)

            self.output_entry.delete(0, tk.END) # Update output filename field
            self.output_entry.insert(0, os.path.splitext(self.filename)[0] + "_processed.h5")

            self.dataset_structure = self.load_hdf5_structure() # Load HDF5 file structure
            self.update_treeview() # Display structure in treeview


    def load_hdf5_structure(self):
        """Load the groups and datasets structure of an HDF5 file.

        Returns:
            dict: structure of the HDF5 file in dictionary format
        """

        structure = {}
        with h5py.File(self.filename, 'r') as file: # Recursively visit groups and subgroups to populate structure dictionary
            def recursively_add(name, obj):
                if isinstance(obj, h5py.Dataset):
                    structure[name] = {"shape": obj.shape, "dtype": obj.dtype, "attrs": dict(obj.attrs)}
            file.visititems(recursively_add)
        return structure

    def update_treeview(self):
        """Updates treeview once the HDF5 structure is loaded
        """

        self.dataset_tree.delete(*self.dataset_tree.get_children())
        for path, info in self.dataset_structure.items():
            self.dataset_tree.insert("", "end", iid=path, text=path, values=[info["shape"], info["dtype"]])

    def toggle_output_entry(self):
        """Toggles the output entry depending on if the process without copy checkbox is checked or not
        """
        if self.process_without_copy_var.get():
            self.output_entry.config(state=tk.DISABLED)
        else:
            self.output_entry.config(state=tk.NORMAL)

            self.output_entry.delete(0, tk.END) # Update output filename field
            self.output_entry.insert(0, os.path.splitext(self.input_entry.get())[0] + "_processed.h5")

    def update_attributes_display(self, _=None):
        """Updates the attributes display when a dataset is selected

        Args:
            _: Unused argument; must remain there because there will sometimes be a tk.Event object passed via button press
        """
        selected_item = self.dataset_tree.selection()
        if selected_item: # If there is a dataset selected, display its attributes (metadata)
            dataset_path = selected_item[-1]
            self.attributes_text.config(state=tk.NORMAL)
            self.attributes_text.delete(1.0, tk.END)
            self.attributes_text.insert(tk.END, f"Attributes for {dataset_path}:\n")
            for key, value in self.dataset_structure.get(dataset_path, {}).get("attrs", {}).items():
                self.attributes_text.insert(tk.END, f"{key}: {value}\n")
            self.attributes_text.config(state=tk.DISABLED)

    def open_add_dataset_dialog(self):
        """Opens a dialog box to add datasets to HDF5 file
        """

        def browse_data_file():
            filepath = filedialog.askopenfilename(filetypes=VALID_EXTENSIONS)
            if filepath:
                file_entry.delete(0, tk.END)
                file_entry.insert(0, filepath)

        def add_attribute_row():
            row = len(attribute_entries) + 1  # +1 to account for header row
            name_entry = tk.Entry(attr_frame, width=15)
            type_entry = tk.Entry(attr_frame, width=10)
            value_entry = tk.Entry(attr_frame, width=15)
            name_entry.grid(row=row, column=0, padx=2, pady=2)
            type_entry.grid(row=row, column=1, padx=2, pady=2)
            value_entry.grid(row=row, column=2, padx=2, pady=2)
            attribute_entries.append((name_entry, type_entry, value_entry))

        def string_to_tuple(s):
            try:
                value = ast.literal_eval(s)
                if isinstance(value, tuple):
                    return value
                else:
                    raise ValueError("Input string does not represent a tuple.")
            except Exception as e:
                raise ValueError(f"Invalid tuple string: {e}")

        def cast_value(value, value_type):
            try:
                if value_type == "int":
                    return int(value)
                elif value_type == "float":
                    return float(value)
                elif value_type == "bool":
                    return value.lower() in ("true", "1", "yes")
                elif value_type == "str":
                    return value
                elif value_type == "tuple":
                    return string_to_tuple(value)
                else:
                    raise ValueError(f"Unsupported type '{value_type}'")
            except Exception as e:
                raise ValueError(f"Cannot convert '{value}' to {value_type}: {e}")

        def on_submit():
            dataset_path = path_entry.get()
            file_path = file_entry.get()
            if not dataset_path or not file_path:
                messagebox.showerror("Missing Information", "Dataset path and file must be provided.")
                return
            
            def read_data(filename):
                """Read data in tiff or npy files
                """
                    
                if filename[-4:] == ".tif":
                    return tifffile.imread(filename)
                
                elif filename[-4:] == ".npy":
                    return np.load(filename)
                
                raise ValueError("Could not recognize file extension.")

            try:
                data = read_data(file_path)

                attrs = {}
                attrs["filepath"] = file_path
                for name_entry, type_entry, value_entry in attribute_entries:
                    key = name_entry.get().strip()
                    value_type = type_entry.get().strip().lower()
                    raw_value = value_entry.get()

                    if key:
                        if not value_type:
                            raise ValueError(f"Missing type for attribute '{key}'")
                        value = cast_value(raw_value, value_type)
                        attrs[key] = value


                self.dataset_structure[dataset_path] = {
                    "shape": data.shape,
                    "dtype": data.dtype,
                    "attrs": attrs
                }

                path, name = dataset_path.rsplit('/', 1) if '/' in dataset_path else (None, dataset_path)
                add_data_to_hdf5(self.filename, name, data, path, attributes=attrs)

                self.update_treeview()
                top.destroy()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load dataset: {e}")

        top = tk.Toplevel()
        top.title("Add Dataset")

        tk.Label(top, text="Dataset Path:").pack()
        path_entry = tk.Entry(top, width=40)
        path_entry.pack()

        tk.Label(top, text="Select .npy or .tif file:").pack()
        file_frame = tk.Frame(top)
        file_frame.pack()
        file_entry = tk.Entry(file_frame, width=30)
        file_entry.pack(side=tk.LEFT)
        tk.Button(file_frame, text="Browse", command=browse_data_file).pack(side=tk.RIGHT)

        tk.Label(top, text="Attributes (optional):").pack(pady=(10, 0))
        attr_frame = tk.Frame(top)
        attr_frame.pack()

        # Column headers
        tk.Label(attr_frame, text="Name").grid(row=0, column=0, padx=2)
        tk.Label(attr_frame, text="Type").grid(row=0, column=1, padx=2)
        tk.Label(attr_frame, text="Value").grid(row=0, column=2, padx=2)

        attribute_entries = []
        add_attribute_row()  # Add one row below the headers

        tk.Button(top, text="Add Attribute", command=add_attribute_row).pack(pady=5)
        tk.Button(top, text="Submit", command=on_submit).pack(pady=5)


    def delete_selected_dataset(self):
        """Prompts user to confirm deletion of selected dataset, and deletes it if user confirms
        """
        selected = self.dataset_tree.selection()
        if not selected:
            messagebox.showwarning("Warning", "No dataset selected.")
            return

        dataset_path = selected[0]
        confirm = messagebox.askyesno("Delete Dataset", f"Are you sure you want to delete '{dataset_path}'?")
        if confirm:
            delete_hdf5_dataset(self.filename, dataset_path)
            if dataset_path in self.dataset_structure:
                del self.dataset_structure[dataset_path]
                self.update_treeview()
            else:
                messagebox.showerror("Error", "Dataset not found in structure.")

    
    def on_script_select(self, _=None):
        """Runs on script selection to update selected scripts
        """
        
        selected_scripts = self.script_listbox.curselection()
        selected_script_names = [self.script_listbox.get(i) for i in selected_scripts]
        self.selected_script_order = [script for script in self.selected_script_order if script in selected_script_names]
        for script in selected_script_names:
            if script not in self.selected_script_order:
                self.selected_script_order.append(script)
        self.update_script_display()

    def update_script_display(self):
        """Updates script order display
        """

        self.order_listbox.delete(0, tk.END)
        for i, script in enumerate(self.selected_script_order, start=1):
            self.order_listbox.insert(tk.END, f"{i}. {os.path.basename(script)}")

    def load_scripts(self):
        """Loads scripts contained in the "operations" directory to display them
        """
        self.script_listbox.delete(0, tk.END)
        self.order_listbox.delete(0, tk.END)
        operations_dir = SCRIPT_DIRECTORY
        if os.path.exists(operations_dir):
            for filename in os.listdir(operations_dir):
                if filename.endswith(".py"):
                    # self.script_listbox.insert(tk.END, os.path.join(operations_dir, filename))
                    self.script_listbox.insert(tk.END, filename)

    def on_preview_button_click(self):
        """Displays preview of data when Preview button is clicked
        """

        selected_datasets = self.dataset_tree.selection()

        if not selected_datasets:
            messagebox.showerror("Error", "No dataset selected.")
            return
        
        if len(selected_datasets) == 1: # Single dataset selected, show preview

            dataset_path = selected_datasets[0]
            data, attrs = get_data_from_dataset(self.input_entry.get(), dataset_path)
            
            # Handle different data dimensions (1D, 2D, 3D)
            if data.ndim == 1: # 1D
                # 1D data handling (plot the timecourse)
                plt.plot(data)
                plt.title(f"1D Data Timecourse - {dataset_path}")
                plt.xlabel("Timepoints")
                plt.ylabel("Value")
                plt.show()
            elif data.ndim == 2: # 2D
                # 2D data handling (imshow the data)
                colormap = 'viridis' if self.colormap_var.get() == 1 else 'coolwarm'
                plt.imshow(data, cmap=colormap)
                plt.title(f"2D Data Preview - {dataset_path}")
                plt.colorbar()
                plt.show()
            elif data.ndim == 3: # 3D
                # 3D data handling (imshow the first slice)
                colormap = 'viridis' if self.colormap_var.get() == 1 else 'coolwarm'
                if self.psd_var.get(): # Show power spectrum density
                    timecourse_data = np.nanmean(data, axis=(1, 2))
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.psd(timecourse_data, Fs=attrs["fps"])  # Compute and plot PSD
                    ax.set_title('Power Spectral Density (PSD)')
                    plt.show()
                elif self.timecourse_var.get(): # Show 1d timecourse
                    timecourse_data = np.nanmean(data, axis=(1, 2))
                    plt.plot(timecourse_data)
                    plt.title(f"1D Timecourse from 3D Data - {dataset_path}")
                    plt.xlabel("Timepoints")
                    plt.ylabel("Mean Value")
                    plt.show()
                else: # Show data
                    plt.imshow(data[0, :, :], cmap=colormap)  # Show first slice of 3D data
                    plt.title(f"3D Data Preview - {dataset_path} (First Slice)")
                    plt.colorbar()
                    plt.show()
            else:
                messagebox.showerror("Error", "Unsupported data dimensions.")
        
        elif len(selected_datasets) > 1: # 
            messagebox.showerror("Error", "Please select a single dataset for previewing.")
        

    def on_seedbased(self):
        """Opens seed-based correlation window
        """
        selected_items = self.dataset_tree.selection()
        if len(selected_items) != 1:
            messagebox.showwarning("Warning", "Please select a single dataset.")
            return
        dataset_path = self.dataset_tree.item(selected_items[0], 'text')
        filepath = self.input_entry.get()
        colormap = "coolwarm" if self.colormap_var.get() == 0 else "viridis"
        self.open_seedbased_window(filepath, dataset_path, colormap)

    def open_seedbased_window(self, file_path, dataset_path, colormap):
        """Shows the seed-based correlation window

        Args:
            file_path (str): path of the HDF5 file
            dataset_path (str): Group path to the dataset
            colormap (str): matplotlib colormap identifier
        """
        
        initial_data, attr = get_data_from_dataset(file_path, dataset_path)
        selected_coords = [None, None]  # Store selected x, y

        def on_click(event):
            if event.xdata is not None and event.ydata is not None:
                x, y = int(event.xdata), int(event.ydata)

                # Ensure a non-NaN seed is selected
                if np.isnan(np.sum(initial_data[:, y, x])):
                    messagebox.showwarning("Warning", "Please select a non-NaN seed.")
                    return

                # Store the selected coordinates
                selected_coords[0], selected_coords[1] = x, y
                coord_label.config(text=f"Selected: ({x}, {y})")
                marker.set_data([x], [y])

                # Compute new data instantly
                new_data = compute_seedbased(initial_data, x, y)
                im.set_data(new_data)
                im.set_clim(np.nanmin(new_data), np.nanmax(new_data))  # Update colorbar range
                colorbar.update_normal(im)  # Refresh colorbar

                canvas.draw()

        new_window = tk.Toplevel()
        new_window.title("2D Data View")
        
        fig, ax = plt.subplots()
        im = ax.imshow(initial_data[0], cmap=colormap)
        marker, = ax.plot([], [], 'kx', markersize=10)  # Black "X" marker
        
        canvas = FigureCanvasTkAgg(fig, master=new_window)
        canvas.get_tk_widget().pack()
        canvas.mpl_connect("button_press_event", on_click)
        
        coord_label = tk.Label(new_window, text="Selected: None")
        coord_label.pack()

        colorbar = plt.colorbar(im, ax=ax)  # Store colorbar reference
        canvas.draw()

    
    def run_processing(self):
        """Once the dataset and the scripts are selected, launch processing
        """

        input_file = self.input_entry.get()
        output_file = input_file if self.process_without_copy_var.get() else self.output_entry.get()
        selected_datasets = self.dataset_tree.selection()
        if not input_file or not output_file or not selected_datasets or not self.selected_script_order:
            return
        
        if not self.process_without_copy_var.get():
            self.log_text.insert(tk.END, "Copying original data.\n\n")
            self.log_text.see(tk.END)
            shutil.copy(input_file, output_file)
        
        dataset_paths = list(selected_datasets)
        for i, script_path in enumerate(self.selected_script_order, start=1):
            script_name = os.path.basename(script_path)
            self.log_text.insert(tk.END, f"{i}. Running: {script_name}\n")
            self.log_text.see(tk.END)
            
            process = subprocess.run(["python", script_path, output_file, *dataset_paths], capture_output=True, text=True, check=True)
            self.log_text.insert(tk.END, process.stdout + "\n" + process.stderr + "\n")
            self.log_text.see(tk.END)
            
            # Ensure HDF5 changes are fully written and reloaded
            with h5py.File(output_file, 'r+') as f:
                f.flush()

            # Reload
            self.browse_file(filename=self.filename)

        
        self.log_text.insert(tk.END, "Finished!")
        self.log_text.see(tk.END)


# if __name__ == "__main__":
#     app = HDF5EditorApp()
#     app.mainloop()