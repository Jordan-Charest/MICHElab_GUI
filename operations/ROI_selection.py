import numpy as np
import h5py
import subprocess
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from toolbox_jocha.hdf5 import get_data_from_dataset, save_data_to_dataset
import tifffile
import imageio
import tempfile
from tempfile import NamedTemporaryFile
import os
from pathlib import Path
import sys

# prefix = "../" # Use when launching as a standalone script
prefix = "" # Use when launching from the GUI launcher

macro_path = prefix+"FIJI_macros/macro_launch.ijm"
ROI_macro_path = prefix+"FIJI_macros/ROI.ijm"
fiji_path = prefix+"../../FIJI_local/Fiji.app/ImageJ-win64.exe"

def apply_roi_mask(filename, dataset_paths, mask, make_copy, copy_name, invert_mask, invert_copy_name):
    for dataset_path in dataset_paths:

        data, attributes = get_data_from_dataset(filename, dataset_path)

        mask = np.array(mask, dtype='bool')

        if invert_mask: inverted_data = np.copy(data)
        
        if data.ndim == 1:
            raise ValueError(f"Cannot apply ROI mask to 1D dataset: {dataset_path}")
        
        if data.ndim == 3:
            for t in range(data.shape[0]):
                data[t][~mask] = np.nan
        else:
            data[~mask] = np.nan
        
        # Crop data to smallest bounding box
        valid_rows = np.any(mask, axis=1)  # Check for valid values along height (axis 1)
        valid_cols = np.any(mask, axis=0)  # Check for valid values along width (axis 0)

        # Crop data to the smallest bounding box in spatial dimensions (height, width) using the mask
        if data.ndim == 2:
            cropped_data = data[valid_rows, :][:, valid_cols]
        elif data.ndim == 3:
            cropped_data = data[:, valid_rows, :][:, :, valid_cols]
        
        if make_copy:
            save_data_to_dataset(filename, dataset_path + "_" + copy_name, cropped_data, attributes)
        else:
            save_data_to_dataset(filename, dataset_path, cropped_data, attributes)
        
        if invert_mask:
            if data.ndim == 3:
                for t in range(data.shape[0]):
                    inverted_data[t][mask] = np.nan
            else:
                inverted_data[mask] = np.nan
            
            save_data_to_dataset(filename, dataset_path + "_" + invert_copy_name, inverted_data, attributes)

def get_roi_mask_from_fiji(macro_path, temp_tif_path):
    # Run the FIJI macro
    result = subprocess.run(
        [fiji_path, "--run", macro_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # merge stderr into stdout
        text=True
    )

    # The macro will save the mask to a temporary TIFF file, which is passed through <output_path>
    # Read the mask from the saved TIFF file
    mask = tifffile.imread(temp_tif_path.as_posix())

    return mask

def select_parameters(filename, dataset_list):
    root = tk.Tk()
    root.title("ROI Mask Parameters")

    tk.Label(root, text="Select Dataset to Load in FIJI").grid(row=0, column=0, padx=10, pady=5)
    dataset_combobox = ttk.Combobox(root, values=dataset_list, state="readonly")
    dataset_combobox.current(0)
    dataset_combobox.grid(row=0, column=1, padx=10, pady=5)
    
    tk.Label(root, text="Or Select Image File").grid(row=1, column=0, padx=10, pady=5)
    image_path_entry = tk.Entry(root, width=40)
    image_path_entry.grid(row=1, column=1, padx=10, pady=5)
    def browse_image():
        image_path_entry.delete(0, tk.END)
        image_path_entry.insert(0, filedialog.askopenfilename(filetypes=[("TIFF files", "*.tif;*.tiff")]))
    tk.Button(root, text="Browse", command=browse_image).grid(row=1, column=2, padx=10, pady=5)
    
    make_copy_var = tk.BooleanVar()
    tk.Checkbutton(root, text="Make Copy", variable=make_copy_var).grid(row=2, column=0, columnspan=2, pady=5)
    copy_name_entry = tk.Entry(root, width=20, state="disabled")
    copy_name_entry.grid(row=2, column=2, padx=10, pady=5)
    
    invert_mask_var = tk.BooleanVar()
    tk.Checkbutton(root, text="Save Outside (Invert Mask)", variable=invert_mask_var).grid(row=3, column=0, columnspan=2, pady=5)
    invert_copy_name_entry = tk.Entry(root, width=20, state="disabled")
    invert_copy_name_entry.grid(row=3, column=2, padx=10, pady=5)
    
    def toggle_copy_name():
        copy_name_entry.config(state="normal" if make_copy_var.get() else "disabled")
    make_copy_var.trace_add("write", lambda *args: toggle_copy_name())
    
    def toggle_invert_copy_name():
        invert_copy_name_entry.config(state="normal" if invert_mask_var.get() else "disabled")
    invert_mask_var.trace_add("write", lambda *args: toggle_invert_copy_name())
    
    def submit():
        root.quit()
    
    tk.Button(root, text="OK", command=submit).grid(row=4, column=0, columnspan=3, pady=10)
    root.mainloop()
    
    selected_dataset = dataset_combobox.get()
    selected_image_path = image_path_entry.get()
    make_copy = make_copy_var.get()
    copy_name = copy_name_entry.get().strip() if make_copy else None
    invert_mask = invert_mask_var.get()
    invert_copy_name = invert_copy_name_entry.get().strip() if invert_mask else None
    
    return selected_dataset, selected_image_path, make_copy, copy_name, invert_mask, invert_copy_name

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py <HDF5 file> <dataset1> [<dataset2> ...]")
        sys.exit(1)

    hdf5_file = sys.argv[1]
    datasets = sys.argv[2:]
    selected_dataset, selected_image_path, make_copy, copy_name, invert_mask, invert_copy_name = select_parameters(hdf5_file, datasets)

    # Create temp image file if not provided
    if selected_image_path:
        image_path = Path(selected_image_path).resolve()
    else:
        data, _ = get_data_from_dataset(hdf5_file, selected_dataset)
        with NamedTemporaryFile(delete=False, suffix=".tif") as temp_img:
            image_path = Path(temp_img.name).resolve()
        tifffile.imwrite(image_path, data.astype(np.float32))

    # Create temporary text file for ROI output
    with NamedTemporaryFile(delete=False, suffix=".tif") as temp_tif:
        temp_ROI_tif_path = Path(temp_tif.name).resolve()

    # Modify and create temporary ROI macro file
    with open(ROI_macro_path, "r") as temp_macro:
        macro_content = temp_macro.read().replace("<output_path>", temp_ROI_tif_path.as_posix())

    with NamedTemporaryFile(delete=False, suffix=".ijm", mode="w") as temp_macro_file:
        temp_macro_file.write(macro_content)
        temp_ROI_macro_path = Path(temp_macro_file.name).resolve()

    # Modify and create the launching macro file
    with open(macro_path, "r") as temp_macro:
        macro_content = temp_macro.read() \
            .replace("<stack_path>", image_path.as_posix()) \
            .replace("<macro_path>", temp_ROI_macro_path.as_posix())

    with NamedTemporaryFile(delete=False, suffix=".ijm", mode="w") as temp_macro_file:
        temp_macro_file.write(macro_content)
        temp_macro_path = Path(temp_macro_file.name).resolve()

    # Run FIJI and get ROI mask
    mask = get_roi_mask_from_fiji(temp_macro_path.as_posix(), temp_ROI_tif_path)

    # Apply ROI mask to datasets
    apply_roi_mask(hdf5_file, datasets, mask, make_copy, copy_name, invert_mask, invert_copy_name)

    # Cleanup temporary files
    os.remove(image_path)
    os.remove(temp_macro_path)
    os.remove(temp_ROI_macro_path)
    os.remove(temp_ROI_tif_path)

    
