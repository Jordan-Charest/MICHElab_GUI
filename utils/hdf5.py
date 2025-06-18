import h5py
import os
import tkinter as tk
from tkinter import messagebox
import numpy as np

def create_hdf5(filepath, overwrite=False):
    """
    Creates an empty HDF5 file with the specified name.

    Parameters:
        filepath (str): Path to the HDF5 file to create (e.g., "example.h5").

    Returns:
        None
    """

    if os.path.exists(filepath) and not overwrite:
        print(f"HDF5 file '{filepath}' already exists and overwrite is set to False. Skipping file creation.")
        return

    with h5py.File(filepath, 'w') as f:
        # The file is created and ready to use.
        print(f"HDF5 file '{filepath}' created successfully.")

def add_data_to_hdf5(filename, dataset_name, dataset, group_path, attributes=None, overwrite=True):
    """
    Add data to an HDF5 file, dynamically creating groups and handling attributes.

    Parameters:
        filename (str): Path to the HDF5 file.
        dataset_name (str): Name of the dataset to add.
        dataset (np.ndarray): The data to store in the dataset.
        group_path (str): Path to the group where the dataset will be stored.
        attributes (dict, optional): Dictionary of attributes to add to the dataset.
        overwrite (bool, optional): Whether to overwrite the dataset if it already exists. Default is True.

    Raises:
        ValueError: If `attributes` is not a dictionary.
    """
    if attributes is not None and not isinstance(attributes, dict):
        raise ValueError("Attributes must be provided as a dictionary.")

    with h5py.File(filename, 'r+') as f:
        # Ensure the group exists
        group = f.require_group(group_path)

        # Handle existing dataset
        if dataset_name in group:
            if not overwrite:
                print(f"Dataset '{dataset_name}' exists in group '{group_path}' and overwrite is set to False. Skipping.")
                return
            del group[dataset_name]

        # Create the new dataset
        dset = group.create_dataset(dataset_name, data=dataset)

        # Add attributes if provided
        if attributes:
            for key, value in attributes.items():
                dset.attrs[key] = value

def add_attributes_to_dataset(filepath, dataset_path, attributes):
    """
    Adds attributes to an existing dataset in an HDF5 file.

    Parameters:
        filepath (str): Path to the HDF5 file.
        dataset_path (str): Full path to the dataset (e.g., "Group1/Group2/dataset").
        attributes (dict): Dictionary of attribute names and their values.

    Returns:
        None
    """
    with h5py.File(filepath, 'r+') as f:
        if dataset_path in f:
            dataset = f[dataset_path]
            for key, value in attributes.items():
                dataset.attrs[key] = value
            print(f"Attributes added to dataset '{dataset_path}': {attributes}")
        else:
            raise KeyError(f"Dataset '{dataset_path}' not found in file.")
        
def get_data_from_dataset(filepath, dataset_path):
    """
    Retrieves data from a specified dataset in an HDF5 file along with its attributes.

    Parameters:
        filepath (str): Path to the HDF5 file.
        dataset_path (str): Path to the dataset within the file (e.g., "Group1/Group2/dataset").

    Returns:
        dataset (numpy.ndarray): The data stored in the dataset.
        attributes (dict): A dictionary containing the attributes of the dataset.
    """
    with h5py.File(filepath, 'r') as f:
        if dataset_path in f:
            # Get the dataset
            dataset = f[dataset_path][:]
            
            # Get the attributes as a dictionary
            attributes = dict(f[dataset_path].attrs)
            
            return dataset, attributes
        else:
            raise KeyError(f"Dataset '{dataset_path}' not found in the file.")
        
def save_data_to_dataset(filepath, dataset_path, data, attributes={}):
    """
    Saves data to a specified dataset in an HDF5 file along with optional attributes.
    Creates groups and subgroups if they do not exist.

    Parameters:
        filepath (str): Path to the HDF5 file.
        dataset_path (str): Path to the dataset within the file (e.g., "Group1/Group2/dataset").
        data (numpy.ndarray): The data to store in the dataset.
        attributes (dict, optional): A dictionary containing attributes to store (default: {}).
    """
    with h5py.File(filepath, 'a') as f:  # Open in append mode to allow modifications
        group_path = '/'.join(dataset_path.split('/')[:-1])  # Extract group path
        
        # Create groups if they do not exist
        if group_path and group_path not in f:
            f.require_group(group_path)
        
        # Create or overwrite dataset
        if dataset_path in f:
            del f[dataset_path]  # Remove existing dataset before writing new data
 
        f.create_dataset(dataset_path, data=data)

        
        # Add attributes
        for key, value in attributes.items():
            f[dataset_path].attrs[key] = value

def print_hdf5_structure(filepath):
    """
    Prints the group and dataset structure of an HDF5 file in a readable format.

    Parameters:
        filepath (str): Path to the HDF5 file.
    """
    def explore_group(group, indent=0):
        """
        Recursively explores and prints groups and datasets in the HDF5 file.
        
        Parameters:
            group (h5py.Group): Current group to explore.
            indent (int): Indentation level for displaying nested structures.
        """
        for name, item in group.items():
            if isinstance(item, h5py.Group):
                print('  ' * indent + f"- {name}")
                # Recursively print sub-groups
                explore_group(item, indent + 1)
            elif isinstance(item, h5py.Dataset):
                # Print the dataset (no further recursion needed)
                print('  ' * (indent) + f"  * {name} (Dataset)")

    with h5py.File(filepath, 'r') as f:
        print(f"Structure of {filepath}:")
        explore_group(f)

def print_dataset_and_attributes(filepath, group_path, dataset_name):
    """
    Prints the data and attributes of a specific dataset in an HDF5 file.

    Parameters:
        filepath (str): Path to the HDF5 file.
        group_path (str): Path to the group containing the dataset.
        dataset_name (str): Name of the dataset.
    """
    with h5py.File(filepath, 'r') as f:
        full_path = f"{group_path}/{dataset_name}"
        
        # Check if the group and dataset exist
        if group_path in f:
            group = f[group_path]
            if dataset_name in group:
                dataset = group[dataset_name]
                
                # Print dataset values
                print(f"Data in {full_path}:")
                print(dataset[:])  # Print the entire dataset
                
                # Print attributes
                print(f"\nAttributes for {full_path}:")
                for key, value in dataset.attrs.items():
                    print(f"  {key}: {value}")
            else:
                print(f"Dataset '{dataset_name}' does not exist in group '{group_path}'.")
        else:
            print(f"Group '{group_path}' does not exist in the file.")

def get_dataset_paths(filepath, group_path):
    """
    Collects all dataset paths within a specified group (including subgroups).

    Parameters:
        filepath (str): Path to the HDF5 file.
        group_path (str): Path to the group to start from.

    Returns:
        list: A list of full paths to all datasets within the group and its subgroups,
              or an empty list if the group path does not exist.
    """
    dataset_paths = []

    with h5py.File(filepath, 'r') as f:
        # Check if the group exists
        if group_path not in f:
            return []  # Return empty list if group doesn't exist

        # Process the group if it exists
        def visit_datasets(name, obj):
            if isinstance(obj, h5py.Dataset):
                dataset_paths.append(f"{group_path}/{name}")

        group = f[group_path]
        group.visititems(visit_datasets)

    return dataset_paths

def dataset_exists(filepath, dataset_path):
    """
    Checks whether a dataset exists at a given path in an HDF5 file.

    Parameters:
        filepath (str): Path to the HDF5 file.
        dataset_path (str): Path to the dataset within the file (e.g., "Group1/Group2/dataset").

    Returns:
        bool: True if the dataset exists, False otherwise.
    """
    with h5py.File(filepath, 'r') as f:
        return dataset_path in f and isinstance(f[dataset_path], h5py.Dataset)
    
def save_dict_to_hdf5(filepath, data_dict):
    """
    Saves a nested dictionary to an HDF5 file. Appends data to the file if it exists
    and overwrites datasets only if the dictionary contains a matching path.

    Parameters:
        filepath (str): Path to the HDF5 file.
        data_dict (dict): Nested dictionary where datasets are stored with the last key.
    """
    def save_item(group, key, value):
        """Recursively saves dictionary items as groups and datasets."""
        if isinstance(value, dict):
            # Create a subgroup (or get it if it exists) and recurse
            subgroup = group.require_group(key)
            for subkey, subvalue in value.items():
                save_item(subgroup, subkey, subvalue)
        else:
            # Overwrite dataset if it exists, otherwise create it
            if key in group:
                del group[key]  # Remove the existing dataset
            group.create_dataset(key, data=value)

    with h5py.File(filepath, 'a') as f:  # Open in append mode
        for key, value in data_dict.items():
            if isinstance(value, dict):
                # Start recursion if the value is a nested dictionary
                group = f.require_group(key)
                for subkey, subvalue in value.items():
                    save_item(group, subkey, subvalue)
            else:
                # Overwrite or create dataset at the root level
                if key in f:
                    del f[key]  # Remove the existing dataset
                f.create_dataset(key, data=value)

def load_hdf5_to_dict(filepath, group_path='/'):
    """
    Loads the contents of a group in an HDF5 file into a nested dictionary.

    Parameters:
        filepath (str): Path to the HDF5 file.
        group_path (str): Path to the group to load. Default is the root ('/').

    Returns:
        dict: Nested dictionary representing the group structure.
    """
    def load_item(obj):
        """Recursively loads groups and datasets into a dictionary."""
        result = {}
        if isinstance(obj, h5py.Group):
            for key, item in obj.items():
                result[key] = load_item(item)  # Recurse for subgroups
        elif isinstance(obj, h5py.Dataset):
            result = obj[()]  # Load dataset as numpy array
        return result

    with h5py.File(filepath, 'r') as f:
        # Ensure group exists
        if group_path not in f:
            raise KeyError(f"Group path '{group_path}' does not exist in the HDF5 file.")
        group = f[group_path]
        return load_item(group)
    
def get_attributes(filename, path):
    """
    Retrieve attributes of a specific group or dataset in an HDF5 file.

    Parameters:
        filename (str): Path to the HDF5 file.
        path (str): Path to the group or dataset inside the file.

    Returns:
        dict: Dictionary containing attribute names and values.
    """
    with h5py.File(filename, 'r') as f:
        if path in f:
            return dict(f[path].attrs)
        else:
            raise KeyError(f"Path '{path}' not found in the HDF5 file.")
        


###################### GUI AND PROCESSING

def parameter_GUI(parameter_list, box_text="Parameter Input"):
    """
    Prompts the user for multiple parameters using a Tkinter GUI.
    
    Each parameter is defined by a tuple (parameter_name, default_value).
    Boolean parameters are shown as checkboxes, while others are shown as text entries.
    
    Parameters:
        parameter_list (iterable of tuples): Each tuple contains a parameter name (str)
                                               and its default value (e.g., float, int, bool).
    
    Returns:
        dict: Dictionary of parameter values keyed by parameter names.
    """
    # Create the Tkinter root window.
    root = tk.Tk()
    root.title(box_text)
    
    # Dictionaries to store widget references.
    entries = {}       # For non-boolean parameters.
    check_vars = {}    # For boolean parameters.
    # Map parameter names to their conversion functions (derived from default value types).
    conversion = {}
    
    # Create a row for each parameter.
    for i, (param_name, default_value) in enumerate(parameter_list):
        tk.Label(root, text=f"Enter {param_name}:").grid(row=i, column=0, padx=5, pady=5, sticky="w")
        conversion[param_name] = type(default_value)
        
        if isinstance(default_value, bool):
            # For booleans, use a Checkbutton.
            var = tk.BooleanVar(value=default_value)
            tk.Checkbutton(root, variable=var).grid(row=i, column=1, padx=5, pady=5, sticky="w")
            check_vars[param_name] = var
        else:
            # For other types, use an Entry widget.
            ent = tk.Entry(root)
            ent.insert(0, str(default_value))
            ent.grid(row=i, column=1, padx=5, pady=5, sticky="w")
            entries[param_name] = ent
    
    # Function to handle the Submit button press.
    def on_submit():
        result = {}
        # Retrieve values from the checkbuttons.
        for param_name, var in check_vars.items():
            result[param_name] = var.get()
        # Retrieve and convert values from entry widgets.
        for param_name, ent in entries.items():
            val_str = ent.get()
            try:
                conv = conversion[param_name]
                result[param_name] = conv(val_str)
            except ValueError:
                messagebox.showerror("Invalid Input", f"Invalid value for {param_name}. Please enter a valid value.")
                return  # Do not exit if there is an error.
        
        # Store the result in the root so we can access it after mainloop.
        root.result = result
        root.quit()  # Exit the mainloop.
    
    # Submit button.
    submit_button = tk.Button(root, text="Submit", command=on_submit)
    submit_button.grid(row=len(parameter_list), column=0, columnspan=2, pady=10)
    
    # Start the GUI event loop.
    root.mainloop()
    
    # Retrieve the result stored in the root (if on_submit was successful).
    result = getattr(root, "result", None)
    root.destroy()
    return result

def delete_hdf5_dataset(file_path, dataset_path):
    """
    Delete a dataset from an HDF5 file.

    Parameters:
    - file_path (str): Path to the HDF5 file.
    - dataset_path (str): Full path to the dataset inside the HDF5 file (e.g., '/group1/datasetA').

    Raises:
    - FileNotFoundError: If the file doesn't exist.
    - KeyError: If the dataset path doesn't exist.
    - OSError: If the dataset cannot be deleted.
    """
    with h5py.File(file_path, 'a') as f:
        if dataset_path in f:
            del f[dataset_path]
            print(f"Deleted dataset '{dataset_path}' from '{file_path}'.")
        else:
            raise KeyError(f"Dataset path '{dataset_path}' not found in file.")

def log(message, level=1, verbosity=0):
    if verbosity >= level:
        print(message)