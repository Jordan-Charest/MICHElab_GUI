import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pickle
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from utils.hdf5 import get_data_from_dataset
from utils.ets import format_array
from utils.community_detection import communities_to_window
from utils.correlation import compute_correlation_map
from sklearn.cluster import HDBSCAN, KMeans
from matplotlib.colors import ListedColormap, BoundaryNorm
import os

class visualize_DimRedGUI(ttk.Frame):
    """
    A Tkinter-based GUI for selecting models, filepaths, and datasets
    for analysis, based on a user-provided layout.
    """
    def __init__(self, parent):
        super().__init__(parent)

        # --- Attributes to store the state of the GUI ---
        self.model_filepath = tk.StringVar()
        self.model_list = []  # To be populated from the model file
        self.selected_model = None

        # --- Store Entry widgets and the currently selected one ---
        self.filepath_entries = []
        self.dataset_entries = []
        self.selected_filepath_entry = None
        self.selected_dataset_entry = None

        # --- UI Setup ---
        self._setup_widgets()

    def _setup_widgets(self):
        """Creates and arranges all the widgets in the main window."""
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- 1. Model Filepath Section ---
        model_frame = ttk.Frame(main_frame)
        model_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(model_frame, text="Model:").pack(side=tk.LEFT, padx=(0, 5))
        self.model_path_entry = ttk.Entry(model_frame, textvariable=self.model_filepath)
        self.model_path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ttk.Button(model_frame, text="Browse", command=self._browse_model_file).pack(side=tk.LEFT)

        # --- 2 & 3. Model and Attributes Section ---
        model_selection_frame = ttk.Frame(main_frame)
        model_selection_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        model_selection_frame.columnconfigure(0, weight=1)
        model_selection_frame.columnconfigure(1, weight=1)
        # model_selection_frame.rowconfigure(1, weight=1)

        # Select Model Box
        ttk.Label(model_selection_frame, text="Select Model").grid(row=0, column=0, sticky="w", pady=(0,5))
        self.model_listbox = tk.Listbox(model_selection_frame, exportselection=False, height=6)
        self.model_listbox.grid(row=1, column=0, sticky="nsew", padx=(0, 10))
        model_scrollbar = ttk.Scrollbar(model_selection_frame, orient=tk.VERTICAL, command=self.model_listbox.yview)
        model_scrollbar.grid(row=1, column=0, sticky="nse", padx=(0,10))
        self.model_listbox.config(yscrollcommand=model_scrollbar.set)
        self.model_listbox.bind("<<ListboxSelect>>", self._on_model_select)

        # Model Attributes Box
        ttk.Label(model_selection_frame, text="Model attributes").grid(row=0, column=1, sticky="w", pady=(0,5))
        self.attr_listbox = tk.Listbox(model_selection_frame, exportselection=False, height=6)
        self.attr_listbox.grid(row=1, column=1, sticky="nsew")
        attr_scrollbar = ttk.Scrollbar(model_selection_frame, orient=tk.VERTICAL, command=self.attr_listbox.yview)
        attr_scrollbar.grid(row=1, column=1, sticky="nse")
        self.attr_listbox.config(yscrollcommand=attr_scrollbar.set)

        # --- 4 & 5. Filepaths and Datasets Section ---
        data_selection_frame = ttk.Frame(main_frame)
        data_selection_frame.pack(fill=tk.X, pady=(0, 10))
        data_selection_frame.columnconfigure(0, weight=1)
        data_selection_frame.columnconfigure(2, weight=1)
        # data_selection_frame.rowconfigure(1, weight=1)

        # --- Select Filepaths Box (using a bordered LabelFrame) ---
        filepath_labelframe = ttk.LabelFrame(data_selection_frame, text="Select filepaths")
        filepath_labelframe.grid(row=1, column=0, sticky="nsew", padx=(0, 10))
        self.filepath_entries_frame, fp_canvas = self._create_scrollable_content(filepath_labelframe)
        
        # Bind mousewheel scrolling only when the mouse is over this specific canvas
        fp_canvas.bind("<Enter>", lambda e, c=fp_canvas: self._bind_mousewheel(e, c))
        fp_canvas.bind("<Leave>", self._unbind_mousewheel)

        # Filepath Buttons
        fp_button_frame = ttk.Frame(data_selection_frame)
        fp_button_frame.grid(row=1, column=1, sticky="n", padx=(0, 20))
        ttk.Button(fp_button_frame, text="Add", command=self._add_filepath).pack(fill=tk.X, pady=2)
        ttk.Button(fp_button_frame, text="Delete", command=self._delete_filepath).pack(fill=tk.X, pady=2)
        ttk.Button(fp_button_frame, text="Browse", command=self._browse_h5_file).pack(fill=tk.X, pady=2)

        # --- Select Datasets Box (using a bordered LabelFrame) ---
        dataset_labelframe = ttk.LabelFrame(data_selection_frame, text="Select datasets")
        dataset_labelframe.grid(row=1, column=2, sticky="nsew", padx=(0, 10))
        self.dataset_entries_frame, ds_canvas = self._create_scrollable_content(dataset_labelframe)

        # Bind mousewheel scrolling only when the mouse is over this specific canvas
        ds_canvas.bind("<Enter>", lambda e, c=ds_canvas: self._bind_mousewheel(e, c))
        ds_canvas.bind("<Leave>", self._unbind_mousewheel)


        # Dataset Buttons
        ds_button_frame = ttk.Frame(data_selection_frame)
        ds_button_frame.grid(row=1, column=3, sticky="n")
        ttk.Button(ds_button_frame, text="Add", command=self._add_dataset).pack(fill=tk.X, pady=2)
        ttk.Button(ds_button_frame, text="Delete", command=self._delete_dataset).pack(fill=tk.X, pady=2)

        # --- 6. Action Buttons Section ---
        action_button_frame = ttk.Frame(main_frame)
        action_button_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=(10, 0))
        action_button_frame.columnconfigure(0, weight=1)
        action_button_frame.columnconfigure(1, weight=1)
        
        ttk.Button(action_button_frame, text="Preview", command=self._open_preview_wrapper).grid(row=0, column=0, sticky="e", padx=5)
        ttk.Button(action_button_frame, text="Loadings", command=self._open_loadings_wrapper).grid(row=0, column=1, sticky="w", padx=5)
    
    def _on_mousewheel(self, event, canvas):
        """Scrolls the canvas that the mouse is currently over."""
        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        
    def _bind_mousewheel(self, event, canvas_to_bind):
        """Binds the mousewheel event to the specified canvas."""
        self.bind_all("<MouseWheel>", lambda e: self._on_mousewheel(e, canvas_to_bind))

    def _unbind_mousewheel(self, event):
        """Unbinds the global mousewheel event."""
        self.unbind_all("<MouseWheel>")

    def _create_scrollable_content(self, parent_container):
        """Creates a scrollable area inside a given parent container (like a LabelFrame)."""
        parent_container.rowconfigure(0, weight=1)
        parent_container.columnconfigure(0, weight=1)

        canvas = tk.Canvas(parent_container, highlightthickness=0, borderwidth=0)
        scrollbar = ttk.Scrollbar(parent_container, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas) # This is where entry widgets will be added

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas_window_id = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        
        canvas.bind(
            "<Configure>",
            lambda e: canvas.itemconfig(canvas_window_id, width=e.width)
        )

        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)
        scrollbar.grid(row=0, column=1, sticky="ns")
        
        return scrollable_frame, canvas

    # --- Placeholder Methods (for user implementation) ---

    def _load_models_from_file(self, filepath):
        """
        Placeholder: Loads model names from the specified file.
        Replace this with your actual file parsing logic.
        """

        self.model_list = []
        
        with open(filepath, 'rb') as f:
            self.data = pickle.load(f)

        for key in self.data.keys():
            if "model" in key:
                split_key = key.rsplit(sep="_", maxsplit=1)
                self.model_list.append(split_key[0])

    def _get_model_attributes(self, model_name):
        """
        Placeholder: Gets attributes for a given model name.
        Replace this with your logic to retrieve model details.
        """

        attributes = []

        for key in self.data:
            if model_name in key:
                attributes.append(f"{key}: {self.data[key]}")

        return attributes


    # --- Event Handlers and Logic ---

    def _browse_model_file(self):
        """Opens a file dialog to select a model file and updates the GUI."""
        filepath = filedialog.askopenfilename(
            title="Select a Model File",
            filetypes=(("Pickle files", "*.pkl"), ("HDF5 files", "*.h5"), ("All files", "*.*"))
        )
        if not filepath:
            return

        self.model_filepath.set(filepath)
        self._load_models_from_file(filepath)
        self._update_model_listbox()

    def _update_model_listbox(self):
        """Refreshes the 'Select Model' listbox with current self.model_list."""
        self.model_listbox.delete(0, tk.END)
        self.attr_listbox.delete(0, tk.END) # Clear attributes as well
        self.selected_model = None
        for model in self.model_list:
            self.model_listbox.insert(tk.END, model)

    def _on_model_select(self, event):
        """Handles selection change in the model listbox."""
        selection_indices = self.model_listbox.curselection()
        if not selection_indices:
            return

        selected_index = selection_indices[0]
        self.selected_model = self.model_listbox.get(selected_index)
        
        self._update_attr_listbox()

    def _update_attr_listbox(self):
        """Refreshes the 'Model attributes' listbox."""
        self.attr_listbox.delete(0, tk.END)
        if self.selected_model:
            attributes = self._get_model_attributes(self.selected_model)
            for attr in attributes:
                self.attr_listbox.insert(tk.END, attr)

    def _highlight_entry(self, entry_to_highlight, entry_list_type):
        """Highlights the selected entry and deselects others."""
        if entry_list_type == 'filepath':
            entry_list = self.filepath_entries
            self.selected_filepath_entry = entry_to_highlight
        elif entry_list_type == 'dataset':
            entry_list = self.dataset_entries
            self.selected_dataset_entry = entry_to_highlight
        else:
            return

        for entry in entry_list:
            if entry.winfo_exists():
                entry.config(bg="white")

        if entry_to_highlight.winfo_exists():
            entry_to_highlight.config(bg="lightblue")

    def _add_entry(self, parent_frame, text, entry_list_type):
        """Adds a new editable entry field."""
        entry = tk.Entry(parent_frame, width=40)
        entry.insert(0, text)
        entry.pack(fill="x", padx=2, pady=2, expand=True)
        
        if entry_list_type == 'filepath':
            entry.bind("<Button-1>", lambda e, ent=entry: self._highlight_entry(ent, 'filepath'))
            self.filepath_entries.append(entry)
            self._highlight_entry(entry, 'filepath')
        elif entry_list_type == 'dataset':
            entry.bind("<Button-1>", lambda e, ent=entry: self._highlight_entry(ent, 'dataset'))
            self.dataset_entries.append(entry)
            self._highlight_entry(entry, 'dataset')

    def _browse_h5_file(self):
        """Browses for an H5 file and adds it to the filepaths list."""
        filepath = filedialog.askopenfilename(
            title="Select H5 File",
            filetypes=(("HDF5 files", "*.h5 *.hdf5"), ("All files", "*.*"))
        )
        if filepath:
            self._add_entry(self.filepath_entries_frame, filepath, 'filepath')

    def _add_filepath(self):
        """Adds a new placeholder item to the filepaths list."""
        self._add_entry(self.filepath_entries_frame, "New filepath - edit here", 'filepath')
    
    def _delete_filepath(self):
        """Deletes the selected filepath entry."""
        if not self.selected_filepath_entry or not self.selected_filepath_entry.winfo_exists():
            return
        
        self.selected_filepath_entry.destroy()
        self.filepath_entries.remove(self.selected_filepath_entry)
        self.selected_filepath_entry = None

        if self.filepath_entries:
            self._highlight_entry(self.filepath_entries[-1], 'filepath')

    def _add_dataset(self):
        """Adds a new placeholder item to the datasets list."""
        self._add_entry(self.dataset_entries_frame, "New dataset - edit here", 'dataset')

    def _delete_dataset(self):
        """Deletes the selected dataset entry."""
        if not self.selected_dataset_entry or not self.selected_dataset_entry.winfo_exists():
            return
        
        self.selected_dataset_entry.destroy()
        self.dataset_entries.remove(self.selected_dataset_entry)
        self.selected_dataset_entry = None
        
        if self.dataset_entries:
            self._highlight_entry(self.dataset_entries[-1], 'dataset')


    def _open_preview_wrapper(self):
        # Retrieve actual values from your main GUI
        model = self.selected_model # Your actual model object
        h5_filepath = self.selected_filepath_entry.get()
        h5_dataset = self.selected_dataset_entry.get()
        window_title = "Data Preview" # Default title, or make it dynamic

        self._open_preview(model, h5_filepath, h5_dataset, window_title)

    def _open_preview(self, model, h5_filepath, h5_dataset, window_title):
        """
        Opens a new, non-modal window for 'Preview', displaying a scatter plot
        of transformed data with configurable colormaps and components.
        """
        if not self.selected_model:
            messagebox.showwarning("Warning", "Please select a model before opening Preview.")
            return

        # Instantiate the new PreviewWindow class
        preview_window = PreviewWindow(self, model, h5_filepath, h5_dataset, window_title)

    def _open_loadings_wrapper(self):
        # Retrieve actual values from your main GUI
        model = self.selected_model # Your actual model object string
        elements_mask = self.data["elements_mask"]
        shape = self.data["shape"]

        if model not in ["pca", "umap"]:
            messagebox.showwarning("Warning", f"Loadings for the {model} model are currently not implemented.")
            return

        window_title = "Data Preview" # Default title, or make it dynamic

        self._open_loadings(model, elements_mask, shape, window_title)

    def _open_loadings(self, model, elements_mask, shape, window_title):
        """
        Opens a new, non-modal window for 'Preview', displaying a scatter plot
        of transformed data with configurable colormaps and components.
        """
        if not self.selected_model:
            messagebox.showwarning("Warning", "Please select a model before opening Preview.")
            return
        
        h5_filepath = self.selected_filepath_entry.get()
        h5_dataset = self.selected_dataset_entry.get()

        # Instantiate the new PreviewWindow class
        # Removed preview_window.wait_window() to make it non-modal
        loadings_window = LoadingsWindow(self, model, elements_mask, shape, window_title, h5_filepath=h5_filepath, h5_dataset=h5_dataset)
        # The line below is no longer needed because wait_window() is removed.
        # The main loop continues running, allowing multiple windows.
        # preview_window.wait_window() 


# --- New Class for the Preview Window ---
class PreviewWindow(tk.Toplevel):
    # TODO: check the arguments, model is superfluous here.
    def __init__(self, parent, model, h5_filepath, h5_dataset, window_title):
        super().__init__(parent)
        window_title = f"Model: {parent.selected_model} from {parent.model_filepath.get()}\nFile: {h5_filepath}\nDataset: {h5_dataset}"
        self.title(window_title)
        self.geometry("900x700")

        self.protocol("WM_DELETE_WINDOW", self._on_close) # Handle close event


        # Store arguments and initialized data as instance attributes
        self.model_filepath = parent.model_filepath.get()
        self.model = parent.data[parent.selected_model+"_model"]
        self.h5_filepath = parent.selected_filepath_entry.get()
        self.h5_dataset = parent.selected_dataset_entry.get()
        self.transformed_data = None
        self.raw_data = None
        self.fig = None
        self.ax = None
        self.scatter_plot = None
        self.canvas_widget = None
        self.colorbar_obj = None # To keep track of the colorbar for clearing
        self.labels = None
        self.methods_dict = {
            "HDBSCAN": ("Min. cluster size", 20),
            "KMeans": ("N. clusters", 4)
        }

        # --- Load and transform data initially ---
        try:
            self.raw_data = self._load_h5_data(self.h5_filepath, self.h5_dataset)

            self.transformed_data = self._transform_h5_data(self.model, self.raw_data)

            if self.transformed_data.ndim < 2:
                messagebox.showerror("Error", "Transformed data must have at least 2 dimensions for plotting.")
                self.destroy()
                return

        except Exception as e:
            messagebox.showerror("Data Error", f"Failed to load or transform data: {e}")
            self.destroy()
            return

        self._create_widgets(window_title)
        self._plot_scatter(0, 1, self.colormap_var.get()) # Initial plot

    def _on_close(self):
        """Handles the closing of the preview window."""
        self.destroy()

    def _create_widgets(self, window_title):
        """Creates all the GUI elements for the preview window."""
        # --- Top Title ---
        title_label = ttk.Label(self, text=window_title, font=("Helvetica", 18, "bold"))
        title_label.pack(pady=10)

        # --- Main content frame (left for plot, right for controls) ---
        main_frame = ttk.Frame(self)
        main_frame.pack(expand=True, fill="both", padx=10, pady=10)

        # Configure columns to expand
        main_frame.grid_columnconfigure(0, weight=3) # Plot column
        main_frame.grid_columnconfigure(1, weight=1) # Controls column
        main_frame.grid_rowconfigure(0, weight=1)

        # --- Left Frame: Scatter Plot ---
        plot_frame = ttk.Frame(main_frame, relief="sunken", borderwidth=2)
        plot_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        self.fig = Figure(figsize=(5, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Add Matplotlib toolbar
        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        toolbar.update()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # --- Right Frame: Controls ---
        controls_frame = ttk.Frame(main_frame)
        controls_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        controls_frame.grid_columnconfigure(0, weight=1) # Allow column to expand

        # --- Colormap Selection ---
        colormap_label = ttk.Label(controls_frame, text="Colormap:", font="-weight bold")
        colormap_label.pack(pady=(10, 0), anchor="w")

        self.colormap_var = tk.StringVar(value="time") # Default to 'time'
        time_radio = ttk.Radiobutton(controls_frame, text="time", variable=self.colormap_var, value="time")
        time_radio.pack(anchor="w", padx=10)
        mean_act_radio = ttk.Radiobutton(controls_frame, text="mean act", variable=self.colormap_var, value="mean_act")
        mean_act_radio.pack(anchor="w", padx=10)

        # --- Components Input ---
        components_label = ttk.Label(controls_frame, text="Components:", font="-weight bold")
        components_label.pack(pady=(10, 0), anchor="w")

        x_comp_frame = ttk.Frame(controls_frame)
        x_comp_frame.pack(fill="x", padx=10, pady=2)
        ttk.Label(x_comp_frame, text="X:").pack(side="left")
        self.x_comp_var = tk.StringVar(value="0") # Default X component
        x_comp_entry = ttk.Entry(x_comp_frame, textvariable=self.x_comp_var)
        x_comp_entry.pack(side="right", expand=True, fill="x")

        y_comp_frame = ttk.Frame(controls_frame)
        y_comp_frame.pack(fill="x", padx=10, pady=2)
        ttk.Label(y_comp_frame, text="Y:").pack(side="left")
        self.y_comp_var = tk.StringVar(value="1") # Default Y component
        y_comp_entry = ttk.Entry(y_comp_frame, textvariable=self.y_comp_var)
        y_comp_entry.pack(side="right", expand=True, fill="x")

        # --- Clustering Section ---
        clustering_label = ttk.Label(controls_frame, text="Other:", font="-weight bold")
        clustering_label.pack(pady=(10, 0), anchor="w")

        clustering_frame = ttk.Frame(controls_frame)
        clustering_frame.pack(fill="x", padx=10, pady=5)

        clustering_label = ttk.Label(clustering_frame, text="Clustering:")
        clustering_label.pack(side="left", padx=(0, 10))

        # Dropdown options
        clustering_options = ["None"] + list(self.methods_dict.keys())

        self.clustering_method_var = tk.StringVar()
        clustering_dropdown = ttk.Combobox(
            clustering_frame,
            textvariable=self.clustering_method_var,
            values=clustering_options,
            state="readonly"
        )
        clustering_dropdown.current(0)  # Default to "None"
        clustering_dropdown.pack(side="left", fill="x", expand=True)

        # Entry section (label + entry)
        entry_frame = ttk.Frame(controls_frame)
        entry_frame.pack(fill="x", padx=10, pady=10)

        self.clustering_param_label = ttk.Label(entry_frame, text="Clustering param")
        self.clustering_param_label.pack(anchor="w")

        self.cluster_param_var = tk.IntVar()
        self.clustering_param_entry = ttk.Entry(entry_frame, textvariable=self.cluster_param_var, width=10)
        self.clustering_param_entry.pack(anchor="w")
        self.clustering_param_entry.state(["disabled"])

        clustering_dropdown.bind("<<ComboboxSelected>>", self.on_clustering_method_change)

        # --- Save Clustering Button ---
        reload_button = ttk.Button(entry_frame, text="Save Clustering Labels", command=self._save_labels)
        reload_button.pack(side="left", padx=5)

        # Explained variance frame
        variance_explained_frame = ttk.Frame(controls_frame)
        variance_explained_frame.pack(fill="x", padx=10, pady=5)

        self.explained_variance_var = tk.BooleanVar(value=False)
        self.explained_variance_log = tk.BooleanVar(value=False)

        explained_variance_box = ttk.Checkbutton(
            variance_explained_frame,
            text="Plot expl. var. ratio",
            variable=self.explained_variance_var
        )
        explained_variance_box.pack(side="left", padx=(0, 20))

        explained_variance_log_box = ttk.Checkbutton(
            variance_explained_frame,
            text="y axis log",
            variable=self.explained_variance_log
        )
        explained_variance_log_box.pack(side="left", padx=(0, 20))

        # TODO: Make comps a spinbox instead

        # Frame for label + entry
        comp_frame = ttk.Frame(variance_explained_frame)
        comp_frame.pack(side="left", fill="x", expand=True)

        ttk.Label(comp_frame, text="# comp. (0 for all)").pack(anchor="w")
        self.no_comp_var = tk.IntVar(value=0)
        no_comp_entry = ttk.Entry(comp_frame, textvariable=self.no_comp_var, width=6)
        no_comp_entry.pack(anchor="w")

        # --- Reload Button ---
        reload_button = ttk.Button(controls_frame, text="Reload", command=self._update_plot)
        reload_button.pack(pady=10)

        # --- Separator ---
        separator = ttk.Separator(controls_frame, orient="h")
        separator.pack(fill="x", pady=15)

        # --- Save Figure Section ---
        filename_label = ttk.Label(controls_frame, text="Filename:")
        filename_label.pack(pady=(5,0), anchor="w", padx=10)

        self.filename_var = tk.StringVar()
        filename_entry = ttk.Entry(controls_frame, textvariable=self.filename_var)
        filename_entry.pack(fill="x", padx=10, pady=2)

        save_button = ttk.Button(controls_frame, text="Save Figure", command=self._save_figure)
        save_button.pack(pady=10)

    def on_clustering_method_change(self, event=None):
        method = self.clustering_method_var.get()

        if method in self.methods_dict:
            label_text, default_value = self.methods_dict[method]
            self.clustering_param_label.config(text=label_text)
            self.cluster_param_var.set(default_value)
            self.clustering_param_entry.state(["!disabled"])
        else:
            self.clustering_param_label.config(text="Clustering param")
            self.clustering_param_entry.state(["disabled"])

    def _load_h5_data(self, h5_filepath, h5_dataset):
        """
        Placeholder function to simulate loading data from an HDF5 file.
        Returns a dummy NumPy array.
        Shape: (time_steps, features)
        """
        
        data, _ = get_data_from_dataset(h5_filepath, h5_dataset)

        data = format_array(data)

        # TODO: This will need to be updated to make sure it is compatible with registered data
        return data

    def _transform_h5_data(self, model_object, data_to_transform):
        """
        Placeholder function to simulate model transformation.
        Returns a dummy NumPy array representing transformed data.
        Shape: (num_samples, num_components)
        """

        transformed_data = model_object.transform(data_to_transform)

        return transformed_data

    def _plot_scatter(self, x_comp, y_comp, colormap_option):
        """
        Updates the scatter plot in the preview window.
        """
        if self.transformed_data is None:
            messagebox.showerror("Plot Error", "No transformed data available to plot.")
            return

        # --- IMPORTANT: Handle colorbar removal and then recreate axes ---

        # 1. First, explicitly remove the old colorbar's axes from the figure if it exists.
        # This prevents the "Failed to remove old colorbar" warning.
        if self.colorbar_obj is not None:
            if hasattr(self.colorbar_obj, 'ax') and self.colorbar_obj.ax is not None:
                try:
                    self.fig.delaxes(self.colorbar_obj.ax)
                except ValueError: # Catch cases where the axis might already be removed
                    pass
            self.colorbar_obj = None # Nullify the reference to the colorbar object

        # 2. Remove the existing main plot axes (self.ax) from the figure.
        # This is the key change to prevent the shrinking.
        if self.ax is not None and self.ax in self.fig.axes:
            self.fig.delaxes(self.ax)

        # 3. Create a brand new Axes object for the main plot.
        self.ax = self.fig.add_subplot(111) # 111 means 1x1 grid, first subplot

        # --- Continue with your plotting logic as before ---
        x_data = self.transformed_data[:, x_comp]
        y_data = self.transformed_data[:, y_comp]

        c_data = None
        label_order = None
        cmap = 'viridis' # Default colormap

        if colormap_option == "cluster":
            # TODO: Add kmeans
            custom_colors = ["grey", "red", "blue", "green", "purple", "orange", "brown", "black"]
            label_order = [-1, 0, 1, 2, 3, 4, 5, 6]

            # TODO: add self._perform_clustering call here to compute the labels instead
            # hdb = HDBSCAN(min_cluster_size=int(self.cluster_size_var.get()))
            # hdb.fit(self.transformed_data[:,[x_comp,y_comp]])
            # unique_labels = np.unique(hdb.labels_).tolist()
            # # label_order = unique_labels
            # labels = hdb.labels_
            self._perform_clustering()
            labels = self.labels

            # Build colormap and norm
            cmap = ListedColormap(custom_colors)
            bounds = np.array(label_order + [label_order[-1] + 1]) - 0.5  # e.g., [-1.5, -0.5, ..., 5.5]
            norm = BoundaryNorm(bounds, cmap.N)

            c_data=labels

        elif colormap_option == "time":
            c_data = np.arange(self.raw_data.shape[0]) # Time order
            self.ax.set_title("Scatter Plot (Color by Time)")
        elif colormap_option == "mean_act":
            if self.raw_data is not None:
                c_data = np.mean(self.raw_data, axis=1) # Mean activity
                self.ax.set_title("Scatter Plot (Color by Mean Activity)")
            else:
                messagebox.showwarning("Colormap Warning", "Raw data not available for Mean Activity colormap. Plotting without color.")
                self.ax.set_title("Scatter Plot")

        

        # Add colorbar if colormap data is present
        if c_data is not None:
            if label_order is not None:
                self.scatter_plot = self.ax.scatter(x_data, y_data, c=c_data, cmap=cmap, s=20, alpha=0.7, norm=norm)
                self.colorbar_obj = self.fig.colorbar(self.scatter_plot, ax=self.ax, ticks=label_order, label=f"{colormap_option.replace('_', ' ').title()}")
            else:
                self.scatter_plot = self.ax.scatter(x_data, y_data, c=c_data, cmap=cmap, s=20, alpha=0.7)
                self.colorbar_obj = self.fig.colorbar(self.scatter_plot, ax=self.ax, label=f"{colormap_option.replace('_', ' ').title()}")
        else:
            self.scatter_plot = self.ax.scatter(x_data, y_data, c=c_data, cmap=cmap, s=20, alpha=0.7)

        self.ax.set_xlabel(f"Component {x_comp}")
        self.ax.set_ylabel(f"Component {y_comp}")
        self.ax.grid(True, linestyle='--', alpha=0.6)

        # Re-apply tight_layout after creating new axes and adding elements
        self.fig.tight_layout()

        self.canvas.draw_idle() # Redraw the canvas

    def plot_explained_variance(self, log_yaxis=False, n_components=0):

        try:
            data = self.model.explained_variance_ratio_
            if int(n_components) != 0:
                data = data[:int(n_components)]
        except:
            messagebox.showwarning("Explained variance warning", "The selected model does not have an explained variance attribute, or the number of selected components is out of bounds.")
            return
        

        # 1. First, explicitly remove the old colorbar's axes from the figure if it exists.
        # This prevents the "Failed to remove old colorbar" warning.
        if self.colorbar_obj is not None:
            if hasattr(self.colorbar_obj, 'ax') and self.colorbar_obj.ax is not None:
                try:
                    self.fig.delaxes(self.colorbar_obj.ax)
                except ValueError: # Catch cases where the axis might already be removed
                    pass
            self.colorbar_obj = None # Nullify the reference to the colorbar object


        # 2. Remove the existing main plot axes (self.ax) from the figure.
        # This is the key change to prevent the shrinking.
        if self.ax is not None and self.ax in self.fig.axes:
            self.fig.delaxes(self.ax)

        # 3. Create a brand new Axes object for the main plot.
        self.ax = self.fig.add_subplot(111) # 111 means 1x1 grid, first subplot

        # --- Continue with your plotting logic as before ---
    

        self.ax.set_title("Variance explained by components")

        x = np.arange(0, len(data), 1)
        self.ax.plot(x, data, "-o", markersize=3)

        self.ax.set_xlabel(f"Component (0-indexing)")

        if not log_yaxis:
            self.ax.set_ylabel(f"Explained variance ratio")
        else:
            self.ax.set_ylabel(f"Explained variance ratio [log]")
            self.ax.set_yscale("log")

        self.ax.grid(True, linestyle='--', alpha=0.6)

        # Re-apply tight_layout after creating new axes and adding elements
        self.fig.tight_layout()

        self.canvas.draw_idle() # Redraw the canvas

    def _perform_clustering(self):

        x_comp = int(self.x_comp_var.get())
        y_comp = int(self.y_comp_var.get())
        method = self.clustering_method_var.get()

        if method == "HDBSCAN":
            hdb = HDBSCAN(min_cluster_size=int(self.clustering_param_entry.get()))
            hdb.fit(self.transformed_data[:,[x_comp,y_comp]])
            unique_labels = np.unique(hdb.labels_).tolist()
            self.labels = hdb.labels_
        elif method == "KMeans":
            km = KMeans(n_clusters=int(self.clustering_param_entry.get()), n_init="auto")
            km.fit(self.transformed_data[:,[x_comp,y_comp]])
            self.labels = km.labels_

    def _save_labels(self):

        if self.clustering_method_var.get() == "None":
            messagebox.showwarning("Error", "Please select a clustering method before saving.")
            return

        self._perform_clustering()

        print(self.model_filepath)

        self.append_to_pickle({"labels": self.labels, "labels_clustering_method": self.clustering_method_var.get(), "labels_clustering_param": self.clustering_param_entry.get()}, self.model_filepath)

    def append_to_pickle(self, new_data: dict, filename: str):
        """Loads an existing pickle file, updates it with new data, and saves it back."""
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File '{filename}' does not exist. You must create it first.")

        with open(filename, 'rb') as f:
            existing_data = pickle.load(f)

        if not isinstance(existing_data, dict):
            raise ValueError("Pickle file does not contain a dictionary.")

        existing_data.update(new_data)

        with open(filename, 'wb') as f:
            pickle.dump(existing_data, f)




    def _update_plot(self):
        """
        Validates component inputs and updates the scatter plot.
        This method is called when the Reload button is pressed.
        """

        if self.explained_variance_var.get():
            self.plot_explained_variance(log_yaxis=self.explained_variance_log.get(), n_components=self.no_comp_var.get())
            return

        x_comp_str = self.x_comp_var.get()
        y_comp_str = self.y_comp_var.get()

        x_comp, y_comp = -1, -1 # Initialize with invalid values

        # --- Input Validation ---
        try:
            x_comp = int(x_comp_str)
        except ValueError:
            messagebox.showwarning("Input Error", "X component must be an integer.")
            return

        try:
            y_comp = int(y_comp_str)
        except ValueError:
            messagebox.showwarning("Input Error", "Y component must be an integer.")
            return

        # Check bounds
        num_components = self.transformed_data.shape[1]
        if not (0 <= x_comp < num_components):
            messagebox.showwarning("Input Error", f"X component {x_comp} is out of bounds. Max component index is {num_components - 1}.")
            return
        if not (0 <= y_comp < num_components):
            messagebox.showwarning("Input Error", f"Y component {y_comp} is out of bounds. Max component index is {num_components - 1}.")
            return

        # --- Update plot if validation passes ---
        colormap_option = self.colormap_var.get()
        if self.clustering_method_var.get() != "None":
            colormap_option = "cluster"

        self._plot_scatter(x_comp, y_comp, colormap_option)

    def _save_figure(self):
        """
        Saves the current Matplotlib figure to a file.
        """
        filename = self.filename_var.get().strip()

        if not filename:
            messagebox.showwarning("Save Warning", "Please enter a filename.")
            return

        # Warn if no extension, but still proceed
        if "." not in filename:
            messagebox.showwarning("File Extension", "No file extension specified. The figure will be saved without one, which might cause issues.")

        try:
            self.fig.savefig(filename, bbox_inches='tight', dpi=300)
            messagebox.showinfo("Save Success", f"Figure saved to: {filename}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save figure: {e}")


# --- New Class for the Loadings Window ---
class LoadingsWindow(tk.Toplevel):
    # TODO: check the arguments, model is superfluous here.
    def __init__(self, parent, model, elements_mask, shape, window_title, h5_filepath=None, h5_dataset=None):
        # TODO: the model argument and how it is handled is quite confusing.
        super().__init__(parent)
        window_title = f"Model: {parent.selected_model} from {parent.model_filepath.get()}"
        self.title(window_title)
        self.geometry("900x700")

        self.protocol("WM_DELETE_WINDOW", self._on_close) # Handle close event


        # Store arguments and initialized data as instance attributes
        # TODO: clean this up, as well as variables names
        self.model = parent.data[parent.selected_model+"_model"]
        self.model_str = model
        self.elements_mask = elements_mask
        self.shape = shape

        # TODO: replace the passed argument with parent.selected_dataset....get()
        self.h5_filepath = h5_filepath
        self.h5_dataset = h5_dataset
        self.fig = None
        self.ax = None
        self.scatter_plot = None
        self.canvas_widget = None
        self.colorbar_obj = None # To keep track of the colorbar for clearing

        self._compute_loadings()
        self._create_widgets(window_title)
        self._plot_loadings(0) # Initial plot

    def _on_close(self):
        """Handles the closing of the preview window."""
        # Removed self.grab_release() as grab_set() is removed
        self.destroy()

    def _create_widgets(self, window_title):
        """Creates all the GUI elements for the preview window."""
        # --- Top Title ---
        title_label = ttk.Label(self, text=window_title, font=("Helvetica", 18, "bold"))
        title_label.pack(pady=10)

        # --- Main content frame (left for plot, right for controls) ---
        main_frame = ttk.Frame(self)
        main_frame.pack(expand=True, fill="both", padx=10, pady=10)

        # Configure columns to expand
        main_frame.grid_columnconfigure(0, weight=3) # Plot column
        main_frame.grid_columnconfigure(1, weight=1) # Controls column
        main_frame.grid_rowconfigure(0, weight=1)

        # --- Left Frame: Scatter Plot ---
        plot_frame = ttk.Frame(main_frame, relief="sunken", borderwidth=2)
        plot_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        self.fig = Figure(figsize=(5, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Add Matplotlib toolbar
        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        toolbar.update()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # --- Right Frame: Controls ---
        controls_frame = ttk.Frame(main_frame)
        controls_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        controls_frame.grid_columnconfigure(0, weight=1) # Allow column to expand

        # --- Component Input ---

        comp_frame = ttk.Frame(controls_frame)
        comp_frame.pack(fill="x", padx=10, pady=2)
        ttk.Label(comp_frame, text="Component (0-indexing):").pack(side="left")
        self.comp_var = tk.StringVar(value="0") # Default X component
        comp_entry = ttk.Entry(comp_frame, textvariable=self.comp_var)
        comp_entry.pack(side="right", expand=True, fill="x")

        # --- Reload Button ---
        reload_button = ttk.Button(controls_frame, text="Reload", command=self._update_plot)
        reload_button.pack(pady=10)

        # --- Separator ---
        separator = ttk.Separator(controls_frame, orient="h")
        separator.pack(fill="x", pady=15)

        # --- Save Figure Section ---
        filename_label = ttk.Label(controls_frame, text="Filename:")
        filename_label.pack(pady=(5,0), anchor="w", padx=10)

        self.filename_var = tk.StringVar()
        filename_entry = ttk.Entry(controls_frame, textvariable=self.filename_var)
        filename_entry.pack(fill="x", padx=10, pady=2)

        save_button = ttk.Button(controls_frame, text="Save Figure", command=self._save_figure)
        save_button.pack(pady=10)

    def _compute_loadings(self):

        if self.model_str in ["pca"]:
            self.loadings = self.model.components_
        if self.model_str in ["umap"]:
            self._compute_umap_loadings()

    def _compute_umap_loadings(self):

        data_3d, _ = get_data_from_dataset(self.h5_filepath, self.h5_dataset)

        data_2d = format_array(data_3d)

        # In shape (T,2)
        timeseries = self._transform_data(self.model, data_2d)

        ts0 = timeseries[:,0]
        ts1 = timeseries[:,1]

        # Correlation map for umap0 and umap1
        corr_map0 = compute_correlation_map(data_3d, ts0)
        corr_map1 = compute_correlation_map(data_3d, ts1)

        cleaned_corr_map0 = corr_map0.flatten()
        cleaned_corr_map0 = cleaned_corr_map0[~np.isnan(cleaned_corr_map0)]

        cleaned_corr_map1 = corr_map1.flatten()
        cleaned_corr_map1 = cleaned_corr_map1[~np.isnan(cleaned_corr_map1)]

        self.loadings = np.empty((2, len(cleaned_corr_map0)))
        self.loadings[0,:] = cleaned_corr_map0
        self.loadings[1,:] = cleaned_corr_map1


    def _transform_data(self, model_object, data_to_transform):
        """
        Placeholder function to simulate model transformation.
        Returns a dummy NumPy array representing transformed data.
        Shape: (num_samples, num_components)
        """

        transformed_data = model_object.transform(data_to_transform)

        return transformed_data
        

    def _plot_loadings(self, comp):
        """
        Displays a 2D matrix (derived from loadings) using imshow,
        along with a colorbar.
        """
        if self.loadings is None:
            messagebox.showerror("Plot Error", "No loadings data available for image plot.")
            return

        # Ensure 'comp' is a valid index for self.loadings
        if not (0 <= comp < self.loadings.shape[0]):
            messagebox.showwarning("Input Error", f"Component {comp} is out of bounds for loadings data.")
            return

        # --- Handle colorbar and recreate axes (robust method) ---
        # 1. Explicitly remove the old colorbar's axes from the figure if it exists.
        if self.colorbar_obj is not None:
            if hasattr(self.colorbar_obj, 'ax') and self.colorbar_obj.ax is not None:
                try:
                    self.fig.delaxes(self.colorbar_obj.ax)
                except ValueError:
                    pass
            self.colorbar_obj = None # Nullify the reference

        # 2. Remove the existing main plot axes (self.ax) from the figure.
        if self.ax is not None and self.ax in self.fig.axes:
            self.fig.delaxes(self.ax)

        # 3. Create a brand new Axes object for the main plot.
        self.ax = self.fig.add_subplot(111) # 1x1 grid, first subplot

        # --- Data preparation for imshow ---
        data_2d = communities_to_window(self.loadings[comp,:], self.elements_mask, self.shape)


        # Ensure data_2d is actually 2D for imshow
        if data_2d.ndim != 2:
            messagebox.showerror("Plot Error", "Processed data is not 2D for imshow.")
            return

        # --- Plotting with imshow ---
        cmap_image = 'viridis' # You can choose a different colormap like 'gray', 'plasma', 'cividis', etc.
        
        # Use imshow to display the 2D matrix
        # 'origin='lower'' often makes images display correctly with (0,0) at bottom-left
        im = self.ax.imshow(data_2d, cmap=cmap_image, origin='lower', aspect='auto')

        # Set title and labels specific to the image plot
        self.ax.set_title(f"Loadings Plot for Component {comp}")
        self.ax.set_xlabel("Column Index") # Or more descriptive label like "X-position"
        self.ax.set_ylabel("Row Index")   # Or more descriptive label like "Y-position"

        # Add the colorbar for the imshow plot
        # The colorbar is associated directly with the image object 'im'
        self.colorbar_obj = self.fig.colorbar(im, ax=self.ax, label="Loadings Value")

        # You might want to remove grid for imshow plots, or make it less prominent
        self.ax.grid(False) # Typically no grid for pixel data
        self.ax.set_aspect('equal')

        # --- Re-apply tight_layout and redraw ---
        self.fig.tight_layout()
        self.canvas.draw_idle() # Redraw the canvas

    def _update_plot(self):
        """
        Validates component inputs and updates the scatter plot.
        This method is called when the Reload button is pressed.
        """
        comp_str = self.comp_var.get()

        comp = -1

        # --- Input Validation ---
        try:
            comp = int(comp_str)
        except ValueError:
            messagebox.showwarning("Input Error", "X component must be an integer.")
            return

        # Check bounds
        num_components = self.loadings.shape[0]
        if not (0 <= comp < num_components):
            messagebox.showwarning("Input Error", f"Component {comp} is out of bounds. Max component index is {num_components - 1}.")
            return

        # --- Update plot if validation passes ---
        self._plot_loadings(comp)

    def _save_figure(self):
        """
        Saves the current Matplotlib figure to a file.
        """
        filename = self.filename_var.get().strip()

        if not filename:
            messagebox.showwarning("Save Warning", "Please enter a filename.")
            return

        # Warn if no extension, but still proceed
        if "." not in filename:
            messagebox.showwarning("File Extension", "No file extension specified. The figure will be saved without one, which might cause issues.")

        try:
            self.fig.savefig(filename, bbox_inches='tight', dpi=300)
            messagebox.showinfo("Save Success", f"Figure saved to: {filename}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save figure: {e}")



# if __name__ == "__main__":
#     app = ModelAnalyzerGUI()
#     app.mainloop()
