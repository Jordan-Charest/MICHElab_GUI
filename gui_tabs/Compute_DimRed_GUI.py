import tkinter as tk
import numpy as np
from tkinter import filedialog, messagebox, ttk
import sys
from datetime import datetime
from utils.hdf5 import get_data_from_dataset
from utils.ets import format_array
from utils.other import merge_dictionaries
import umap
from sklearn.decomposition import PCA
import pickle
import os


class ConsoleRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, message):
        self.text_widget.config(state="normal") # Enable for writing
        self.text_widget.insert(tk.END, message)
        self.text_widget.see(tk.END)  # Auto-scroll to the end
        self.text_widget.config(state="disabled") # Disable after writing

    def flush(self):
        pass # Required for file-like objects

class compute_DimRedGUI(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        style = ttk.Style(self)

        self.filepaths = []
        self.datasets = []
        self.highlighted_entry = None # To keep track of the currently highlighted entry

        self.create_scrollable_frame() # New method to create the scrollable area
        self.create_widgets()

        # Redirect stdout to the console text widget
        # self.original_stdout = sys.stdout
        # sys.stdout = ConsoleRedirector(self.console_text)

        # Update scroll region when the main_frame (content inside canvas) size changes
        self.main_frame.bind("<Configure>", self.on_frame_configure)
        # Bind the canvas itself to resize the window (in case the main_frame dictates a larger size)
        self.canvas.bind("<Configure>", self.on_canvas_configure)


    def __del__(self):
        # Restore original stdout when the application closes
        sys.stdout = self.original_stdout

    def create_scrollable_frame(self):
        self.canvas = tk.Canvas(self)
        self.scrollbar = tk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.main_frame = tk.Frame(self.canvas)
        # We need to bind the size of the window to the main_frame, not just the content inside it
        self.main_frame.bind("<Configure>", self.on_frame_configure)
        self.canvas_frame_id = self.canvas.create_window((0, 0), window=self.main_frame, anchor="nw")

    def on_frame_configure(self, event=None):
        # Update the scroll region of the canvas whenever the 'main_frame' size changes
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def on_canvas_configure(self, event=None):
        # Adjust the width of the main_frame inside the canvas when canvas changes size
        canvas_width = event.width if event else self.canvas.winfo_width()
        self.canvas.itemconfig(self.canvas_frame_id, width=canvas_width)

    def create_widgets(self):
        # All widgets will now be placed inside self.main_frame

        data_selection_frame = ttk.Frame(self.main_frame)
        data_selection_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        data_selection_frame.columnconfigure(0, weight=1)
        data_selection_frame.columnconfigure(1, weight=0)
        data_selection_frame.rowconfigure(0, weight=0)
        data_selection_frame.rowconfigure(1, weight=0)

        # --- Filepaths Section ---
        filepath_labelframe = ttk.LabelFrame(data_selection_frame, text="Input Files")
        filepath_labelframe.grid(row=0, column=0, sticky="nsew", padx=(0, 8), pady=(0, 2))
        self.filepath_entries_frame, fp_canvas = self._create_scrollable_content(filepath_labelframe)

        # Mousewheel binding
        fp_canvas.bind("<Enter>", lambda e: self._bind_mousewheel(e, fp_canvas))
        fp_canvas.bind("<Leave>", self._unbind_mousewheel)
        fp_canvas.config(height=100, width=200)

        fp_button_frame = ttk.Frame(data_selection_frame)
        fp_button_frame.grid(row=0, column=1, sticky="n", padx=(0, 8), pady=(0, 2))
        ttk.Button(fp_button_frame, text="Add", command=self.add_filepath_entry).pack(fill="x", pady=2)
        ttk.Button(fp_button_frame, text="Delete", command=self.delete_filepath_entry).pack(fill="x", pady=2)
        ttk.Button(fp_button_frame, text="Browse", command=self.browse_filepath).pack(fill="x", pady=2)

        # --- Datasets Section ---
        dataset_labelframe = ttk.LabelFrame(data_selection_frame, text="Datasets")
        dataset_labelframe.grid(row=1, column=0, sticky="nsew", padx=(0, 8))
        self.dataset_entries_frame, ds_canvas = self._create_scrollable_content(dataset_labelframe)

        ds_canvas.bind("<Enter>", lambda e: self._bind_mousewheel(e, ds_canvas))
        ds_canvas.bind("<Leave>", self._unbind_mousewheel)
        ds_canvas.config(height=100, width=200)

        ds_button_frame = ttk.Frame(data_selection_frame)
        ds_button_frame.grid(row=1, column=1, sticky="n", padx=(0, 8))
        ttk.Button(ds_button_frame, text="Add", command=self.add_dataset_entry).pack(fill="x", pady=2)
        ttk.Button(ds_button_frame, text="Delete", command=self.delete_dataset_entry).pack(fill="x", pady=2)

        # --- Output File ---
        output_frame = ttk.LabelFrame(self.main_frame, text="Output File")
        output_frame.pack(padx=10, pady=5, fill="x")

        self.output_filepath_entry = ttk.Entry(output_frame, width=50) # Set width for output file entry
        self.output_filepath_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))

        browse_output_btn = ttk.Button(output_frame, text="Browse", command=self.browse_output_folder)
        browse_output_btn.pack(side="right")

        # --- Checkboxes ---
        checkbox_frame = ttk.Frame(self.main_frame)
        checkbox_frame.pack(pady=10)

        self.umap_var = tk.BooleanVar()
        self.pca_var = tk.BooleanVar()

        umap_checkbox = ttk.Checkbutton(checkbox_frame, text="UMAP", variable=self.umap_var)
        umap_checkbox.pack(side="left", padx=10)

        pca_checkbox = ttk.Checkbutton(checkbox_frame, text="PCA", variable=self.pca_var)
        pca_checkbox.pack(side="left", padx=10)

        # --- Run Button ---
        run_button = ttk.Button(self.main_frame, text="Run", command=self.run_process)
        run_button.pack(pady=10)

        # --- Console Box ---
        # console_frame = ttk.LabelFrame(self.main_frame, text="Console Output")
        # console_frame.pack(padx=10, pady=5, fill="both", expand=True)

        # self.console_text = tk.Text(console_frame, wrap="word", height=10, state="disabled")
        # self.console_text.pack(side="left", fill="both", expand=True)

        # console_scrollbar = ttk.Scrollbar(console_frame, command=self.console_text.yview)
        # console_scrollbar.pack(side="right", fill="y")
        # self.console_text.config(yscrollcommand=console_scrollbar.set)

    def add_entry_field(self, parent_frame, entry_list):
        entry = tk.Entry(parent_frame, width=40) # Set a fixed width for Entry widgets
        entry.pack(fill="x", padx=2, pady=2)
        entry.bind("<Button-1>", lambda event, e=entry: self.highlight_entry(e, entry_list))
        entry_list.append(entry)
        self.clear_highlight(entry_list) # Clear previous highlights when adding new
        self.update_scroll_region() # Update scroll region after adding content

    def delete_entry_field(self, entry_list):
        # Delete selected (highlighted) entry if valid
        if self.highlighted_entry and self.highlighted_entry in entry_list:
            entry_list.remove(self.highlighted_entry)
            self.highlighted_entry.destroy()
            self.highlighted_entry = None
        # If nothing selected, delete the last entry in the list
        elif entry_list:
            last_entry = entry_list.pop()
            last_entry.destroy()
            self.highlighted_entry = None
        else:
            # Nothing to delete
            return

        self.clear_highlight(entry_list)  # Clear any remaining highlights
        self.update_scroll_region()       # Refresh the scroll region

    def add_filepath_entry(self):
        self.add_entry_field(self.filepath_entries_frame, self.filepaths)

    def add_dataset_entry(self):
        self.add_entry_field(self.dataset_entries_frame, self.datasets)

    def delete_filepath_entry(self):
        self.delete_entry_field(self.filepaths)

    def delete_dataset_entry(self):
        self.delete_entry_field(self.datasets)

    def highlight_entry(self, selected_entry, entry_list):
        self.clear_highlight(entry_list)
        selected_entry.config(bg="lightblue")
        self.highlighted_entry = selected_entry

    def clear_highlight(self, entry_list):
        for entry in entry_list:
            entry.config(bg="white")
        self.highlighted_entry = None

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

    def browse_filepath(self):
        filepath = filedialog.askopenfilename()
        if filepath:
            if self.highlighted_entry and self.highlighted_entry in self.filepaths:
                # If an entry is highlighted, update that one
                self.highlighted_entry.delete(0, tk.END)
                self.highlighted_entry.insert(0, filepath)
            else:
                # Otherwise, add a new entry and put the path into it
                self.add_filepath_entry()
                self.filepaths[-1].delete(0, tk.END)
                self.filepaths[-1].insert(0, filepath)
                self.highlight_entry(self.filepaths[-1], self.filepaths)
        self.clear_highlight(self.filepaths)

    def browse_output_folder(self):
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            output_path = f"{folder_selected}/output_dimred.pkl"
            self.output_filepath_entry.delete(0, tk.END)
            self.output_filepath_entry.insert(0, output_path)

    def run_process(self):
        print(f"[{self.current_time()}] Starting.")

        # Collect filepaths
        actual_filepaths = [entry.get() for entry in self.filepaths if entry.get().strip()]
        if not actual_filepaths:
            messagebox.showerror("Error", "Please provide at least one input filepath.")
            return

        # Collect datasets
        actual_datasets = [entry.get() for entry in self.datasets if entry.get().strip()]
        if not actual_datasets:
            messagebox.showerror("Error", "Please provide at least one dataset.")
            return # Decide if this is an error or just a warning based on your use case

        # Check checkbox states
        umap_selected = self.umap_var.get()
        pca_selected = self.pca_var.get()

        if not umap_selected and not pca_selected:
            messagebox.showerror("Error", "Please select at least one option: UMAP or PCA.")
            print(f"[{self.current_time()}] Error: UMAP or PCA not selected.")
            return

        output_file = self.output_filepath_entry.get().strip()
        if not output_file:
            messagebox.showerror("Error", "Please specify an output file.")
            print(f"[{self.current_time()}] Error: Output file not specified.")
            return

        
        # Execution

        # 1. Update console
        self.show_variables(actual_filepaths, actual_datasets, umap_selected, pca_selected, output_file)

        # 2. Retrieve and format input data
        input_data, elements_mask, window_shape = self.format_input_data(actual_filepaths, actual_datasets)
        print(f"[{self.current_time()}] Data loaded.")

        # 3. Perform UMAP and/or PCA

        to_save = {}

        if umap_selected:

            print(f"[{self.current_time()}] Performing UMAP.")

            # UMAP parameters
            attr = {"n_neighbors": 300, "min_dist": 0.1, "metric": "euclidean"}

            # Perform UMAP
            umap_model, umap_embedding = self.perform_umap(input_data, **attr)

            # Save data
            umap_to_save = {"umap_model": umap_model, "umap_embedding": umap_embedding,
                            "umap_params": attr,
                            "umap_input_filepaths": actual_filepaths, "umap_input_datasets": actual_datasets}

            if elements_mask is not None:
                umap_to_save["elements_mask"] = elements_mask
            if window_shape is not None:
                umap_to_save["shape"] = window_shape

            merge_dictionaries(to_save, umap_to_save)

            print(f"[{self.current_time()}] UMAP done.")

        if pca_selected:

            print(f"[{self.current_time()}] Performing PCA.")

            # PCA parameters
            attr = {"n_components": None}

            # Perform PCA
            pca_model, pca_embedding = self.perform_pca(input_data, **attr)

            # Save data
            pca_to_save = {"pca_model": pca_model, "pca_embedding": pca_embedding,
                            "pca_params": attr,
                            "pca_input_filepaths": actual_filepaths, "pca_input_datasets": actual_datasets}

            if elements_mask is not None:
                pca_to_save["elements_mask"] = elements_mask
            if window_shape is not None:
                pca_to_save["shape"] = window_shape

            merge_dictionaries(to_save, pca_to_save)

            print(f"[{self.current_time()}] PCA done.")

        self.save_dict_to_pkl(to_save, output_file)

        print(f"[{self.current_time()}] Finished.")

    def show_variables(self, filepaths, datasets, umap_selected, pca_selected, output_file):
    
        print(f"[{self.current_time()}] Filepaths received: {filepaths}")
        print(f"[{self.current_time()}] Datasets received: {datasets}")
        print(f"[{self.current_time()}] UMAP selection: {umap_selected}")
        print(f"[{self.current_time()}] PCA selection: {pca_selected}")
        print(f"[{self.current_time()}] Output File: {output_file}")


    def update_scroll_region(self):
        # This function updates the scrollable area of the canvas.
        # It should be called whenever the size of content in main_frame changes.
        # Use after_idle to ensure the widget has been rendered before calculating bbox
        self.after_idle(lambda: self.canvas.config(scrollregion=self.canvas.bbox("all")))

    def format_input_data(self, filepaths, datasets):

        data_dim = None
        input_data = None
        elements_mask = None
        window_shape = None

        for filepath in filepaths:
            for dataset in datasets:

                # Load data
                data, attr = get_data_from_dataset(filepath, dataset)

                # Validate that the data dimension matches with prior data
                dim = data.ndim

                if data_dim is None:
                    data_dim = dim
                if data_dim != dim:
                    raise ValueError("All input data must be of the same dimension (3d or 2d).")
                
                # Format data
                if data_dim == 3:
                    window_shape = (data.shape[1], data.shape[2])
                    data, elements_mask = format_array(data, return_mask=True)

                if input_data is None:
                    input_data = data
                else:
                    data = np.concatenate((input_data, data), axis=0)


        return input_data, elements_mask, window_shape


    def save_dict_to_pkl(self, data_dict: dict, filepath: str):
        """
        Saves a Python dictionary to a .pkl file.

        Args:
            data_dict (dict): The dictionary to be saved.
            filepath (str): The full path to the .pkl file (e.g., "data/my_settings.pkl").
                            The directory containing the file must exist.
        """
        try:
            # Ensure the directory exists
            directory = os.path.dirname(filepath)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
                print(f"Created directory: {directory}")

            with open(filepath, 'wb') as f:
                pickle.dump(data_dict, f)

        except Exception as e:
            print(f"Error saving dictionary to {filepath}: {e}")
                
        
    def perform_pca(self, data, **kwargs):
        
        # Fit the PCA object
        pca_model = PCA(**kwargs)
        pca_model.fit(data)

        # Embed the data
        embedding = pca_model.transform(data)

        return pca_model, embedding

    def perform_umap(self, data, **kwargs):

        # Fit the UMAP object
        umap_model = umap.UMAP(**kwargs)
        umap_model.fit(data)

        # Embed the data
        embedding = umap_model.transform(data)

        return umap_model, embedding

    def current_time(self):

        return datetime.now().strftime('%H:%M:%S')



# if __name__ == "__main__":
#     app = Application()
#     app.mainloop()