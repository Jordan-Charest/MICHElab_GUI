import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from utils.ets import split_into_bins
from utils.hdf5 import get_data_from_dataset, save_data_to_dataset, get_dataset_paths
from utils.community_detection import compute_coassignment_probability_kmeans, compute_coassignment_probability_louvain, communities_to_window, reorder_communities, communities_colormap, hierarchical_clustering, intercommunity
import matplotlib.colors as mcolors
from matplotlib.colors import TwoSlopeNorm
from sklearn.cluster import KMeans
import bct
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from utils.other import cast_str_to_float_and_int, flat_to_symmetric
from scipy.stats import linregress

class CommunityDetectionGUI(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.method_params = {
            "Louvain": (("gamma_start", 0.9), ("gamma_end", 1.1), ("n_iter", 10)),
            "KMeans": (("min_communities", 2), ("max_communities", 6), ("n_iter", 10)),
            "None": ()
        }
        self.param_widgets = []  # Hold dynamic param fields
        self.create_widgets()

    def create_widgets(self):
        # --- Filepath Section ---
        file_frame = ttk.Frame(self)
        file_frame.grid(row=0, column=0, sticky="ew", pady=(5, 10), padx=10)
        # file_frame.columnconfigure(0, weight=1)

        self.file_entry = ttk.Entry(file_frame, width=80)
        self.file_entry.grid(row=0, column=0, sticky="ew")
        browse_button = ttk.Button(file_frame, text="Browse", command=self.browse_file)
        browse_button.grid(row=0, column=1, padx=(5, 0))

        # --- Method and Parameters Section ---
        param_frame = ttk.Frame(self)
        param_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10), padx=10)
        row = 0

        ttk.Label(param_frame, text="Method:").grid(row=row, column=0, sticky="w")
        self.method_var = tk.StringVar(value="Louvain")
        method_menu = ttk.OptionMenu(param_frame, self.method_var, "Louvain", *self.method_params.keys(), command=self.update_params)
        method_menu.grid(row=row, column=1, sticky="w", padx=(0, 10))

        self.dynamic_param_frame = ttk.Frame(param_frame)
        self.dynamic_param_frame.grid(row=0, column=2, sticky="w")

        self.param_frame = param_frame  # Store for later dynamic use
        self.update_params("Louvain")

        # --- Restrict Segment Section ---
        restrict_frame = ttk.Frame(self)
        restrict_frame.grid(row=2, column=0, sticky="w", pady=(0, 10), padx=10)

        self.restrict_var = tk.BooleanVar(value=False)
        restrict_cb = ttk.Checkbutton(restrict_frame, text="Use only segment", variable=self.restrict_var, command=self.toggle_restrict)
        restrict_cb.grid(row=0, column=0, sticky="w")

        self.from_spinbox = tk.Spinbox(restrict_frame, from_=0, to=9999, width=5, state="disabled")
        self.from_spinbox.grid(row=0, column=1, padx=(5, 0))
        ttk.Label(restrict_frame, text="out of").grid(row=0, column=2, padx=5)
        self.to_spinbox = tk.Spinbox(restrict_frame, from_=0, to=9999, width=5, state="disabled")
        self.to_spinbox.grid(row=0, column=3)
        ttk.Label(restrict_frame, text="(0-indexing)").grid(row=0, column=4, padx=5)

        # --- Compute Button ---
        compute_button = ttk.Button(self, text="Compute coassignment matrix", command=self.compute_coassignment)
        compute_button.grid(row=3, column=0, pady=(0, 10), padx=10, sticky="w")

        # --- Separator ---
        separator = ttk.Separator(self, orient="horizontal")
        separator.grid(row=4, column=0, sticky="ew", padx=10, pady=(5, 10))

        # --- Preview Button ---
        preview_button = ttk.Button(self, text="Preview communities", command=self.preview_communities)
        preview_button.grid(row=5, column=0, padx=10, pady=(0, 10), sticky="w")

    def browse_file(self):
        filepath = filedialog.askopenfilename(filetypes=[("HDF5 files", "*.h5")])
        if filepath:
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, filepath)

    def update_params(self, method_name):
        # Clear existing parameter widgets
        for widget in self.dynamic_param_frame.winfo_children():
            widget.destroy()
        self.param_widgets.clear()

        params = self.method_params.get(method_name, ())
        for idx, (label_text, default_val) in enumerate(params):
            label = ttk.Label(self.dynamic_param_frame, text=f"{label_text}:")
            entry = ttk.Entry(self.dynamic_param_frame, width=7)
            entry.insert(0, str(default_val))

            label.grid(row=0, column=idx * 2, sticky="w", padx=(0, 2))
            entry.grid(row=0, column=idx * 2 + 1, sticky="w", padx=(0, 10))
            self.param_widgets.append((label, entry))

    def toggle_restrict(self):
        state = "normal" if self.restrict_var.get() else "disabled"
        self.from_spinbox.config(state=state)
        self.to_spinbox.config(state=state)

    def compute_coassignment(self):
        if self.restrict_var.get():
            try:
                start = int(self.from_spinbox.get())
                end = int(self.to_spinbox.get())
                if start >= end:
                    messagebox.showwarning("Invalid Range", "Segment value out of bounds")
                    return
            except ValueError:
                messagebox.showwarning("Invalid Input", "Spinbox values must be integers.")
                return

        
        print("Computing coassignment matrix")
        print(f"Selected file: {self.file_entry.get()}")
        print(f"Method: {self.method_var.get()}")
        param_dict = {label.cget("text")[:-1]: cast_str_to_float_and_int(entry.get()) for label, entry in self.param_widgets}
        print(f"Parameters: {param_dict}")

        dfc, attr = get_data_from_dataset(self.file_entry.get(), "dfc")

        if self.restrict_var.get():
            cts, _ = get_data_from_dataset(self.file_entry.get(), "cts")
            mats, indices = split_into_bins(cts, int(self.to_spinbox.get()))
            indices = indices[int(self.from_spinbox.get())] # Keeps only the indices from the specified segment

            dfc = dfc[indices,:]
            FC = np.mean(dfc, axis=0)
        else:
            indices = list(range(dfc.shape[0])) # Keep all indices
            FC = np.mean(dfc, axis=0)

        N = int((np.sqrt(8*len(FC)+1)-1)/2)
        FC = flat_to_symmetric(FC, N)

        print(FC.shape)
        print(f"Keeping {len(indices)} indices")

        del dfc, indices, attr

        

        if self.method_var.get() == "KMeans":
            consensus_mat = compute_coassignment_probability_kmeans(FC, threshold=False, N_iters=param_dict["n_iter"],
                                                            k_min=param_dict["min_communities"], k_max=param_dict["max_communities"])
        elif self.method_var.get() == "Louvain":
            consensus_mat = compute_coassignment_probability_louvain(FC, threshold=False, N_iters=param_dict["n_iter"],
                                                            gamma_min=param_dict["gamma_start"], gamma_max=param_dict["gamma_end"])
        else:
            messagebox.showwarning("Invalid method", f"Selected method '{self.method_var.get()}' invalid or not implemented.")
            return
            
        attributes = param_dict
        attributes.update({"method": self.method_var.get()})
        dataset_name = f"consensus_matrix/{attributes['method']}"
        
        if self.restrict_var.get():
            attributes.update({"selected_segment": int(self.from_spinbox.get()), "n_segments": int(self.to_spinbox.get())})
            dataset_name += f"_segment_{attributes['selected_segment']}_of_{attributes['n_segments']}"

        save_data_to_dataset(self.file_entry.get(), dataset_name, consensus_mat,
                             attributes=attributes)
        
        print(f"Computed and saved consensus matrix under dataset {dataset_name}.")


    def preview_communities(self):

        selected_file = self.file_entry.get()
        # print(f"Previewing communities for file: {selected_file}")

        dataset_paths = get_dataset_paths(selected_file, "consensus_matrix")

        clustering_window = ClusteringGUI(self, selected_file, dataset_paths)

    # def open_clustering_gui(self):
    #     selected_datasets = list(self.tree.selection())
    #     if selected_datasets:
    #         clustering_window = tk.Toplevel(self.root)
    #         ClusteringGUI(clustering_window, self.file_path, selected_datasets)


class ClusteringGUI(tk.Toplevel):
    def __init__(self, root, filepath, dataset_paths):
        super().__init__(root)
        self.title("Clustering")

        self.dataset_paths = dataset_paths
        self.filepath = filepath

        self.display_matrix = tk.StringVar(value="FC")
        self.clustering_method = tk.StringVar(value="Louvain")
        self.n_communities = tk.IntVar(value=4)
        self.gamma = tk.DoubleVar(value=1.0)
        self.n_bins = 0  # updates when selecting a dataset

        self.constant_lims_var = tk.IntVar()
        self.diverging_colormap_var = tk.IntVar()
        self.reorder_mat_var = tk.IntVar()

        self.create_widgets()  # call the widget builder

    def create_widgets(self):
        # ---------------- Dropdown for dataset selection ----------------
        tk.Label(self, text="Select Dataset:").pack(padx=10, pady=(10, 0))
        self.dataset_dropdown = ttk.Combobox(self, values=self.dataset_paths,
                                             state="readonly", width=70)
        self.dataset_dropdown.pack(fill=tk.X, padx=10, pady=(5, 10))
        self.dataset_dropdown.bind("<<ComboboxSelected>>", self.update_fields)

        # ---------------- Main layout frames ----------------
        container = tk.Frame(self)  # holds left/right frames
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.left_frame = tk.Frame(container)
        self.left_frame.pack(side=tk.LEFT, padx=1, pady=10, fill=tk.BOTH, expand=True)

        self.right_frame = tk.Frame(container)
        self.right_frame.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.BOTH, expand=True)

        # ---------------- Left panel controls ----------------
        tk.Label(self.left_frame, text="Matrix to display:").pack()
        for matrix in ["FC", "Consensus"]:
            tk.Radiobutton(self.left_frame, text=matrix,
                           variable=self.display_matrix, value=matrix).pack()

        tk.Label(self.left_frame, text="Clustering Method:").pack()
        for method in ["Louvain", "k-means", "Hierarchical"]:
            tk.Radiobutton(self.left_frame, text=method,
                           variable=self.clustering_method, value=method,
                           command=self.update_fields).pack()

        tk.Label(self.left_frame, text="# of Communities:").pack()
        self.n_communities_entry = tk.Entry(self.left_frame, textvariable=self.n_communities,
                                            state=tk.DISABLED)
        self.n_communities_entry.pack()

        tk.Label(self.left_frame, text="Gamma:").pack()
        self.gamma_entry = tk.Entry(self.left_frame, textvariable=self.gamma)
        self.gamma_entry.pack()

        tk.Label(self.left_frame, text="Component #").pack()
        self.component_dropdown = ttk.Combobox(self.left_frame, state=tk.DISABLED)
        self.component_dropdown.pack()
        self.update_component_dropdown()

        self.constant_lims_box = tk.Checkbutton(
            self.left_frame, text="Constant colormap limits",
            variable=self.constant_lims_var, onvalue=1, offvalue=0, state=tk.DISABLED)
        self.constant_lims_box.pack()

        self.diverging_colormap_box = tk.Checkbutton(
            self.left_frame, text="Diverging colormap",
            variable=self.diverging_colormap_var, onvalue=1, offvalue=0)
        self.diverging_colormap_box.pack()

        self.reorder_mat_box = tk.Checkbutton(
            self.left_frame, text="Reorder matrix",
            variable=self.reorder_mat_var, onvalue=1, offvalue=0)
        self.reorder_mat_box.pack()

        self.preview_button = tk.Button(self.left_frame, text="Preview", command=self.display_preview)
        self.preview_button.pack(pady=(10, 0))

        # ---------------- Right panel previews ----------------
        self.figure, self.ax = plt.subplots(1, 1, figsize=(6, 6))
        self.canvas = FigureCanvasTkAgg(self.figure, self.right_frame)
        self.canvas.get_tk_widget().pack(side=tk.LEFT)

        self.small_preview_frame = tk.Frame(self.right_frame)
        self.small_preview_frame.pack(side=tk.RIGHT, padx=10, pady=10)

        self.small_canvas1 = FigureCanvasTkAgg(plt.Figure(figsize=(3, 3)), self.small_preview_frame)
        self.small_canvas1.get_tk_widget().pack(side=tk.TOP, padx=5)

        self.small_canvas2 = FigureCanvasTkAgg(plt.Figure(figsize=(3, 3)), self.small_preview_frame)
        self.small_canvas2.get_tk_widget().pack(side=tk.TOP, padx=5)

    def update_fields(self, event=None):

        self.update_component_dropdown()

        if self.clustering_method.get() in ["k-means", "Hierarchical"]:
            self.n_communities_entry.config(state=tk.NORMAL)
            self.gamma_entry.config(state=tk.DISABLED)
        else:
            self.n_communities_entry.config(state=tk.DISABLED)
            self.gamma_entry.config(state=tk.NORMAL)

        # TODO: the check below could maybe be combined with update_component dropdown since they both activate at the same time. Issue is when constant_lims_box is defined in the current code.
        if self.dataset_dropdown.get().split("/")[1].startswith("consensus"): # Whole
            self.constant_lims_box.config(state=tk.DISABLED)
        else:
            self.constant_lims_box.config(state=tk.NORMAL)

    def update_component_dropdown(self, event=None):
        """Update the component dropdown based on n_bins"""

        # Update self.n_bins first:
        try:
            dataset = self.dataset_dropdown.get()
            last_part = dataset.split('/')[-1]
            self.n_bins = int(last_part.split('_')[0])
        except:
            self.n_bins = 0

        if self.n_bins > 0:
            # Populate dropdown with numbers from 0 to n_bins-1
            self.component_dropdown.config(state="normal", values=list(range(self.n_bins)))
            self.component_dropdown.set(0)
        else:
            # Disable the dropdown if n_bins is 0
            self.component_dropdown.config(state="disabled", values=[])

    def get_FC_dataset_path(self, dataset_path):

        # Retrieve FC data
        dataset_path_parts = dataset_path.split('/')
        if dataset_path_parts[1].startswith("consensus"): # Whole
            FC_dataset_path = dataset_path_parts[0] + "/whole"
            comp = None
            vmin, vmax = None, None
        else:
            last_part_split = dataset_path_parts[-1].split("_")
            FC_dataset_path = dataset_path_parts[0] + "/" + dataset_path_parts[1] + "/" + last_part_split[0] + "_" + last_part_split[1]
            comp = int(self.component_dropdown.get())
            vmin, vmax = None, None

        return FC_dataset_path, comp, vmin, vmax
    
    def get_FC(self, dataset_path):

        dFC, _ = get_data_from_dataset(self.filepath, "dfc")
        cts, _ = get_data_from_dataset(self.filepath, "cts")

        dataset_path_parts = dataset_path.split('/')

        if 'segment' in dataset_path_parts[1]: # Segmented FC
            last_part_split = dataset_path_parts[1].split('_')
            segment_index = int(last_part_split[2])
            n_segments = int(last_part_split[4])

            mats, indices = split_into_bins(cts, n_segments)

            kept_indices = indices[segment_index]

            dFC = dFC[kept_indices, :]

        else: # Whole FC
            kept_indices = None

        FC = np.mean(dFC, axis=0)
        FC = flat_to_symmetric(FC, N=int((np.sqrt(8*len(FC)+1)-1)/2))

        return FC, kept_indices

    
    def update_smaller_previews(self, consensus_mat, communities, cmap):
        """Function to update the two smaller preview images, including a color swatch for clusters."""

        # Retrieve FC data REPLACED BY METHOD
        # dataset_path_parts = self.dataset.split('/')
        # if dataset_path_parts[1].startswith("consensus"): # Whole
        #     FC_dataset_path = dataset_path_parts[0] + "/whole"
        #     comp = None
        #     vmin, vmax = None, None
        # else:
        #     last_part_split = dataset_path_parts[-1].split("_")
        #     FC_dataset_path = dataset_path_parts[0] + "/" + dataset_path_parts[1] + "/" + last_part_split[0] + "_" + last_part_split[1]
        #     comp = int(self.component_dropdown.get())

        # FC_dataset_path, comp, vmin, vmax = self.get_FC_dataset_path(self.dataset)

        # data, attr = get_data_from_dataset(self.filepath, FC_dataset_path)

        data, comp = self.get_FC(self.dataset)

        # Check which component
        if comp is not None:
            vmin, vmax = np.nanmin(data), np.nanmax(data)
            data = data[comp]

        # Choose which matrix to display in fig1
        if self.display_matrix.get() == "FC":
            if self.reorder_mat_var:
                mat_to_display = reorder_communities(data, communities)
            else:
                mat_to_display = data
            fig1_title = "FC"
        elif self.display_matrix.get() == "Consensus":
            if self.reorder_mat_var:
                mat_to_display = reorder_communities(consensus_mat, communities)
            else:
                mat_to_display = consensus_mat
            fig1_title = "Consensus matrix"
            

        # Set vlims for colorbars:
        if self.constant_lims_var.get():
            vmin_bottom, vmax_bottom = vmin, vmax
            if self.display_matrix.get() == "Consensus": # Special case; do not use vmin and vmax
                vmin_top, vmax_top = None, None
            elif self.display_matrix.get() == "FC":
                vmin_top, vmax_top = vmin, vmax
        else:
            vmin_bottom, vmax_bottom, vmin_top, vmax_top = None, None, None, None

        # Set Norms for colormaps
        if self.diverging_colormap_var.get():
            norm_top = TwoSlopeNorm(vmin=vmin_top, vcenter=0, vmax=vmax_top)
            norm_bottom = TwoSlopeNorm(vmin=vmin_bottom, vcenter=0, vmax=vmax_bottom)
        else:
            norm_top = mcolors.Normalize(vmin=vmin_top, vmax=vmax_top)
            norm_bottom = mcolors.Normalize(vmin=vmin_bottom, vmax=vmax_bottom)


        # Compute the order of rows based on communities
        unique_communities, cluster_sizes = np.unique(communities, return_counts=True)

        # Generate a color swatch matching the clustering
        colors = [cmap(i / (len(unique_communities) - 1)) for i in range(len(unique_communities))]
        cluster_colors = np.concatenate([np.full((size, 1, 3), colors[i][:3]) for i, size in enumerate(cluster_sizes)], axis=0)

        # Create figure with two subplots: one for matrix, one for color swatch
        fig1 = self.small_canvas1.figure
        fig1.clear()

        gs = fig1.add_gridspec(1, 2, width_ratios=[1, 10])  # Adjust width ratio for proper spacing
        ax1 = fig1.add_subplot(gs[0, 1])
        ax2 = fig1.add_subplot(gs[0, 0])

        # Plot consensus matrix with fixed aspect ratio
        im1 = ax1.imshow(mat_to_display, cmap="coolwarm", aspect="auto", norm=norm_top)
        ax1.axis('off')
        ax1.set_title(fig1_title)
        colorbar1 = fig1.colorbar(im1, ax=ax1)

        # Plot color swatch, ensuring it matches the matrix height
        ax2.imshow(cluster_colors, aspect="auto")
        ax2.axis('off')

        self.small_canvas1.draw()

        ###### SECOND GRAPH

        intercommunities = intercommunity(data, communities).T

        cluster_colors = np.concatenate([np.full((1, 1, 3), colors[i][:3]) for i, size in enumerate(cluster_sizes)], axis=0) # TODO: Change this; unnecessarily complicated


        fig2 = self.small_canvas2.figure
        fig2.clear()
        gs2 = fig2.add_gridspec(1, 2, width_ratios=[1, 10])  # Adjust width ratio for proper spacing
        ax3 = fig2.add_subplot(gs2[0, 1])
        ax4 = fig2.add_subplot(gs2[0, 0])


        ax3.set_title("Ave. community connectivity")

        im3 = ax3.imshow(intercommunities, cmap="coolwarm", aspect="auto", norm=norm_bottom)  # Replace with relevant data
        ax3.axis('off')
        colorbar3 = fig2.colorbar(im3, ax=ax3, )

        ax4.imshow(cluster_colors, aspect="auto")
        ax4.axis('off')

        self.small_canvas2.draw()

        return fig1, fig2

    def display_preview(self):
        try:
            self.dataset = self.dataset_dropdown.get()
            clustering_method = self.clustering_method.get()
            if clustering_method in ["k-means", "Hierarchical"]:
                n_communities = int(self.n_communities_entry.get())
            elif clustering_method in ["Louvain"]:
                gamma = float(self.gamma_entry.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Enter valid parameters.")
            return
        
        data, _ = get_data_from_dataset(self.filepath, self.dataset)

        _, dfc_attr = get_data_from_dataset(self.filepath, "dfc")
        window_shape = dfc_attr["window_shape"][1:]
        element_mask = dfc_attr["elements_mask"]

        if data.ndim == 3:  # Multiple components
            comp_n = int(self.component_dropdown.get())
            consensus_mat = data[comp_n, :, :]
        else:
            consensus_mat = data

        if clustering_method == "k-means":
            kmeans = KMeans(n_clusters=n_communities, random_state=0, n_init="auto").fit(consensus_mat)
            communities = kmeans.labels_

        elif clustering_method == "Hierarchical":
            linkage_mat, communities = hierarchical_clustering(consensus_mat, max_clusters=n_communities, show=False, method='ward')

        elif clustering_method == "Louvain":
            communities = bct.modularity_louvain_und(consensus_mat, gamma=gamma, seed=None)[0]
            
        window = communities_to_window(communities, element_mask, window_shape)
        cmap, _ = communities_colormap(communities)
        
        self.ax.clear()
        self.ax.imshow(window, cmap=cmap)
        self.ax.axis('off')
        
        # Update the two smaller previews as well
        self.update_smaller_previews(consensus_mat, communities, cmap)

        self.canvas.draw()

        np.save("clustering.npy", window)

