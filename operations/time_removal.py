import numpy as np
import warnings
import sys
import os
import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from copy import copy
from matplotlib.patches import Rectangle

warnings.filterwarnings('ignore')

# Necessary for running as a subprocess
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.parsing import parse_key_value_args
from utils.hdf5 import get_data_from_dataset, save_data_to_dataset, parameter_GUI, log

# TODO: Implement method for 1d data
# TODO: Implement frame selection method in addition to frame average

def remove_time(filename, dataset_paths, indices=[], radius=1, ts_len=1, verbosity=5, indices_to_remove=None):

    if indices_to_remove is None:
        indices_to_remove = []

        # Build the list of indices to remove using the selected indices and radius
        for index in indices:
            index_range = list(np.arange(index-radius, index+radius+1, 1))
            for i in index_range:
                indices_to_remove.append(i)

        indices_to_remove = np.asarray(indices_to_remove)
        indices_to_remove = np.delete(indices_to_remove, np.where(~(0<=indices_to_remove)))
        indices_to_remove = np.delete(indices_to_remove, np.where(~(indices_to_remove<ts_len)))

    for dataset_path in dataset_paths:
        
        data, attributes = get_data_from_dataset(filename, dataset_path)

        if data.ndim == 3:
            trimmed_data = np.delete(data, indices_to_remove, axis=0)
        elif data.ndim == 1: # TODO: not sure this check is necessary, should probably work with axis=0 as well
            trimmed_data = np.delete(data, indices_to_remove)
        else:
            raise ValueError("The dataset for trimming must be 1d or 3d.")

        attributes["removed_time_indices"] = indices_to_remove

        save_data_to_dataset(filename, dataset_path, trimmed_data, attributes=attributes)

    log(f"\nIndices removal finished.", level=1, verbosity=verbosity)

    return indices_to_remove

class TimeSeriesSelector:
    """
    Timeseries marker selector with spinbox and dataset dropdown.

    Features:
      - Left-click: add/move marker
      - Drag: move marker
      - Right-click: delete marker
      - Spinbox for "Removal radius [indices]"
      - Dropdown to choose preview dataset
      - Markers and spinbox persist across dataset switches

    Usage:
        selector = TimeSeriesSelector(dataset_names, default_dataset="optional")
        indices, radius, dataset = selector.run()
    """

    def __init__(self, filename, dataset_names, default_dataset=None,
                 figsize=(8, 4), px_tol=6, title="Timeseries Marker Selector"):
        if not dataset_names:
            raise ValueError("dataset_names must be a non-empty list of strings.")

        self.filename = filename
        self.dataset_names = dataset_names
        if default_dataset is None:
            self.current_dataset = dataset_names[0]
        else:
            if default_dataset not in dataset_names:
                raise ValueError("default_dataset must be in dataset_names.")
            self.current_dataset = default_dataset

        self.figsize = figsize
        self.px_tol = px_tol
        self.title = title

        # internal state
        self._xs = []             # marker x positions (floats)
        self._selected_idx = None # marker being dragged
        self._result = None
        self._root = None

        # spinbox variable
        self._radius_var = None
        # dropdown variable
        self._dataset_var = None

        # matplotlib objects
        self._fig = None
        self._ax = None
        self._canvas = None
        self._artists = []
        self._line = None
        self._n = 0   # length of current timeseries

    # -------------------------
    # Placeholder loader
    # -------------------------
    def load_timeseries(self, dataset_name):
        
        timeseries, _ = get_data_from_dataset(self.filename, dataset_name)

        if timeseries.ndim == 3:
            timeseries = np.nanmean(timeseries, axis=(1,2))

        self.length = timeseries.shape[0]

        return timeseries

    # -------------------------
    # Figure / drawing helpers
    # -------------------------
    def _build_figure(self):
        self._fig = Figure(figsize=self.figsize)
        self._ax = self._fig.add_subplot(111)
        self._plot_timeseries(self.current_dataset, initial=True)

    def _plot_timeseries(self, dataset_name, initial=False):
        ts = np.asarray(self.load_timeseries(dataset_name)).ravel()
        self._n = len(ts)

        if self._line is None:
            x = np.arange(self._n)
            self._line, = self._ax.plot(x, ts, linewidth=1)
        else:
            x = np.arange(self._n)
            self._line.set_data(x, ts)
            self._ax.set_xlim(0, max(1, self._n - 1))
            self._ax.relim()
            self._ax.autoscale_view()

        self._draw_lines(initial=initial)
        if not initial and self._canvas:
            self._canvas.draw_idle()

    def _draw_lines(self, initial=False):
        # Remove old artists
        for art in self._artists:
            try:
                art.remove()
            except Exception:
                pass
        self._artists = []

        try:
            radius = int(self._radius_var.get()) if self._radius_var else 0
        except Exception:
            radius = 0

        for x in self._xs:
            # Thin red vertical line
            vline = self._ax.axvline(x, ymin=0, ymax=1, linewidth=1.2, color='red', linestyle='-')
            self._artists.append(vline)

            # Semi-transparent rectangle spanning [x - radius, x + radius]
            left = max(0, x - radius)
            width = min(self._n - 1, x + radius) - left
            rect = Rectangle(
                (left, self._ax.get_ylim()[0]),
                width,
                self._ax.get_ylim()[1] - self._ax.get_ylim()[0],
                color='red',
                alpha=0.2,
                zorder=0
            )
            self._ax.add_patch(rect)
            self._artists.append(rect)

        if not initial and self._canvas:
            self._canvas.draw_idle()

    def _data_x_to_canvas_px(self, x):
        trans = self._ax.transData
        xp, _ = trans.transform((x, 0))
        return xp

    def _nearest_marker_index(self, event):
        if not self._xs or event.x is None:
            return None
        dists = [abs(self._data_x_to_canvas_px(xi) - event.x) for xi in self._xs]
        min_idx = int(np.argmin(dists))
        if dists[min_idx] <= self.px_tol:
            return min_idx
        return None

    # -------------------------
    # Matplotlib event handlers
    # -------------------------
    def _on_button_press(self, event):
        if event.inaxes != self._ax:
            return
        if event.button == 1:
            idx = self._nearest_marker_index(event)
            if idx is not None:
                self._selected_idx = idx
                return
            if event.xdata is None:
                return
            x = min(max(0.0, event.xdata), self._n - 1)
            self._xs.append(x)
            self._selected_idx = len(self._xs) - 1
            self._draw_lines()
        elif event.button == 3:
            idx = self._nearest_marker_index(event)
            if idx is not None:
                self._xs.pop(idx)
                self._selected_idx = None
                self._draw_lines()

    def _on_motion(self, event):
        if self._selected_idx is None or event.inaxes != self._ax or event.xdata is None:
            return
        x = min(max(0.0, event.xdata), self._n - 1)
        self._xs[self._selected_idx] = x
        self._draw_lines()

    def _on_button_release(self, event):
        self._selected_idx = None

    def _on_key_press(self, event):
        if event.key == 'escape':
            self._result = ([], 0, self.length)
            self._close_gui()

    # -------------------------
    # GUI building / control
    # -------------------------
    def _build_gui(self):
        
        self._root = tk.Tk()
        self._root.title(self.title)

        # spinbox variable
        self._radius_var = tk.StringVar(value="26")
        # dropdown variable
        self._dataset_var = tk.StringVar(value=self.current_dataset)

        # Plot frame
        plot_frame = ttk.Frame(self._root)
        plot_frame.pack(fill=tk.BOTH, expand=True)

        # Canvas
        self._canvas = FigureCanvasTkAgg(self._fig, master=plot_frame)
        widget = self._canvas.get_tk_widget()
        widget.pack(fill=tk.BOTH, expand=True)

        # Connect matplotlib events
        self._fig.canvas.mpl_connect("button_press_event", self._on_button_press)
        self._fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self._fig.canvas.mpl_connect("button_release_event", self._on_button_release)
        self._fig.canvas.mpl_connect("key_press_event", self._on_key_press)

        # Bottom controls
        bottom = ttk.Frame(self._root)
        bottom.pack(fill=tk.X, padx=6, pady=6)

        info = ttk.Label(bottom, text="Left-click: place/move. Drag: move. Right-click: delete. Esc: cancel")
        info.pack(side=tk.TOP, anchor="w")

        # Spinbox
        spin_frame = ttk.Frame(bottom)
        spin_frame.pack(side=tk.LEFT, padx=5)

        ttk.Label(spin_frame, text="Removal radius [indices]:").pack(side=tk.LEFT)
        spin = ttk.Spinbox(spin_frame, from_=0, to=9999, textvariable=self._radius_var, width=6,
                   command=lambda: self._draw_lines())
        spin.pack(side=tk.LEFT)

        # Dropdown
        drop_frame = ttk.Frame(bottom)
        drop_frame.pack(side=tk.LEFT, padx=10)

        ttk.Label(drop_frame, text="Preview dataset:").pack(side=tk.LEFT)
        dropdown = ttk.OptionMenu(drop_frame, self._dataset_var, self.current_dataset, *self.dataset_names,
                                  command=self._on_dataset_change)
        dropdown.pack(side=tk.LEFT)

        # Continue button
        btn = ttk.Button(bottom, text="Continue", command=self._on_continue)
        btn.pack(side=tk.RIGHT)

        widget.focus_set()
        self._canvas.draw_idle()

    def _on_dataset_change(self, value):
        self.current_dataset = value
        self._plot_timeseries(value)

    def _on_continue(self):
        idxs = []
        for x in self._xs:
            if x is None:
                continue
            i = int(round(x))
            i = max(0, min(self._n - 1, i))
            idxs.append(i)
        idxs = sorted(set(idxs))

        try:
            radius = int(self._radius_var.get())
        except ValueError:
            radius = 0

        self._result = (idxs, radius, self.length)
        self._close_gui()

    def _close_gui(self):
        if self._root:
            try:
                self._root.quit()
                self._root.destroy()
            except Exception:
                pass

    # -------------------------
    # Public API
    # -------------------------
    def run(self):
        self._build_figure()
        self._build_gui()
        try:
            self._root.mainloop()
        except Exception:
            pass
        return ([], 0, self.length) if self._result is None else self._result


def main():
    if "--batch" in sys.argv:  # Batch mode

        filename = sys.argv[2]
        datasets = sys.argv[3].split(',')
        args = parse_key_value_args(sys.argv[4:])

        use_saved_indices = args["use_saved_indices"] # TODO: should probably be changed to a file string like how ROI_selection.py functions instead
        indices_file = os.path.join(filename, "../indices_removal.npy")

        if use_saved_indices:
            try:
                print(f"Using removal indices already saved in {indices_file}.")
                indices_to_remove = np.load(indices_file)
                indices = None
                radius = None
                length = None

            except:
                print("Error while trying to load the file. Manual selection is required.")
                indices_to_remove = None
                use_saved_indices = False
            
        if not use_saved_indices:

            indices_to_remove = None

            selector = TimeSeriesSelector(filename, datasets, default_dataset=args["preview_dataset"])
            print("Place markers with left click, drag to move, right click to remove. Click Continue when done.")
            indices, radius, length = selector.run()
            print("Returned indices:", indices)

        indices_to_remove = remove_time(filename, datasets, indices=indices, radius=radius, ts_len=length, indices_to_remove=indices_to_remove)
        np.save(indices_file, indices_to_remove)

    else:  # GUI mode
        if len(sys.argv) < 3:
            log("Usage: python script.py <HDF5 file> <dataset1> [<dataset2> ...]", verbosity=5)
        else:
            hdf5_file = str(sys.argv[1])
            datasets = [str(x) for x in sys.argv[2:]]

            selector = TimeSeriesSelector(hdf5_file, datasets, default_dataset=None)
            print("Place markers with left click, drag to move, right click to remove. Click Continue when done.")
            indices, radius, length = selector.run()
            print("Returned indices:", indices)

            indices_to_remove = remove_time(hdf5_file, datasets, indices=indices, radius=radius, ts_len=length)

            indices_file = os.path.join(hdf5_file, "../indices_removal.npy")
            np.save(indices_file, indices_to_remove)

if __name__ == "__main__":
    main()