import cv2
from tqdm import tqdm
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog
import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import cv2
from toolbox_jocha.hdf5 import get_data_from_dataset
import matplotlib.pyplot as plt
import sys
import os
import ast
import tifffile

def read_data(filename):
    """Read data in tiff or npy files
    """
        
    if filename[-4:] == ".tif":
        return tifffile.imread(filename)
    
    elif filename[-4:] == ".npy":
        return np.load(filename)
    
    raise ValueError("Could not recognize file extension.")

# Change according to file location
def return_filepaths(mouse_num):

    root = f"D:/mouse_data/new_data/M{mouse_num}"

    mouse_num_no_age = mouse_num.split("-")[0]

    video_input = os.path.join(root, f"raw_data/RS_M{mouse_num_no_age}_video.mp4")
    video_output = video_input[:-4] + "_trimmed.mp4"

    face_motion_output = os.path.join(root, f"pupillo_face/M{mouse_num}_face_motion.npy")

    HbT_input = os.path.join(root, f"raw_data/dHbT.tif")

    datafile_input = f"D:/mouse_data/new_data/M{mouse_num}/data.txt"

    return video_input, video_output, face_motion_output, HbT_input, datafile_input

def return_video_path(mouse_num):
    input_path = f"D:/mouse_data/new_data/M{mouse_num}/raw_data/RS_M{mouse_num}_video.mp4"
    
    output_path = input_path[:-4] + "_trimmed.mp4"

    return input_path, output_path

def return_datafile_path(mouse_num):

    return f"D:/mouse_data/new_data/M{mouse_num}/data.txt"

def return_face_motion_path(mouse_num):
    input_path = 1

    output_path = 2

    return input_path, output_path

def smart_cast(value: str):
    """
    Recursively casts a string value according to these rules:
      1) Pure digits -> int
      2) Digits with decimal point -> float
      3) Parentheses "(...)" -> tuple with elements casted recursively
      4) Brackets "[...]" -> list with elements casted recursively
      5) Otherwise -> original string
    """
    value = value.strip()

    # 1) Check for int
    if value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
        return int(value)

    # 2) Check for float
    try:
        # float conversion but exclude cases like "1.2.3"
        if '.' in value and all(part.strip('-').isdigit() for part in value.split('.') if part != ''):
            return float(value)
    except ValueError:
        pass

    # 3) Check for tuple
    if value.startswith('(') and value.endswith(')'):
        inner = value[1:-1].strip()
        if inner == '':
            return tuple()
        parts = split_args(inner)
        return tuple(smart_cast(part) for part in parts)

    # 4) Check for list
    if value.startswith('[') and value.endswith(']'):
        inner = value[1:-1].strip()
        if inner == '':
            return []
        parts = split_args(inner)
        return [smart_cast(part) for part in parts]

    # 5) Return as string
    return value

def split_args(s: str):
    """
    Split a comma-separated string into parts, respecting nested () or [].
    Example: "1, (2,3), [4,5]" -> ['1', '(2,3)', '[4,5]']
    """
    parts = []
    buf = []
    depth_paren = 0
    depth_brack = 0

    for ch in s:
        if ch == ',' and depth_paren == 0 and depth_brack == 0:
            part = ''.join(buf).strip()
            if part:
                parts.append(part)
            buf = []
        else:
            if ch == '(':
                depth_paren += 1
            elif ch == ')':
                depth_paren -= 1
            elif ch == '[':
                depth_brack += 1
            elif ch == ']':
                depth_brack -= 1
            buf.append(ch)

    # last part
    last = ''.join(buf).strip()
    if last:
        parts.append(last)

    return parts

def get_arg_from_file(filepath: str, key: str):
    """
    Opens a .txt file in 'key:arg' format, searches for the given key,
    and returns the associated argument if found, or None otherwise.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                # Strip whitespace and ignore empty lines
                line = line.strip()
                if not line or ':' not in line:
                    continue
                k, v = line.split(':', 1)  # split only on the first ':'
                if k.strip() == key:
                    return smart_cast(v.strip())
        return None
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")

def select_frame_range(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Failed to open video.")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame_idx = [0]
    start_idx = [None]
    end_idx = [None]

    def get_frame(index):
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = cap.read()
        if not ret:
            return None
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame)

    root = tk.Tk()
    root.title("Frame Range Selector")

    img_label = tk.Label(root)
    img_label.pack()

    tk_img = [None]

    def update_frame():
        frame_img = get_frame(current_frame_idx[0])
        if frame_img:
            tk_img[0] = ImageTk.PhotoImage(frame_img)
            img_label.configure(image=tk_img[0])
        scrollbar.set(current_frame_idx[0])

    def on_left(event=None):
        if current_frame_idx[0] > 0:
            current_frame_idx[0] -= 1
            update_frame()

    def on_right(event=None):
        if current_frame_idx[0] < total_frames - 1:
            current_frame_idx[0] += 1
            update_frame()

    def on_scroll(val):
        idx = int(val)
        current_frame_idx[0] = idx
        update_frame()

    def mark_start():
        start_idx[0] = current_frame_idx[0]
        start_entry_var.set(str(start_idx[0]))

    def mark_end():
        end_idx[0] = current_frame_idx[0]
        end_entry_var.set(str(end_idx[0]))

    def on_finish():
        try:
            start_idx[0] = int(start_entry_var.get())
            end_idx[0] = int(end_entry_var.get())
        except ValueError:
            print("Start or end frame is not a valid number.")
            return

        if start_idx[0] is None or end_idx[0] is None:
            print("Please select both start and end frames.")
            return

        if not (0 <= start_idx[0] < total_frames) or not (0 <= end_idx[0] < total_frames):
            print("Start or end frame is out of bounds.")
            return

        root.selected_range = (start_idx[0], end_idx[0])
        root.quit()

    root.bind('<Left>', on_left)
    root.bind('<Right>', on_right)

    # Scrollbar
    scrollbar = tk.Scale(root, from_=0, to=total_frames-1, orient=tk.HORIZONTAL,
                         length=500, command=on_scroll)
    scrollbar.set(current_frame_idx[0])
    scrollbar.pack()

    # Start/End mark + display
    control_frame = tk.Frame(root)
    control_frame.pack()

    tk.Button(control_frame, text="Start", command=mark_start).grid(row=0, column=0, padx=5)
    start_entry_var = tk.StringVar()
    tk.Entry(control_frame, textvariable=start_entry_var, width=10).grid(row=0, column=1)

    tk.Button(control_frame, text="End", command=mark_end).grid(row=0, column=2, padx=5)
    end_entry_var = tk.StringVar()
    tk.Entry(control_frame, textvariable=end_entry_var, width=10).grid(row=0, column=3)

    def update_from_start_entry(*args):
        try:
            idx = int(start_entry_var.get())
            if 0 <= idx < total_frames:
                current_frame_idx[0] = idx
                update_frame()
        except ValueError:
            pass

    def update_from_end_entry(*args):
        try:
            idx = int(end_entry_var.get())
            if 0 <= idx < total_frames:
                current_frame_idx[0] = idx
                update_frame()
        except ValueError:
            pass

    start_entry_var.trace_add("write", update_from_start_entry)
    end_entry_var.trace_add("write", update_from_end_entry)


    # Finish button
    tk.Button(root, text="Finished", command=on_finish).pack(pady=10)

    update_frame()
    root.mainloop()

    selected_range = getattr(root, 'selected_range', None)
    root.destroy()
    cap.release()
    return selected_range

def get_arg_from_file(filepath: str, key: str):
    """
    Opens a .txt file in 'key:arg' format, searches for the given key,
    and returns the associated argument if found, or None otherwise.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                # Strip whitespace and ignore empty lines
                line = line.strip()
                if not line or ':' not in line:
                    continue
                k, v = line.split(':', 1)  # split only on the first ':'
                if k.strip() == key:
                    return smart_cast(v.strip())
        return None
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")

def space_video(input_path, output_path, indices):
    # Open the input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {input_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    codec = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'avc1' or 'H264' if needed

    # Prepare video writer
    out = cv2.VideoWriter(output_path, codec, fps, (width, height))

    # Ensure indices are sorted and within bounds
    indices_set = set(i for i in indices if 0 <= i < total_frames)
    current_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if current_index in indices_set:
            out.write(frame)

        current_index += 1
        if current_index > max(indices_set):
            break

    cap.release()
    out.release()

def evenly_spread_vector_elements(vec, M):
    N = len(vec)
    if M > N:
        raise ValueError("M cannot be greater than the number of elements in the vector")

    # Compute M indices spaced between [0, N), non-inclusive of the final element unless it fits
    indices = np.linspace(0, N, M, endpoint=False, dtype=int)
    return vec[indices]

def compute_correlation_with_lag(signal1, signal2, indices, lag_range, abs_r=False):
        """
        Computes the correlation between signal1 and signal2[a:b] for every lag value in lag_range.

        Parameters:
            signal1 (1d array): The first signal.
            signal2 (1d array): The second signal.
            indices (list or 1d array): A tuple (a, b) specifying the list of indices to consider for signal2.
            lag_range (tuple): A tuple (x, y) specifying the range of lag values to consider.

        Returns:
            correlations (list): An array of correlation values as a function of lag.
        """

        x, y = lag_range
        correlations = []

        # TODO: change to use scipy.signal.correlate and correlation_lags. It will be much faster I think.
        for lag in range(x, y + 1):
            rolled_signal2 = np.roll(signal2, lag)
            correlation = np.corrcoef(signal1, rolled_signal2[indices])[0, 1]
            if abs_r:
                correlation = np.abs(correlation)
            correlations.append(correlation)

        return np.array(correlations)

def slice_video_selected_frames(input_path, output_path, frame_indices):
    """
    Saves a new video with only the specified frame indices from the input video.

    Parameters:
    - input_path (str): Path to the input video file.
    - output_path (str): Path where the output video will be saved.
    - frame_indices (list of int): List of frame indices to keep.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file '{input_path}' does not exist.")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {input_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Output codec for .mp4

    # Remove out-of-range indices and sort
    valid_indices = sorted(set(i for i in frame_indices if 0 <= i < total_frames))
    if not valid_indices:
        raise ValueError("No valid frame indices to process.")

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for i in valid_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            out.write(frame)
        else:
            print(f"Warning: Failed to read frame {i}")

    cap.release()
    out.release()
    print(f"Saved {len(valid_indices)} frames to: {output_path}")