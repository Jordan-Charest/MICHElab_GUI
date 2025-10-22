import numpy as np
import os
import tifffile
import pickle
from scipy.ndimage import zoom, rotate

#### MODIFY FILEPATHS BELOW AS NEEDED

def return_raw_data_root(mouse_num): # Modify as needed according to your file structure
    return f"D:/mouse_data/new_data/M{mouse_num}/raw_data"

def return_filename(mouse_num, output_file_id): # Modify as needed according to your file structure
    # return f"D:/mouse_data/new_data/M{mouse_num}/formatted/M{mouse_num}_{output_file_id}.h5"
    return f"D:/mouse_data/new_data/M{mouse_num}/formatted/M{mouse_num}_{output_file_id}.h5"

def return_avg_data(mouse_num):

    input_path = 1

    data = read_data(input_path)
    avg_data = np.nanmean(data, axis=0)

    avg_data_str = "green_avg"

    return input_path, avg_data, avg_data_str

def return_datafile_contents(mouse_num):

    filename = f"D:/mouse_data/new_data/M{mouse_num}/data.txt"

    attr_dict = parse_txt_to_dict(filename)

    return attr_dict


def parse_txt_to_dict(filepath: str):
    """
    Parse a txt file with lines like 'key:arg' into a dictionary.
    Each value is casted with smart_cast().
    Lines without ':' or empty lines are ignored.
    """
    result = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line:
                continue
            k, v = line.split(':', 1)  # split only on first colon
            key = k.strip()
            arg = v.strip()
            result[key] = smart_cast(arg)
    return result


def get_transformed_crop(atlas, params):    
    
    x = params['x']
    y = params['y']
    h = params['h']
    w = params['w']
    pad = params['pad']
    scale = params['scale']
    angle = params['angle']
    
    canvas_h, canvas_w = h + 2 * pad, w + 2 * pad

    scaled = zoom(atlas, scale, order=1)
    rotated = rotate(scaled, angle, reshape=False, order=1, mode='constant', cval=0.0)

    H, W = rotated.shape
    overlay = np.zeros((canvas_h, canvas_w), dtype=rotated.dtype)

    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(canvas_w, x + W)
    y1 = min(canvas_h, y + H)

    rx0 = max(0, -x)
    ry0 = max(0, -y)
    rx1 = rx0 + (x1 - x0)
    ry1 = ry0 + (y1 - y0)

    if x1 > x0 and y1 > y0:
        overlay[y0:y1, x0:x1] = rotated[ry0:ry1, rx0:rx1]

    atlas_registered = overlay[pad:pad + h, pad:pad + w]

    return atlas_registered
    

#########################################################

def read_data(filename):
    """Read data in tiff or npy files
    """
        
    if filename[-4:] == ".tif":
        return tifffile.imread(filename)
    
    elif filename[-4:] == ".npy":
        return np.load(filename)
    
    elif filename[-4:] == ".pkl":

        with open(filename, 'rb') as file:
            data = pickle.load(file)

        return data

    raise ValueError("Could not recognize file extension.")

def split_dataset_path(dataset_path):
    """
    Splits a full dataset path into the group path and the dataset name.

    Parameters:
        dataset_path (str): Full path to the dataset (e.g., "group1/group2/group3/dataset").

    Returns:
        group_path (str): Path to the group (e.g., "group1/group2/group3").
        dataset_name (str): Name of the dataset (e.g., "dataset").
    """
    # Split the dataset path
    group_path, dataset_name = os.path.split(dataset_path)
    
    return group_path, dataset_name

def smart_cast(value: str):
    """
    Recursively casts a string value according to these rules:
      1) Pure digits -> int
      2) Digits with decimal point -> float
      3) Parentheses "(...)" -> tuple with elements casted recursively
      4) Brackets "[...]" -> list with elements casted recursively
      5) True/False -> bool
      5) Otherwise -> original string
    """
    if not isinstance(value, str):
        return value
    
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
    
    # 5) Check for bool
    if value in ("True", "true"):
        return True
    elif value in ("False", "false"):
        return False

    # 6) Return as string
    return value

def split_args(s: str):
    """
    Split a comma-separated string into parts, but ignore commas inside
    nested parentheses () or brackets [].
    Example: "1, (2, 3), [4,5]" -> ['1', '(2, 3)', '[4,5]']
    """
    parts = []
    buf = []
    depth_paren = 0
    depth_brack = 0

    for ch in s:
        if ch == ',' and depth_paren == 0 and depth_brack == 0:
            # Reached a top-level comma: split here
            part = ''.join(buf).strip()
            if part:
                parts.append(part)
            buf = []
        else:
            # Update depth counters
            if ch == '(':
                depth_paren += 1
            elif ch == ')':
                depth_paren -= 1
            elif ch == '[':
                depth_brack += 1
            elif ch == ']':
                depth_brack -= 1
            buf.append(ch)

    # Add the last buffered part
    last = ''.join(buf).strip()
    if last:
        parts.append(last)

    return parts

def read_regions_file(filepath):

    region_list = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if ":" not in line:
                raise ValueError(f"Invalid line format: {line}")
            region, name = line.split(":", 1)
            region_list.append((region, name.strip()))
    return region_list

