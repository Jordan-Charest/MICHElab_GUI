import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from scipy.ndimage import rotate, zoom
import tifffile as tiff
import os
import sys
import pickle

### ADAPTED FROM CODE BY ALEXANDRE CLÉROUX CUILLERIER

def load_first_frame(tiff_path):
    with tiff.TiffFile(tiff_path) as tif:
        first_frame = tif.pages[0].asarray()
    return first_frame.astype(np.float32)


def interactive_register_atlas(atlas, data, previous_params=None):
    h, w = data.shape
    pad = 100
    canvas_h, canvas_w = h + 2 * pad, w + 2 * pad

    last_cropped = np.zeros_like(data)

    # Pad data into a canvas
    data_canvas = np.zeros((canvas_h, canvas_w), dtype=data.dtype)
    data_canvas[pad:pad+h, pad:pad+w] = data

    defaults = {
    'x': (canvas_w - atlas.shape[1]) // 2,
    'y': -175, # Previous value used was (canvas_h - atlas.shape[0]) // 2
    'angle': 0,
    'alpha': 0.3,   
    'scale': 1.0,
    'contrast': 6.0
}

    # Initial parameters
    if previous_params is not None:
        init_x = previous_params['x']
        init_y = previous_params['y']
        init_angle = previous_params['angle']
        init_alpha = previous_params['alpha']
        init_scale = previous_params['scale']
        init_contrast = previous_params['contrast']

    else:
        init_x = defaults['x']
        init_y = defaults['y']
        init_angle = defaults['angle']
        init_alpha = defaults['alpha']
        init_scale = defaults['scale']
        init_contrast = defaults['contrast']

    # === Setup figure ===
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.45)
    ax.set_xlim(0, canvas_w)
    ax.set_ylim(canvas_h, 0)

    # Initial display
    data_canvas_display = data_canvas
    img_data = ax.imshow(data_canvas_display, cmap='gray', zorder=1)

    # Initial atlas overlay
    rotated = rotate(atlas, init_angle, reshape=False, order=1, mode='constant', cval=0.0)
    cropped = np.zeros_like(data_canvas)
    y0 = max(0, init_y)
    x0 = max(0, init_x)
    y1 = min(init_y + rotated.shape[0], canvas_h)
    x1 = min(init_x + rotated.shape[1], canvas_w)
    ry0 = y0 - init_y
    rx0 = x0 - init_x
    ry1 = ry0 + (y1 - y0)
    rx1 = rx0 + (x1 - x0)
    cropped[y0:y1, x0:x1] = rotated[ry0:ry1, rx0:rx1]

    img_overlay = ax.imshow(cropped, cmap='hot', alpha=init_alpha, zorder=2)

    # === Sliders ===
    ax_x = plt.axes([0.15, 0.35, 0.7, 0.03])
    ax_y = plt.axes([0.15, 0.30, 0.7, 0.03])
    ax_angle = plt.axes([0.15, 0.25, 0.7, 0.03])
    ax_alpha = plt.axes([0.15, 0.20, 0.7, 0.03])
    ax_scale = plt.axes([0.15, 0.15, 0.7, 0.03])
    ax_contrast = plt.axes([0.15, 0.10, 0.7, 0.03])

    slider_x = Slider(ax_x, 'X Offset', -canvas_w//2, canvas_w//2, valinit=init_x)
    slider_y = Slider(ax_y, 'Y Offset', -canvas_h, canvas_h, valinit=init_y)
    slider_angle = Slider(ax_angle, 'Rotation (°)', -45, 45, valinit=init_angle)
    slider_alpha = Slider(ax_alpha, 'Alpha', 0, 1, valinit=init_alpha)
    slider_scale = Slider(ax_scale, 'Scale', 0.90, 1.10, valinit=init_scale, valstep=0.01)
    slider_contrast = Slider(ax_contrast, 'Data Contrast', 0.1, 10.0, valinit=init_contrast, valstep=0.1)

    # === Reset button ===
    ax_reset = plt.axes([0.8, 0.025, 0.1, 0.04])
    button_reset = Button(ax_reset, 'Reset')

    def reset(event):
        slider_x.set_val(defaults['x'])
        slider_y.set_val(defaults['y'])
        slider_angle.set_val(defaults['angle'])
        slider_alpha.set_val(defaults['alpha'])
        slider_scale.set_val(defaults['scale'])
        slider_contrast.set_val(defaults['contrast'])

    button_reset.on_clicked(reset)

    def get_transformed_crop(x, y, angle, scale):
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

        return overlay

    def update(val=None):
        x = int(slider_x.val)
        y = int(slider_y.val)
        angle = slider_angle.val
        alpha = slider_alpha.val
        scale = slider_scale.val
        contrast = slider_contrast.val

        # Update data image brightness
        img_data.set_data(data_canvas * contrast)

        # Update atlas overlay
        cropped = get_transformed_crop(x, y, angle, scale)
        img_overlay.set_data(cropped)
        img_overlay.set_alpha(alpha)

        fig.canvas.draw_idle()

        nonlocal last_cropped
        last_cropped = cropped[pad:pad + h, pad:pad + w].copy()

    # Connect sliders
    for s in [slider_x, slider_y, slider_angle, slider_alpha, slider_scale, slider_contrast]:
        s.on_changed(update)

    result = {}

    def on_key(event):
        if event.key == 'enter':
            x = int(slider_x.val)
            y = int(slider_y.val)
            angle = slider_angle.val
            alpha = slider_alpha.val
            scale = slider_scale.val
            contrast = slider_contrast.val

            result.update({
                'x': x,
                'y': y,
                'angle': angle,
                'alpha': alpha,
                'scale': scale,
                'contrast': contrast,
                'output_shape': (h, w)  # NEW LINE
            })

            # Final transformed atlas (cropped to data size)
            full_overlay = get_transformed_crop(x, y, angle, scale)
            atlas_registered = full_overlay[pad:pad + h, pad:pad + w].copy()
            result['final'] = atlas_registered
            result['params'] = {"x": x, "y": y, "angle": angle, "alpha": alpha, "scale": scale, "contrast": contrast, "pad": pad, "h": h, "w": w}

            plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', on_key)

    update()
    plt.show()

    return result


def apply_registration(result, source_array, pad=100, cval=0.0):
    output_shape = result['output_shape']
    h, w = output_shape
    canvas_h, canvas_w = h + 2 * pad, w + 2 * pad

    # Extract transform params
    x = result['x']
    y = result['y']
    angle = result['angle']
    scale = result['scale']

    # Scale and rotate the array
    scaled = zoom(source_array, scale, order=1)
    rotated = rotate(scaled, angle, reshape=False, order=1, mode='constant', cval=cval)

    H, W = rotated.shape
    canvas = np.full((canvas_h, canvas_w), cval, dtype=rotated.dtype)

    # Paste into canvas at the given offset
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(canvas_w, x + W)
    y1 = min(canvas_h, y + H)

    rx0 = max(0, -x)
    ry0 = max(0, -y)
    rx1 = rx0 + (x1 - x0)
    ry1 = ry0 + (y1 - y0)

    if x1 > x0 and y1 > y0:
        canvas[y0:y1, x0:x1] = rotated[ry0:ry1, rx0:rx1]

    # Crop to output shape
    cropped = canvas[pad:pad + h, pad:pad + w]
    return cropped


def main(folder_path_func, mice_num, overwrite=False, save_output=False):

    params = None

    for mouse_num in mice_num:

        folder_path = folder_path_func(mouse_num) # TO MODIFY: as appropriate depending on your file structure; path
        rawdata_path = os.path.join(folder_path, "./raw_data/rawdata_green.tif") # TO MODIFY: path to the raw data to display beneath the atlas mask in the GUI
        atlas_path = r"D:/mouse_data/new_data/atlas/outline_mask_coarse.npy" # TO MODIFY: path to the atlas to use
        atlas_output_path = os.path.join(folder_path, "atlas.npy") # TO MODIFY: output atlas
        params_output_path = os.path.join(folder_path, "atlas_params.pkl") # TO MODIFY: where the save the atlas transformation parameters

        if not overwrite and os.path.exists(atlas_output_path):
            print(f"Registration already exists at {atlas_output_path}. Skipping.")
            return

        data = load_first_frame(rawdata_path)
        print(f"Processing mouse {mouse_num}")
        atlas = np.load(atlas_path)
        result = interactive_register_atlas(atlas, data, previous_params=params)

        params = result['params']

        if save_output:
            registration = result["final"]
            np.save(atlas_output_path, registration)

            with open(params_output_path, "wb") as file:
                pickle.dump(params, file)


# UNUSED
def generate_atlas_variation(folder_path):
    rawdata_path = os.path.join(folder_path, "rawdata_green.tif")
    atlas_path = r"C:\Users\alexc\Documents\GitHub\WF-analysis\Allen-Atlas\outline_mask_coarse.npy"
    atlas_folder_path = os.path.join(folder_path, "atlas_variations")
    data = load_first_frame(rawdata_path)
    atlas = np.load(atlas_path)

    scale = [95, 97, 100, 103, 105]
    y_offset = np.arange(-50, 52, 2)
    for s in scale:
        print("Doing scale = {}%".format(s))
        result = interactive_register_atlas(atlas, data)
        y0 = result["y"]
        for dy in y_offset:
            result["y"] = y0 + dy
            registration = apply_registration(result, atlas)
            output_path = os.path.join(atlas_folder_path, r"atlas_{}_{}".format(s, dy))
            np.save(output_path, registration)

    return None


if __name__ == "__main__":

    mice_num = sys.argv[1].split(",")

    def return_folder_path(mouse_num): # TO MODIFY: path to your mouse data
        return f"D:/mouse_data/new_data/M{mouse_num}/"
    
    main(return_folder_path, mice_num, save_output=True, overwrite=True)

