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

from funcs import return_filepaths, select_frame_range, get_arg_from_file, read_data, compute_correlation_with_lag, space_video, evenly_spread_vector_elements, slice_video_selected_frames

abs_r = True

max_lag_frames = 88 # About 5 seconds at 17.5 fps
lag_range = (-max_lag_frames, max_lag_frames)

skip_face_motion = False

# "308-16", "316-14", "316-16", "322-14", 
mice_num = ["308-10"]

for mouse_num in mice_num:

    print(f"Running mouse M{mouse_num}.")

    # COMPUTE FACE MOTION #
    ### FACE MOTION CODE MADE BY ANTOINE DAIGLE

    if not skip_face_motion:

        video_input, video_output, face_motion_output, cortical_signal_input, datafile_input = return_filepaths(mouse_num)
        print(f"Opening video {video_input}.")

        cap = cv2.VideoCapture(video_input)
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Check if the video opened successfully
        if not cap.isOpened():
            print("Error: Could not open video.")
            exit()

        ret, frame = cap.read()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        x_mot, y_mot, w_mot, h_mot = cv2.selectROI("Select motion zone", gray_frame)

        motion = np.zeros(shape=video_length)

        for i in tqdm(range(video_length)):
            # Read frame-by-frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()

            # Convert the frame to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            motion_zone = gray_frame[y_mot:y_mot+h_mot, x_mot:x_mot+w_mot]

            if i == 0:
                old_motion_zone = motion_zone.copy()

            else:
                motion[i] = cv2.mean(cv2.absdiff(motion_zone, old_motion_zone))[0]
                old_motion_zone = motion_zone.copy()

        # Release the video capture object and close the display window
        cap.release()
        cv2.destroyAllWindows()

        np.save(face_motion_output, motion)

        print(f"Face motion saved.")

    elif skip_face_motion:

        video_input, video_output, face_motion_output, cortical_signal_input, datafile_input = return_filepaths(mouse_num)
        face_motion_signal = np.load(face_motion_output)

    # TRIM VIDEO AND FACE MOTION TO APPROPRIATE FPS

    frame_range = get_arg_from_file(datafile_input, "monitoring_frame_range")

    if frame_range is not None:
        print(f"Frame range {frame_range} found in data.txt. Skipping frame selection GUI.")
    else:
        frame_range = select_frame_range(video_input)
        print(f"Frame range {frame_range} selected.")

        with open(datafile_input, 'a') as file:
            file.write(f"\nmonitoring_frame_range:{frame_range}")

        print("Monitoring frame range written to data.txt file.")


    # Load datasets

    # TODO: the only issue here is that it considers the whole HbT window, not just the ROI since it loads the raw file. Maybe to fix eventually but should be similar in any case.
    cortical_signal = read_data(cortical_signal_input)
    duration = cortical_signal.shape[0] # At 12 fps, should be 5760 (8 minutes). 1440 for 3 fps.
    print(f"HbT data with {duration} frames loaded. Trimming the video and face motion to match.")

    cortical_signal = np.nanmean(cortical_signal, axis=(1, 2))

    face_motion_signal = np.load(face_motion_output)

    # Find the indices that correctly spread the monitoring frames across the cortical (HbT/GCaMP) timeseries
    indices_vec = np.arange(frame_range[0], frame_range[1], 1)
    # print(indices_vec)
    indices = evenly_spread_vector_elements(indices_vec, duration)
    # print(indices)

    # Cross correlate face_motion with the cortical signal (HbT or GCaMP)
    correlations = compute_correlation_with_lag(cortical_signal, face_motion_signal, indices, lag_range, abs_r=abs_r)
    max_correl_index = np.argmax(correlations)
    max_correl_lag = np.arange(lag_range[0], lag_range[1]+1, 1)[max_correl_index]

    # Apply spacing indices and save the sliced face_motion, sliced+lagged face_motion, and sliced video
    lagged_signal= np.roll(face_motion_signal, max_correl_lag)
    sliced_lagged_signal = lagged_signal[indices]
    sliced_signal = face_motion_signal[indices]

    np.save(face_motion_output[:-4]+"_sliced.npy", sliced_signal)
    np.save(face_motion_output[:-4]+"_sliced_lagged.npy", sliced_lagged_signal)

    slice_video_selected_frames(video_input, video_output, indices)

    print(f"All data saved for M{mouse_num}.\n")

