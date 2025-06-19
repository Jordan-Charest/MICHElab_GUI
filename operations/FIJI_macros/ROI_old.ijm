// Ensure an image is open
if (nImages == 0) {
    exit("No image is open.");
}

// Get active image and its dimensions
img = getImageID();
width = getWidth();
height = getHeight();

// Ensure an ROI is selected (check selection type)
selectionType = getSelectionType();
if (selectionType == 0) {
    exit("No ROI selected.");
}

// Get the image's processor
processor = getProcessor();

// Create a binary mask from the ROI
mask = processor.createMask();

// Print image dimensions
print("Width:", width, "Height:", height);

// Print binary mask (flattened 1D array)
for (y = 0; y < height; y++) {
    for (x = 0; x < width; x++) {
        print(mask.get(x, y) > 0 ? 1 : 0);
    }
}
