// Define the output file (change path if needed)
outputPath = "<output_path>"; // This will be replaced by the actual file path for saving the mask

// Ensure an image is open
if (nImages == 0) {
    exit("No image is open.");
}

// Get image dimensions
width = getWidth();
height = getHeight();

// Ensure an ROI is selected
if (selectionType() == -1) {
    exit("No ROI selected.");
}

// Ensure the ROI is in the ROI Manager
roiManager("Add");

// Duplicate the image and create a mask from the ROI
maskID = getImageID();
run("Duplicate...", "title=ROI_Mask");
selectImage(maskID);
run("Create Mask");

// Save the mask as a TIFF file
saveAs("Tiff", outputPath);

// Notify Python that processing is complete
print("ROI mask saved to: " + outputPath);

run("Quit");