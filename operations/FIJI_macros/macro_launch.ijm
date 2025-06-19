open("<stack_path>");

wait(500); // Optional: Ensure time for images to be displayed

// Open the Brightness/Contrast dialog
run("Brightness/Contrast...");

// Wait for the user to select the ROI and adjust settings
waitForUser("Adjust the brightness/contrast as needed, select the ROI, then click OK.");

// Run the macro from the given path
runMacro("<macro_path>");

print("Macro finished successfully!");