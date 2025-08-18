import sys
import subprocess
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BATCH_INSTRUCTIONS_DIR = os.path.join(BASE_DIR, "batch_processing_instructions")
OPERATIONS_DIR = os.path.join(BASE_DIR, "../operations")

def parse_command_line(line, mouse_num):
    # Ignore lines starting with '#'
    line = line.strip()
    if not line:
        return None
    if line.startswith('#'):
        print(f"Comment: {line}")
        return None  # Skip comments
    
    # Replace {mouse_num} with the provided mouse_num
    line = line.replace("{mouse_num}", str(mouse_num))

    parts = line.split()
    if len(parts) < 3:
        raise ValueError(f"Invalid command line format: {line}")
    
    filename = parts[0]
    operation = parts[1]
    datasets = parts[2]
    params = parts[3:]
    return filename, operation, datasets, params

def run_operations(input_file, mouse_num):
    with open(input_file, 'r') as f:
        for line in f:
            try:
                # Parse the command line, including replacing mouse_num
                result = parse_command_line(line, mouse_num)
                if result is None:
                    continue  # Skip comments and empty lines

                filename, operation, datasets, params = result
            except ValueError as e:
                print(e)
                continue
            
            script_path = os.path.join(OPERATIONS_DIR, f"{operation}.py")  # Assuming each operation has a corresponding script named <operation>.py
            
            command = [sys.executable, script_path, "--batch", filename] + [datasets] + params
            
            print(f"Executing: {' '.join(command)}")
            result = subprocess.run(command, capture_output=False, text=True, cwd=PROJECT_ROOT)
            
            if result.returncode != 0:
                print(f"Error in {operation}: {result.stderr}")
            else:
                print(f"{operation} completed successfully.")
                print(result.stdout)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python batch_processing.py <input_file> <mouse_num>")
        sys.exit(1)
    
    input_file = os.path.join(BATCH_INSTRUCTIONS_DIR, sys.argv[1])
    mice_num = str(sys.argv[2]).split(',')  # Convert mice_num to a list of strings

    for mouse_num in mice_num:
        run_operations(input_file, mouse_num)
