import sys
import os

def parse_key_value_args(argv):

    # Create a dictionary to store the parameters and their values
    param_dict = {}

    # Iterate over the arguments
    for arg in argv[0:]:
        # Check if the argument contains an '=' sign, indicating it's a key-value pair
        if '=' in arg:
            key, value = arg.split('=', 1)
            
            # Check if the value contains a type specification (e.g., '_int', '_float', etc.)
            if ';' in value:
                value, type_spec = value.rsplit(';', 1)
            else:
                type_spec = None

            if value == 'None':
                value = None
            # Convert the value to its appropriate type based on type_spec
            elif type_spec == 'int':
                value = int(value)
            elif type_spec == 'float':
                value = float(value)
            elif type_spec == 'str' or type_spec is None:
                value = str(value)
            else:
                print(f"Warning: Unsupported type '{type_spec}' for parameter '{key}'. Defaulting to string.")
                value = str(value)

            # Store the value in the dictionary
            param_dict[key] = value
        else:
            print(f"Warning: Argument '{arg}' is not in 'key=value' format and will be ignored.")
    
    return param_dict