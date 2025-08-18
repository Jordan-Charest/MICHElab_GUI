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
            elif type_spec == 'bool':
                value = bool(value)
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

def smart_cast(value: str):
    """
    Recursively casts a string value according to these rules:
      1) Pure digits -> int
      2) Digits with decimal point -> float
      3) Parentheses "(...)" -> tuple with elements casted recursively
      4) Brackets "[...]" -> list with elements casted recursively
      5) True/False -> bool
      6) "None" -> None
      7) Otherwise -> original string
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
    if value.lower() == "true":
        return True
    elif value.lower() == "false":
        return False
    
    # 6) Check for None
    if value.lower() == "none":
        return None

    # 7) Return as string
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