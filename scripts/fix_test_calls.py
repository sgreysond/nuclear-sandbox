#!/usr/bin/env python3
"""Fix all build_fusor_cylindrical_model calls to have height_cm first."""

import re

# Read the test file
with open("test_openmc_hybrid.py", "r") as f:
    content = f.read()

# Pattern to find build_fusor_cylindrical_model calls
pattern = r'build_fusor_cylindrical_model\((.*?)\)'

def fix_call(match):
    """Fix a single function call to have height_cm first."""
    call_content = match.group(1)
    
    # Find height_cm parameter
    height_match = re.search(r'height_cm=(\d+\.?\d*)', call_content)
    if not height_match:
        return match.group(0)  # No height_cm, leave as is
    
    height_value = height_match.group(1)
    
    # Remove height_cm from wherever it is
    call_content = re.sub(r',?\s*height_cm=\d+\.?\d*,?\s*', '', call_content)
    
    # Clean up any trailing commas or spaces
    call_content = call_content.strip().rstrip(',')
    
    # Add height_cm at the beginning
    new_call = f'build_fusor_cylindrical_model(\n            height_cm={height_value},\n            {call_content.strip()}\n        )'
    
    return new_call

# Fix all calls (need to handle multiline)
content = re.sub(
    r'build_fusor_cylindrical_model\([^)]*\)',
    fix_call,
    content,
    flags=re.DOTALL
)

# Write back
with open("test_openmc_hybrid.py", "w") as f:
    f.write(content)

print("Fixed all build_fusor_cylindrical_model calls")
