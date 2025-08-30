#!/usr/bin/env python3
"""Regenerate cross_sections.xml from existing .h5 files."""

import os
import glob
import xml.etree.ElementTree as ET
from xml.dom import minidom


def main():
    data_dir = "openmc-data"
    
    # Find all .h5 files
    h5_files = glob.glob(os.path.join(data_dir, "*.h5"))
    h5_files.sort()
    
    # Create XML structure
    root = ET.Element("cross_sections")
    
    for h5_file in h5_files:
        basename = os.path.basename(h5_file)
        # Extract nuclide name from filename (e.g., ENDFB-8.0-NNDC_H1.h5 -> H1)
        if "_" in basename:
            nuclide = basename.split("_")[1].replace(".h5", "")
            library = ET.SubElement(root, "library")
            library.set("materials", nuclide)
            library.set("path", basename)
            library.set("type", "neutron")
    
    # Pretty print XML
    rough_string = ET.tostring(root, encoding='unicode')
    reparsed = minidom.parseString(rough_string)
    pretty_xml = reparsed.toprettyxml(indent="  ")
    
    # Write to file
    output_path = os.path.join(data_dir, "cross_sections.xml")
    with open(output_path, "w") as f:
        f.write(pretty_xml)
    
    print(f"Regenerated {output_path} with {len(h5_files)} nuclide files")
    
    # List what we found
    print("\nIncluded nuclides:")
    for h5_file in h5_files:
        basename = os.path.basename(h5_file)
        if "_" in basename:
            nuclide = basename.split("_")[1].replace(".h5", "")
            print(f"  - {nuclide}")


if __name__ == "__main__":
    main()
