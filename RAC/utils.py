import pandas as pd

# import models.rac_mlp_concat as rac_mlp_concat
import os
import re
import pandas as pd
import os

def find_subfolder_with_prefix(folder_path, prefix):
    """
    Find a subfolder inside the given folder that starts with the specified prefix.
    
    Args:
        folder_path (str): Path to the parent folder.
        prefix (str): Prefix string to match subfolder names.
    
    Returns:
        str: Path to the first subfolder that matches the prefix or None if no match is found.
    """
    try:
        subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
        for subfolder in subfolders:
            if subfolder.startswith(prefix):
                return os.path.join(folder_path, subfolder)
        return None
    except Exception as e:
        return f"Error: {e}"


def get_last_epoch_data(parent_folder):
    """
    From the given parent_folder, find all subfolders starting with `Epoch_`,
    determine the last epoch folder, and return the paths to the CSV files.
    """
    # List all entries in the parent folder that start with "Epoch_"
    epoch_folders = [d for d in os.listdir(parent_folder) if d.startswith("Epoch_")]
    
    # Extract the numeric portion of the folder name using a regular expression
    epoch_numbers = []
    for folder_name in epoch_folders:
        match = re.match(r"Epoch_(\d+)", folder_name)
        if match:
            epoch_numbers.append(int(match.group(1)))
    
    if not epoch_numbers:
        raise ValueError(f"No Epoch_ subfolders found in {parent_folder}")

    # Identify the maximum (last) epoch number
    last_epoch_num = max(epoch_numbers)
    
    # Construct the path to the last epoch folder
    last_epoch_folder = os.path.join(parent_folder, f"Epoch_{last_epoch_num}/")
    return last_epoch_folder