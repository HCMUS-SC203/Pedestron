import os
import shutil
import sys

def move_and_delete_subfolders(folder_path):
    for subfolder_name in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder_name)

        if os.path.isdir(subfolder_path):
            for file_name in os.listdir(subfolder_path):
                file_path = os.path.join(subfolder_path, file_name)
                shutil.move(file_path, os.path.join(folder_path, file_name))

            os.rmdir(subfolder_path)

if __name__ == "__main__":
    # Check if the correct number of command-line arguments is provided
    if len(sys.argv) != 2:
        print("Usage: python script.py <folder_path>")
        sys.exit(1)

    # Get the folder path from command-line argument
    folder_path = sys.argv[1]

    # Call the function to move pictures and delete subfolders
    move_and_delete_subfolders(folder_path)