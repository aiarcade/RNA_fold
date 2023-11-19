import os

directory_path = "../testdata"

# Get a list of file names in the directory


# Process the files in ascending order
for file_name in sorted_files:
    file_path = os.path.join(directory_path, str(file_name)+".npz")
    
    # Your processing code here
    # For example, you can print the file names
    print(file_path)
