import numpy as np
 
# Path to your .npz file
npz_file_path = "home/arptracker/rmunusamy/human_tracker_MSC/human_tracker/calibration_camera/extrinsic/stereo_extrinsics.npz"
 
# Load the .npz file
data = np.load(npz_file_path)
 
# Method 1: List all keys (array names)
print("=== ARRAYS IN THE NPZ FILE ===")
print(data.files)
 
# Method 2: Print content of each array
print("\n=== CONTENT OF EACH ARRAY ===")
for key in data.files:
    print(f"\n[{key}]")
    print(data[key])
 
# Close the file when done
data.close()