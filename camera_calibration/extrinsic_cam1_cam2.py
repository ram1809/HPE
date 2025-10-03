import cv2
from pupil_apriltags import Detector
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from datetime import datetime

# === SETTINGS ===
# AprilTag size in meters
tag_size = 0.45  # Adjust to your actual tag size

# Paths to intrinsic calibration files
intrinsic_files = {
    "Camera0": r"C:\Windows\System32\Dissertation\Images\intrinsic\calibration_Cam0.npz",
    "Camera1": r"C:\Windows\System32\Dissertation\Images\intrinsic\calibration_Cam1.npz"
}

# Paths to extrinsic calibration image folders
camera_folders = {
    "Camera0": r"C:\Windows\System32\Dissertation\New_folder\captured_images\Camera0",
    "Camera1": r"C:\Windows\System32\Dissertation\New_folder\captured_images\Camera1"
}

# Output files
output_file = r"C:\Windows\System32\Dissertation\stereo_extrinsics.npz"
debug_dir = r"C:\Windows\System32\Dissertation\extrinsic_debug"
os.makedirs(debug_dir, exist_ok=True)

# Create log file with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(debug_dir, f"extrinsic_calibration_log_{timestamp}.txt")

# Setup logging
def log(message):
    print(message)
    with open(log_file, 'a') as f:
        f.write(message + '\n')

log(f"\n=== STEREO EXTRINSIC CALIBRATION - {timestamp} ===")

# Load intrinsic parameters
intrinsics = {}
for cam_name, calib_file in intrinsic_files.items():
    try:
        data = np.load(calib_file)
        # Handle different variable names based on checkerboard calibration
        camera_matrix = data['camera_matrix'] if 'camera_matrix' in data else data['mtx'] 
        dist_coeffs = data['dist_coeffs'] if 'dist_coeffs' in data else data['dist']
        image_size = data['image_size'] if 'image_size' in data else None
        intrinsics[cam_name] = {
            'camera_matrix': camera_matrix,
            'dist_coeffs': dist_coeffs,
            'image_size': image_size
        }
        log(f"Loaded intrinsic parameters for {cam_name}")
        log(f"Camera Matrix ({cam_name}):\n{camera_matrix}")
        log(f"Distortion Coeffs ({cam_name}):\n{dist_coeffs.ravel()}")
    except Exception as e:
        log(f"[ERROR] Could not load intrinsics for {cam_name}: {e}")
        log("Verify that intrinsic calibration has been completed.")
        exit(1)

# # Initialize AprilTag detector
detector = Detector(
    families="tagStandard41h12",
    nthreads=1,
    quad_decimate=1.0,    # Use full resolution
    quad_sigma=0.0,       # No blurring
    refine_edges=True,    # Better corner accuracy
    decode_sharpening=0.25
)

# Storage for stereo calibration
objpoints = []  # 3D object points
imgpoints_cam0 = []  # 2D points in Cam0
imgpoints_cam1 = []  # 2D points in Cam1

# Get image files
images_cam0 = sorted(glob.glob(os.path.join(camera_folders["Camera0"], "*.jpg")))
images_cam1 = sorted(glob.glob(os.path.join(camera_folders["Camera1"], "*.jpg")))

if not images_cam0:
    images_cam0 = sorted(glob.glob(os.path.join(camera_folders["Camera0"], "*.png")))
if not images_cam1:
    images_cam1 = sorted(glob.glob(os.path.join(camera_folders["Camera1"], "*.png")))

if len(images_cam0) != len(images_cam1):
    log(f"[ERROR] Image count mismatch: Camera0 ({len(images_cam0)}) vs Camera1 ({len(images_cam1)})")
    exit(1)

if not images_cam0 or not images_cam1:
    log(f"[ERROR] No images found in one or both camera folders")
    exit(1)

log(f"Found {len(images_cam0)} image pairs")

# Process each image pair
successful_matches = 0
for i, (fname_cam0, fname_cam1) in enumerate(zip(images_cam0, images_cam1)):
    base_cam0 = os.path.basename(fname_cam0)
    base_cam1 = os.path.basename(fname_cam1)
    
    log(f"\nProcessing pair {i+1}/{len(images_cam0)}: {base_cam0} & {base_cam1}")
    
    img_cam0 = cv2.imread(fname_cam0)
    img_cam1 = cv2.imread(fname_cam1)
    
    if img_cam0 is None or img_cam1 is None:
        log(f"[MISS] Could not load image pair: {base_cam0}")
        continue
    
    # Convert to grayscale for tag detection
    gray_cam0 = cv2.cvtColor(img_cam0, cv2.COLOR_BGR2GRAY)
    gray_cam1 = cv2.cvtColor(img_cam1, cv2.COLOR_BGR2GRAY)
    
    # Detect tags
    tags_cam0 = detector.detect(gray_cam0)
    tags_cam1 = detector.detect(gray_cam1)
    
    log(f"  Detected {len(tags_cam0)} tags in Cam0, {len(tags_cam1)} tags in Cam1")
    
    # Skip if no tags detected
    if not tags_cam0 or not tags_cam1:
        log(f"[MISS] No tags found in pair: {base_cam0}")
        continue
    
    # Find tags that are visible in both images
    tags_cam0_dict = {tag.tag_id: tag for tag in tags_cam0}
    tags_cam1_dict = {tag.tag_id: tag for tag in tags_cam1}
    common_ids = set(tags_cam0_dict.keys()) & set(tags_cam1_dict.keys())
    
    if not common_ids:
        log(f"[MISS] No matching tag IDs in: {base_cam0}")
        log(f"  Cam0 tag IDs: {list(tags_cam0_dict.keys())}")
        log(f"  Cam1 tag IDs: {list(tags_cam1_dict.keys())}")
        continue
    
    # Create visualizations
    vis_cam0 = img_cam0.copy()
    vis_cam1 = img_cam1.copy()
    
    # Process each common tag
    pair_matches = 0
    for tag_id in common_ids:
        tag_cam0 = tags_cam0_dict[tag_id]
        tag_cam1 = tags_cam1_dict[tag_id]
        
        # Get corners
        corners_cam0 = np.array(tag_cam0.corners, dtype=np.float32)
        corners_cam1 = np.array(tag_cam1.corners, dtype=np.float32)
        
        # Create 3D object points (Z=0 plane)
        objp = np.array([
            [0, 0, 0],
            [tag_size, 0, 0],
            [tag_size, tag_size, 0],
            [0, tag_size, 0]
        ], dtype=np.float32)
        
        # Store points
        objpoints.append(objp)
        imgpoints_cam0.append(corners_cam0)
        imgpoints_cam1.append(corners_cam1)
        pair_matches += 1
        successful_matches += 1
        
        # # Draw detected tags
        # for j in range(4):
        #     # Draw lines connecting corners
        #     pt1_cam0 = tuple(corners_cam0[j].astype(int))
        #     pt2_cam0 = tuple(corners_cam0[(j+1)%4].astype(int))
        #     cv2.line(vis_cam0, pt1_cam0, pt2_cam0, (0, 255, 0), 2)
            
        #     pt1_cam1 = tuple(corners_cam1[j].astype(int))
        #     pt2_cam1 = tuple(corners_cam1[(j+1)%4].astype(int))
        #     cv2.line(vis_cam1, pt1_cam1, pt2_cam1, (0, 255, 0), 2)
        
        # Draw tag IDs
        center_cam0 = tuple(np.mean(corners_cam0, axis=0).astype(int))
        center_cam1 = tuple(np.mean(corners_cam1, axis=0).astype(int))
        
        cv2.putText(vis_cam0, f"ID:{tag_id}", 
                   (corners_cam0[0][0].astype(int), corners_cam0[0][1].astype(int)-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(vis_cam1, f"ID:{tag_id}", 
                   (corners_cam1[0][0].astype(int), corners_cam1[0][1].astype(int)-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    log(f"[OK] Matched {pair_matches} tags in pair: {base_cam0}")
    

# Perform stereo calibration
if len(objpoints) > 0:
    log(f"\nPerforming stereo calibration with {successful_matches} tag matches...")
    
    # Get image size (from first image)
    h, w = gray_cam0.shape
    img_size = (w, h)
    log(f"Image size: {img_size}")
    
    # Extract camera matrices and distortion coefficients
    K0 = intrinsics["Camera0"]["camera_matrix"]
    D0 = intrinsics["Camera0"]["dist_coeffs"]
    K1 = intrinsics["Camera1"]["camera_matrix"]
    D1 = intrinsics["Camera1"]["dist_coeffs"]

    # Run stereo calibration - use intrinsics from checkerboard calibration
    flags = cv2.CALIB_FIX_INTRINSIC
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
    
    ret, CM1, DC1, CM2, DC2, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_cam0, imgpoints_cam1,
        K0, D0, K1, D1, img_size, 
        flags=flags, criteria=criteria
    )
    
    log(f"Stereo calibration complete")
    log(f"RMS error: {ret}")
    log(f"Rotation matrix:\n{R}")
    log(f"Translation vector:\n{T}")

    # Calculate stereo rectification
    log("Computing rectification parameters...")
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        CM1, DC1, CM2, DC2, img_size, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.175
    )

    log(f"Valid ROI (Left): {roi1}")
    log(f"Valid ROI (Right): {roi2}")
    
    # Save results
    np.savez(output_file,
        R=R, T=T, E=E, F=F,
        R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
        roi1=roi1, roi2=roi2,
        camera_matrix1=CM1, dist_coeffs1=DC1,
        camera_matrix2=CM2, dist_coeffs2=DC2,
        image_size=img_size
    )
    
    log(f"[SUCCESS] Stereo extrinsic calibration completed")
    log(f" Saved {output_file}")
    
  # Create and save test rectification
    try:
        log("Generating test rectification...")
        # Safety check
        if not images_cam0 or not images_cam1:
            raise ValueError("Image paths for cam0 or cam1 are empty!")    
        
        # Use the first image pair for testing rectification
        test_img_left = cv2.imread(images_cam0[0])
        test_img_right = cv2.imread(images_cam1[0])
        if test_img_left is None or test_img_right is None:
            raise ValueError("Could not load test images!")

        h, w = test_img_left.shape[:2]
        # Create undistortion and rectification maps
        map1x, map1y = cv2.initUndistortRectifyMap(CM1, DC1, R1, P1, img_size, cv2.CV_32FC1)
        map2x, map2y = cv2.initUndistortRectifyMap(CM2, DC2, R2, P2, img_size, cv2.CV_32FC1)
        
        # Apply rectification
        rect_left = cv2.remap(test_img_left, map1x, map1y, cv2.INTER_LINEAR)
        rect_right = cv2.remap(test_img_right, map2x, map2y, cv2.INTER_LINEAR)
        
        # Draw horizontal lines to check rectification
        for y in range(0, h, 50):  # Draw a line every 50 pixels
            cv2.line(rect_left, (0, y), (w, y), (0, 0, 255), 1)
            cv2.line(rect_right, (0, y), (w, y), (0, 0, 255), 1)
        
        # Save rectified images
        cv2.imwrite(os.path.join(debug_dir, "rectified_left.jpg"), rect_left)
        cv2.imwrite(os.path.join(debug_dir, "rectified_right.jpg"), rect_right)
        
        # Create side-by-side comparison
        rect_combined = np.zeros((h, 2*w, 3), dtype=np.uint8)
        rect_combined[:, :w] = rect_left
        rect_combined[:, w:] = rect_right
        cv2.imwrite(os.path.join(debug_dir, "rectified_stereo.jpg"), rect_combined)
        
        log(f"Test rectification saved to {debug_dir}")
    except Exception as e:
        log(f"Could not generate test rectification: {e}")
        
else:
    log(f"[ERROR] Not enough matching tags for calibration")
    log("Make sure AprilTags are clearly visible in both camera views")

log("\n=== EXTRINSIC CALIBRATION PROCESS COMPLETED ===")
log(f"Log saved to: {log_file}")
log(f"Debug images saved to: {debug_dir}")
log(f"Calibration data saved to: {output_file}")

cv2.destroyAllWindows()