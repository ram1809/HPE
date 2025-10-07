import cv2
import os
from datetime import datetime
 
# === Camera device paths ===
CAMERA_DEVICES = {
    "Camera0": "/dev/video0",
    "Camera1": "/dev/video2"
}
 
# Create output directory
OUTPUT_DIR = "high_res_images"
for cam_name in CAMERA_DEVICES:
    os.makedirs(os.path.join(OUTPUT_DIR, cam_name), exist_ok=True)
 
# Active camera
selected_camera = "Camera0"  # Default to first camera
selected_device = CAMERA_DEVICES[selected_camera]
 
# Initialize camera
print(f"Opening {selected_camera} at {selected_device}...")
cap = cv2.VideoCapture(selected_device)
 
if not cap.isOpened():
    print(f"Error: Could not open {selected_camera} at {selected_device}")
    exit(1)
 
# Print all camera properties
print("Camera properties:")
for i in range(46):  # OpenCV supports properties 0-45
    try:
        value = cap.get(i)
        print(f"Property {i}: {value}")
    except:
        pass
 
# Try to set highest resolution
resolutions = [
    (4096, 2160),  # 4K
    (3840, 2160),  # 4K UHD
    (2560, 1440),  # 2K
    (1920, 1080),  # Full HD
    (1280, 720)    # HD
]
 
# Try resolutions from highest to lowest
for width, height in resolutions:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    # Check if the resolution was actually set
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if actual_width == width and actual_height == height:
        print(f"Successfully set resolution to: {width}x{height}")
        break
 
# Print the actual resolution we're using
actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Camera resolution: {actual_width}x{actual_height}")
 
# Image counter for each camera
img_counters = {name: 0 for name in CAMERA_DEVICES}
 
print("\nInstructions:")
print("Press 1 to select Camera0")
print("Press 2 to select Camera1")
print("Press SPACE to capture an image")
print("Press Q to quit")
 
while True:
    # Read a frame
    ret, frame = cap.read()
    if not ret:
        print(f"Error reading from {selected_camera}")
        break
 
    # Display the frame
    display = frame.copy()
    # Add camera and resolution info
    cv2.putText(display, f"Selected: {selected_camera} ({actual_width}x{actual_height})", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    # Add instructions
    cv2.putText(display, "1/2: Switch Camera | SPACE: Capture | Q: Quit", 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    # Scale down display window if resolution is too high
    scale_factor = min(1.0, 1200 / max(actual_width, actual_height))
    if scale_factor < 1.0:
        display = cv2.resize(display, (0, 0), fx=scale_factor, fy=scale_factor)
    cv2.imshow("High Resolution Camera", display)
    # Process key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Quitting...")
        break
    elif key == ord('1') or key == ord('2'):
        # Close current camera
        cap.release()
        # Select new camera
        if key == ord('1'):
            selected_camera = "Camera0"
        else:
            selected_camera = "Camera1"
        selected_device = CAMERA_DEVICES[selected_camera]
        print(f"Switching to {selected_camera} at {selected_device}...")
        # Open new camera
        cap = cv2.VideoCapture(selected_device)
        if not cap.isOpened():
            print(f"Error: Could not open {selected_camera} at {selected_device}")
            continue
        # Print all camera properties for the new camera
        print(f"Properties for {selected_camera}:")
        for i in range(46):
            try:
                value = cap.get(i)
                print(f"Property {i}: {value}")
            except:
                pass
        # Try to set highest resolution again
        for width, height in resolutions:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if actual_width == width and actual_height == height:
                break
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera resolution: {actual_width}x{actual_height}")
    elif key == ord(' '):  # Space key
        # Simple timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_idx = img_counters[selected_camera]
        # Create filename
        filename = os.path.join(OUTPUT_DIR, selected_camera, 
                              f"image_{timestamp}_{selected_camera}_{img_idx:03d}.png")
        # Save the image at full resolution
        cv2.imwrite(filename, frame)
        print(f"📸 Saved: {filename} ({actual_width}x{actual_height})")
        img_counters[selected_camera] += 1
 
# Release camera and close windows
cap.release()
cv2.destroyAllWindows()
print(f"Total images captured: {sum(img_counters.values())}")
print(f"All images saved to: {os.path.abspath(OUTPUT_DIR)}")