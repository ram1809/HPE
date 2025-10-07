import cv2

cap = cv2.VideoCapture(3)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("nope")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("nope 2")
        break

    cv2.imshow("windowname", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

