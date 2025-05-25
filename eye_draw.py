import cv2
import numpy as np

# Load Haar cascades for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Start webcam (confirmed working config)
cap = cv2.VideoCapture(0)

# Create a blank canvas for drawing
canvas = np.zeros((480, 640, 3), dtype=np.uint8)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            # Draw a dot at the center of each detected eye on the canvas
            cx, cy = x + ex + ew//2, y + ey + eh//2
            cv2.circle(canvas, (cx, cy), 4, (0, 0, 255), -1)
    # Show both the webcam and the drawing canvas
    cv2.imshow('Webcam (q to quit, c to clear)', frame)
    cv2.imshow('Eye-Draw Canvas', canvas)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('c'):
        canvas[:] = 0  # Clear canvas
cap.release()
cv2.destroyAllWindows() 