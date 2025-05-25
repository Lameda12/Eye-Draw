import cv2
import numpy as np
import threading
import tkinter as tk
from tkinter import colorchooser, filedialog
from PIL import Image, ImageTk

# --- Eye tracking and drawing logic ---
class EyeDrawApp:
    def __init__(self, root):
        self.root = root
        self.root.title('Eye-Draw GUI')
        self.running = True
        self.drawing_mode = 'freehand'  # Modes: freehand, line, shape, text
        self.draw_color = (0, 0, 255)
        self.sensitivity = 70
        self.last_gaze = None
        self.blink_detected = False
        self.canvas_img = np.zeros((480, 640, 3), dtype=np.uint8)
        self.calibrated = False
        self.calibration_points = [(50, 50), (590, 50), (320, 240), (50, 430), (590, 430)]
        self.calibration_data = []
        self.calibrating = False
        self.calib_index = 0
        self.line_start = None
        self.shape_start = None
        self.text_to_draw = "Hello!"
        self.setup_gui()
        self.cap = cv2.VideoCapture(0)
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.thread = threading.Thread(target=self.video_loop)
        self.thread.start()
        self.root.protocol('WM_DELETE_WINDOW', self.on_close)

    def setup_gui(self):
        # Video frame
        self.video_label = tk.Label(self.root)
        self.video_label.grid(row=0, column=0, columnspan=5)
        # Canvas frame
        self.canvas_label = tk.Label(self.root)
        self.canvas_label.grid(row=1, column=0, columnspan=5)
        # Drawing mode buttons
        tk.Button(self.root, text='Freehand', command=lambda: self.set_mode('freehand')).grid(row=2, column=0)
        tk.Button(self.root, text='Line', command=lambda: self.set_mode('line')).grid(row=2, column=1)
        tk.Button(self.root, text='Shape', command=lambda: self.set_mode('shape')).grid(row=2, column=2)
        tk.Button(self.root, text='Text', command=lambda: self.set_mode('text')).grid(row=2, column=3)
        # Calibration button
        tk.Button(self.root, text='Calibrate', command=self.start_calibration).grid(row=2, column=4)
        # Color picker
        tk.Button(self.root, text='Pick Color', command=self.pick_color).grid(row=3, column=0)
        # Save button
        tk.Button(self.root, text='Save', command=self.save_canvas).grid(row=3, column=1)
        # Clear button
        tk.Button(self.root, text='Clear', command=self.clear_canvas).grid(row=3, column=2)
        # Sensitivity slider
        self.sens_slider = tk.Scale(self.root, from_=30, to=120, orient=tk.HORIZONTAL, label='Sensitivity')
        self.sens_slider.set(self.sensitivity)
        self.sens_slider.grid(row=3, column=3)
        # Mode and calibration status label
        self.status_label = tk.Label(self.root, text=self.status_text())
        self.status_label.grid(row=3, column=4)

    def status_text(self):
        return f"Mode: {self.drawing_mode} | Calibrated: {self.calibrated}"

    def set_mode(self, mode):
        self.drawing_mode = mode
        self.status_label.config(text=self.status_text())
        if mode == 'line':
            self.line_start = None
        if mode == 'shape':
            self.shape_start = None

    def pick_color(self):
        color_code = colorchooser.askcolor(title="Choose color")
        if color_code[0]:
            self.draw_color = tuple(int(c) for c in color_code[0][::-1])

    def save_canvas(self):
        filename = filedialog.asksaveasfilename(defaultextension='.png', filetypes=[('PNG files', '*.png')])
        if filename:
            cv2.imwrite(filename, self.canvas_img)

    def clear_canvas(self):
        self.canvas_img[:] = 0

    def on_close(self):
        self.running = False
        self.cap.release()
        self.root.destroy()

    def start_calibration(self):
        self.calibrating = True
        self.calib_index = 0
        self.calibration_data = []
        self.calibrated = False
        self.status_label.config(text=self.status_text())

    def map_gaze(self, gaze):
        # Simple mapping: if calibrated, use the closest calibration point offset
        if not self.calibrated or not self.calibration_data:
            return gaze
        # Find the closest calibration point
        dists = [np.linalg.norm(np.array(gaze) - np.array(g)) for g in self.calibration_data]
        min_idx = np.argmin(dists)
        # Map to the corresponding screen point
        return self.calibration_points[min_idx]

    def video_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            gaze_point = None
            blink = False
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                eyes = self.eye_cascade.detectMultiScale(roi_gray)
                for (ex, ey, ew, eh) in eyes:
                    eye_gray = roi_gray[ey:ey+eh, ex:ex+ew]
                    # --- Pupil tracking ---
                    blurred = cv2.GaussianBlur(eye_gray, (5, 5), 0)
                    _, thresh = cv2.threshold(blurred, self.sens_slider.get(), 255, cv2.THRESH_BINARY_INV)
                    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
                    if contours:
                        (cx, cy), radius = cv2.minEnclosingCircle(contours[0])
                        cx, cy = int(cx), int(cy)
                        abs_x = x + ex + cx
                        abs_y = y + ey + cy
                        gaze_point = (abs_x, abs_y)
                        cv2.circle(frame, (abs_x, abs_y), int(radius), (255, 0, 0), 2)
                        # --- Blink detection (simple: if radius is very small) ---
                        # TODO: Replace with facial landmark-based blink detection (dlib/mediapipe)
                        if radius < 3:
                            blink = True
                        break
                break  # Only use first face/eye for demo
            # --- Calibration routine ---
            if self.calibrating and gaze_point:
                # Show calibration point on canvas
                pt = self.calibration_points[self.calib_index]
                temp_canvas = self.canvas_img.copy()
                cv2.circle(temp_canvas, pt, 10, (0, 255, 255), -1)
                self.show_canvas(temp_canvas)
                # Wait for blink to record calibration
                if not self.blink_detected and blink:
                    self.calibration_data.append(gaze_point)
                    self.calib_index += 1
                    if self.calib_index >= len(self.calibration_points):
                        self.calibrating = False
                        self.calibrated = True
                        self.status_label.config(text=self.status_text())
                self.blink_detected = blink
                self.show_video(frame)
                continue
            # --- Drawing logic ---
            if gaze_point and self.calibrated:
                mapped_gaze = self.map_gaze(gaze_point)
                if self.drawing_mode == 'freehand':
                    if not self.blink_detected and blink:
                        self.last_gaze = mapped_gaze
                    if self.last_gaze and not blink:
                        cv2.line(self.canvas_img, self.last_gaze, mapped_gaze, self.draw_color, 2)
                        self.last_gaze = mapped_gaze
                elif self.drawing_mode == 'line':
                    if not self.line_start and not self.blink_detected and blink:
                        self.line_start = mapped_gaze
                    elif self.line_start and not blink and not self.blink_detected:
                        cv2.line(self.canvas_img, self.line_start, mapped_gaze, self.draw_color, 2)
                        self.line_start = None
                elif self.drawing_mode == 'shape':
                    if not self.shape_start and not self.blink_detected and blink:
                        self.shape_start = mapped_gaze
                    elif self.shape_start and not blink and not self.blink_detected:
                        cv2.rectangle(self.canvas_img, self.shape_start, mapped_gaze, self.draw_color, 2)
                        self.shape_start = None
                elif self.drawing_mode == 'text':
                    if not self.blink_detected and blink:
                        cv2.putText(self.canvas_img, self.text_to_draw, mapped_gaze, cv2.FONT_HERSHEY_SIMPLEX, 1, self.draw_color, 2)
            self.blink_detected = blink
            self.show_video(frame)
            self.show_canvas(self.canvas_img)

    def show_video(self, frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        imgtk = ImageTk.PhotoImage(image=img_pil)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

    def show_canvas(self, canvas_img):
        canvas_rgb = cv2.cvtColor(canvas_img, cv2.COLOR_BGR2RGB)
        canvas_pil = Image.fromarray(canvas_rgb)
        canvas_imgtk = ImageTk.PhotoImage(image=canvas_pil)
        self.canvas_label.imgtk = canvas_imgtk
        self.canvas_label.configure(image=canvas_imgtk)

if __name__ == '__main__':
    root = tk.Tk()
    app = EyeDrawApp(root)
    root.mainloop() 