# Eye-Draw: Eye Tracking and Drawing with OpenCV

A simple Python app to detect your face and eyes and draw on a canvas using your gaze.

## Features
- Real-time face and eye detection using OpenCV Haar cascades
- Draw on a canvas with your eyes (dots appear where your eyes are detected)
- Press 'q' to quit, 'c' to clear the canvas

## Requirements
- Python 3.x
- opencv-python
- numpy

## Usage
```sh
pip install opencv-python numpy
python eye_draw.py
```

## How it works
- The webcam feed is shown with rectangles around detected faces and eyes.
- A separate window shows a canvas where a dot is drawn at the center of each detected eye.
- You can clear the canvas at any time by pressing 'c'.

## License
MIT 