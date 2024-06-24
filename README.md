# Real-Time Eye Tracking and Gaze Direction Detection

This project demonstrates real-time eye tracking and gaze direction detection using OpenCV and YOLOv5. The goal is to detect the position of the user's eyes and determine their gaze direction using computer vision techniques.

## Introduction

This project uses YOLOv5 for face detection and OpenCV's Haar cascades for eye detection. By extracting the eye regions and detecting the pupil's position within the eyes, we can infer the user's gaze direction. The gaze direction is determined by calculating the angle between the eye center and face center.

## Requirements

- Python 3.6+
- OpenCV
- Torch
- Ultralytics YOLOv5

## Installation

1. **Install OpenCV**:

```sh
 pip install opencv-python
```

2. **Install Torch**:

```sh
 pip install torch
```

3. **Install ultralytics**:

```sh
 pip install ultralytics
```

## Explanation

### Face and Eye Detection

- **Face Detection**: YOLOv5 is used to detect faces in the video feed. YOLO (You Only Look Once) is a state-of-the-art, real-time object detection system.
- **Eye Detection**: OpenCV's Haar cascades are used to detect eye regions within the detected faces.

### Pupil Detection

- **Thresholding and Contour Detection**: The eye region is converted to grayscale, and binary thresholding is applied to create a binary image. Contours are then detected within the binary image to find the largest contour, assumed to be the pupil.

### Gaze Direction Determination

- **Angle Calculation**: The direction of the gaze is determined by calculating the angle between the eye center and face center.
- **Direction Mapping**: The angle is mapped to one of eight directions (_Right, Up-Right, Up, Up-Left, Left, Down-Left, Down, Down-Right_) based on predefined angular ranges.

### Visualization

Here's a visual representation of how the gaze direction is determined based on the angle:

                        Up
                        |
                        |
           Up-Left      |       Up-Right
               \        |       /
                \       |      /
                 \      |     /
                  \     |    /
                   \    |   /
                    \   |  /
                     \  | /
        Left -------------------- Right
                      /|\
                     / | \
                    /  |  \
                   /   |   \
                  /    |    \
                 /     |     \
                /      |      \
               /       |       \
    Down-Left          |        Down-Right
                        |
                        |
                      Down

## References

- [OpenCV Documentation](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- [YOLOv5 Documentation](https://docs.ultralytics.com/yolov5/#explore-and-learn)
