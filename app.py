import cv2
import math
import torch


def detect_faces(model, frame):
    results = model(frame)
    # Assuming cls is at index 5 and 0 represents 'Person'
    return [result for result in results.xyxy[0] if result[5] == 0]


def detect_eyes(eye_cascade, face_roi):
    # Convert the frame to grayscale
    face_roi_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

    # Detect eyes in the grayscale image
    eyes = eye_cascade.detectMultiScale(face_roi_gray)
    return eyes


def detect_eyes_direction(eyes, face_center):
    for x, y, w, h in eyes:
        # Draw the eye rectangle
        cv2.rectangle(face_roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Calculate the eye center
        eye_center = (x + w // 2, y + h // 2)
        cv2.circle(frame, eye_center, 2, (0, 0, 255), -1)

        # Calculate the angle between the eye center and face center
        dx = eye_center[0] - face_center[0]
        dy = eye_center[1] - face_center[1]
        angle = math.atan2(dy, dx) * 180 / math.pi

        # Determine the direction based on the angle
        if -22.5 <= angle < 22.5:
            direction = "Right"
        elif 22.5 <= angle < 67.5:
            direction = "Up-Right"
        elif 67.5 <= angle <= 112.5:
            direction = "Up"
        elif 112.5 < angle <= 157.5:
            direction = "Up-Left"
        elif abs(angle) > 157.5:
            direction = "Left"
        elif -157.5 <= angle < -112.5:
            direction = "Down-Left"
        elif -112.5 <= angle < -67.5:
            direction = "Down"
        else:
            direction = "Down-Right"
        return direction


if __name__ == "__main__":

    # Start the webcam
    cap = cv2.VideoCapture(0)

    # Load YOLOv8n model
    model = torch.hub.load("ultralytics/yolov5", "yolov5s")

    # Load the Haar cascade model for eye detection
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        if not ret:
            break

        # Detect faces
        faces = detect_faces(model, frame)
        for result in faces:
            x1, y1, x2, y2, conf, cls = result
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

            # Draw Region of interest in the face
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            face_roi = frame[y1:y2, x1:x2]
            # Calculate the face center
            face_center_x = x1 + (x2 - x1) // 2
            face_center_y = y1 + (y2 - y1) // 2
            face_center = (face_center_x, face_center_y)

            # Detect eyes
            eyes = detect_eyes(eye_cascade, face_roi)

            # Detect eyes direction
            direction = detect_eyes_direction(eyes, face_center)

            # Display the direction text on the frame
            cv2.putText(
                frame,
                f"Looking {direction}",
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (36, 255, 12),
                2,
            )

        cv2.imshow("Eyes Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release video capture and close windows
    cap.release()
    cv2.destroyAllWindows()
