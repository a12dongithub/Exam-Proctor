

import cv2
from gaze_tracking import GazeTracking

gaze = GazeTracking()
webcam = cv2.VideoCapture(0)

while True:
    _, frame = webcam.read()
    gaze.refresh(frame)
    frame = gaze.annotated_frame()
    text = ""
    if gaze.is_blinking():
        text = "Blinking"
    elif gaze.is_right():
        text = "Distracted Looking right"
    elif gaze.is_left():
        text = "Distracted Looking left"
    elif gaze.is_center():
        text = "Focus on Screen: Looking center"
    cv2.putText(frame, text, (50, 60), cv2.FONT_HERSHEY_DUPLEX, 1, (147, 58, 31), 2)
    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    cv2.imshow("Demo", frame)
    if cv2.waitKey(1) == 27:
        break
webcam.release()
cv2.destroyAllWindows()
