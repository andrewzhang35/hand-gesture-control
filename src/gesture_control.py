import cv2
import sys
import pyautogui
import numpy as np
import keyboard
import math
from copy import deepcopy

# if case of error below in PyCharm, right click hand-tracking-lib directory and mark directory as sources root
sys.path.append("../hand-tracking-lib")
import hand_tracking_lib
from hand_shapes import *

# Set to "true" to see OpenCV window
show_window = False

# Setting OpenCV window border to 80 (e.g. x or y of coordinate at 80 on OpenCV window will be equivalent to
# x or y of 0 on computer screen)
window_border = 80

# pyautogui failsafe
pyautogui.FAILSAFE = True


def main():
    # Getting webcam input
    cap = cv2.VideoCapture(0)

    # Initializing variables and objects
    clicked = False
    previous_hand_x = 0
    previous_hand_y = 0

    hand_detector = hand_tracking_lib.HandDetector(min_detection_confidence=0.75, min_tracking_confidence=0.9)
    hand_shape_detector = hand_tracking_lib.HandShapeDetector()

    while cap.isOpened():
        # Get frame
        success, frame = cap.read()

        if not success:
            print("Ignoring empty camera frame")
            continue

        # Draw landmarks (coordinates) on frame using mediapipe
        frame = hand_detector.draw_landmarks_on(frame)

        # Get 2D list of landmarks (in format: [[x, y], [x, y], etc.])
        landmark_list = hand_detector.find_landmark_positions(frame, draw=False)
        landmark_list_deep_copy = deepcopy(landmark_list)

        # If landmarks detected:
        if len(landmark_list) != 0:
            # Process raw landmark list by normalizing data about the wrist and flattening to 1D
            processed_landmark_list = hand_detector.process_landmarks(landmark_list_deep_copy)

            # Get hand shape prediction and confidence of prediction
            inference, confidence = hand_shape_detector.get_prediction([processed_landmark_list])

            # If confidence level is higher than confidence threshold set in hand_shapes.py:
            if confidence[0][inference[0]] >= CONFIDENCE_THRESHOLD:
                # Set state = detected hand state
                hand_shape_detector.state = hand_shapes[inference[0]]
            else:
                # In case of low confidence:
                hand_shape_detector.state = "No Shape Detected"

            hand_shape = hand_shape_detector.state

            # If detected hand shape is not associated with a clicking action, set clicked to False
            if hand_shape != "Pinch" and hand_shape != "Spider-man" and hand_shape != "Three"\
                    and hand_shape != "Vertical":
                clicked = False

            # Execute computer actions using hand shapes
            if hand_shape == "Palm Open":
                # Move mouse cursor

                # Get location of the 9th landmark (joint right below the middle finger)
                hand_x, hand_y = landmark_list[9][0], landmark_list[9][1]

                # Get width and height of frame (640 x 480 by default)
                height, width, _ = frame.shape

                # Calculate distance moved compared to previous position
                distance_moved = math.hypot(hand_x - previous_hand_x, hand_y - previous_hand_y)

                # Check if distance moved is sufficient to move mouse cursor
                if distance_moved > 2:
                    # Map OpenCV window coordinate values to screen coordinate values
                    screen_x = np.interp(hand_x, [window_border, width-window_border], [pyautogui.size()[0], 0])
                    screen_y = np.interp(hand_y, [window_border, height-window_border], [0, pyautogui.size()[1]])

                    # Move to screen coordinate
                    pyautogui.moveTo(screen_x, screen_y, _pause=False)

                    # Set previous position to current position
                    previous_hand_x = hand_x
                    previous_hand_y = hand_y

            elif hand_shape == "Peace Sign":
                # Scroll down
                pyautogui.scroll(-30, _pause=False)

            elif hand_shape == "Closed Hand":
                # Scroll up
                pyautogui.scroll(30, _pause=False)

            elif hand_shape == "Pinch":
                # Click left mouse button and set clicked to true
                if not clicked:
                    pyautogui.click()
                    clicked = True

            elif hand_shape == "Spider-man":
                # Right click
                if not clicked:
                    pyautogui.click(button="right")
                    clicked = True

            elif hand_shape == "Thumbs Up":
                # Volume up
                pyautogui.press('volumeup')

            elif hand_shape == "Thumbs Down":
                # Volume down
                pyautogui.press('volumedown')

            elif hand_shape == "Three":
                # Show all available windows
                if not clicked:
                    pyautogui.hotkey('win', 'tab', _pause=False)
                    clicked = True

            elif hand_shape == "Vertical":
                # Open snipping tool
                if not clicked:
                    pyautogui.hotkey('win', 'shift', 's', _pause=False)
                    clicked = True

        # Show cv2 window
        if show_window:
            if len(landmark_list) != 0:
                cv2.putText(frame, hand_shape_detector.state,
                            (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("Window", frame)
            cv2.waitKey(0)

        # Close program on escape
        if keyboard.is_pressed('esc'):
            break

    cap.release()


if __name__ == "__main__":
    main()
