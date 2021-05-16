import csv
import cv2
import sys

sys.path.append("../hand-tracking-lib")
from hand_shapes import *
import hand_tracking_lib

# Defining path to csv and default hand shape ID (index for defining hand shapes)
hand_shape_id = -1
csv_path = "landmarks_classifier.csv"


def select_hand_shape(key):
    # Changes selected hand shape ID on key press from 0-9
    global hand_shape_id
    # 0 - 9
    if 48 <= key <= (48 + len(hand_shapes) - 1) and len(hand_shapes) < 10:
        hand_shape_id = key - 48
    # space bar
    elif key == 32:
        hand_shape_id = -1


def clear_data():
    # Deletes all training data
    with open(csv_path, 'r+') as data_file:
        data_file.truncate(0)
    print("CSV file contents deleted")


def write_data(processed_landmark_list):
    # Writes one row of data to csv in following format: hand shape id, processed landmark list elements
    with open(csv_path, 'a', newline="") as data_file:
        csv_writer = csv.writer(data_file)
        csv_writer.writerow([hand_shape_id, *processed_landmark_list])


def main():
    # Output available hand shapes and corresponding index
    for index, hand_shape in enumerate(hand_shapes):
        print(str(index) + ": " + hand_shape)

    cap = cv2.VideoCapture(0)

    hand_detector = hand_tracking_lib.HandDetector(min_detection_confidence=0.75, min_tracking_confidence=0.9)

    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            break

        frame = hand_detector.draw_landmarks_on(frame)
        landmark_list = hand_detector.find_landmark_positions(frame, draw=False)

        if len(landmark_list) != 0 and hand_shape_id != -1:
            # Collect data
            cv2.putText(frame, "Collecting data for: " + hand_shapes[hand_shape_id],
                        (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            write_data(hand_detector.process_landmarks(landmark_list))

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(0)

        # Escape key press to exit
        if key == 27:
            break
        elif key == 8:
            clear_data()

        # Pass through key presses to check if hand shape index is being selected
        select_hand_shape(key)

    cap.release()


if __name__ == "__main__":
    main()
