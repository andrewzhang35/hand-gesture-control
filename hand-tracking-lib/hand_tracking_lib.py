import cv2
import mediapipe as mp
from pandas import read_csv
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from itertools import chain
from hand_shapes import *
from copy import deepcopy
from datetime import datetime

# Relative path to csv training data
csv_path = "../model-data/landmarks_classifier.csv"


# Class for detecting hands, drawing landmarks, and returning/processing landmark data
class HandDetector:
    def __init__(self, static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.static_image_mode, self.max_num_hands,
                                         self.min_detection_confidence, self.min_tracking_confidence)
        self.mp_draw = mp.solutions.drawing_utils
        self.results = None

    def draw_landmarks_on(self, image, draw=True):
        # Adds 'landmarks' to frame

        # Process frame; get inference results
        self.results = self.hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # If landmarks detected:
        if self.results.multi_hand_landmarks:
            # Draw each landmark and connections
            for hand_landmark in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(image, hand_landmark, self.mp_hands.HAND_CONNECTIONS)

        # Return frame with landmarks
        return image

    def find_landmark_positions(self, image, hand_number=0, draw=False):
        # Returns position of the 21 landmarks in a 2D list [[x, y], [x, y], etc.]

        hand_landmark_list = []

        if self.results.multi_hand_landmarks:
            hand_landmarks = self.results.multi_hand_landmarks[hand_number]

            for index, landmark in enumerate(hand_landmarks.landmark):
                # Convert relative x and y coordinate of a landmark into coordinate based on width and height of
                # frame. e.g. (0.2, 0.2) -> (128, 96) on 640 x 480 frame
                height, width, _ = image.shape
                x_coord, y_coord = int(landmark.x * width), int(landmark.y * height)

                # Append each landmark point to list
                hand_landmark_list.append([x_coord, y_coord])

                # if draw parameter is true: draw each landmark
                if draw:
                    cv2.circle(image, (x_coord, y_coord), 15, (255, 0, 255), cv2.FILLED)

        return hand_landmark_list

    def process_landmarks(self, landmark_list):
        # Converts all landmarks to coordinates relative to wrist landmark (normalizes landmarks around
        # wrist landmark)
        # Convert 2D landmark list to 1D list for writing to csv

        wrist_x, wrist_y = 0, 0

        for index, landmark in enumerate(landmark_list):
            if index == 0:
                wrist_x, wrist_y = landmark[0], landmark[1]

            # Normalize landmark x and y around wrist x and y
            landmark_list[index][0] -= wrist_x
            landmark_list[index][1] -= wrist_y

        # Flatten 2D array to 1D
        landmark_list = list(chain.from_iterable(landmark_list))

        return landmark_list


class HandShapeDetector:
    def __init__(self):
        # Training model using .csv in /model-data
        print("Starting to read dataset... " + str(datetime.now()))
        dataset = read_csv(csv_path, header=None)
        hand_shape_data = dataset.values
        x_train = hand_shape_data[:, 1:43]
        y_train = hand_shape_data[:, 0]

        self.model = KNeighborsClassifier()
        self.model.fit(x_train, y_train)
        print("Model trained. " + str(datetime.now()))

        # State variable for current hand shape state (e.g. "Palm Open", "Closed Hand", etc.)
        self.state = None

    def get_prediction(self, processed_landmark_list):
        # Get prediction and confidence from model
        prediction = self.model.predict(processed_landmark_list)
        confidence = self.model.predict_proba(processed_landmark_list)
        return prediction, confidence


def main():
    # Testing gesture recognition/classification

    cap = cv2.VideoCapture(0)

    hand_detector = HandDetector(min_detection_confidence=0.75, min_tracking_confidence=0.9)
    hand_shape_detector = HandShapeDetector()

    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            print("Ignoring empty camera frame")
            continue

        frame = hand_detector.draw_landmarks_on(frame)
        landmark_list = hand_detector.find_landmark_positions(frame, draw=False)
        landmark_list_deep_copy = deepcopy(landmark_list)

        if len(landmark_list) != 0:
            processed_landmark_list = hand_detector.process_landmarks(landmark_list_deep_copy)
            inference, confidence = hand_shape_detector.get_prediction([processed_landmark_list])

            if confidence[0][inference[0]] >= CONFIDENCE_THRESHOLD:
                cv2.putText(frame, hand_shapes[inference[0]],
                            (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, "No shape detected",
                            (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Frame", frame)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()


if __name__ == "__main__":
    main()
