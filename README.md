# hand-gesture-control
Python application for navigating and controlling a Windows computer using various static hand gestures.

Please note that the computer commands run in this program will likely only be functional on Windows. Currently only gestures with the right hand are supported.

## Set Up
- Clone this repo
- Set up a virtual environment
- Run "pip install -r requirements.txt" while in the project directory

## How To Run
- To run the program in its entirety, run gesture_control.py in src/
- To run only the gesture recognition portion of the program, run hand_tracking_lib.py in hand-tracking-lib/
- To collect more training data or collect all new training data, run ground_truth.py in model-data/
- Run evaluate_models.py to compare the performance of different models on the training/validation data

### Resources
🔗 https://google.github.io/mediapipe/solutions/hands

🔗 https://machinelearningmastery.com/machine-learning-in-python-step-by-step/

🔗 https://www.youtube.com/watch?v=NZde8Xt78Iw

🔗 https://www.youtube.com/watch?v=9iEPzbG-xLE

🔗 https://github.com/andrew-zhan139/Ctrl-Air-Space

🔗 https://github.com/kinivi/hand-gesture-recognition-mediapipe
