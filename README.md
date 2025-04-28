# Real-time Exercise Classification with Rep Counter

This project focuses on real-time exercise classification, enabling the identification of exercises being performed and providing a rep counter for the user’s convenience. The classification is achieved using the Mediapipe library, which maps the locations of the joints on the body in a video feed.

## Overview

In this project, we use **Mediapipe Pose** for human pose estimation and **LSTM (Long Short-Term Memory)** networks for exercise classification. The system can recognize different types of exercises such as squats, push-ups, etc., and count the number of repetitions in real-time. This application can be beneficial for users tracking their workout performance.

### Key Features:
- **Pose Estimation**: The Mediapipe library is used to detect 33 key landmarks (joints) on the human body in real-time.
- **Exercise Classification**: The detected joint coordinates are used as inputs for an LSTM model, which classifies the type of exercise being performed.
- **Rep Counting**: The model counts the repetitions of each exercise in real-time.
- **Real-time Video Processing**: The system processes video frames in real-time and updates the classification and rep count on the live feed.

## Dataset
The model is trained using a custom dataset of exercise videos. The data consists of labeled video sequences of different exercises, such as squats and push-ups, and the pose landmarks are extracted from the video frames.

## Model Architecture
The core of the model is based on an LSTM (Long Short-Term Memory) neural network. The model uses a sequence of 33 joint coordinates for each frame and learns to classify exercises. The architecture consists of:

Three LSTM layers (128, 256, and 128 units)

Fully connected layers to output the classification

The output layer uses the softmax activation to classify the exercises.

## Data Collection
The data for training is collected using a video capture method. The pose landmarks are extracted from each frame, and keypoints (coordinates) are saved into a NumPy array file. This data is used to train the LSTM model.

Exercise Categories: Currently, the system can classify Squats and Push-ups

## Training the Model
Steps for Training:
Collect the training data by capturing video sequences of the exercise types you want to recognize.

Extract keypoints using Mediapipe Pose.

Organize the data into sequences of keypoints for training.

Train the LSTM model with the processed data.

Save the trained model for future use.

The model is trained with early stopping and learning rate adjustments to prevent overfitting and to improve convergence.

## Real-time Prediction
Once the model is trained, you can use it for real-time exercise classification using a webcam or any video source. The process includes:

Capturing video frames in real-time.

Estimating the human pose in each frame using Mediapipe.

Extracting the pose landmarks (joint positions) from each frame.

Feeding the extracted keypoints into the trained LSTM model for classification.

Displaying the current exercise type and rep count in real-time.

Results:
The LSTM model achieves a classification accuracy of approximately 82.5% on the test data.
