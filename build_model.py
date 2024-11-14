import os
from random import shuffle
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from keras.applications.mobilenet_v2 import MobileNetV2

IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
SEQUENCE_LENGTH = 16
DATASET_DIR = "./dataset"
CLASSES_LIST = ["NonViolence", "Violence"]

def frames_extraction(video_path):
    frames_list = []
    video_reader = cv2.VideoCapture(video_path)
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_frames_window = max(1, video_frames_count // SEQUENCE_LENGTH)

    for frame_counter in range(SEQUENCE_LENGTH):
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
        success, frame = video_reader.read()
        if not success:
            break
        frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        frame = frame / 255.0
        frames_list.append(frame)

    video_reader.release()
    return frames_list

def create_model():
    mobilenet = MobileNetV2(include_top=False, weights="imagenet")
    mobilenet.trainable=True

    for layer in mobilenet.layers[:-40]:
        layer.trainable=False

    model = Sequential()

    model.add(Input(shape = (SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 3)))    

    model.add(TimeDistributed(mobilenet))
    model.add(Dropout(0.25))
    model.add(TimeDistributed(Flatten()))
    
    lstm_fw = LSTM(units=32)
    lstm_bw = LSTM(units=32, go_backwards = True)  

    model.add(Bidirectional(lstm_fw, backward_layer = lstm_bw))
    model.add(Dropout(0.25))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.25))
       
    model.add(Dense(len(CLASSES_LIST), activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=["accuracy"])
    return model

if __name__ == '__main__':
    model = create_model()
    model.summary()

    features, labels = [], []
    for class_index, class_name in enumerate(CLASSES_LIST):
        class_dir = os.path.join(DATASET_DIR, class_name)
        for file_name in os.listdir(class_dir):
            video_path = os.path.join(class_dir, file_name)
            frames = frames_extraction(video_path)
            if len(frames) == SEQUENCE_LENGTH:
                features.append(frames)
                labels.append(class_index)

    features = np.array(features)
    labels = np.array(labels)
    
    # One-hot encode the labels
    labels = tf.keras.utils.to_categorical(labels, num_classes=len(CLASSES_LIST))

    # Train the model
    model.fit(features, labels, epochs=50, validation_split=0.2, batch_size=8, shuffle=True)

    # Save the model
    model.save('violence_detection_model.keras')
