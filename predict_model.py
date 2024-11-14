import os
import cv2
import numpy as np
import tensorflow as tf


IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
SEQUENCE_LENGTH = 16
CLASSES_LIST = ["NonViolence", "Violence"]

model = tf.keras.models.load_model('violence_detection_model.h5')


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


def predict_video_class(video_path):
    frames = frames_extraction(video_path)
    
    if len(frames) != SEQUENCE_LENGTH:
        print("Video does not have enough frames for prediction.")
        return None
    
    frames_array = np.expand_dims(frames, axis=0)
    predictions = model.predict(frames_array)[0]
    
    for i, class_name in enumerate(CLASSES_LIST):
        print(f"{class_name}: {predictions[i] * 100:.2f}%")
    
    predicted_class = np.argmax(predictions)
    predicted_label = CLASSES_LIST[predicted_class]
    print(f"\nPredicted class: {predicted_label} (Confidence: {predictions[predicted_class] * 100:.2f}%)")
    
    return predicted_label

if __name__ == '__main__':
    video_path = './dataset/Violence/V_387.mp4'
    predict_video_class(video_path)
