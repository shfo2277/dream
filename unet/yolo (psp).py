import torch
import cv2
import numpy as np
from pytube import YouTube

# Load the pre-trained models
person_model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolo_pt_file/best_person.pt')
car_model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolo_pt_file/best_car.pt')


# Replace with your YouTube video URL
youtube_url = "https://www.youtube.com/watch?v=7t3xBqAWLaU"

# Get the stream from YouTube using pytube
yt = YouTube(youtube_url)
stream = yt.streams.filter(progressive=True, file_extension='mp4').first()

# Open the video stream using OpenCV
cap = cv2.VideoCapture(stream.url)

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break

    # Run inference for person detection
    person_results = person_model(frame)
    
    # Run inference for car detection
    car_results = car_model(frame)

    # Draw bounding boxes for person
    person_results.render()  # or use custom rendering

    # Draw bounding boxes for car
    car_results.render()

    # Show the frame with detections
    cv2.imshow('Person and Car Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
