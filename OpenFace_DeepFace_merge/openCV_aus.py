import cv2
import dlib
import numpy as np

# Load the face detector and landmark predictor
faceDetPath = r'C:\Users\hwojc\Desktop\Diplomka\Repo\OpenFace_DeepFace_merge\haarcascade_frontalface_default.xml'
faceDetector = cv2.CascadeClassifier(faceDetPath)
landmarkDetector = dlib.get_frontal_face_detector()

def detect_AUs(image_path, face_cascade_path, landmark_detector_path):
    # Load face detector and landmark detector
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    landmark_detector = dlib.shape_predictor(landmark_detector_path)

    # Read the image
    img = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Detect Action Units for each face
    for (x, y, w, h) in faces:
        # Detect landmarks for the face
        rect = dlib.rectangle(int(x), int(y), int(x+w), int(y+h))
        landmarks = landmark_detector(gray, rect)
        
        # Use the detected landmarks to detect action units
        # ... your code here ...
    
    # Display the output
    cv2.imshow('img', img)
    cv2.waitKey()

# Define the function to extract facial features from landmarks
def extract_features(landmarks):
    # Convert the landmarks to a NumPy array
    landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])

    # Compute the distances between landmark points
    distances = np.zeros((68, 68))
    for i in range(68):
        for j in range(i + 1, 68):
            distances[i, j] = np.linalg.norm(landmarks[i] - landmarks[j])
            distances[j, i] = distances[i, j]

    # Compute the angles between landmark points
    angles = np.zeros((68, 68, 68))
    for i in range(68):
        for j in range(68):
            for k in range(68):
                if i != j and i != k and j != k:
                    a = distances[i, j]
                    b = distances[i, k]
                    c = distances[j, k]
                    angle = np.arccos((a ** 2 + b ** 2 - c ** 2) / (2 * a * b))
                    angles[i, j, k] = angle

    # Flatten the distances and angles into a feature vector
    features = np.concatenate((distances.flatten(), angles.flatten()))

    # Return the feature vector
    return features

# Define the function to perform AU detection using the features
def detect_AU(features):
    # Placeholder function - replace with your own AU detection code
    return []

# Load an example image
imgPath = r'C:\Users\hwojc\Desktop\Diplomka\Repo\OpenFace_DeepFace_merge\images\frame.jpg'
image = cv2.imread(imgPath)

# Detect the AUs in the image
AUs = detect_AUs(image)

# Print the AUs
print(AUs)
