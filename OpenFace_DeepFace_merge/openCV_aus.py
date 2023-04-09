import cv2
import dlib
import numpy as np

# Load the face detector and landmark predictor
faceDetPath = r'C:\Users\hwojc\Desktop\Diplomka\Repo\OpenFace_DeepFace_merge\haarcascade_frontalface_default.xml'
faceDetector = cv2.CascadeClassifier(faceDetPath)
landmarkDetector = dlib.get_frontal_face_detector()

# Define the function to detect AUs
def detect_AUs(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces using the Haar cascade
    faces = faceDetector.detectMultiScale(gray, 1.3, 5)

    # Loop through the faces and detect landmarks
    for (x, y, w, h) in faces:
        # Convert the face region to a Dlib rectangle
        rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))

        # Detect the facial landmarks
        landmarks = landmarkDetector(gray, rect)

        # Extract the features from the landmarks
        features = extract_features(landmarks)

        # Perform AU detection using the features
        AUs = detect_AU(features)

        # Return the AUs for this face
        return AUs

    # If no faces were detected, return an empty array
    return []

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
