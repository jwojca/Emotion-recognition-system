import subprocess
import pandas as pd
import time
import csv
import os
import cv2  
from deepface import DeepFace
import numpy as np
import math
import argparse
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.tree import plot_tree

import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay
from collections import Counter

def deleteFolderContents(folder_path):
    """
    Deletes all files and subdirectories inside a folder, but not the folder itself.
    """
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)  # remove file
            elif os.path.isdir(file_path):
                deleteFolderContents(file_path)  # recurse and remove subdirectories
                os.rmdir(file_path)  # remove directory
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

def angleFromVertical(p1, p2):
    """Calculate the angle in degrees between the line connecting points p1 and p2
    and the vertical axis."""
    deltaY = p2[1] - p1[1]
    deltaX = p2[0] - p1[0]
    angle = math.atan2(deltaX, deltaY)
    return -math.degrees(angle)

def opDrawSkeleton(frame, points, POSE_PAIRS):
    # Draw Skeleton
    for pair in POSE_PAIRS:
        partA = pair[0]
        partB = pair[1]

        if points[partA] and points[partB]:
            cv2.line(frame, points[partA], points[partB], (0, 255, 255), 2)
            cv2.circle(frame, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
            
        if points[0] and points[1]:
            diameter = math.dist(points[0], points[1])
            midPoint = [0, 0]
            # Define two points
            pointArr0 = np.array(points[0])
            pointArr1 = np.array(points[1])

            # Calculate the midpoint
            midpoint = (pointArr0 + pointArr1) / 2
            center = tuple(np.round(midpoint).astype(int))
            ellAngle = angleFromVertical(points[0], points[1])
            #cv2.circle(frame, center, round((diameter/2) * 1.2), (255, 0, 0), thickness = 2)
            center2 = tuple(np.round(points[1]).astype(int))
            cv2.ellipse(frame, center, (int(diameter/1.5), int(diameter/1.2)), ellAngle, 0, 360, (255, 0, 0), thickness = 2)

            e = Ellipse((int(center[0]), int(center[1])), 2*int(diameter/1.5), 2*int(diameter/1.2), ellAngle)
            rightWrist = points[4]
            leftWrist = points[7]
            if(rightWrist and not leftWrist):
                if e.contains_point(rightWrist):
                    print("Right hand in face area")
            elif(leftWrist and not rightWrist):
                if e.contains_point(leftWrist):
                    print("Left hand in face area")
            elif(rightWrist and leftWrist):
                if e.contains_point(leftWrist) and e.contains_point(rightWrist):
                    print("Both hands in face area")
            else:
                print("No hand in face area")
    return frame

def opGetPoints(output, frame, frameCopy):
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    threshold = 0.1
     
    H = output.shape[2]
    W = output.shape[3]

    # Empty list to store the detected keypoints
    points = []

    for i in range(nPoints):
        # confidence map of corresponding body's part.
        probMap = output[0, i, :, :]

        # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
        
        # Scale the point to fit on the original image
        x = (frameWidth * point[0]) / W
        y = (frameHeight * point[1]) / H

        if prob > threshold : 
            cv2.circle(frameCopy, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frameCopy, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)

            # Add the point to the list if the probability is greater than the threshold
            points.append((int(x), int(y)))
        else :
            points.append(None)
    return points

def ofGetDominantEmotion(data):
    """
    Returns the dominant emotion and its percentage in a given array of emotions.

    Args:
        data: A NumPy array of emotions.

    Returns:
        A tuple containing the dominant emotion (string) and its percentage (float).
    """
    counts = Counter(data)
    dominantEmotion = counts.most_common(1)[0][0]
    dominantPercentage = counts[dominantEmotion] / len(data) * 100
    return (dominantEmotion, dominantPercentage)

#Decision tree
# Function importing Dataset
def importdata():
	trainData = pd.read_csv(trainCSV, sep= ';', header=None)
	testData = pd.read_csv(testCSV, sep= ';', header=None)
	
	# Printing the dataset obseravtions
	#print ("Dataset: ", trainData.head())
	#print ("Dataset: ", testData.head())

	return trainData, testData

# Function to load the dataset
def loaddataset(trainData, testData):

	# Separating the target variable
	X_train = trainData.values[1:len(trainData), 1:35]
	y_train = trainData.values[1:len(trainData):, 36]

	X_test = testData.values[1:len(testData), 1:35]
	y_test = testData.values[1:len(testData):, 36]
	
	return X_train, X_test, y_train, y_test
		
# Function to perform training with entropy.
def train_using_entropy(X_train, X_test, y_train, maxDepth, minSamplesLeaf):

	# Decision tree with entropy
	clf_entropy = DecisionTreeClassifier(
			criterion = "entropy", random_state = 100,
			max_depth = maxDepth, min_samples_leaf = minSamplesLeaf)

	# Performing training
	clf_entropy.fit(X_train, y_train)
	return clf_entropy


# Function to make predictions
def prediction(X_test, clf_object):

	# Predicton on test with giniIndex
	y_pred = clf_object.predict(X_test)
	#print("Predicted values:")
	#print(y_pred)
	return y_pred

def cal_accuracy(y_test, y_pred):
	
	print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))
	print ("Accuracy : ", accuracy_score(y_test,y_pred)*100)
	print("Report : \n",classification_report(y_test, y_pred))
	acc = accuracy_score(y_test,y_pred)*100
	return acc

trainCSV = r'C:\Users\hwojc\OneDrive - Vysoké učení technické v Brně\Magisterské studium\Diplomka\02 Modely\Validace\OpenFace\rozhodovaci strom\trainAUcAUr.csv'
testCSV =  r'C:\Users\hwojc\OneDrive - Vysoké učení technické v Brně\Magisterské studium\Diplomka\02 Modely\Validace\OpenFace\rozhodovaci strom\testAUcAUr.csv'

classList = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

featureList = ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r',
 			   'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r',
 			   'AU25_r', 'AU26_r', 'AU45_r',
	           'AU01_c', 'AU02_c', 'AU04_c', 'AU05_c', 'AU06_c', 'AU07_c', 'AU09_c',
 			   'AU10_c', 'AU12_c', 'AU14_c', 'AU15_c', 'AU17_c', 'AU20_c', 'AU23_c',
 			   'AU25_c', 'AU26_c', 'AU28_c', 'AU45_c']

#train decision tree
trainData, testData = importdata()
X_train, X_test, y_train, y_test = loaddataset(trainData, testData)
clf_entropy = train_using_entropy(X_train, X_test, y_train, 7, 37)
y_pred_entropy = prediction(X_test, clf_entropy)
acc= cal_accuracy(y_test, y_pred_entropy)




#Open Pose
MODE = "MPI"

if MODE == "COCO":
    protoFile = "pose/coco/pose_deploy_linevec.prototxt"
    weightsFile = "pose/coco/pose_iter_440000.caffemodel"
    nPoints = 18
    POSE_PAIRS = [ [1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],[0,14],[0,15],[14,16],[15,17]]

elif MODE == "MPI" :
    protoFile = r'C:\Users\hwojc\Desktop\Diplomka\OpenPose\repo\openpose\models\pose\mpi\pose_deploy_linevec_faster_4_stages.prototxt'
    weightsFile = r'C:\Users\hwojc\Desktop\Diplomka\OpenPose\repo\openpose\models\pose\mpi\pose_iter_160000.caffemodel'
    nPoints = 15
    POSE_PAIRS = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14], [14,8], [8,9], [9,10], [14,11], [11,12], [12,13] ]

net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
imgPath = r'C:\Users\hwojc\Desktop\Diplomka\Repo\OpenPose\OpenPose\single.jpeg'
parser = argparse.ArgumentParser(description='Run keypoint detection')
parser.add_argument("--device", default="gpu", help="Device to inference on")
parser.add_argument("--image_file", default=imgPath, help="Input image")

args = parser.parse_args()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# Path to FaceLandmarkImg.exe and image file
#exePath  = r"C:\Users\hwojc\Desktop\Diplomka\Open Face\OpenFace_2.2.0_win_x64\FeatureExtraction.exe"
exePath  = r"C:\Users\hwojc\Desktop\Diplomka\Open Face\OpenFace_2.2.0_win_x64\FeatureExtraction.exe"

imageDirPathBase  = r"C:\Users\hwojc\Desktop\Diplomka\Repo\OpenFace_DeepFace_merge\images"
imgID = 0

# Output directory
outDir = r"C:\Users\hwojc\Desktop\Diplomka\Repo\OpenFace_DeepFace_merge\processed"

#Delete previous data
deleteFolderContents(imageDirPathBase)
deleteFolderContents(outDir)

outputFilePath = outDir + r"\images.csv"
checkCSV = False
gCheckProc = False

start = time.time()

imgDirIndex = 1
imgSavedCount = 0
imageDirPath =  os.path.join(imageDirPathBase, str(imgDirIndex))
if not os.path.exists(imageDirPath):
    os.makedirs(imageDirPath)


frameCount = 0
skippedFrames = 10
dfPredEm = "None"
points = []

while True:

    frameRecieved, frame = cap.read()
   
    if frameRecieved:
        frameCount = frameCount + 1
        if frameCount % skippedFrames == 0:
            try:
                result = DeepFace.analyze(frame, actions = ['emotion'], enforce_detection= True)
                dfPredEm = result['dominant_emotion']
                imgName = str(imgID) + '.jpg'
                imgPath = os.path.join(imageDirPath, imgName)
                cv2.imwrite(imgPath, frame)
                imgSavedCount = imgSavedCount + 1
                imgID = imgID + 1

                if(imgSavedCount % 30 == 0):
                    #process = subprocess.run([exePath, "-fdir", imageDirPath , "-aus","-out_dir", outDir])
                    process = subprocess.Popen([exePath, "-fdir", imageDirPath , "-aus", "-tracked","-out_dir", outDir])
                    csvName = str(imgDirIndex) + '.csv'
                    outputFilePath = os.path.join(outDir, csvName)
                    imgDirIndex = imgDirIndex + 1
                    imgSavedCount = 0
                    imageDirPath = os.path.join(imageDirPathBase, str(imgDirIndex))
                    if not os.path.exists(imageDirPath):
                        os.makedirs(imageDirPath)
                    gCheckProc = True
                #cv2.imshow('Video', frame)
            except:
                frame = cv2.putText(frame,'Cannot detect face',(50,50), cv2.FONT_ITALIC, 1, (0,0,0), 2, cv2.LINE_4)
                #cv2.imshow('Video', frame)

            if gCheckProc:
                if process.poll() == None:
                    checkCSV = False
                else:
                    checkCSV = True
                    gCheckProc = False

            

            if checkCSV:
                try:
                    ofData = pd.read_csv(outputFilePath, sep= ',', header=None)
                    rows, cols = ofData.shape
                    aus = ofData.values[1:rows, 5:cols-1]
                    #print(aus)
                    emPred = prediction(aus, clf_entropy)
                    dominantEm, dominantEmPct = ofGetDominantEmotion(emPred)
                    print(dominantEm, dominantEmPct)
                    frame = cv2.putText(frame,dominantEm, (300,50), cv2.FONT_ITALIC, 1, (0,0,255), 2, cv2.LINE_4)
                    """
                    with open(outputFilePath, 'r') as csvFile:         
                        # Create a new CSV reader object
                        reader = csv.reader(csvFile, delimiter = ",")
                        # Process the new data in the file
                        # Skip header row
                        next(reader)
                        # Load data into list of lists, selecting only columns from 5 to end-1
                        data = [[row[i] for i in range(5, len(row)-1)] for row in reader]
                        print(data)
                    
                        for row in reader:
                            # Do something with the row data
                            print(row)         
                        # Wait for a short time before checking the file again
                        #time.sleep(0.1)
                
                    """
                    
                except FileNotFoundError:
                    # Handle the case where the file doesn't exist yet
                    #time.sleep(0.1)
                    print("CSV doesnt exist yet")
        
        frameCopy = np.copy(frame)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        # input image dimensions for the network
        inWidth = 240
        inHeight = 160
        inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                                (0, 0, 0), swapRB=False, crop=False)
        net.setInput(inpBlob)
        output = net.forward()
        points = opGetPoints(output, frame, frameCopy)

        frame = opDrawSkeleton(frame, points, POSE_PAIRS)
        frame = cv2.putText(frame, dfPredEm,(50,50), cv2.FONT_ITALIC, 1, (0,0,0), 2, cv2.LINE_4)
        frame = cv2.putText(frame, str(frameCount), (50,100), cv2.FONT_ITALIC, 1, (0,0,0), 2)
        
        cv2.imshow('Output-Skeleton', frame)
        
    else:
        print("Frame not recieved")
    
  
    

    #Get key, if ESC, then end loop
    c = cv2.waitKey(1)
    if c == 27:
        break

end = time.time()
print(end - start, " seconds")
cap.release()
cv2.destroyAllWindows()



   




