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

import openPose
import openFace
import decTree



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
 






def displayTableOnFrame(frame, deepfaceOutput, openfaceOutput, numFrames):
    # Define the table contents
    table = [
        ["DeepFace output:", str(deepfaceOutput)],
        ["OpenFace output:", str(openfaceOutput)],
        ["Frames:", str(numFrames)]
    ]

    # Define the font and font scale
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5

    # Define the size of each cell in the table
    cellSize = (150, 25)

    # Loop over the rows and columns of the table
    for i in range(3):
        for j in range(2):
            # Define the position of the cell
            x = j * cellSize[0]
            y = i * cellSize[1]

            # Draw the black rectangle behind the text
            cv2.rectangle(frame, (x, y), (x + cellSize[0], y + cellSize[1]), (0, 0, 0), -1)

            # Draw the text in the cell
            cv2.putText(frame, table[i][j], (x + 5, y + 20), font, fontScale, (255, 255, 255), 1)

    return frame


#train decision tree
trainData, testData = decTree.importdata()
X_train, X_test, y_train, y_test = decTree.loaddataset(trainData, testData)
clf_entropy = decTree.train_using_entropy(X_train, X_test, y_train, 7, 37)
y_pred_entropy = decTree.prediction(X_test, clf_entropy)
acc = decTree.cal_accuracy(y_test, y_pred_entropy)




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

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# Path to FaceLandmarkImg.exe and image file
#exePath  = r"C:\Users\hwojc\Desktop\Diplomka\Open Face\OpenFace_2.2.0_win_x64\FeatureExtraction.exe"
exePath  = r"C:\Users\hwojc\Desktop\Diplomka\Open Face\OpenFace_2.2.0_win_x64\FeatureExtraction.exe"

imageDirPathBase  = r"C:\Users\hwojc\Desktop\Diplomka\Repo\OpenFace_DeepFace_merge\images"
imgID = 0
lastPosition = 0

# Output directory
outDir = r"C:\Users\hwojc\Desktop\Diplomka\Repo\OpenFace_DeepFace_merge\processed"

#Delete previous data
deleteFolderContents(imageDirPathBase)
deleteFolderContents(outDir)

csvFilePath = outDir + r"\images.csv"
checkCSV = False

start = time.time()


frameCount = 0
skippedFrames = 10
dfPredEm = "None"
ofDominantEm = "None"
gSkipHeader = True

args = ["-device", "2", "-cam_width", "640", "-cam_height", "480", "-vis-aus", "-aus", "-out_dir", outDir]

# start the subprocess
process = subprocess.Popen([exePath] + args)

while True:
    files = os.listdir(outDir)
    csvFiles = False
    for f in files:
        if f.endswith(".csv"):
            csvFiles = True
            csvFilePath = os.path.join(outDir, f)
            print(csvFilePath)
    if csvFiles:
        print("CSV file found! Emotion analysis starting...")
        start = time.time()
        break
    else:
        print("No CSV files found.")
        time.sleep(1)


while True:

    frameRecieved, frame = cap.read()
    if frameRecieved:
        frameCount = frameCount + 1
        if frameCount % skippedFrames == 0:
            try:
                result = DeepFace.analyze(frame, actions = ['emotion'], enforce_detection= True)
                dfPredEm = result['dominant_emotion']
            except:
                dfPredEm = "Cannot detect face"
         
            try:
                # Get the current size of the file
                currentSize = os.path.getsize(csvFilePath)
                
                # Check if the file has grown since we last read it
                if currentSize > lastPosition:
                    with open(csvFilePath, 'r') as csvFile:
                        # Move the file pointer to the last position
                        csvFile.seek(lastPosition)
                        
                        # Create a new CSV reader object
                        reader = csv.reader(csvFile)
                        
                        ofDataArr = []
                        # Process the new data in the file
                        for row in reader:
                            # Do something with the row data
                            #print(row)
                            ofDataArr.append(row)
                        ofData = pd.DataFrame(ofDataArr)
                        rows, cols = ofData.shape
                        conf = []

                        if gSkipHeader:
                            aus = ofData.values[1:rows, 5:cols-1]
                            conf = ofData.values[1:rows, 3]
                            gSkipHeader = False
                        else:
                            aus = ofData.values[:, 5:cols-1]
                            conf = ofData.values[:, 3]
                        
                        conf = np.array(conf, dtype=np.float)
                        avgConf = np.mean(conf)
                        
                        if rows > 0:
                            if avgConf < 0.5:
                                ofDominantEm = "Low confidence"
                            else:
                                emPred = decTree.prediction(aus, clf_entropy)
                                ofDominantEm, dominantEmPct = openFace.GetDominantEmotion(emPred)
                                print(ofDominantEm, dominantEmPct)
                       

                        # Update the last position to the current size of the file
                        lastPosition = currentSize
            except FileNotFoundError:
                print("CSV file doesnt exist!")


        frameCopy = np.copy(frame)
        # input image dimensions for the network
        inWidth = 256
        inHeight = 144
        inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                                (0, 0, 0), swapRB=False, crop=False)
        net.setInput(inpBlob)
        output = net.forward()
        points = openPose.GetPoints(output, frame, frameCopy)

        frame = openPose.DrawSkeleton(frame, points, POSE_PAIRS)
  
        frame = displayTableOnFrame(frame, dfPredEm, ofDominantEm, frameCount)
        """
        frame = cv2.putText(frame, dfPredEm,(50,50), cv2.FONT_ITALIC, 1, (0,0,0), 2, cv2.LINE_4)
        frame = cv2.putText(frame, ofDominantEm, (300,50), cv2.FONT_ITALIC, 1, (0,0,255), 2, cv2.LINE_4)
        frame = cv2.putText(frame, str(frameCount), (50,100), cv2.FONT_ITALIC, 1, (0,0,0), 2)
        """
        cv2.imshow('Output-Skeleton', frame)
        #print("Elapsed time: {:.2f} seconds".format(time.time() - start))
        
    else:
        print("Frame not recieved")
    
  
    

    #Get key, if ESC, then end loop
    c = cv2.waitKey(1)
    if c == 27:
        break

end = time.time()
print(end - start, " seconds")
process.terminate()
cap.release()
cv2.destroyAllWindows()



   




