import subprocess
import pandas as pd
import time

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
net = openPose.loadModel()

#Init webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# Path to FaceLandmarkImg.exe and image file
#exePath  = r"C:\Users\hwojc\Desktop\Diplomka\Open Face\OpenFace_2.2.0_win_x64\FeatureExtraction.exe"

lastPosition = 0

# Output directory
outDir = r"C:\Users\hwojc\Desktop\Diplomka\Repo\OpenFace_DeepFace_merge\processed"

#Delete previous data
deleteFolderContents(outDir)

csvFilePath = outDir


start = time.time()


frameCount = 0
skippedFrames = 10
dfPredEm = "None"
ofDominantEm = "None"
gSkipHeader = True

process = openFace.featuresExtraxtionWebcam()
csvFilePath = openFace.checkCSV()


start = time.time()
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
            ofDominantEm, lastPosition = openFace.predict(csvFilePath, clf_entropy, lastPosition, gSkipHeader)
            gSkipHeader = False
         
            
        frameCopy = np.copy(frame)
        # input image dimensions for the network
        inWidth = 256
        inHeight = 144
        inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                                (0, 0, 0), swapRB=False, crop=False)
        net.setInput(inpBlob)
        output = net.forward()
        points = openPose.GetPoints(output, frame, frameCopy)
        frame = openPose.DrawSkeleton(frame, points)
  
        frame = displayTableOnFrame(frame, dfPredEm, ofDominantEm, frameCount)

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



   




