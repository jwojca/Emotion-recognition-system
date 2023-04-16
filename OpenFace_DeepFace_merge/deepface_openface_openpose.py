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

def displayTableInWindow(deepfaceOutput, openfaceOutput, numFrames, handsPoints):
    # Define the table contents
    table = [
        ["DeepFace output:", str(deepfaceOutput)],
        ["OpenFace output:", str(openfaceOutput)],
        ["Frames:", str(numFrames)],
        ["RH in Face:", handsPoints[0]],
        ["LH in Face:", handsPoints[1]],
        ["RH raised:", handsPoints[2]],
        ["LH raised:", handsPoints[3]]
    ]

    # Define the font and font scale
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5

    # Define the size of each cell in the table
    cellSize = (180, 25)

    # Define the circle radius
    circleRadius = 5

    # Calculate the size of the window based on the number of rows in the table
    windowWidth = cellSize[0] * 2
    windowHeight = cellSize[1] * len(table)

    # Create a new window using OpenCV
    cv2.namedWindow("Table", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Table", windowWidth, windowHeight)

    # Create an empty image for the table
    tableImage = np.zeros((windowHeight, windowWidth, 3), dtype=np.uint8)

    # Loop over the rows and columns of the table
    for i in range(len(table)):
        for j in range(2):
            # Define the position of the cell
            x = j * cellSize[0]
            y = i * cellSize[1]

            # Draw the black rectangle behind the text
            cv2.rectangle(tableImage, (x, y), (x + cellSize[0], y + cellSize[1]), (0, 0, 0), -1)

            

            # Draw the circle for boolean values
            if j == 1 and isinstance(table[i][j], bool):
                circleCenter = (x + circleRadius, y + 15)
                if table[i][j]:
                    cv2.circle(tableImage, circleCenter, circleRadius, (0, 255, 0), -1)
                else:
                    cv2.circle(tableImage, circleCenter, circleRadius, (0, 0, 255), 1)
            else:
                # Draw the text in the cell
                 cv2.putText(tableImage, table[i][j], (x + 5, y + 20), font, fontScale, (255, 255, 255), 1)

    # Display the table in the window
    cv2.imshow("Table", tableImage)





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

gLastPosCsv = 0
gSkipCsvHead = True
gHandsPoints = [False, False, False]

start = time.time()


frameCount = 0
skippedFrames = 10
dfPredEm = "None"
ofDominantEm = "None"


process = openFace.featuresExtractionWebcam()
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
            ofDominantEm, gLastPosCsv = openFace.predict(csvFilePath, clf_entropy, gLastPosCsv, gSkipCsvHead)
            gSkipCsvHead = False
         
            
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
        frame, gHandsPoints = openPose.handsPos(frame, points)
  
        frame = displayTableOnFrame(frame, dfPredEm, ofDominantEm, frameCount)
        displayTableInWindow(dfPredEm, ofDominantEm, frameCount, gHandsPoints)

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



   




