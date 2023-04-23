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

import win32gui
import win32con

import tkinter as tk
from PIL import Image, ImageTk

# Define global variables
startButt = False
button2State = False
button3_state = False
button4State = False
button5State = False
button6State = False

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
gHandsPoints = []
gDfOutput = ('None', 0.0)
gOfOutput = ('None', 0.0)



gFrameCount = 0
skippedFrames = 10
gDfPredEm = "None"
gOfDomEm = "None"
gOfDomEmPct = 0.0
gFPS = 0

gFearOrSur = False

gFinalEmotion = "None"

testStart = 0
testEnd = 0

gWebcamCanvasShape = (640, 420)
gTableCanvasShape = (500, 500)


process = openFace.featuresExtractionWebcam()
csvFilePath = openFace.checkCSV()

# Create a window of size 1920x1080 with color (240, 240, 240)
windowHeight, windowWidth = 1080, 1920
mainWindow = np.zeros((windowHeight, windowWidth, 3), np.uint8)
mainWindow.fill(240)





#TODO as functions

gOfWindow = win32gui.FindWindow(None, "tracking result")
# Check if the window was found
if gOfWindow != 0:
    print("Window found with handle", gOfWindow)
else:
    print("Window not found")

gAusWindow = win32gui.FindWindow(None, "action units")
# Check if the window was found
if gAusWindow != 0:
    print("Window found with handle", gAusWindow)
else:
    print("Window not found")



# Define functions for button actions
def butt1Cmd():
    global startButt
    startButt = not startButt
    if button1["text"] == "Start":
        button1["text"] = "Stop"
    else:
        button1["text"] = "Start"
    print(f"Button 1 state: {startButt}")

def butt2Cmd():
    global button2State
    #global tableCanvas
    button2State = not button2State
    if button2State:
        button2["text"] = "Turn off"
    else:
        button2["text"] = "Turn on"
    print(f"Button 2 state: {button2State}")

    

def butt3Cmd():
    global button3_state
    button3_state = not button3_state

    if button3_state:
        win32gui.ShowWindow(gOfWindow, win32con.SW_SHOWNORMAL)
        left, top, right, bottom = win32gui.GetWindowRect(gOfWindow)
        width, height = right - left, bottom - top
        print(width, height)
        height = 420
        width = 640
        style = win32gui.GetWindowLong(gOfWindow, win32con.GWL_STYLE)
        # Modify the window style to remove the title bar
        style &= ~win32con.WS_CAPTION
        # Set the new window style
        win32gui.SetWindowLong(gOfWindow, win32con.GWL_STYLE, style)
        win32gui.MoveWindow(gOfWindow, 850, 35, width, height, True)
        win32gui.SetWindowPos(gOfWindow, win32con.HWND_TOPMOST, 0, 0 ,0, 0, win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
        button3["text"] = "Turn off"
    else:
        win32gui.ShowWindow(gOfWindow, win32con.SW_MINIMIZE)
        button3["text"] = "Turn on"
    print(f"Button 3 state: {button3_state}")

def butt4Cmd():
    global button4State
    button4State = not button4State
    if button4State:
        win32gui.ShowWindow(gAusWindow, win32con.SW_SHOWNORMAL)
        left, top, right, bottom = win32gui.GetWindowRect(gAusWindow)
        width, height = right - left, bottom - top
        print(width, height)
        width = 540
        height = 370
        style = win32gui.GetWindowLong(gAusWindow, win32con.GWL_STYLE)
        # Modify the window style to remove the title bar
        style &= ~win32con.WS_CAPTION
        # Set the new window style
        win32gui.SetWindowLong(gAusWindow, win32con.GWL_STYLE, style)
        win32gui.MoveWindow(gAusWindow, 850, 450, width, height, True)
        #win32gui.SetWindowPos(gAusWindow, None, 0, 0, 0, 0, win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_NOZORDER | win32con.SWP_FRAMECHANGED)
        win32gui.SetWindowPos(gAusWindow, win32con.HWND_TOPMOST, 0, 0, 0, 0, win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
        button4["text"] = "Turn off"
    else:
        win32gui.ShowWindow(gAusWindow, win32con.SW_MINIMIZE)
        button4["text"] = "Turn on"
    print(f"Button 4 state: {button4}")

def butt5Cmd():
    global button5State
    button5State = not button5State
    if button5State:
        button5["text"] = "Turn off"
    else:
        button5["text"] = "Turn on"
    print(f"Button 5 state: {button5}")

def butt6Cmd():
    global button6State
    button6State = not button6State
    if button6State:
        button6["text"] = "Turn off"
    else:
        button6["text"] = "Turn on"
    print(f"Button 6 state: {button6}")


def drawFrameOnWindow(windowImage, frame, topLeft):

    # Add the frame to the specified position in the window
    x, y = topLeft
    frameHeight, frameWidth = frame.shape[:2]
    windowImage[y:y+frameHeight, x:x+frameWidth] = frame
    return windowImage


def displayTableInWindow(deepfaceOutput, openfaceOutput, finalEmotion, numFrames, handsPoints):
    # Define the table contents
    numOfDec = 2
    table = [
        ["DeepFace output:", str(deepfaceOutput[0]) + ": " + str(round(deepfaceOutput[1], numOfDec)) + "%"],
        ["OpenFace output:", str(openfaceOutput[0]) + ": " + str(round(openfaceOutput[1], numOfDec)) + "%"],
        ["Final emotion:", str(finalEmotion)],
        ["Frames:", str(numFrames)],
        ["RH in Bot Face:", handsPoints[0]],
        ["LH in Bot Face:", handsPoints[1]],
        ["RH in Top Face:", handsPoints[2]],
        ["LH in Top Face:", handsPoints[3]],
        ["RH raised:", handsPoints[4]],
        ["LH raised:", handsPoints[5]],
        ["RH in chest:", handsPoints[6]],
        ["LH in chest:", handsPoints[7]]
    ]

    # Define the font and font scale
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5

    # Define the size of each cell in the table
    cellSize = (180, 25)
    cellSize2 = (250, 25)

    # Define the circle radius
    circleRadius = 5

    # Calculate the size of the window based on the number of rows in the table
    windowWidth = cellSize[0]  + cellSize2[0]
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
    return tableImage

# Define function to update the image in the GUI
def update_image():
    global img, webcamCanvas, gWebcamCanvasShape, gTableCanvasShape
    global gLastPosCsv
    global gSkipCsvHead
    global gHandsPoints
    global gDfOutput
    global gOfOutput
    global gFrameCount
    global skippedFrames 
    global gDfPredEm 
    global gOfDomEm 
    global gOfDomEmPct 
    global gFPS 
    global gFearOrSur 
    global gFinalEmotion 
    global process 
    global csvFilePath 
    global start
    global net
    global cap
    global mainWindow
    global gOfWindow,gAusWindow
    global testStart, testEnd
    global stripe

    start = time.time()
        
    while True:
        emotionDict = {
        "Angry" : 0.0,
        "Disgust" : 0.0,
        "Fear" : 0.0,
        "Happy" : 0.0,
        "Sad" : 0.0,
        "Surprise" : 0.0,
        "Neutral" : 0.0
        }   
        stripe = np.zeros((60, 640, 3), np.uint8)
        stripe = cv2.rectangle(stripe, (0, 0), (640, 60), (0, 0, 0), -1)

        gFearOrSur = False

    
        frameRecieved, frame = cap.read()
        if frameRecieved and startButt:
            global gFrameCount
            gFrameCount = gFrameCount + 1
            if gFrameCount % skippedFrames == 0:
                try:
                    result = DeepFace.analyze(frame, actions = ['emotion'], enforce_detection= True)
                    gDfPredEm = result['dominant_emotion']
                    domEmStr = str(result['dominant_emotion'])
                    gDfPredEmPct = result['emotion'][domEmStr]
                    gDfOutput = (gDfPredEm, gDfPredEmPct)
                    emotionDict[domEmStr] = gDfPredEmPct
                    if gDfPredEm == "Fear" or "Surprise":
                        gFearOrSur = True
                except:
                    gDfPredEm = "Cannot detect face"
                    gDfPredEmPct = 0.0
                    gDfOutput = (gDfPredEm, gDfPredEmPct)
                
                try:
                    gOfDomEm, gOfDomEmPct, gLastPosCsv = openFace.predict(csvFilePath, clf_entropy, gLastPosCsv, gSkipCsvHead)
                    gOfOutput = (gOfDomEm, gOfDomEmPct)
                    gSkipCsvHead = False

                    handInChest = gHandsPoints[6] or gHandsPoints[7]
                    handInBotFace = gHandsPoints[0] or gHandsPoints[1]
                    handsInTopFace = gHandsPoints[2] and gHandsPoints[3]

                    if gOfDomEm == "Fear" or "Surprise":
                        gFearOrSur = True

                    if gOfDomEm != "Low confidence":
                        if handsInTopFace:
                            emotionDict["Surprise"] += 100.0
                        if handInBotFace:
                            if gFearOrSur:
                                emotionDict["Surprise"] += 100.0
                                emotionDict["Fear"] += 100.0
                            else:
                                emotionDict["Surprise"] += 20.0
                                emotionDict["Fear"] += 20.0


                    emotionDict[str(gOfDomEm)] += gOfDomEmPct
                    gFinalEmotion = max(emotionDict, key = emotionDict.get)
                    
                
                    if handInChest and gFearOrSur:
                        emotionDict["Fear"] += 200.0
                    
                    gFinalEmotion = max(emotionDict, key = emotionDict.get)
                except:
                    print("Low confidence")
                    handInBotFace = gHandsPoints[0] or gHandsPoints[1]
                    if handInBotFace:
                        emotionDict["Fear"] += 100.0
                    gFinalEmotion = max(emotionDict, key = emotionDict.get)
            
            testStart = time.time()
            frameCopy = np.copy(frame)
            # input image dimensions for the network
            inWidth = 256
            inHeight = 144
            inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                                    (0, 0, 0), swapRB=False, crop=False)
            net.setInput(inpBlob)
            output = net.forward()
            points = openPose.GetPoints(output, frame)
            if button5State:
             frame = openPose.DrawSkeleton(frame, points)
            frame, gHandsPoints = openPose.handsPos(frame, points, button6State)
    
            #frame = displayTableOnFrame(frame, gDfPredEm, gOfDomEm, gFrameCount)
            if gFrameCount % skippedFrames == 0:
                end = time.time()
                procTime = end - start
                gFPS = round(skippedFrames/procTime)
                start = end

            guiTable = displayTableInWindow(gDfOutput, gOfOutput, gFinalEmotion, gFrameCount, gHandsPoints)
            #frame = cv2.rectangle(frame, (0, 0), (640, 60), (0, 0, 0), -1)
            #frame = cv2.putText(frame, "FPS: " + str(gFPS), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            #frame = cv2.putText(frame, str(gFinalEmotion), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


            
            stripe = cv2.putText(stripe, "FPS: " + str(gFPS), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            stripe = cv2.putText(stripe, str(gFinalEmotion), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            

            #mainWindow = drawFrameOnWindow(mainWindow,frame, (0, 0))
            #mainWindow = drawFrameOnWindow(mainWindow, guiTable, (1000, 0))
            #cv2.putText(tableImage, table[i][j], (x + 5, y + 20), font, fontScale, (255, 255, 255), 1)
            
            #cv2.imshow('Output-Skeleton', mainWindow)

            try:
                frameFinal = np.concatenate((stripe, frame), axis = 0)
                if root.winfo_exists() and webcamCanvas.winfo_exists():
                    #create webcam canvas
                    frameFinal = cv2.cvtColor(frameFinal, cv2.COLOR_BGR2RGB)
                    img = ImageTk.PhotoImage(Image.fromarray(frameFinal))
                    webcamCanvas.create_image(0, 0, anchor=tk.NW, image=img)

                    #create table canvas
                    if button2State:
                        guiTable = cv2.cvtColor(guiTable, cv2.COLOR_BGR2RGB)  
                    else:
                        h, w, c = guiTable.shape    
                        guiTable = np.zeros((h, w, 3), np.uint8)
                        guiTable.fill(240)
                     
                    tableImg = ImageTk.PhotoImage(Image.fromarray(guiTable))
                    tableCanvas.create_image(0, 0, anchor=tk.NW, image=tableImg, tags = 'tableImg')

                
                    #if not button2State and state != 'hidden':     
                    #    tableCanvas.itemconfig('tableImg', state = 'hidden')  
                                              
                    root.update()
                    
            except:
                print("App has been destroyed")
                break 
        
        
        elif not startButt:
            frameFinal = np.concatenate((stripe, frame), axis = 0)
            #print(frameFinal.shape)

            print("Waiting for start...")
            try:
                if root.winfo_exists() and webcamCanvas.winfo_exists():
                    frameFinal = cv2.rectangle(frameFinal, (0, 0), gWebcamCanvasShape, (0, 0, 0), -1)
                    img = ImageTk.PhotoImage(Image.fromarray(frameFinal))
                    webcamCanvas.create_image(0, 0, anchor=tk.NW, image=img)
                    root.update()
            except:
                print("App has been destroyed")
                break 
            
        else:
            print("Frame not recieved")
            break

        
        testEnd = time.time()
        #print(testEnd - testStart)
    
        

        #Get key, if ESC, then end loop
        c = cv2.waitKey(1)
        if c == 27:
            break
       
        



# Create the tkinter window and canvas
root = tk.Tk()
root.geometry("1920x1080")
root.state("zoomed")
root.title('Emotion Recognition System')
webcamPos = (10, 10)

buttPosOrig = (180, 10)
buttYOffset = 40

buttTextPosOrig = (50, 10)


webcamCanvas = tk.Canvas(root, width=gWebcamCanvasShape[0], height=gWebcamCanvasShape[1])
webcamCanvas.place(x = webcamPos[0], y = webcamPos[1])

rectWidth = 250
rectHeight = 350
rectPos = (10, 450)
rectBorder = 1
rectCanvas = tk.Canvas(root, width = rectWidth + 1, height = rectHeight + 1)
rectCanvas.create_rectangle(rectBorder + 1, rectBorder + 1, rectWidth, rectHeight, width = rectBorder)
rectCanvas.create_text(buttTextPosOrig[0], buttTextPosOrig[1], text = "Text text", fill="black", anchor=tk.NW)
rectCanvas.create_text(buttTextPosOrig[0], buttTextPosOrig[1] + buttYOffset, text = "Table", fill="black",anchor=tk.NW)
rectCanvas.create_text(buttTextPosOrig[0], buttTextPosOrig[1] + 2 * buttYOffset, text = "OpenFace webcam", fill="black", anchor=tk.NW)
rectCanvas.create_text(buttTextPosOrig[0], buttTextPosOrig[1] + 3 * buttYOffset, text = "Action units", fill="black", anchor=tk.NW)
rectCanvas.create_text(buttTextPosOrig[0], buttTextPosOrig[1] + 4 * buttYOffset, text = "OpenPose skeleton", fill="black", anchor=tk.NW)
rectCanvas.create_text(buttTextPosOrig[0], buttTextPosOrig[1] + 5 * buttYOffset, text = "Face areas", fill="black", anchor=tk.NW)
rectCanvas.place(x = rectPos[0], y = rectPos[1])

tableCanvas = tk.Canvas(root, width=gTableCanvasShape[0], height=gTableCanvasShape[1])
tableCanvas.place(x = 300, y = 450)


# Create the buttons
button1Pos = (buttPosOrig[0], buttPosOrig[1])
button1 = tk.Button(rectCanvas, text="Start", command=butt1Cmd)
button1.place(x = button1Pos[0], y = button1Pos[1])

button2Pos = (buttPosOrig[0], buttPosOrig[1] + buttYOffset)
button2 = tk.Button(rectCanvas, text="Turn on", command=butt2Cmd)
button2.place(x = button2Pos[0], y = button2Pos[1])

button3Pos = (buttPosOrig[0], buttPosOrig[1] + 2*buttYOffset)
button3 = tk.Button(rectCanvas, text="Turn on", command=butt3Cmd)
button3.place(x = button3Pos[0], y = button3Pos[1])

button4Pos = (buttPosOrig[0], buttPosOrig[1] + 3*buttYOffset)
button4 = tk.Button(rectCanvas, text="Turn on", command=butt4Cmd)
button4.place(x = button4Pos[0], y = button4Pos[1])

button5Pos = (buttPosOrig[0], buttPosOrig[1] + 4*buttYOffset)
button5 = tk.Button(rectCanvas, text="Turn on", command=butt5Cmd)
button5.place(x = button5Pos[0], y = button5Pos[1])

button6Pos = (buttPosOrig[0], buttPosOrig[1] + 5*buttYOffset)
button6 = tk.Button(rectCanvas, text="Turn on", command=butt6Cmd)
button6.place(x = button6Pos[0], y = button6Pos[1])

# Start the GUI loop and update the image
update_image()
root.mainloop()

cap.release()
openFace.destroyProcess(process)
cv2.destroyAllWindows()



   




