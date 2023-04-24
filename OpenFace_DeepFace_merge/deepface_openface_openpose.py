import time
import cv2  
from deepface import DeepFace
import numpy as np
import openPose
import openFace
import decTree
import gui

import tkinter as tk
from PIL import Image, ImageTk

# Define global variables
startButt = gui.startButt
button2State = gui.button2State
button5State = gui.button5State
button6State = gui.button6State

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
skippedFrames = 5
gDfPredEm = "None"
gOfDomEm = "None"
gOfDomEmPct = 0.0
gFPS = 0

gFearOrSur = False

gFinalEmotion = "None"

testStart = 0
testEnd = 0

gWebcamCanvasShape = gui.gWebcamCanvasShape


process = openFace.featuresExtractionWebcam()
csvFilePath = openFace.checkCSV()







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
    #cv2.namedWindow("Table", cv2.WINDOW_NORMAL)
    #cv2.resizeWindow("Table", windowWidth, windowHeight)

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
    #cv2.imshow("Table", tableImage)
    return tableImage

# Define function to update the image in the GUI
def update_image():
    global img, webcamCanvas, gWebcamCanvasShape
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
    global testStart, testEnd
    global stripe
    global tableCanvas, webcamCanvas

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

        startButt = gui.startButt
        button2State = gui.button2State
        button5State = gui.button5State
        button6State = gui.button6State

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
            stripe = cv2.putText(stripe, "FPS: " + str(gFPS), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            stripe = cv2.putText(stripe, str(gFinalEmotion), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
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
                                              
                    root.update()
                    
            except:
                print("App has been destroyed")
                break 
        
        
        elif not startButt:
            frameFinal = np.concatenate((stripe, frame), axis = 0)
            #print(frameFinal.shape)

            print("Waiting for start...")
            time.sleep(0.1)
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
root = gui.tkInit()
#root = gui.root
tableCanvas = gui.tableCanvas
webcamCanvas = gui.webcamCanvas

# Start the GUI loop and update the image
update_image()
root.mainloop()

cap.release()
openFace.destroyProcess(process)
cv2.destroyAllWindows()



   




