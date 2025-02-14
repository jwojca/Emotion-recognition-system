import win32gui
import win32con

import tkinter as tk
from tkinter import *
from tkinter import ttk
import os
import openFace
from PIL import Image, ImageTk
from tkinter import messagebox
import decTree


# Define global variables
startButt = False
button2State = False
button3State = False
button4State = False
button5State = False
button6State = False
button7State = False
button8State = False
button9State = False
gWebcamCanvasShape = (640, 420)
gTableCanvasShape = (500, 500)

gFirstTimeTrain = True

OPTIONS = [
"Angry",
"Happy",
"Sad",
"Surprise",
"Neutral",
"Disgust",
"Fear"
] 


# Define functions for button actions
def butt1Cmd():
    global startButt
    startButt = not startButt
    if button1["text"] == "Start":
        button1["text"] = "Stop"
    else:
        button1["text"] = "Start"
        butt2Cmd()
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
    global button3State
    button3State = not button3State
    window = findWindow("tracking result")

    if window:
        if button3State:
            win32gui.ShowWindow(window, win32con.SW_SHOWNORMAL)
            #left, top, right, bottom = win32gui.GetWindowRect(window)
            #width, height = right - left, bottom - top
            #print(width, height)
            position = (800, 35)
            height = 420
            width = 640
            style = win32gui.GetWindowLong(window, win32con.GWL_STYLE)
            # Modify the window style to remove the title bar
            style &= ~win32con.WS_CAPTION
            # Set the new window style
            win32gui.SetWindowLong(window, win32con.GWL_STYLE, style)
            win32gui.MoveWindow(window, position[0], position[1], width, height, True)
            win32gui.SetWindowPos(window, win32con.HWND_TOPMOST, 0, 0 ,0, 0, win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
            button3["text"] = "Turn off"
        else:
            win32gui.ShowWindow(window, win32con.SW_MINIMIZE)
            button3["text"] = "Turn on"
    print(f"Button 3 state: {button3State}")

def butt4Cmd():
    global button4State
    button4State = not button4State
    window = findWindow("action units")
    if window:
        if button4State:
            win32gui.ShowWindow(window, win32con.SW_SHOWNORMAL)
            left, top, right, bottom = win32gui.GetWindowRect(window)
            width, height = right - left, bottom - top
            print(width, height)
            position = (800, 455)
            width = 585
            height = 370
            style = win32gui.GetWindowLong(window, win32con.GWL_STYLE)
            # Modify the window style to remove the title bar
            style &= ~win32con.WS_CAPTION
            # Set the new window style
            win32gui.SetWindowLong(window, win32con.GWL_STYLE, style)
            win32gui.MoveWindow(window, position[0], position[1], width, height, True)
            win32gui.SetWindowPos(window, win32con.HWND_TOPMOST, 0, 0, 0, 0, win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
            button4["text"] = "Turn off"
        else:
            win32gui.ShowWindow(window, win32con.SW_MINIMIZE)
            button4["text"] = "Turn on"
    print(f"Button 4 state: {button4State}")

def butt5Cmd():
    global button5State
    button5State = not button5State
    if button5State:
        button5["text"] = "Turn off"
    else:
        button5["text"] = "Turn on"
    print(f"Button 5 state: {button5State}")

def butt6Cmd():
    global button6State
    button6State = not button6State
    if button6State:
        button6["text"] = "Turn off"
    else:
        button6["text"] = "Turn on"
    print(f"Button 6 state: {button6State}")

def butt7Cmd(analyzeCanvas, trainCanvas, rectPos, headerCanvas):
    global button7State
    button7State = not button7State
    
    if button7State:
        button7["text"] = "To analyze"
        analyzeCanvas.place_forget()
        trainCanvas.place(x = rectPos[0], y = rectPos[1])
        headerCanvas.itemconfig(modeTxtId, text = "Train mode")
    else:
        analyzeCanvas.config(state='normal')
        button7["text"] = "To train"
        analyzeCanvas.place(x = rectPos[0], y = rectPos[1])
        trainCanvas.place_forget()
        headerCanvas.itemconfig(modeTxtId, text = "Analyze mode")
    print(f"Button 7 state: {button7State}")

def butt8Cmd(emOption):

    #Defining window parameters
    window1 = findWindow("tracking result")
    position1 = (800, 35)
    height1 = 420
    width1 = 640
    
    window2 = findWindow("action units")
    position2 = (800, 455)
    width2 = 585
    height2 = 370

    #When trained for first time - create folder
    global gFirstTimeTrain
    if gFirstTimeTrain:
        openFace.createCustomCsv()
        gFirstTimeTrain = False

    #Get selected emotion
    selectedEmotion = emOption.get()
    
    #Inform user that the record will start 
    messagebox.showinfo("Info", "Recording of " + selectedEmotion + " emotion will follow after hitting OK button. \
    Recording will take approximately 8 seconds. Perform the emotion until the camera output hides.")

    #Show OpenFace output to user
    showWindow(window1, position1, width1, height1)
    showWindow(window2, position2, width2, height2)

    #Check if csv exists and write data to custom csv file
    csvFilePath = openFace.checkCSV()
    openFace.writeToCustomCSV(csvFilePath, selectedEmotion)
    
    #Hide windows
    hideWindow(window1)
    hideWindow(window2)

    #Inform user that the recording was succesful 
    messagebox.showinfo("Info", "Succesfully recorded emotion " + selectedEmotion + " . For updating tree use Train button.")

def butt9Cmd():
    global button9State
    global trainedTree
    trainedTree = decTree.trainTree(FALSE)
    button9State = True
    messagebox.showinfo("Info", "Decision tree was trained.")
    print(f"Button 9 state: {button9State}")

def findWindow(windowName):
    window = win32gui.FindWindow(None, windowName)
    # Check if the window was found
    if window != 0:
        print("Window found with handle", window)
    else:
        print("Window not found")
    return window

def showWindow(window, position, width, height):
    win32gui.ShowWindow(window, win32con.SW_SHOWNORMAL)
    style = win32gui.GetWindowLong(window, win32con.GWL_STYLE)
    style &= ~win32con.WS_CAPTION
    win32gui.SetWindowLong(window, win32con.GWL_STYLE, style)
    win32gui.MoveWindow(window, position[0], position[1], width, height, True)
    win32gui.SetWindowPos(window, win32con.HWND_TOPMOST, 0, 0, 0, 0, win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)

def hideWindow(window):
    win32gui.ShowWindow(window, win32con.SW_MINIMIZE)

def tkInit():
    global root, tableCanvas, webcamCanvas, analyzeCanvas
    global button1, button2, button3, button4, button5, button6, button7, button8, button9
    global modeTxtId

    buttPosOrig = (130, 7)
    buttYOffset = 40
    buttTextPosOrig = (5, 10)
    webcamPos = (90, 10)

    root = tk.Tk()
    root.geometry("1920x1080")
    root.state("zoomed")
    root.title('Emotion Recognition System')  
    
    #Webcam canvas
    webcamCanvas = tk.Canvas(root, width=gWebcamCanvasShape[0], height=gWebcamCanvasShape[1])
    webcamCanvas.place(x = webcamPos[0], y = webcamPos[1])

    #Table canvas
    tableCanvas = tk.Canvas(root, width=gTableCanvasShape[0], height=gTableCanvasShape[1])
    tableCanvas.place(x = 316, y = 450)
    
    #Analyze canvas
    rectWidth = 200
    rectHeight = 350
    rectPos = (85, 520)
    rectBorder = 0
    analyzeCanvas = tk.Canvas(root, width = rectWidth + 1, height = rectHeight + 1)
    analyzeCanvas.create_rectangle(rectBorder + 1, rectBorder + 1, rectWidth, rectHeight, width = rectBorder)
    analyzeCanvas.create_text(buttTextPosOrig[0], buttTextPosOrig[1], text = "Start button", fill="black", anchor=tk.NW)
    analyzeCanvas.create_text(buttTextPosOrig[0], buttTextPosOrig[1] + buttYOffset, text = "Table", fill="black",anchor=tk.NW)
    analyzeCanvas.create_text(buttTextPosOrig[0], buttTextPosOrig[1] + 2 * buttYOffset, text = "OpenFace webcam", fill="black", anchor=tk.NW)
    analyzeCanvas.create_text(buttTextPosOrig[0], buttTextPosOrig[1] + 3 * buttYOffset, text = "Action units", fill="black", anchor=tk.NW)
    analyzeCanvas.create_text(buttTextPosOrig[0], buttTextPosOrig[1] + 4 * buttYOffset, text = "OpenPose skeleton", fill="black", anchor=tk.NW)
    analyzeCanvas.create_text(buttTextPosOrig[0], buttTextPosOrig[1] + 5 * buttYOffset, text = "Face areas", fill="black", anchor=tk.NW)
    analyzeCanvas.place(x = rectPos[0], y = rectPos[1])

    button1Pos = (buttPosOrig[0], buttPosOrig[1])
    button1 = tk.Button(analyzeCanvas, text= "Start" , command = butt1Cmd)
    button1.place(x = button1Pos[0], y = button1Pos[1])

    button2Pos = (buttPosOrig[0], buttPosOrig[1] + buttYOffset)
    button2 = tk.Button(analyzeCanvas, text="Turn on", command=butt2Cmd)
    button2.place(x = button2Pos[0], y = button2Pos[1])

    button3Pos = (buttPosOrig[0], buttPosOrig[1] + 2*buttYOffset)
    button3 = tk.Button(analyzeCanvas, text="Turn on", command=butt3Cmd)
    button3.place(x = button3Pos[0], y = button3Pos[1])

    button4Pos = (buttPosOrig[0], buttPosOrig[1] + 3*buttYOffset)
    button4 = tk.Button(analyzeCanvas, text="Turn on", command=butt4Cmd)
    button4.place(x = button4Pos[0], y = button4Pos[1])

    button5Pos = (buttPosOrig[0], buttPosOrig[1] + 4*buttYOffset)
    button5 = tk.Button(analyzeCanvas, text="Turn on", command=butt5Cmd)
    button5.place(x = button5Pos[0], y = button5Pos[1])

    button6Pos = (buttPosOrig[0], buttPosOrig[1] + 5*buttYOffset)
    button6 = tk.Button(analyzeCanvas, text="Turn on", command=butt6Cmd)
    button6.place(x = button6Pos[0], y = button6Pos[1])


    #Header canvas
    headerHeight = 80
    headerPos = (rectPos[0], 440)
    headerCanvas = tk.Canvas(root, width=rectWidth, height= headerHeight)
    headerCanvas.place(x = headerPos[0], y = headerPos[1])
    headerCanvas.create_text(buttTextPosOrig[0], buttTextPosOrig[1], text = "Change mode", fill="black", anchor=tk.NW)
    modeTxtId = headerCanvas.create_text(buttTextPosOrig[0], buttTextPosOrig[1] + buttYOffset, text = "Analyze mode", font=('freemono', 11,'bold'), fill="black", anchor=tk.NW)

    button7Pos = (buttPosOrig[0], buttPosOrig[1])
    button7 = tk.Button(headerCanvas, text="To train", command= lambda: butt7Cmd(analyzeCanvas, trainCanvas, rectPos, headerCanvas))
    button7.place(x = button7Pos[0], y = button7Pos[1])

    #Train canvas
    trainCanvas = tk.Canvas(root, width = rectWidth + 1, height = rectHeight + 1)
    trainCanvas.create_text(buttTextPosOrig[0], buttTextPosOrig[1], text = "Select emotion", fill="black", anchor=tk.NW)
    trainCanvas.create_text(buttTextPosOrig[0], buttTextPosOrig[1] + buttYOffset, text = "Start recording", fill="black",anchor=tk.NW)
    trainCanvas.create_text(buttTextPosOrig[0], buttTextPosOrig[1] + 2 * buttYOffset, text = "Train decision tree", fill="black",anchor=tk.NW)

    emOption = StringVar(trainCanvas)
    emOption.set(OPTIONS[0])
    dropDown = OptionMenu(trainCanvas, emOption, *OPTIONS)
    dropDown.place(x = buttPosOrig[0], y = buttPosOrig[1] - 3)

    button8Pos = (buttPosOrig[0], buttPosOrig[1] + buttYOffset)
    button8 = tk.Button(trainCanvas, text="Start", command= lambda: butt8Cmd(emOption))
    button8.place(x = button8Pos[0], y = button8Pos[1])

    button9Pos = (buttPosOrig[0], buttPosOrig[1] + 2 * buttYOffset)
    button9 = tk.Button(trainCanvas, text="Train", command= butt9Cmd)
    button9.place(x = button9Pos[0], y = button9Pos[1])

    return root
