import win32gui
import win32con

import tkinter as tk
from PIL import Image, ImageTk

# Define global variables
startButt = False
button2State = False
button3State = False
button4State = False
button5State = False
button6State = False
gWebcamCanvasShape = (640, 420)
gTableCanvasShape = (500, 500)


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

def findWindow(windowName):
    window = win32gui.FindWindow(None, windowName)
    # Check if the window was found
    if window != 0:
        print("Window found with handle", window)
    else:
        print("Window not found")
    return window

def tkInit():
    global root, tableCanvas, webcamCanvas
    global button1, button2, button3, button4, button5, button6

    root = tk.Tk()
    root.geometry("1920x1080")
    root.state("zoomed")
    root.title('Emotion Recognition System')
    webcamPos = (90, 10)

    buttPosOrig = (180, 10)
    buttYOffset = 40

    buttTextPosOrig = (50, 10)


    webcamCanvas = tk.Canvas(root, width=gWebcamCanvasShape[0], height=gWebcamCanvasShape[1])
    webcamCanvas.place(x = webcamPos[0], y = webcamPos[1])

    rectWidth = 250
    rectHeight = 350
    rectPos = (40, 450)
    rectBorder = 0
    rectCanvas = tk.Canvas(root, width = rectWidth + 1, height = rectHeight + 1)
    rectCanvas.create_rectangle(rectBorder + 1, rectBorder + 1, rectWidth, rectHeight, width = rectBorder)
    rectCanvas.create_text(buttTextPosOrig[0], buttTextPosOrig[1], text = "Start button", fill="black", anchor=tk.NW)
    rectCanvas.create_text(buttTextPosOrig[0], buttTextPosOrig[1] + buttYOffset, text = "Table", fill="black",anchor=tk.NW)
    rectCanvas.create_text(buttTextPosOrig[0], buttTextPosOrig[1] + 2 * buttYOffset, text = "OpenFace webcam", fill="black", anchor=tk.NW)
    rectCanvas.create_text(buttTextPosOrig[0], buttTextPosOrig[1] + 3 * buttYOffset, text = "Action units", fill="black", anchor=tk.NW)
    rectCanvas.create_text(buttTextPosOrig[0], buttTextPosOrig[1] + 4 * buttYOffset, text = "OpenPose skeleton", fill="black", anchor=tk.NW)
    rectCanvas.create_text(buttTextPosOrig[0], buttTextPosOrig[1] + 5 * buttYOffset, text = "Face areas", fill="black", anchor=tk.NW)
    rectCanvas.place(x = rectPos[0], y = rectPos[1])

    tableCanvas = tk.Canvas(root, width=gTableCanvasShape[0], height=gTableCanvasShape[1])
    tableCanvas.place(x = 316, y = 450)


    # Create the buttons
    button1Pos = (buttPosOrig[0], buttPosOrig[1])
    button1 = tk.Button(rectCanvas, text= "Start" , command = butt1Cmd)
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
    return root
