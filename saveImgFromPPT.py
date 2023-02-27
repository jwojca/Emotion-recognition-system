import pyautogui as pya
import cv2

x0 = 1152
y0 = 587
time = 0.4
i = 0

for i in range (25):
    # Select 
    pya.leftClick(x0, y0)
    pya.rightClick(x0, y0)

    # Save as
    pya.moveTo(1295, 815, time)
    pya.leftClick()
    pya.moveTo(x0, y0, time)

    # Save
    pya.moveTo(747, 562, time)
    pya.leftClick()

    # Delete
    pya.moveTo(x0, y0, time)
    pya.press('delete')
    ++i







