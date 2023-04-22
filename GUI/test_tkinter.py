import tkinter as tk
import cv2
from PIL import Image, ImageTk

# Define global variables
button1_state = False
button2_state = False
button3_state = False

# Define functions for button actions
def toggle_button1():
    global button1_state
    button1_state = not button1_state
    print(f"Button 1 state: {button1_state}")

def toggle_button2():
    global button2_state
    button2_state = not button2_state
    print(f"Button 2 state: {button2_state}")

def toggle_button3():
    global button3_state
    button3_state = not button3_state
    print(f"Button 3 state: {button3_state}")

# Define function to update the image in the GUI
def update_image():
    global img, canvas
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = ImageTk.PhotoImage(Image.fromarray(frame))
    canvas.create_image(0, 0, anchor=tk.NW, image=img)
    root.after(100, update_image)

# Create the tkinter window and canvas
root = tk.Tk()
root.geometry("1920x1080")
canvas = tk.Canvas(root, width=640, height=480)
canvas.pack(side=tk.TOP)

# Create the buttons
button1 = tk.Button(root, text="Button 1", command=toggle_button1)
button1.pack(side=tk.LEFT, padx=10, pady=10)

button2 = tk.Button(root, text="Button 2", command=toggle_button2)
button2.pack(side=tk.LEFT, padx=10, pady=10)

button3 = tk.Button(root, text="Button 3", command=toggle_button3)
button3.pack(side=tk.LEFT, padx=10, pady=10)

# Open the webcam
cap = cv2.VideoCapture(0)

# Start the GUI loop and update the image
update_image()
root.mainloop()

# Release the webcam and destroy all windows
cap.release()
cv2.destroyAllWindows()
