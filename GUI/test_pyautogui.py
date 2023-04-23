import cv2
import numpy
import pyautogui

# Get the initial position and size of the window with the specified title
window_title = "tracking result"
window = pyautogui.getWindowsWithTitle(window_title)[0]
window_position = (window.left, window.top)
window_size = (window.width, window.height)

# Continuously capture and display screenshots
while True:
    # Get the current position and size of the window
    window = pyautogui.getWindowsWithTitle(window_title)[0]
    window_position = (window.left, window.top)
    window_size = (window.width, window.height)

    # Capture a screenshot of the current region of the window
    screenshot = pyautogui.screenshot(region=(*window_position, *window_size))

    # Display the screenshot in a new OpenCV window
    cv2.imshow('External App Window', cv2.cvtColor(numpy.array(screenshot), cv2.COLOR_RGB2BGR))

    # Wait for a key event and check if the 'q' key was pressed to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Close the OpenCV window and release resources
cv2.destroyAllWindows()
