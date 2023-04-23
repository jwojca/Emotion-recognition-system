import win32gui
import win32con

# Find the handle of the window with the title "tracking result"
hwnd = win32gui.FindWindow(None, "tracking result")
left, top, right, bottom = win32gui.GetWindowRect(hwnd)
width, height = right - left, bottom - top

# Check if the window was found
if hwnd != 0:
    print("Window found with handle", hwnd)
else:
    print("Window not found")

while True:
    win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 200,100,0,0, win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
    win32gui.MoveWindow(hwnd, 0, 0, width, height, True)
