def butt1Cmd(startButt, butt1Text):
    startButt = not startButt
    if butt1Text == "Start":
        butt1Text = "Stop"
    else:
        butt1Text = "Start"
    print(f"Button 1 state: {startButt}")