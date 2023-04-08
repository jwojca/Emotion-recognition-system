import subprocess
import pandas as pd
import time
import csv
import os
import cv2  
from deepface import DeepFace

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError("Cannot open webcam")

# Path to FaceLandmarkImg.exe and image file
exePath  = r"C:\Users\hwojc\Desktop\Diplomka\Open Face\OpenFace_2.2.0_win_x64\FeatureExtraction.exe"
imageDirPath  = r"C:\Users\hwojc\Desktop\Diplomka\Repo\OpenFace_DeepFace_merge\images"
imgID = 0

# Output directory
outDir = r"C:\Users\hwojc\Desktop\Diplomka\Open Face\OpenFace_2.2.0_win_x64\processed"

outputFilePath = outDir + r"\images.csv"
lastPosition = 0

#save first image
imgName = str(imgID) + '.jpg'
frameRecieved, frame = cap.read()
cv2.imwrite(os.path.join(imageDirPath, imgName), frame)




start = time.time()
# Run FaceLandmarkImg.exe with image file as argument
#process = subprocess.Popen([exePath, "-fdir", imageDirPath , "-aus","-out_dir", outDir], stdout=subprocess.PIPE)
#process = subprocess.Popen([exePath, "-fdir", imageDirPath , "-aus","-out_dir", outDir])

while True:

    frameRecieved, frame = cap.read()
    #frame = cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_AREA)
    #model name VGG-Face, Facenet, OpenFace, DeepFace, DeepID, Dlib or Ensemble
    if frameRecieved:
        try:
            result = DeepFace.analyze(frame, actions = ['emotion'], enforce_detection= True)
            cv2.putText(frame,result['dominant_emotion'],(50,50), cv2.FONT_ITALIC, 1, (0,0,0), 2, cv2.LINE_4)
            imgName = str(imgID) + '.jpg'
            imgPath = os.path.join(imageDirPath, imgName)
            cv2.imwrite(imgPath, frame)
            
            process = subprocess.run([exePath, "-f", imgPath , "-aus","-out_dir", outDir])
            csvName = str(imgID) + '.csv'
            outputFilePath = os.path.join(outDir, csvName)

            imgID = imgID + 1

        except:
            cv2.putText(frame,'Cannot detect face',(50,50), cv2.FONT_ITALIC, 1, (0,0,0), 2, cv2.LINE_4)
        cv2.imshow('Video', frame)
    else:
        print("Frame not recieved")
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    try:
        # Get the current size of the file
     
        with open(outputFilePath, 'r') as csvFile:         
            # Create a new CSV reader object
            reader = csv.reader(csvFile)
            # Process the new data in the file
            for row in reader:
                # Do something with the row data
                print(row)
                    
        # Wait for a short time before checking the file again
        time.sleep(0.1)
        
    except FileNotFoundError:
        # Handle the case where the file doesn't exist yet
        time.sleep(0.1)
    
        
    #Get key, if ESC, then end loop
    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()



   




