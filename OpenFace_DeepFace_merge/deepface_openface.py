import subprocess
import pandas as pd
import time
import csv
import os
import cv2  
from deepface import DeepFace

def deleteFolderContents(folder_path):
    """
    Deletes all files and subdirectories inside a folder, but not the folder itself.
    """
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)  # remove file
            elif os.path.isdir(file_path):
                deleteFolderContents(file_path)  # recurse and remove subdirectories
                os.rmdir(file_path)  # remove directory
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# Path to FaceLandmarkImg.exe and image file
#exePath  = r"C:\Users\hwojc\Desktop\Diplomka\Open Face\OpenFace_2.2.0_win_x64\FeatureExtraction.exe"
exePath  = r"C:\Users\hwojc\Desktop\Diplomka\Open Face\OpenFace_2.2.0_win_x64\FeatureExtraction.exe"

imageDirPathBase  = r"C:\Users\hwojc\Desktop\Diplomka\Repo\OpenFace_DeepFace_merge\images"
imgID = 0

# Output directory
outDir = r"C:\Users\hwojc\Desktop\Diplomka\Repo\OpenFace_DeepFace_merge\processed"

#Delete previous data
deleteFolderContents(imageDirPathBase)
deleteFolderContents(outDir)

outputFilePath = outDir + r"\images.csv"
lastPosition = 0
checkCSV = False
gCheckProc = False

#save first image
"""imgName = str(imgID) + '.jpg'
frameRecieved, frame = cap.read()
cv2.imwrite(os.path.join(imageDirPath, imgName), frame)"""

start = time.time()

imgDirIndex = 1
imgSavedCount = 0
imageDirPath =  os.path.join(imageDirPathBase, str(imgDirIndex))
if not os.path.exists(imageDirPath):
    os.makedirs(imageDirPath)


while True:

    frameRecieved, frame = cap.read()
    if frameRecieved:
        try:
            result = DeepFace.analyze(frame, actions = ['emotion'], enforce_detection= True)
            frame = cv2.putText(frame, result['dominant_emotion'],(50,50), cv2.FONT_ITALIC, 1, (0,0,0), 2, cv2.LINE_4)
            frame = cv2.putText(frame, str(imgID), (50,100), cv2.FONT_ITALIC, 1, (0,0,0), 2)

            imgName = str(imgID) + '.jpg'
            imgPath = os.path.join(imageDirPath, imgName)
            cv2.imwrite(imgPath, frame)
            imgSavedCount = imgSavedCount + 1
            imgID = imgID + 1

            if(imgSavedCount % 30 == 0):
                #process = subprocess.run([exePath, "-fdir", imageDirPath , "-aus","-out_dir", outDir])
                process = subprocess.Popen([exePath, "-fdir", imageDirPath , "-aus","-out_dir", outDir])
                csvName = str(imgDirIndex) + '.csv'
                outputFilePath = os.path.join(outDir, csvName)
                imgDirIndex = imgDirIndex + 1
                imgSavedCount = 0
                imageDirPath = os.path.join(imageDirPathBase, str(imgDirIndex))
                if not os.path.exists(imageDirPath):
                    os.makedirs(imageDirPath)
                gCheckProc = True
            cv2.imshow('Video', frame)
        except:
            frame = cv2.putText(frame,'Cannot detect face',(50,50), cv2.FONT_ITALIC, 1, (0,0,0), 2, cv2.LINE_4)
            cv2.imshow('Video', frame)

    else:
        print("Frame not recieved")
        
    """
    try:
        with open(outputFilePath, 'r') as csvFile:         
            # Create a new CSV reader object
            reader = csv.reader(csvFile)
            # Process the new data in the file
            for row in reader:
                # Do something with the row data
                print(row)         
        # Wait for a short time before checking the file again
        #time.sleep(0.1)
        
    except FileNotFoundError:
        # Handle the case where the file doesn't exist yet
        #time.sleep(0.1)
        print("CSV doesnt exist yet")
    """
    #Get key, if ESC, then end loop
    c = cv2.waitKey(1)
    if c == 27:
        break

end = time.time()
print(end - start, " seconds")
cap.release()
cv2.destroyAllWindows()



   




