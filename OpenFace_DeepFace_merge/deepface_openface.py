import subprocess
import pandas as pd
import time
import csv
import os
import cv2  #computer vision library/package?
from deepface import DeepFace

#faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Path to FaceLandmarkImg.exe and image file
exePath  = r"C:\Users\hwojc\Desktop\Diplomka\Open Face\OpenFace_2.2.0_win_x64\FeatureExtraction.exe"
#imagePath  = r"C:\Users\hwojc\Desktop\Diplomka\Repo\OpenFace_DeepFace_merge\images\frame.jpg"
imageDirPath  = r"C:\Users\hwojc\OneDrive - Vysoké učení technické v Brně\Magisterské studium\Diplomka\04 Datasety\AffectNet\Train\angry"

# Output directory
outDir = r"C:\Users\hwojc\Desktop\Diplomka\Open Face\OpenFace_2.2.0_win_x64\processed"

outputFilePath = outDir + r"\angry.csv"
last_modified_time = 0

start = time.time()
# Run FaceLandmarkImg.exe with image file as argument
#process = subprocess.Popen([exePath, "-fdir", imageDirPath , "-aus","-out_dir", outDir], stdout=subprocess.PIPE)
process = subprocess.Popen([exePath, "-fdir", imageDirPath , "-aus","-out_dir", outDir])

while process.poll() is None:
    # Do something else while the process is running
    print("Process is running...")
    try:
        # Get the current modified time of the file
        modified_time = os.path.getmtime(outputFilePath)
        
        # Check if the file has been modified since we last read it
        if modified_time > last_modified_time:
            with open(outputFilePath, 'r') as csv_file:
                reader = csv.reader(csv_file)
                # Process the contents of the CSV file
                for row in reader:
                    # Do something with the row data
                    print(row)
                    
            # Update the last modified time
            last_modified_time = modified_time
        
        # Wait for a short time before checking the file again
        time.sleep(1)
    
    except FileNotFoundError:
        # Handle the case where the file doesn't exist yet
        time.sleep(1)




   




# Wait for the process to finish and capture the output
#output, error = process.communicate()
# Print the output
#print(output.decode("utf-8"))

# Read the output CSV file into a pandas DataFrame
outputFilePath = outDir + r"\angry.csv"
df = pd.read_csv(outputFilePath)

# Print the DataFrame
print(df)

end = time.time()
print(f"Execution time: {end - start} seconds")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while False:
    frameRecieved, frame = cap.read()
    #frame = cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_AREA)
    #model name VGG-Face, Facenet, OpenFace, DeepFace, DeepID, Dlib or Ensemble
    if frameRecieved:
        try:
            result = DeepFace.analyze(frame, actions = ['emotion'], enforce_detection= True)
            cv2.putText(frame,result['dominant_emotion'],(50,50), cv2.FONT_ITALIC, 1, (0,0,0), 2, cv2.LINE_4)
        except:
            cv2.putText(frame,'Cannot detect face',(50,50), cv2.FONT_ITALIC, 1, (0,0,0), 2, cv2.LINE_4)
        cv2.imshow('Video', frame)
    else:
        print("Frame not recieved")
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    
    #Get key, if ESC, then end loop
    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()
