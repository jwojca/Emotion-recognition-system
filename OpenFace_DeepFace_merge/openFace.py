from collections import Counter
import subprocess
import os
import time
import csv
import decTree
import pandas as pd
import numpy as np

outDir = r"C:\Users\hwojc\Desktop\Diplomka\Repo\OpenFace_DeepFace_merge\processed"
exePath  = r"C:\Users\hwojc\Desktop\Diplomka\Open Face\OpenFace_2.2.0_win_x64\FeatureExtraction.exe"
#args = ["-device", "2", "-cam_width", "640", "-cam_height", "480", "-vis-aus", "-aus", "-out_dir", outDir]
args = ["-device", "2", "-cam_width", "640", "-cam_height", "480", "-vis-aus", "-aus", "-out_dir", "-mloc", "model/main_clnf_general.txt", outDir]

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


def GetDominantEmotion(data):
    """
    Returns the dominant emotion and its percentage in a given array of emotions.

    Args:
        data: A NumPy array of emotions.

    Returns:
        A tuple containing the dominant emotion (string) and its percentage (float).
    """
    counts = Counter(data)
    dominantEmotion = counts.most_common(1)[0][0]
    dominantPercentage = counts[dominantEmotion] / len(data) * 100
    return (dominantEmotion, dominantPercentage)

def featuresExtractionWebcam():
    #Delete previous data
    deleteFolderContents(outDir)
    # start the subprocess
    process = subprocess.Popen([exePath] + args)
    return process

def checkCSV():
    while True:
        files = os.listdir(outDir)
        csvFiles = False
        for f in files:
            if f.endswith(".csv"):
                csvFiles = True
                csvFilePath = os.path.join(outDir, f)
                print(csvFilePath)
        if csvFiles:
            print("CSV file found! Emotion analysis starting...")
            break
        else:
            print("No CSV files found.")
            time.sleep(1)
    return csvFilePath


def predict(csvFilePath, clf_entropy, lastPosition, gSkipHeader):
    ofDominantEm = "None"
    try:
        # Get the current size of the file
        currentSize = os.path.getsize(csvFilePath)
        
        # Check if the file has grown since we last read it
        if currentSize > lastPosition:
            with open(csvFilePath, 'r') as csvFile:
                # Move the file pointer to the last position
                csvFile.seek(lastPosition)
                
                # Create a new CSV reader object
                reader = csv.reader(csvFile)
                
                ofDataArr = []
                # Process the new data in the file
                for row in reader:
                    # Do something with the row data
                    ofDataArr.append(row)
                ofData = pd.DataFrame(ofDataArr)
                rows, cols = ofData.shape
                conf = []

                if gSkipHeader:
                    aus = ofData.values[1:rows, 5:cols]
                    conf = ofData.values[1:rows, 3]
                else:
                    aus = ofData.values[:, 5:cols]
                    conf = ofData.values[:, 3]
                
                conf = np.array(conf, dtype=np.float)
                avgConf = np.mean(conf)
                
                if rows > 0:
                    if avgConf < 0.5:
                        ofDominantEm = "Low confidence"
                        dominantEmPct = 0.0
                    else:
                        emPred = decTree.prediction(aus, clf_entropy)
                        ofDominantEm, dominantEmPct = GetDominantEmotion(emPred)
                        #print(ofDominantEm, dominantEmPct)
                

                # Update the last position to the current size of the file
                lastPosition = currentSize
    except FileNotFoundError:
        print("CSV file doesnt exist!")

    return (ofDominantEm, dominantEmPct, lastPosition)

def destroyProcess(process):
    process.terminate()
