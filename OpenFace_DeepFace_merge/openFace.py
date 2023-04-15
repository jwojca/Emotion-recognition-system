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
args = ["-device", "2", "-cam_width", "640", "-cam_height", "480", "-vis-aus", "-aus", "-out_dir", outDir]




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

def featuresExtraxtionWebcam():
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
                    #print(row)
                    ofDataArr.append(row)
                ofData = pd.DataFrame(ofDataArr)
                rows, cols = ofData.shape
                conf = []

                if gSkipHeader:
                    aus = ofData.values[1:rows, 5:cols-1]
                    conf = ofData.values[1:rows, 3]
                else:
                    aus = ofData.values[:, 5:cols-1]
                    conf = ofData.values[:, 3]
                
                conf = np.array(conf, dtype=np.float)
                avgConf = np.mean(conf)
                
                if rows > 0:
                    if avgConf < 0.5:
                        ofDominantEm = "Low confidence"
                    else:
                        emPred = decTree.prediction(aus, clf_entropy)
                        ofDominantEm, dominantEmPct = GetDominantEmotion(emPred)
                        print(ofDominantEm, dominantEmPct)
                

                # Update the last position to the current size of the file
                lastPosition = currentSize
    except FileNotFoundError:
        print("CSV file doesnt exist!")

    return (ofDominantEm, lastPosition)