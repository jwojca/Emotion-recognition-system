from collections import Counter
import subprocess
import os
import time
import csv
import decTree
import pandas as pd
import numpy as np
import shutil
from tkinter import filedialog as fd

outDir = r"C:\Users\hwojc\Desktop\Diplomka\Repo\OpenFace_DeepFace_merge\processed"
exePath  = r"C:\Users\hwojc\Desktop\Diplomka\Open Face\OpenFace_2.2.0_win_x64\FeatureExtraction.exe"
#args = ["-device", "2", "-cam_width", "640", "-cam_height", "480", "-vis-aus", "-aus", "-out_dir", outDir]
args = ["-device", "2", "-cam_width", "640", "-cam_height", "480", "-vis-aus", "-aus", "-out_dir", "-mloc", "model/main_clnf_general.txt", outDir]

filePath = os.path.dirname(__file__)
customTrainCsvPath = os.path.join(filePath, r'csv\custom\trainCustom.csv')
customTestCsvPath = os.path.join(filePath, r'csv\custom\testCustom.csv')

baseTrainCsvPath = os.path.join(filePath, r'csv\base\trainBase.csv')
baseTestCsvPath = os.path.join(filePath, r'csv\base\testBase.csv')

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


def createCustomCsv():

    #Ask to create new directory
    initDir = os.path.join(filePath, r'csv')
    dirPath = fd.askdirectory(title='Create new directorz', initialdir = initDir)

    #Create destination path of csv file
    newTestCsvPath = os.path.join(dirPath, r'testCustom.csv')
    newTrainCsvPath = os.path.join(dirPath, r'trainCustom.csv')
    
    #Copy files
    shutil.copy(baseTestCsvPath, newTestCsvPath)
    shutil.copy(baseTrainCsvPath, newTrainCsvPath)

    #Update path of custom csv files
    global customTestCsvPath
    global customTrainCsvPath
    customTestCsvPath = newTestCsvPath
    customTrainCsvPath = newTrainCsvPath


def writeToCustomCSV(csvReadPath, emotion):
   
    # Get the current size of the file
    lastPosition = os.path.getsize(csvReadPath)
    trainSamples = 200
    testSamples = 50
    numOfSamples = trainSamples + testSamples
    writtenSamples = 0
    trainDataArr = []
    testDataArr = []

    while True:
        time.sleep(0.3)
        try:
            currentSize = os.path.getsize(csvReadPath)
            # Check if the file has grown since we last read it
            if (currentSize > lastPosition):
                with open(csvReadPath, 'r') as readFile:
                    # Move the file pointer to the last position
                    readFile.seek(lastPosition)

                    # Create a new CSV reader and writer object
                    reader = csv.reader(readFile)

                    # Process the new data in the file
                    for row in reader:
                        #append selected emotion to csv file
                        row.append(emotion)
                        if writtenSamples < trainSamples:
                            trainDataArr.append(row)
                        else:
                            testDataArr.append(row)
                        writtenSamples += 1
                        if writtenSamples >= numOfSamples:
                            break
                       
                    lastPosition = currentSize

        except FileNotFoundError:
            print("CSV file doesnt exist!")

        prevData = []
        prevTestData = []
        if writtenSamples >= numOfSamples:
            #Write to train csv file
            with open(customTrainCsvPath, 'r') as writeFile:
                reader = csv.reader(writeFile)
                position = getCsvPos(trainSamples, emotion)
                for row in reader:
                    prevData.append(row)

            with open(customTrainCsvPath, 'w', newline='') as writeFile:
                writer = csv.writer(writeFile, delimiter = ',')
                for row in prevData[:position]:
                    writer.writerow(row)
                for row in trainDataArr:
                    writer.writerow(row)
                for row in prevData[position + trainSamples:]:
                    writer.writerow(row)
            print("Saved to custom train CSV")

            #Write to test csv file
            with open(customTestCsvPath, 'r') as writeFile:
                reader = csv.reader(writeFile)
                position = getCsvPos(testSamples, emotion)
                for row in reader:
                    prevTestData.append(row)

            with open(customTestCsvPath, 'w', newline='') as writeFile:
                writer = csv.writer(writeFile, delimiter = ',')
                for row in prevTestData[:position]:
                    writer.writerow(row)
                for row in testDataArr:
                    writer.writerow(row)
                for row in prevTestData[position + testSamples:]:
                    writer.writerow(row)
            print("Saved to custom test CSV")
            break

   
def getCsvPos(numOfSamples, emotionStr):

    """ 
          0        : Header
          1 -  200 : Angry
        201 -  400 : Disgust
        401 -  600 : Fear
        601 -  800 : Happy
        801 - 1000 : Neutral
       1001 - 1200 : Sad
       1201 - 1400 : Surprise
    """ 
    
    if emotionStr == "Angry":
        position = 0 * numOfSamples
    elif emotionStr == "Disgust":
        position = 1 * numOfSamples
    elif emotionStr == "Fear":
        position = 2 * numOfSamples
    elif emotionStr == "Happy":
        position = 3 * numOfSamples
    elif emotionStr == "Neutral":
        position = 4 * numOfSamples
    elif emotionStr == "Sad":
        position = 5 * numOfSamples
    elif emotionStr == "Surprise":
        position = 6 * numOfSamples
    else:
        print("Wrong emotion string argument")
    
    #Adding the header
    position += 1 
    return position
    

    
    

    


def destroyProcess(process):
    process.terminate()
