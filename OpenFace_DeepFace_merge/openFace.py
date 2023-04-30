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


#TODO make relative path
outDir = r"C:\Users\hwojc\Desktop\Diplomka\Repo\OpenFace_DeepFace_merge\processed"
exePath  = r"C:\Users\hwojc\Desktop\Diplomka\Open Face\OpenFace_2.2.0_win_x64\FeatureExtraction.exe"



filePath = os.path.dirname(__file__)
customTrainCsvPath = os.path.join(filePath, r'csv\custom\trainCustom.csv')
customTestCsvPath = os.path.join(filePath, r'csv\custom\testCustom.csv')

baseTrainCsvPath = os.path.join(filePath, r'csv\base\trainBase.csv')
baseTestCsvPath = os.path.join(filePath, r'csv\base\testBase.csv')

def deleteFolderContents(folder_path):
    """
    Deletes all files and subdirectories inside a folder, but not the folder itself.
    Args:
        folder_path: Path of directory which should be empty.

    Returns:
        Nothing
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
    """
    Starts OpenFace analysis on external webcam via command prompt. 

    Args:
        None

    Returns:
        A process object.
    """
    #Delete previous data
    deleteFolderContents(outDir)

    # start the subprocess
    args = ["-device", "2", "-cam_width", "640", "-cam_height", "480", "-vis-aus", "-aus", "-out_dir", "-mloc", "model/main_clnf_general.txt", outDir]
    process = subprocess.Popen([exePath] + args)
    return process

def checkCSV():
    """
    Checks if OpenFace process already created a csv file with its output data.

    Args:
        None

    Returns:
        A path of csv file.
    """
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


def predict(csvFilePath, treeClass, lastPosition, skipHeader):
    """
    Reads OpenFace output csv file and predict the emotion based on decision tree classification.
    It reads only new frames since last read.

    Args:
        csvFilePath: Path of OpenFace csv file
        treeClass: Decision tree object used to predictions
        lastPosition: Last position of csv file read
        skipHeader: Boolean variable used for skip header

    Returns:
        A tuple
            ofDominantEm: Most frequent emotion from all read frames.
            dominantEmPct: Percentage of occurence of dominant emotion.
            lastPosition: End position of csv file for next read.
    """

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
                
                # Process the new data in the file
                ofDataArr = []
                for row in reader:
                    ofDataArr.append(row)

                ofData = pd.DataFrame(ofDataArr)
                rows, cols = ofData.shape
                conf = []

                #If header was read, skip it
                if skipHeader:
                    aus = ofData.values[1:rows, 5:cols]
                    conf = ofData.values[1:rows, 3]
                else:
                    aus = ofData.values[:, 5:cols]
                    conf = ofData.values[:, 3]
                
                #Calculate average confidence
                conf = np.array(conf, dtype=np.float)
                avgConf = np.mean(conf)
                
                if rows > 0:
                    if avgConf < 0.5:
                        ofDominantEm = "Low confidence eee"
                        dominantEmPct = 0.0
                    else:
                        emPred = decTree.prediction(aus, treeClass)
                        ofDominantEm, dominantEmPct = GetDominantEmotion(emPred)
            
                # Update the last position to the current size of the file
                lastPosition = currentSize
    except FileNotFoundError:
        print("CSV file doesnt exist!")

    return (ofDominantEm, dominantEmPct, lastPosition)


def createCustomCsv():
    """
    Function for asking user to create new folder and copy train and test csv files into the folder.
    It also updates path (Global variables) of the files to be used in the app.

    Args:
        None
    Returns:
        Nothing
    """

    #Ask to create new directory
    initDir = os.path.join(filePath, r'csv')
    dirPath = fd.askdirectory(title='Create new directory', initialdir = initDir)

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
    """
    Function that writes data to custom train and test csv files used to train the decision tree.
    Default is 200 samples for train and 50 samples for test for each emotion.
    Args:
        csvReadPath: Path of OpenFace output csv file
        emotion: Selected emotion to be recorded to new files.
    Returns:
        Nothing
    """
   
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
    Gets position to which data should be written (based on selected emotion).
    For example for default training file of 200 samples per emotion
          0        : Header
          1 -  200 : Angry
        201 -  400 : Disgust
        401 -  600 : Fear
        601 -  800 : Happy
        801 - 1000 : Neutral
       1001 - 1200 : Sad
       1201 - 1400 : Surprise

    Args:
        numOfSamples: Number of samples for emotion class.
        emotionStr: String value of selected emotion.
    Returns:
        Integer value of the position
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
    """ 
    Terminate process.

    Args:
        process: Object of process, which should be terminated.
    Returns:
        Nothing
    """ 
    process.terminate()
