# Run this program on your local python
# interpreter, provided you have installed
# the required libraries.

# Importing the required packages
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.tree import plot_tree

import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay

plotOn = True
loopingOn = False

trainCSV = r'C:\Users\hwojc\Desktop\Diplomka\Open Face\Excel data\Excel data augmented\test 15_03_2023\test_highConf_AuAr_withoutDisgust.csv'
testCSV =  r'C:\Users\hwojc\Desktop\Diplomka\Open Face\Excel data\Excel data augmented\train 15_03_2023\train_highConf_AuAr_withoutDisgust.csv'

#classList = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
classList = ['Angry', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
#classList = ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise']
#classList = ['Angry', 'Fear', 'Sad', 'Surprise']


# Function importing Dataset
def importdata():
	trainData = pd.read_csv(trainCSV, sep= ';', header=None)
	testData = pd.read_csv(testCSV, sep= ';', header=None)
	
	# Printing the dataset obseravtions
	print ("Dataset: ", trainData.head())
	print ("Dataset: ", testData.head())

	return trainData, testData

# Function to load the dataset
def loaddataset(trainData, testData):

	rows, cols = trainData.shape
	featureList = trainData.values[0, 1:cols-1]
	# Separating the target variable
	X_train = trainData.values[1:rows, 1:cols-1]
	y_train = trainData.values[1:rows:, cols-1]

	X_test = testData.values[1:len(testData), 1:cols-1]
	y_test = testData.values[1:len(testData):, cols-1]
	
	return X_train, X_test, y_train, y_test, featureList
	
# Function to perform training with giniIndex.
def train_using_gini(X_train, X_test, y_train, maxDepth, minSamplesLeaf):

	# Creating the classifier object
	clf_gini = DecisionTreeClassifier(criterion = "gini",
			random_state = 100,max_depth= maxDepth, min_samples_leaf= minSamplesLeaf)

	# Performing training
	clf_gini.fit(X_train, y_train)
	return clf_gini
	
# Function to perform training with entropy.
def train_using_entropy(X_train, X_test, y_train, maxDepth, minSamplesLeaf):

	# Decision tree with entropy
	clf_entropy = DecisionTreeClassifier(
			criterion = "entropy", random_state = 100,
			max_depth = maxDepth, min_samples_leaf = minSamplesLeaf)

	# Performing training
	clf_entropy.fit(X_train, y_train)
	return clf_entropy


# Function to make predictions
def prediction(X_test, clf_object):

	# Predicton on test with giniIndex
	y_pred = clf_object.predict(X_test)
	#print("Predicted values:")
	#print(y_pred)
	return y_pred
	
# Function to calculate accuracy
def cal_accuracy(y_test, y_pred):
	
	print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))
	print ("Accuracy : ", accuracy_score(y_test,y_pred)*100)
	print("Report : \n",classification_report(y_test, y_pred))
	acc = accuracy_score(y_test,y_pred)*100
	return acc

# Driver code
def main():
	
	# Building Phase
	trainData, testData = importdata()
	X_train, X_test, y_train, y_test, featureList = loaddataset(trainData, testData)
	maxAcc = -1
	bestPair = ['0', '0']


	if(loopingOn):
		depthLow= 5
		depthHigh = 15
		sampleLeafLow = 5
		sampleLeafHigh = 80
		step = 5

		depthRange = depthHigh - depthLow
		sampleLeafRange = sampleLeafHigh - sampleLeafLow
		recallSumMax = -1
		

		for i in range(depthRange):
			for j in range(0, sampleLeafRange, step):
				print(maxAcc)

				clf_entropy = train_using_entropy(X_train, X_test, y_train, depthLow + i, sampleLeafLow + j)
				#clf_entropy = train_using_gini(X_train, X_test, y_train, depthLow + i, sampleLeafLow + j)
				print("Results Using Entropy:\nMaxDepth: ", depthLow + i, ", MaxSampleLeaf: ", sampleLeafLow + j)
				y_pred_entropy = prediction(X_test, clf_entropy)
				acc= cal_accuracy(y_test, y_pred_entropy)

				recallSum = 0
				#report = classification_report(y_test, y_pred_entropy, output_dict = True)
				#for emotionClass in classList:
				#	recallSum = recallSum + report[emotionClass]['recall']

				if(acc > maxAcc):
					maxAcc = acc
					bestPair[0] = depthLow + i
					bestPair[1] = sampleLeafLow + j

		print("Max accuracy was: ", maxAcc, "with ", bestPair)
	#plot
	if plotOn:
		clf_entropy = train_using_entropy(X_train, X_test, y_train, bestPair[0], bestPair[1])
		#clf_entropy = train_using_entropy(X_train, X_test, y_train, 6, 59)
		y_pred_entropy = prediction(X_test, clf_entropy)
		acc= cal_accuracy(y_test, y_pred_entropy)
		

		

		plt.figure(figsize=(25, 25))
		plot_tree(clf_entropy, filled=True, class_names=classList, feature_names= featureList)
		plt.title("Decision tree - entropy")
		

		plt.show()

	
	
# Calling main function
if __name__=="__main__":
	main()
