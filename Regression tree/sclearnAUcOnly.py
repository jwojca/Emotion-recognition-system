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

trainCSV = r'C:\Users\hwojc\OneDrive - Vysoké učení technické v Brně\Magisterské studium\Diplomka\02 Modely\Validace\OpenFace\rozhodovaci strom\trainAUcOnly.csv'
testCSV = r'C:\Users\hwojc\OneDrive - Vysoké učení technické v Brně\Magisterské studium\Diplomka\02 Modely\Validace\OpenFace\rozhodovaci strom\testAUcOnly.csv'

classList = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
featureList = ['AU01_c', 'AU02_c', 'AU04_c', 'AU05_c', 'AU06_c', 'AU07_c', 'AU09_c',
 			   'AU10_c', 'AU12_c', 'AU14_c', 'AU15_c', 'AU17_c', 'AU20_c', 'AU23_c',
 			   'AU25_c', 'AU26_c', 'AU28_c', 'AU45_c']

# Function importing Dataset
def importdata():
	trainData = pd.read_csv(trainCSV, sep= ';', header=None)
	testData = pd.read_csv(testCSV, sep= ';', header=None)
	
	# Printing the dataset obseravtions
	#print ("Dataset: ", trainData.head())
	#print ("Dataset: ", testData.head())

	return trainData, testData

# Function to load the dataset
def loaddataset(trainData, testData):

	# Separating the target variable
	X_train = trainData.values[1:len(trainData), 1:18]
	y_train = trainData.values[1:len(trainData):, 19]

	X_test = testData.values[1:len(testData), 1:18]
	y_test = testData.values[1:len(testData):, 19]
	
	return X_train, X_test, y_train, y_test
	
# Function to perform training with giniIndex.
def train_using_gini(X_train, X_test, y_train):

	# Creating the classifier object
	clf_gini = DecisionTreeClassifier(criterion = "gini",
			random_state = 100,max_depth=3, min_samples_leaf=5)

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
	
	#print("Confusion Matrix: \n",
	#	confusion_matrix(y_test, y_pred))
	
	print ("Accuracy : ",
	accuracy_score(y_test,y_pred)*100)
	
	#print("Report : ",
	#classification_report(y_test, y_pred))
	acc = accuracy_score(y_test,y_pred)*100
	return acc

# Driver code
def main():
	
	# Building Phase
	trainData, testData = importdata()
	X_train, X_test, y_train, y_test = loaddataset(trainData, testData)
	clf_gini = train_using_gini(X_train, X_test, y_train)


	depthLow= 1
	depthHigh = 15
	sampleLeafLow = 1
	sampleLeafHigh = 40

	depthRange = depthHigh - depthLow
	sampleLeafRange = sampleLeafHigh - sampleLeafLow
	
	maxAcc = -1
	bestPair = ['0', '0']
	for i in range(depthRange):
		for j in range(sampleLeafRange):
			clf_entropy = train_using_entropy(X_train, X_test, y_train, depthLow + i, sampleLeafLow + j)
			print("Results Using Entropy:\nMaxDepth: ", depthLow + i, ", MaxSampleLeaf: ", sampleLeafLow + j)
			y_pred_entropy = prediction(X_test, clf_entropy)
			acc= cal_accuracy(y_test, y_pred_entropy)
			if(acc > maxAcc):
				maxAcc = acc
				bestPair[0] = depthLow + i
				bestPair[1] = sampleLeafLow + j

	print("Max accuracy was: ", maxAcc, "with ", bestPair)
	#plot
	if plotOn:
		plt.figure(figsize=(20,20))
		plot_tree(clf_gini, filled=True, class_names=classList, feature_names=featureList)
		plt.title("Decision tree - gini")
		plt.show()
		
		input("Press Enter to continue...")

		plt.figure(figsize=(20,20))
		plot_tree(clf_entropy, filled=True, class_names=classList, feature_names=featureList)
		plt.title("Decision tree - entropy")

		plt.show()

	
	
# Calling main function
if __name__=="__main__":
	main()
