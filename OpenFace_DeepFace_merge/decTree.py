import pandas as pd
import os
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from tkinter import filedialog as fd
from tkinter import messagebox


#trainCSV = r'C:\Users\hwojc\OneDrive - Vysoké učení technické v Brně\Magisterské studium\Diplomka\02 Modely\Validace\OpenFace\rozhodovaci strom\trainAUcAUr.csv'
#testCSV =  r'C:\Users\hwojc\OneDrive - Vysoké učení technické v Brně\Magisterské studium\Diplomka\02 Modely\Validace\OpenFace\rozhodovaci strom\testAUcAUr.csv'

classList = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

featureList = ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r',
 			   'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r',
 			   'AU25_r', 'AU26_r', 'AU45_r',
	           'AU01_c', 'AU02_c', 'AU04_c', 'AU05_c', 'AU06_c', 'AU07_c', 'AU09_c',
 			   'AU10_c', 'AU12_c', 'AU14_c', 'AU15_c', 'AU17_c', 'AU20_c', 'AU23_c',
 			   'AU25_c', 'AU26_c', 'AU28_c', 'AU45_c']


def importdata():
	"""
    Imports dataset from csv files. 
    Args:
        None:
		Dataset csv files are selected by user.
    Returns:
        trainData: Pandas dataframe representing train data of selected csv file.
		testData: Pandas dataframe representing test data of selected csv file.
    """
	
	messagebox.showinfo("Info", "Choose files for training the decision tree.")
	filePath = os.path.dirname(__file__)
	csvPath = os.path.join(filePath, r'csv')
	testCSV, trainCSV =  fd.askopenfilenames(title='Open a file', initialdir = csvPath, filetypes= (('csv files', '*.csv'),))

	try:
		trainData = pd.read_csv(trainCSV, sep= ',', header=None)
		testData = pd.read_csv(testCSV, sep= ',', header=None)
	except:
		print("bad path")
	
	# Printing the dataset obseravtions
	#print ("Dataset: ", trainData.head())
	#print ("Dataset: ", testData.head())

	return trainData, testData


def loaddataset(trainData, testData):
	"""
    Loads only relevant data of the dataset, such as action units and emotion classes. 
    Args:
        trainData: Pandas dataframe representing train data of selected csv file.
		testData: Pandas dataframe representing test data of selected csv file.
    Returns:
        X_train: Data representing features for training.
		X_test: Data representing features for testing.
		Y_train: Data representing annotated emotions for training.
		Y_test: Data representing annotated emotions for testing.
    """
	rows, cols = trainData.shape

	# Separating the target variable
	X_train = trainData.values[1:rows, 5:cols-1]
	y_train = trainData.values[1:rows:, cols-1]

	X_test = testData.values[1:rows, 5:cols-1]
	y_test = testData.values[1:rows:, cols-1]
	
	return X_train, X_test, y_train, y_test
		

def train_using_entropy(xTrain, yTrain, maxDepth, minSamplesLeaf):
	"""
    Function to perform training with entropy. 
    Args:
        xTrain: Data representing features for training.
		yTrain: Data representing annotated emotions for training.
		maxDepth: Maximal depth of decision tree.
		minSamplesLeaf: Minimal number of samples allowed in leaf node.

    Returns:
        clf_entropy: Object of trained decision tree.
    """

	# Decision tree with entropy
	clf_entropy = DecisionTreeClassifier(
			criterion = "entropy", random_state = 100,
			max_depth = maxDepth, min_samples_leaf = minSamplesLeaf)

	# Performing training
	clf_entropy.fit(xTrain, yTrain)
	return clf_entropy


def prediction(xTest, clf_object):
	"""
    Function to make predictions. 
    Args:
        xTest: Data representing features for making prediction.
		clf_object: Object of trained decision tree.

    Returns:
        yPred: Data representing prediction output.
    """

	yPred = clf_object.predict(xTest)
	#print("Predicted values:")
	#print(y_pred)
	return yPred

def cal_accuracy(yTest, yPred):
	"""
    Function calculate accuracy of predictions. 
    Args:
        yTest: Data representing features for making prediction.
		yPred: Data representing annotated emotions for testing.
    Returns:
        acc: Calculated accuracy.
    """
	
	print("Confusion Matrix: \n", confusion_matrix(yTest, yPred))
	print ("Accuracy : ", accuracy_score(yTest,yPred)*100)
	print("Report : \n",classification_report(yTest, yPred))
	acc = accuracy_score(yTest,yPred)*100
	return acc

def trainTree(plotTree):
	"""
    Function to train decision tree. 
    Args:
        plotTree: Boolean specifies if tree should be visualized or not.
    Returns:
        clf_entropy: Object of trained decision tree.
    """

	#TODO replace constants in training process

	#train decision tree
	trainData, testData = importdata()
	X_train, X_test, y_train, y_test = loaddataset(trainData, testData)
	clf_entropy = train_using_entropy(X_train, y_train, 7, 37)

	y_pred_entropy = prediction(X_test, clf_entropy)
	acc = cal_accuracy(y_test, y_pred_entropy)

	if plotTree:
		plt.figure(figsize=(25, 25))
		plot_tree(clf_entropy, filled=True, class_names=classList, feature_names= featureList)
		plt.title("Decision tree - entropy")
		plt.show()

	return clf_entropy