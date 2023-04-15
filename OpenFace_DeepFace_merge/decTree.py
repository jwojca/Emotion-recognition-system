import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

trainCSV = r'C:\Users\hwojc\OneDrive - Vysoké učení technické v Brně\Magisterské studium\Diplomka\02 Modely\Validace\OpenFace\rozhodovaci strom\trainAUcAUr.csv'
testCSV =  r'C:\Users\hwojc\OneDrive - Vysoké učení technické v Brně\Magisterské studium\Diplomka\02 Modely\Validace\OpenFace\rozhodovaci strom\testAUcAUr.csv'

classList = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

featureList = ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r',
 			   'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r',
 			   'AU25_r', 'AU26_r', 'AU45_r',
	           'AU01_c', 'AU02_c', 'AU04_c', 'AU05_c', 'AU06_c', 'AU07_c', 'AU09_c',
 			   'AU10_c', 'AU12_c', 'AU14_c', 'AU15_c', 'AU17_c', 'AU20_c', 'AU23_c',
 			   'AU25_c', 'AU26_c', 'AU28_c', 'AU45_c']

#Decision tree
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
	X_train = trainData.values[1:len(trainData), 1:35]
	y_train = trainData.values[1:len(trainData):, 36]

	X_test = testData.values[1:len(testData), 1:35]
	y_test = testData.values[1:len(testData):, 36]
	
	return X_train, X_test, y_train, y_test
		
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

def cal_accuracy(y_test, y_pred):
	
	print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))
	print ("Accuracy : ", accuracy_score(y_test,y_pred)*100)
	print("Report : \n",classification_report(y_test, y_pred))
	acc = accuracy_score(y_test,y_pred)*100
	return acc