import numpy as np
import pandas as pd
from collections import deque

path = r'C:\Users\hwojc\OneDrive - Vysoké učení technické v Brně\Magisterské studium\Diplomka\02 Modely\Validace\OpenFace\rozhodovaci strom\merged.csv'

# generate some data
# define features and target values
data = {
    'frame' :  ['0', '1'],
    'AU01_c' :  ['0', '1'],
    'AU02_c' :  ['0', '1'],
    'AU04_c' :  ['0', '1'],
    'AU05_c' :  ['0', '1'],
    'AU06_c' :  ['0', '1'],
    'AU07_c' :  ['0', '1'],
    'AU09_c' :  ['0', '1'],
    'AU10_c' :  ['0', '1'],
    'AU12_c' :  ['0', '1'],
    'AU14_c' :  ['0', '1'],
    'AU15_c' :  ['0', '1'],
    'AU17_c' :  ['0', '1'],
    'AU20_c' :  ['0', '1'],
    'AU23_c' :  ['0', '1'],
    'AU25_c' :  ['0', '1'],
    'AU26_c' :  ['0', '1'],
    'AU28_c' :  ['0', '1'],
    'AU45_c' :  ['0', '1'],
    'Class' :  ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
}

 # create an empty dataframe
data_df = pd.DataFrame(columns=data.keys())

data_df = pd.read_csv(path, delimiter = ';')
Array = data_df.to_numpy()
numOfCols = Array[0].size
numOfRows = int(Array.size/numOfCols)

for i in range(numOfRows):
    data_df.loc[i, 0] =  Array[i, 0]
    data_df.loc[i, 1] =  Array[i, 1]
    data_df.loc[i, 2] =  Array[i, 2]
    data_df.loc[i, 3] =  Array[i, 3]
    data_df.loc[i, 4] =  Array[i, 4]
    data_df.loc[i, 5] =  Array[i, 5]
    data_df.loc[i, 6] =  Array[i, 6]
    data_df.loc[i, 7] =  Array[i, 7]
    data_df.loc[i, 8] =  Array[i, 8]
    data_df.loc[i, 9] =  Array[i, 9]
    data_df.loc[i, 10] =  Array[i, 10]
    data_df.loc[i, 11] =  Array[i, 11]
    data_df.loc[i, 12] =  Array[i, 12]
    data_df.loc[i, 13] =  Array[i, 13]
    data_df.loc[i, 14] =  Array[i, 14]
    data_df.loc[i, 15] =  Array[i, 15]
    data_df.loc[i, 16] =  Array[i, 16]
    data_df.loc[i, 17] =  Array[i, 17]
    data_df.loc[i, 18] =  Array[i, 18]
    data_df.loc[i, 19] =  Array[i, 19]



data_df.head()

# separate target from predictors
X = np.array(data_df.drop(['frame','Class'], axis=1).copy())
y = np.array(data_df['Class'].copy())
feature_names = list(data_df.keys())[1:18]
print(feature_names)
print(X)

# import and instantiate our DecisionTreeClassifier class
from ID3 import DecisionTreeClassifier

# instantiate DecisionTreeClassifier
tree_clf = DecisionTreeClassifier(X=X, feature_names=feature_names, labels=y)
print("System entropy {:.4f}".format(tree_clf.entropy))
# run algorithm id3 to build a tree
tree_clf.id3()
tree_clf.printTree()
#tree_clf.classify()


