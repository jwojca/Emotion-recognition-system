import numpy as np
import pandas as pd
from collections import deque

# generate some data
# define features and target values
data = {
    'color': ['blond', 'dark', 'ginger'],
    'height': ['low', 'middle', 'high'],
    'weight': ['small', 'medium', 'large'],
    'cream': ['yes', 'no'],
    'burn' : ['yes', 'no']
}

 # create an empty dataframe
data_df = pd.DataFrame(columns=data.keys())
"""
np.random.seed(42)
# randomnly create 1000 instances
for i in range(1000):
    data_df.loc[i, 'wind_direction'] = str(np.random.choice(data['wind_direction'], 1)[0])
    data_df.loc[i, 'tide'] = str(np.random.choice(data['tide'], 1)[0])
    data_df.loc[i, 'swell_forecasting'] = str(np.random.choice(data['swell_forecasting'], 1)[0])
    data_df.loc[i, 'good_waves'] = str(np.random.choice(data['good_waves'], 1)[0]) """

data_df.loc[0] = ['blond', 'middle', 'small', 'no', 'yes']
data_df.loc[1] = ['blond', 'high', 'medium', 'yes', 'no']
data_df.loc[2] = ['dark', 'low', 'medium', 'yes', 'no']
data_df.loc[3] = ['blond', 'low', 'medium', 'no', 'yes']
data_df.loc[4] = ['ginger', 'middle', 'large', 'no', 'yes']
data_df.loc[5] = ['dark', 'high', 'large', 'no', 'no']
data_df.loc[6] = ['dark', 'middle', 'large', 'no', 'no']
data_df.loc[7] = ['blond', 'low', 'small', 'yes', 'no']

data_df.head()

# separate target from predictors
X = np.array(data_df.drop('burn', axis=1).copy())
y = np.array(data_df['burn'].copy())
feature_names = list(data_df.keys())[:4]
print(feature_names)

# import and instantiate our DecisionTreeClassifier class
from ID3 import DecisionTreeClassifier

# instantiate DecisionTreeClassifier
tree_clf = DecisionTreeClassifier(X=X, feature_names=feature_names, labels=y)
print("System entropy {:.4f}".format(tree_clf.entropy))
# run algorithm id3 to build a tree
tree_clf.id3()
tree_clf.printTree()



