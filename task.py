from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn import train
from sklearn import metrics
import pandas as pd
import numpy as np

df = pd.read_csv("DATA.csv", sep=';')

feature_df = df[['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11',
       '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23',
       '24', '25', '26', '27', '28', '29', '30', 'COURSE ID']]

ind = np.asarray(feature_df)
dep = np.asarray(df['GRADE'])

# Relu method
print("\nMethod: Relu \n")
classifier = MLPClassifier(max_iter = 1500,activation='relu',random_state=0 ,hidden_layer_sizes=(40,40,40))
kfld = ShuffleSplit(n_splits=10, test_size=0.1)
scores = cross_val_score(classifier, ind, dep, cv = kfld, scoring='accuracy')
print(" Accuracy: %0.2f"%(scores.mean()))
mae = cross_val_score(classifier, ind, dep, cv = kfld, scoring='neg_mean_absolute_error')
print(" MAE: %0.2f"%(abs(mae.mean())),"\n")

# Relu method
print("\nMethod: Tanh \n")
classifier = MLPClassifier(max_iter = 1500,activation='tanh',random_state=0 ,hidden_layer_sizes=(40,40,40))
kfld = ShuffleSplit(n_splits=10, test_size=0.1)
scores = cross_val_score(classifier, ind, dep, cv = kfld, scoring='accuracy')
print(" Accuracy: %0.2f"%(scores.mean()))
mae = cross_val_score(classifier, ind, dep, cv = kfld, scoring='neg_mean_absolute_error')
print(" MAE: %0.2f"%(abs(mae.mean())),"\n")

metrics.confusion_matrix(actual_labels, predicted_labels)
metrics.accuracy_score(actual_labels, predicted_labels)
