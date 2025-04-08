#Answer for third question
import sys
sys.path.append("/home/aakankshay/Desktop/UMC203/A1/")
from oracle import q1_fish_train_test_data, q3_hyper,q2_train_test_emnist
data3= q3_hyper(23647)
print(data3)

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion='entropy',splitter = 'best', max_depth=5 )


import pandas as pd 
import math

data = pd.read_csv("cleve.mod", sep=r'\s+', na_values=["?"], comment="%")

#Doing data cleanup
data_new = data.dropna()
print("Original dataset size:", len(data))
print("Cleaned dataset size:", len(data_new))
#splitting datsets
n = int(math.floor(len(data_new))*0.8)
training_data = data_new.iloc[:n]
test_data = data_new.iloc[n+1:]

print(data_new.columns)

healthy_or_sick = data_new['H'].tolist()
target_values = [0 if i=='H' else 1 for i in healthy_or_sick]

target_values_train = target_values[:n]
target_values_test = target_values[n+1:]

data_new['male'] = data_new['male'].replace({'male': 1, 'fem': 0})
data_new['angina'] = data_new['angina'].replace({'angina': 1, 'abnang': 2, 'notang': 3, 'asympt':4})
data_new['true'] = data_new['true'].replace({'true': 1, 'fal': 0})
data_new['hyp'] = data_new['hyp'].replace({'norm': 0, 'abn': 1, 'hyp': 2})
data_new['fal'] = data_new['fal'].replace({'fal': 0, 'true': 1})
data_new['down'] = data_new['down'].replace({'up': 1, 'flat': 2, 'down': 3})
data_new['fix'] = data_new['fix'].replace({'norm': 3, 'fix': 6, 'rev': 7})

x_values = data_new.drop(columns=['H','buff'])
x_values_train = x_values[:n]
x_values_test = x_values[n+1:]


dtc.fit(x_values_train, target_values_train)

target_values_predict = dtc.predict(x_values_test)


#reporting precision, accuracy, recall, f1 score
from sklearn.metrics import precision_score  , accuracy_score , recall_score , f1_score
accuracy = accuracy_score(target_values_test, target_values_predict)
precision = precision_score(target_values_test, target_values_predict)
recall = recall_score(target_values_test, target_values_predict)
F1 = f1_score(target_values_test, target_values_predict)

print("precision:", precision)
print("accuracy:", accuracy)
print("recall:", recall)
print("F1 score:", F1)


#finding important feature 
features = x_values.columns
fimp = []
for i,feature in enumerate(features):
    fimp.append([feature,dtc.feature_importances_[i]])

print(fimp)
max_fimp = max(dtc.feature_importances_)
for i,f in enumerate(fimp):
    if fimp[i][1]==max_fimp:
        print(fimp[i])

#Visualising the decision tree
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(32, 12))
plot_tree(dtc, 
          max_depth=5, 
          feature_names=['age', 'sex', 'chestpain type', 'trestbps', 'cholestrol', 
                         'fasting blood sugar <120', 'resting ecg', 'max heart rate', 
                         'ex. induced angina', 'oldpeak', 'slope', 'no. of vessels covered','thal'], 
          class_names=["Disease", "No Disease"], 
          filled=True, 
          rounded=True,
          fontsize=10)

plt.show() 
