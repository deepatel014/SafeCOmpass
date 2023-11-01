#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 13:38:41 2023

@author: moneyraheja
"""


import warnings

warnings.filterwarnings("ignore", category=ResourceWarning, message="unclosed socket")

import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
#from sklearn import tree
#from sklearn.tree import DecisionTreeRegressor
import scipy.stats as stats
from sklearn.metrics import accuracy_score

#Get Data
KSI = pd.read_csv('C:/Users/deepp/Desktop/centennial/Sem2/SupervisedLearning/project_test/KSI (1).csv')
print(KSI.head)

#Use Describe 
d=KSI.describe()
print(d.to_string())

'''
###
    co-orelation Matrix

#Changing values in ACCLASS to Fatal and non-fatal only- for easy accsess of target column
KSI.loc[KSI['ACCLASS'] == 'Property Damage Only', 'ACCLASS'] = 'Non-Fatal'
# Replace 'Non-Fatal Injury' with 'Non-Fatal'
KSI.loc[KSI['ACCLASS'] == 'Non-Fatal Injury', 'ACCLASS'] = 'Non-Fatal'


KSI['ACCLASS'] = KSI['ACCLASS'].replace({'Fatal': 1, 'Non-Fatal': 0})
correlation_matrix=KSI.corr()
#print(KSI.corr())

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')
plt.show()
'''
'''
###
    co-orelation Matrix end
'''

#Drop columns which are not required
KSI = KSI.drop(axis=1,columns=['ROAD_CLASS','YEAR','TIME','X','Y','INDEX_','LATITUDE','LONGITUDE','ACCNUM','STREET1','STREET2','OFFSET',
                               'WARDNUM' , 'ACCLOC','INJURY','FATAL_NO','INITDIR','MANOEUVER','PEDTYPE','PEDTYPE','PEDCOND',
                               'CYCLISTYPE','CYCACT','CYCCOND','PASSENGER',
                               'NEIGHBOURHOOD_158','HOOD_140','NEIGHBOURHOOD_140','DIVISION','ObjectId','INVAGE','IMPACTYPE','TRAFFCTL','VEHTYPE','INVTYPE'])

#is it above 80Percent missing values,,, 
#Unnesseary Values are there


print(KSI.isnull().sum())


#Drop missing rows in a ACCLASS - Target column - because its only 5
KSI.dropna(subset=['ACCLASS'], inplace=True)




#Data Exploration
print(KSI.head)
d=KSI.describe()
print(d.to_string())
print(KSI.dtypes)
print(KSI.columns)



#null values in the columns
print(KSI.isnull().sum())

#Checking null values of each column in the KSI dataset
print(KSI.isnull().sum())

#Checking values of ACCLASS 
print(KSI['ACCLASS'].unique())

#Changing values in ACCLASS to Fatal and non-fatal only- for easy accsess of target column
KSI.loc[KSI['ACCLASS'] == 'Property Damage Only', 'ACCLASS'] = 'Non-Fatal'
# Replace 'Non-Fatal Injury' with 'Non-Fatal'
KSI.loc[KSI['ACCLASS'] == 'Non-Fatal Injury', 'ACCLASS'] = 'Non-Fatal'

'''

#decide where to place this
#Checking values of ACCLASS 
print(KSI['INVTYPE'].unique())
KSI.loc[KSI['INVTYPE'] == 'Moped Passenger', 'INVTYPE'] = 'Passenger'

KSI = KSI.drop(KSI[KSI['INVTYPE'] == 'Witness'].index[0])
'''

#column with only two values - Yes and No
columns_with_yes = [col for col in KSI.columns if KSI[col].nunique() == 1 and 'Yes' in KSI[col].unique()]

#Replace nan values with NO
print(KSI[columns_with_yes])
KSI[columns_with_yes] = KSI[columns_with_yes].replace(np.nan, 'No')
print(KSI[columns_with_yes])





#GET SEASONS FROM THE DATE
KSI['DATE'] = pd.to_datetime(KSI['DATE'])

KSI['MONTH'] = KSI['DATE'].dt.month
KSI['DAY'] = KSI['DATE'].dt.day


#Function to get season from the date
def get_season(month, day):
    if (month >= 3 and month <= 5) or (month == 2 and day >= 20):
        return 'Spring'
    elif (month >= 6 and month <= 8) or (month == 9 and day <= 22):
        return 'Summer'
    elif (month >= 9 and month <= 11) or (month == 12 and day <= 20):
        return 'Autumn'
    else:
        return 'Winter'


KSI['SEASON'] = KSI.apply(lambda x: get_season(x['MONTH'], x['DAY']), axis=1)

grouped = KSI.groupby(['SEASON', 'ACCLASS']).size().unstack()

#Checking datatypes
print(KSI.dtypes)

'''

# =============================================================================
# # CHI - SQUARE TEST - for categorical columns
# =============================================================================

'''
columm = KSI.columns
print(columm)

#List
relevant_col=[]
for colum in columm:
    contingency_table = pd.crosstab(KSI[colum], KSI['ACCLASS'])
    
    chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    
    print('\n')
    print(colum)
    print("P-value:", p_value)
  
    
#Dropping columns whouse p-value is less than 0.05
    if p_value<0.05:
        relevant_col.append(colum)
print(relevant_col)
'''
# =============================================================================
# #Making a Dataframe using relevant column
# =============================================================================
'''
KSI=KSI[relevant_col]

#Checking datatypes
print(KSI.dtypes)
print(KSI.columns)

#Checking Null values
print(KSI.isnull().sum())

#Dropping 
KSI = KSI.drop(axis=1,columns=['DATE','MONTH','DAY','PEDACT','DRIVACT','DRIVCOND'])  # DROPPING PEDACT BECAUSE NULL VALUES IS MORE THAN 80%
print(KSI.isnull().sum())


print(KSI.dtypes)

#checking unique values in Dataframe
print(KSI.nunique())




'''
# Get unique values  of each columns and it's value counts 
'''
for col in KSI.columns:
    unique_values = KSI[col].unique()
    value_counts = KSI[col].nunique()
    print(f"Column '{col}':")
    print("Unique Values:", unique_values)
    #print("Value Counts:")
    print(value_counts)
    print("---------------------")

'''
#Checking values with YES/NO - and replace them with 1 and 0
'''
columns_with_yesANDno = [col for col in KSI.columns if KSI[col].nunique() == 2 and 'Yes' in KSI[col].unique()]

print(KSI[columns_with_yesANDno])
KSI[columns_with_yesANDno] = KSI[columns_with_yesANDno].replace('Yes', 1)
KSI[columns_with_yesANDno] = KSI[columns_with_yesANDno].replace('No',0)
KSI[columns_with_yesANDno] = KSI[columns_with_yesANDno].astype(int)
KSI.info()

print(KSI.dtypes)
print(KSI.isnull().sum())
#####


KSI['HOOD_158'].unique()
count_method_2 = (KSI['HOOD_158'] == 'NSA').sum()
print(count_method_2)

#only 124 values of NSA, removing NSA to make it into INT
KSI = KSI.drop(index=KSI[KSI['HOOD_158'] == 'NSA'].index)
KSI.isnull().sum()

KSI['HOOD_158']=KSI['HOOD_158'].astype('int')

#Changing object datatype into category
object_columns = KSI.select_dtypes(["object"]).columns
KSI[object_columns] = KSI[object_columns].astype('category')
'''
# =============================================================================
# Spilliting DATA INTO TRAINING AND TESTING
# =============================================================================

'''
label_mapping = {'Non-Fatal': 0, 'Fatal': 1}
KSI['ACCLASS'] = KSI['ACCLASS'].map(label_mapping)

from sklearn.model_selection import train_test_split
X =  KSI.drop(axis=1,columns=['ACCLASS'])
y =  KSI['ACCLASS']



np.random.seed(34)


KSI.reset_index(drop=True, inplace=True)
#Using Stratified shuffle split
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)

for train_index, test_index in split.split(KSI, KSI["ACCLASS"]):
    strat_train_set = KSI.loc[train_index]
    strat_test_set = KSI.loc[test_index]
    
features = KSI.drop('ACCLASS', axis=1).columns.tolist()
predict = ['ACCLASS']

X_train = strat_train_set[strat_train_set.columns.difference(predict)]
y_train = strat_train_set[predict].values.ravel()

X_test = strat_test_set[strat_train_set.columns.difference(predict)]
y_test = strat_test_set[predict].values.ravel()

'''
#Handling Categorical Columns
'''


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

cat_attribs =  X.select_dtypes(include='category').columns.tolist()
print(cat_attribs)

#num_attribs=['HOOD_158']
num_attribs = X.select_dtypes(include='number').columns.tolist()
print(num_attribs)
num_pipeline_standard = StandardScaler()

cat_pipeline_money = Pipeline(steps=[('cat',SimpleImputer(strategy="most_frequent")),
                           ("one_hot_encode",OneHotEncoder())])


full_pipeline_money = ColumnTransformer(transformers=[
    ("cat", cat_pipeline_money, cat_attribs),
    ("num", num_pipeline_standard, num_attribs)  # Use the StandardScaler for numerical columns
])


X_train_transformed=full_pipeline_money.fit_transform(X_train)
a = X_train_transformed.toarray()
print(X_train_transformed.toarray())

dense_array = X_train_transformed.toarray()

import pickle
with open('full_pipeline_money.pkl', 'wb') as f:
    pickle.dump(full_pipeline_money, f)


'''
##   - - - -- - - - - - -- - - - -  SMOTING------ TO INCREASE FATAL INTO THE CODE ,, 
#AND TO MAKE MODEL MORE BETTER AS WE HAVE NON-FATAL : 10857 AND FATAL :1788
'''
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
# summarize the new class distribution
counter = Counter(y_train)
print(counter)

oversample = SMOTE()
X_train_transformed, y_train = oversample.fit_resample(X_train_transformed, y_train)

counter = Counter(y_train)
print(counter)



'''
#####--- - -- - -- - - - -SVM Model- - - - - -- - - - -- - ---- - -- -  - -- - - - - --


'''

from sklearn.svm import SVC

y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

np.random.seed(15)
# Build the Support Vector Machine classifier
#clf_svm_money= SVC(gamma='auto')
clf_svm_money = SVC()
clf_svm_money.fit(X_train_transformed, y_train)

import joblib 
joblib.dump(clf_svm_money, 'C:/Users/deepp/Desktop/centennial/Sem2/SupervisedLearning/project_test/svm_model_money.pkl')
print("Model dumped!")



'''

'''
#CONVERT Y-TEST INTO Y-TRANSFORMED
X_test_transformed = full_pipeline_money.transform(X_test)

y_pred = clf_svm_money.predict(X_test_transformed)

from sklearn.metrics import accuracy_score

# checking accuracy score

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy Score:", accuracy)





# Assuming X_train_transformed contains the transformed training features after using full_pipeline_money
# Assuming y_train contains the true target labels for the training data

# Predict on the training data
y_train_pred = clf_svm_money.predict(X_train_transformed)

# Calculate the accuracy score for the training data
train_accuracy = accuracy_score(y_train, y_train_pred)
print("Training Accuracy Score:", train_accuracy)


from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Calculate precision, recall, and F1 score
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Calculate ROC curve and AUC
y_pred_probs = clf_svm_money.decision_function(X_test_transformed)
roc_auc = roc_auc_score(y_test, y_pred_probs)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'SVM (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.show()








'''
     LOGISTIC REGRESSION MODEL   
'''
#from sklearn import preprocessing
#scaler = preprocessing.StandardScaler().fit(X_train_transformed)

from sklearn.linear_model import LogisticRegression

# Create an instance of the LogisticRegression class
logreg = LogisticRegression(max_iter=1000)

# Fit the model to your training data
logreg.fit(X_train_transformed, y_train)

X_test_transformed = full_pipeline_money.transform(X_test)







'''
    Grid Search for Logistic Regression
'''

#fine tune the model with Grid Search 
from sklearn.model_selection import GridSearchCV

# Define the parameter grid for grid search
param_grid = {
    'penalty': ['l1', 'l2'],
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'saga']
}

# Create a grid search instance
grid_search = GridSearchCV(estimator=logreg, param_grid=param_grid, cv=5, n_jobs=-1)

# Perform grid search on the training data
grid_search.fit(X_train_transformed, y_train)

# Get the best parameters and score from grid search
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Best Parameters (Grid Search):", best_params)
print("Best Score (Grid Search):", best_score)


logreg_grid_model = grid_search.best_estimator_
print(logreg_grid_model)

y_pred = logreg_grid_model.predict(X_test_transformed)





# checking accuracy score for Grid Search

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy Score:", accuracy)


'''
#Accuracy score using Training Data
y_train_pred = logreg.predict(X_train_transformed)

# Calculate the accuracy score for the training data
train_accuracy = accuracy_score(y_train, y_train_pred)
print("Training Accuracy Score:", train_accuracy)
'''









'''
Randomized Search for Logistic Regression
'''
from sklearn.model_selection import RandomizedSearchCV
# Define the parameter distribution for randomized search
param_dist = {
    'penalty': ['l1', 'l2'],
    'C': np.logspace(-3, 3, 7),  # Varying C values on a logarithmic scale
    'solver': ['liblinear', 'saga']
}

# Create a randomized search instance
random_search = RandomizedSearchCV(
    estimator=logreg,
    param_distributions=param_dist,
    n_iter=10,  # Number of random combinations to try
    cv=5,       # Cross-validation folds
    n_jobs=-1
)

# Perform randomized search on the training data
random_search.fit(X_train_transformed, y_train)

# Get the best parameters and score from randomized search
best_params = random_search.best_params_
best_score = random_search.best_score_

print("Best Parameters (Randomized Search):", best_params)
print("Best Score (Randomized Search):", best_score)

# Predict using the best model from randomized search
logreg_randomzied_model = random_search.best_estimator_
print(logreg_randomzied_model)




#Accuracy Score

models = [logreg_randomzied_model, logreg_grid_model]


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

for model in models:
    y_pred = model.predict(X_test_transformed)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"Model: {type(model).__name__} {model}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("\n")


# ROC Curve and AUC
y_prob = logreg_randomzied_model.predict_proba(X_test_transformed)[:, 1]  # Probability of positive class
roc_auc = roc_auc_score(y_test, y_prob)

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.show()

# I am getting the same accuracy and precision and Recall, F1 score










'''
Decision Trees

'''

from sklearn.tree import DecisionTreeClassifier

# Create an instance of the DecisionTreeClassifier class
clf_decision_tree = DecisionTreeClassifier(random_state=42)

# Fit the model to your training data
clf_decision_tree.fit(X_train_transformed, y_train)

# Make predictions on the test data
y_pred_tree = clf_decision_tree.predict(X_test_transformed)

# Evaluate the model's accuracy
accuracy_tree = accuracy_score(y_test, y_pred_tree)
print("Decision Tree Classifier Accuracy:", accuracy_tree)

# Calculate the accuracy score for the training data
y_train_pred_tree = clf_decision_tree.predict(X_train_transformed)
train_accuracy_tree = accuracy_score(y_train, y_train_pred_tree)
print("Training Accuracy Score (Decision Tree):", train_accuracy_tree)

'''
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Calculate precision, recall, and F1 score
precision = precision_score(y_test, y_pred_tree)
recall = recall_score(y_test, y_pred_tree)
f1 = f1_score(y_test, y_pred_tree)

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_tree)
print("Confusion Matrix:")
print(conf_matrix)

# Plot ROC curve
y_pred_probs = clf_decision_tree.predict_proba(X_test_transformed)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_probs)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'Decision Tree (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.show()
'''

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Define parameter grid for GridSearchCV
param_grid = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Perform Grid Search
grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5)
grid_search.fit(X_train_transformed, y_train)

best_tree_model_grid = grid_search.best_estimator_
best_params_grid = grid_search.best_params_

print("Best Parameters (Grid Search):", best_params_grid)

# Define parameter distribution for RandomizedSearchCV
param_dist = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Perform Randomized Search
random_search = RandomizedSearchCV(DecisionTreeClassifier(random_state=42), param_dist, n_iter=10, cv=5)
random_search.fit(X_train_transformed, y_train)

best_tree_model_random = random_search.best_estimator_
best_params_random = random_search.best_params_

print("Best Parameters (Randomized Search):", best_params_random)

# Now you can use the best models obtained from Grid Search and Randomized Search for evaluation
y_pred_best_grid = best_tree_model_grid.predict(X_test_transformed)
accuracy_best_grid = accuracy_score(y_test, y_pred_best_grid)
print("Best Decision Tree Classifier Accuracy (Grid Search):", accuracy_best_grid)

y_pred_best_random = best_tree_model_random.predict(X_test_transformed)
accuracy_best_random = accuracy_score(y_test, y_pred_best_random)
print("Best Decision Tree Classifier Accuracy (Randomized Search):", accuracy_best_random)


models = [best_tree_model_grid, best_tree_model_random]



from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

for model in models:
    y_pred = model.predict(X_test_transformed)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"Model: {type(model).__name__}")
    print(f"{model}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("\n")



'''
    Neural Networks

'''

from sklearn.neural_network import MLPClassifier
# Create an instance of the MLPClassifier class
clf_neural_net = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
# Fit the model to your training data
clf_neural_net.fit(X_train_transformed, y_train)

# Transform the test data using the same preprocessing steps
X_test_transformed = full_pipeline_money.transform(X_test)

# Make predictions using the trained neural network
y_pred = clf_neural_net.predict(X_test_transformed)


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier

# Create an instance of the MLPClassifier class
clf_neural_net = MLPClassifier(max_iter=1000, random_state=42)

# Define parameter grid for GridSearchCV
param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'sgd'],
    'alpha': [0.0001, 0.001, 0.01],
}

# Perform Grid Search
grid_search = GridSearchCV(clf_neural_net, param_grid, cv=5)
grid_search.fit(X_train_transformed, y_train)

best_neural_net_model_grid = grid_search.best_estimator_
best_params_grid = grid_search.best_params_

print("Best Parameters (Grid Search):", best_params_grid)

# Define parameter distribution for RandomizedSearchCV
param_dist = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'sgd'],
    'alpha': [0.0001, 0.001, 0.01],
}

# Perform Randomized Search
random_search = RandomizedSearchCV(clf_neural_net, param_dist, n_iter=10, cv=5)
random_search.fit(X_train_transformed, y_train)

best_neural_net_model_random = random_search.best_estimator_
best_params_random = random_search.best_params_

print("Best Parameters (Randomized Search):", best_params_random)



from sklearn.metrics import accuracy_score

# Calculate the accuracy score for the predictions
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy Score:", accuracy)


y_train_pred = clf_neural_net.predict(X_train_transformed)

# Calculate the accuracy score for the training data
train_accuracy = accuracy_score(y_train, y_train_pred)
print("Training Accuracy Score:", train_accuracy)


from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Precision
precision = precision_score(y_test, y_pred)
print("Precision:", precision)

# Recall
recall = recall_score(y_test, y_pred)
print("Recall:", recall)

# F1-Score
f1 = f1_score(y_test, y_pred)
print("F1-Score:", f1)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# ROC Curve and AUC
y_prob = clf_neural_net.predict_proba(X_test_transformed)[:, 1]  # Probability of positive class
roc_auc = roc_auc_score(y_test, y_prob)

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.show()



'''
Random Forest
'''

from sklearn.ensemble import RandomForestClassifier

# Create an instance of the RandomForestClassifier class
clf_random_forest = RandomForestClassifier(n_estimators=100, random_state=42)  # You can adjust the number of estimators as needed

# Fit the model to your training data
clf_random_forest.fit(X_train_transformed, y_train)

# Make predictions on the test data
y_pred_rf = clf_random_forest.predict(X_test_transformed)

# Evaluate the model's accuracy
from sklearn.metrics import accuracy_score

accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Random Forest Classifier Accuracy:", accuracy_rf)



y_train_pred = clf_random_forest.predict(X_train_transformed)

# Calculate the accuracy score for the training data
train_accuracy = accuracy_score(y_train, y_train_pred)
print("Training Accuracy Score:", train_accuracy)





'''
K neighbours classifier
'''

'''

from sklearn.neighbors import KNeighborsClassifier

# Create an instance of the KNeighborsClassifier class
clf_knn = KNeighborsClassifier(n_neighbors=5)  # You can adjust the number of neighbors as needed

# Fit the model to your training data
clf_knn.fit(X_train_transformed, y_train)

# Make predictions on the test data
y_pred_knn = clf_knn.predict(X_test_transformed)

# Evaluate the model's accuracy
from sklearn.metrics import accuracy_score

accuracy_knn = accuracy_score(y_test, y_pred_knn)
print("K-Nearest Neighbors (KNN) Classifier Accuracy:", accuracy_knn)



y_train_pred = clf_knn.predict(X_train_transformed)

# Calculate the accuracy score for the training data
train_accuracy = accuracy_score(y_train, y_train_pred)
print("Training Accuracy Score:", train_accuracy)
'''


'''




___-------_____------_____-----_____-___------__________------______------_____------______
'''



'''
    Sample input to check the prediction 
'''

sample_input = {
    'AG_DRIV': 1,
    'ALCOHOL': 0,
    'AUTOMOBILE' :1,
    'CYCLIST':0,
    'DISTRICT':	'Scarborough',
    'HOOD_158':88,
    'LIGHT':'Daylight',
    'LOCCOORD':'Intersection',
    'PEDESTRIAN':1,
    'RDSFCOND':'Dry',
    'SEASON':'Summer',
    'SPEEDING':0,
    'TRSN_CITY_VEH':1,
    'TRUCK':0,
    'VISIBILITY':'Clear'

    # ... and so on for all your input features
}
sample_input_df = pd.DataFrame([sample_input])


sample_input_transformed = full_pipeline_money.transform(sample_input_df)

predicted_class = logreg.predict(sample_input_transformed)


# Convert the predicted class into the corresponding label
predicted_label = 'Fatal' if predicted_class[0] == 1 else 'Non-Fatal'

print("Predicted Class:", predicted_class[0])
print("Predicted Label:", predicted_label)




'''
import pandas as pd

# List of sample input dictionaries
sample_inputs = [
    {
        'AG_DRIV': 1,
        'ALCOHOL': 0,
        'AUTOMOBILE': 1,
        'CYCLIST': 0,
        'DISTRICT': 'Scarborough',
        'HOOD_158': 88,
        'LIGHT': 'Daylight',
        'LOCCOORD': 'Intersection',
        'PEDESTRIAN': 1,
        'RDSFCOND': 'Dry',
        'SEASON': 'Summer',
        'SPEEDING': 0,
        'TRSN_CITY_VEH': 1,
        'TRUCK': 0,
        'VISIBILITY': 'Clear'
    },
    {
        'AG_DRIV': 0,
        'ALCOHOL': 1,
        'AUTOMOBILE': 0,
        'CYCLIST': 1,
        'DISTRICT': 'Etobicoke York',
        'HOOD_158': 6,
        'LIGHT': 'Dusk',
        'LOCCOORD': 'Mid-Block',
        'PEDESTRIAN': 1,
        'RDSFCOND': 'Dry',
        'SEASON': 'Winter',
        'SPEEDING': 1,
        'TRSN_CITY_VEH': 0,
        'TRUCK': 0,
        'VISIBILITY': 'Clear'
    },
    # Add more sample input dictionaries here
    # ...
]

# Create an empty DataFrame to store the results
results_df = pd.DataFrame(columns=['Predicted Class', 'Predicted Label'])

# Generate and test 10 sample inputs using a loop
for _ in range(10):
    # Choose a random sample input dictionary from the list
    sample_input = sample_inputs[_ % len(sample_inputs)]
    
    # Create a DataFrame from the current sample input dictionary
    sample_input_df = pd.DataFrame([sample_input])
    
    # Transform the sample input using the preprocessed pipeline
    sample_input_transformed = full_pipeline_money.transform(sample_input_df)
    
    # Predict the class using the SVM model
    predicted_class = clf_svm_money.predict(sample_input_transformed)
    
    # Convert the predicted class into the corresponding label
    predicted_label = 'Fatal' if predicted_class[0] == 1 else 'Non-Fatal'
    
    # Append the results to the results DataFrame
    results_df = results_df.append({'Predicted Class': predicted_class[0], 'Predicted Label': predicted_label}, ignore_index=True)

# Print the results
print(results_df)


'''





