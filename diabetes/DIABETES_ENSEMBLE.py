################ ADABoosting ####################
import pandas as pd

df=pd.read_csv(r"E:\DESKTOPFILES\suraj\assigments\ensemble technique\Datasets_ET\Diabeted_Ensemble.csv")

df.rename(columns={" Class variable":"cv"}, inplace=True)

df.head()
df.info()

# Input and Output Split
predictors = df.loc[:, df.columns!="cv"]
type(predictors)

target = df["cv"]
type(target)

# Train Test partition of the data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(predictors, target,test_size = 0.2, random_state=0)

from sklearn.ensemble import AdaBoostClassifier
ada_clf = AdaBoostClassifier(learning_rate = 0.02, n_estimators = 5000)
ada_clf.fit(x_train, y_train)

from sklearn.metrics import accuracy_score, confusion_matrix

# Evaluation on Testing Data
confusion_matrix(y_test, ada_clf.predict(x_test))
accuracy_score(y_test, ada_clf.predict(x_test))

# Evaluation on Training Data
accuracy_score(y_train, ada_clf.predict(x_train))

##########BAGGING  CLASSIFICATION#############
import pandas as pd
df=pd.read_csv(r"E:\DESKTOPFILES\suraj\assigments\ensemble technique\Datasets_ET\Diabeted_Ensemble.csv")

df.rename(columns={" Class variable":"cv"}, inplace=True)

# Dummy variables
df.head()
df.info()

# Input and Output Split
predictors = df.loc[:, df.columns!="cv"]
type(predictors)

target = df["cv"]
type(target)

# Train Test partition of the data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(predictors, target,test_size = 0.2, random_state=0)

from sklearn import tree
clftree = tree.DecisionTreeClassifier()

from sklearn.ensemble import BaggingClassifier
bag_clf = BaggingClassifier(base_estimator = clftree, n_estimators = 500, bootstrap = True, n_jobs = 1, random_state = 42)
bag_clf.fit(x_train, y_train)

from sklearn.metrics import accuracy_score, confusion_matrix

# Evaluation on Testing Data
confusion_matrix(y_test, bag_clf.predict(x_test))
accuracy_score(y_test, bag_clf.predict(x_test))

# Evaluation on Training Data
confusion_matrix(y_train, bag_clf.predict(x_train))
accuracy_score(y_train, bag_clf.predict(x_train))

########## GRADIENT BOOSTING ###########

import pandas as pd

df=pd.read_csv(r"E:\DESKTOPFILES\suraj\assigments\ensemble technique\Datasets_ET\Diabeted_Ensemble.csv")

df.rename(columns={" Class variable":"cv"}, inplace=True)

df.head()

df.info()

# Input and Output Split
predictors = df.loc[:, df.columns!="cv"]
type(predictors)

target = df["cv"]
type(target)

# Train Test partition of the data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(predictors, target,test_size = 0.2, random_state=0)

from sklearn.ensemble import GradientBoostingClassifier
boost_clf = GradientBoostingClassifier()
boost_clf.fit(x_train, y_train)

from sklearn.metrics import accuracy_score, confusion_matrix
confusion_matrix(y_test, boost_clf.predict(x_test))
accuracy_score(y_test, boost_clf.predict(x_test))

# Hyperparameters
boost_clf2 = GradientBoostingClassifier(learning_rate = 0.02, n_estimators =1000, max_depth = 1)
boost_clf2.fit(x_train, y_train)

from sklearn.metrics import accuracy_score, confusion_matrix

# Evaluation on Testing Data
confusion_matrix(y_test, boost_clf2.predict(x_test))
accuracy_score(y_test, boost_clf2.predict(x_test))

# Evaluation on Training Data
accuracy_score(y_train, boost_clf2.predict(x_train))


############# XGBOOSTING #############
import pandas as pd

df=pd.read_csv(r"E:\DESKTOPFILES\suraj\assigments\ensemble technique\Datasets_ET\Diabeted_Ensemble.csv")

df.rename(columns={" Class variable":"cv"}, inplace=True)

df.head()
df.info()

df.shape

df.columns

######### LabelEncoding ############
from sklearn.preprocessing import LabelEncoder

lb = LabelEncoder() 
df['cv'] = lb.fit_transform(df['cv'])


df.info()

predictors = df.loc[:, df.columns!="cv"]
type(predictors)

target = df["cv"]
type(target)

# Train Test partition of the data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(predictors, target,test_size = 0.2, random_state=0)


import xgboost as xgb

xgb_clf = xgb.XGBClassifier(max_depths = 5, n_estimators = 10000, learning_rate= 0.3, n_jobs = -1)


# n_jobs – Number of parallel threads used to run xgboost.
# learning_rate (float) – Boosting learning rate (xgb’s “eta”)

xgb_clf.fit(x_train, y_train)


from sklearn.metrics import accuracy_score, confusion_matrix

# Evaluation on Testing Data
confusion_matrix(y_test, xgb_clf.predict(x_test))
accuracy_score(y_test, xgb_clf.predict(x_test))

xgb.plot_importance(xgb_clf)

xgb_clf = xgb.XGBClassifier(n_estimators = 500, learning_rate = 0.1,random_state = 42)

param_test1 = {'max_depth': range(3,10,2), 'gamma': [0.1, 0.2, 0.3],'subsample': [0.8, 0.9], 'colsample_bytree': [0.8, 0,9],'rag_alpha': [1e-2, 0.1, 1]}

# Grid Search
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(xgb_clf, param_test1, n_jobs = -1, cv = 5, scoring ='accuracy')
grid_search.fit(x_train, y_train)
cv_xg_clf = grid_search.best_estimator_

# Evaluation on Testing Data with model with hyperparameter
accuracy_score(y_test, cv_xg_clf.predict(x_test))
grid_search.best_params_


############## VOTING_HARDSOFT #########
# Import the required libraries
from sklearn import linear_model, svm, neighbors, naive_bayes
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score

# Load the dataset

# Split the train and test samples
test_samples = 100
x_train, y_train = predictors[:-test_samples], target[:-test_samples]
x_test, y_test = predictors[-test_samples:], target[-test_samples:]

# Instantiate the learners (classifiers)
learner_1 = neighbors.KNeighborsClassifier(n_neighbors=5)
learner_2 = linear_model.Perceptron(tol=1e-2, random_state=0)
learner_3 = svm.SVC(gamma=0.001)

# Instantiate the voting classifier
voting = VotingClassifier([('KNN', learner_1),
                           ('Prc', learner_2), 
                           ('SVM', learner_3)])

# Fit classifier with the training data
voting.fit(x_train, y_train)

# Predict the most voted class
hard_predictions = voting.predict(x_test)

# Accuracy of hard voting
print('Hard Voting:', accuracy_score(y_test, hard_predictions))

################## Soft Voting ##############
# Instantiate the learners (classifiers)
learner_4 = neighbors.KNeighborsClassifier(n_neighbors = 5)
learner_5 = naive_bayes.GaussianNB()
learner_6 = svm.SVC(gamma = 0.001, probability = True)

# Instantiate the voting classifier
voting = VotingClassifier([('KNN', learner_4), 
                           ('NB', learner_5), 
                           ('SVM', learner_6)], 
                              voting = 'soft')

# Fit classifier with the training data
voting.fit(x_train, y_train)
learner_4.fit(x_train, y_train)
learner_5.fit(x_train, y_train)
learner_6.fit(x_train, y_train)

# Predict the most probable class
soft_predictions = voting.predict(x_test)

# Get the base learner predictions
predictions_4 = learner_4.predict(x_test)
predictions_5 = learner_5.predict(x_test)
predictions_6 = learner_6.predict(x_test)

# Accuracies of base learners
print('L4:', accuracy_score(y_test, predictions_4))
print('L5:', accuracy_score(y_test, predictions_5))
print('L6:', accuracy_score(y_test, predictions_6))

# Accuracy of Soft voting
print('Soft Voting:', accuracy_score(y_test, soft_predictions))
