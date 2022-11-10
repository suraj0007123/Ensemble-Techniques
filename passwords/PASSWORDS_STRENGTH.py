###### Loading libraries ###### 
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

bag=pd.read_excel(r"E:\DESKTOPFILES\suraj\assigments\ensemble technique\Datasets_ET\Ensemble_Password_Strength.xlsx")

bag.head()

bag.info()

bag["condition"]=" "

bag.loc[bag["characters_strength"]>0,"condition"]="strong"

bag.loc[bag["characters_strength"]< 1,"condition"]="weak"


########## LabelEncoding ######### 
lb = LabelEncoder()

bag['condition']=lb.fit_transform(bag['condition'])
bag['characters'] = lb.fit_transform(bag['characters'].astype(str))

# Input and Output Split
predictors = bag.loc[:, bag.columns!="condition"]
type(predictors)

target = bag["condition"]
type(target)

# Train Test partition of the data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(predictors, target, test_size = 0.2,random_state=0)

# Feature Scaling 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler() 
X_train = sc.fit_transform(X_train) 
X_test = sc.transform(X_test)

from sklearn import tree
clftree = tree.DecisionTreeClassifier()

from sklearn.ensemble import BaggingClassifier
bag_clf = BaggingClassifier(base_estimator = clftree, n_estimators = 500,bootstrap = True, n_jobs= 1, random_state = 42)
bag_clf.fit(X_train, Y_train)

from sklearn.metrics import accuracy_score, confusion_matrix
# Evaluation on Testing Data
confusion_matrix(Y_test, bag_clf.predict(X_test))
accuracy_score(Y_test, bag_clf.predict(X_test))

# Evaluation on Training Data
confusion_matrix(Y_train, bag_clf.predict(X_train))
accuracy_score(Y_train, bag_clf.predict(X_train))

########## Bagging ###########
from sklearn import tree
clftree = tree.DecisionTreeClassifier()

from sklearn.ensemble import BaggingClassifier
bag_classf = BaggingClassifier(base_estimator = clftree, n_estimators = 300, bootstrap = True,n_jobs = 1, random_state = 32)
bag_classf.fit(X_train, Y_train)

from sklearn.metrics import accuracy_score, confusion_matrix
# Evaluation on Testing Data
confusion_matrix(Y_test, bag_classf.predict(X_test))
accuracy_score(Y_test, bag_classf.predict(X_test))

# Evaluation on Training Data
confusion_matrix(Y_train, bag_classf.predict(X_train))
accuracy_score(Y_train, bag_classf.predict(X_train)) 

######### BOOSTING #########
####### Adaboosting ##################

from sklearn.ensemble import AdaBoostClassifier
ada_classf = AdaBoostClassifier(learning_rate = 0.03, n_estimators = 500)
ada_classf.fit(X_train, Y_train)

# Evaluation on Testing Data
confusion_matrix(Y_test, ada_classf.predict(X_test))
accuracy_score(Y_test, ada_classf.predict(X_test))

# Evaluation on Training Data
confusion_matrix(Y_train, bag_classf.predict(X_train))
accuracy_score(Y_train, ada_classf.predict(X_train))

########### Gradient boosting ####################
from sklearn.ensemble import GradientBoostingClassifier
boost_classf = GradientBoostingClassifier()
boost_classf.fit(X_train, Y_train)


# Evaluation on Testing Data
confusion_matrix(Y_test, boost_classf.predict(X_test))
accuracy_score(Y_test, boost_classf.predict(X_test))

# Hyperparameters
boost_clf2 = GradientBoostingClassifier(learning_rate = 0.02, n_estimators =1000, max_depth =1)
boost_clf2.fit(X_train, Y_train)

# Evaluation on Testing Data
confusion_matrix(Y_test, boost_clf2.predict(X_test))
accuracy_score(Y_test, boost_clf2.predict(X_test))

# Evaluation on Training Data
confusion_matrix(Y_train, boost_clf2.predict(X_train))
accuracy_score(Y_train, boost_clf2.predict(X_train))


######## XG Boosting ####################
import xgboost as xgb
xgb_classf = xgb.XGBClassifier(max_depths = 3, n_estimators = 500, learning_rate= 0.3, n_jobs =0)
xgb_classf.fit(X_train, Y_train)

# Evaluation on Testing Data
confusion_matrix(Y_test, xgb_classf.predict(X_test))
accuracy_score(Y_test, xgb_classf.predict(X_test))

# Evaluation on Training Data
confusion_matrix(Y_train, xgb_classf.predict(X_train))
accuracy_score(Y_train, xgb_classf.predict(X_train))

xgb.plot_importance(xgb_classf)

xgb_classf = xgb.XGBClassifier(n_estimators = 500, learning_rate = 0.1,random_state = 42)

param_test1 = {'max_depth': range(3,10,2), 'gamma': [0.1, 0.2, 0.3], 'subsample': [0.8, 0.9],
               'colsample_bytree': [0.8, 0,9],'rag_alpha': [1e-2, 0.1, 1]}

######### Grid Search #########
from sklearn.model_selection import GridSearchCV

grid_search = GridSearchCV(xgb_classf, param_test1, n_jobs = -1, cv = 5, scoring= 'accuracy')

grid_search.fit(X_train,Y_train)

cv_xg_clf = grid_search.best_estimator_

# Evaluation on Testing Data with model with hyperparameter
accuracy_score(Y_test, cv_xg_clf.predict(X_test))

grid_search.best_params_


######## Voting hard and soft ################
from sklearn import datasets, linear_model, svm, neighbors, naive_bayes
from sklearn.ensemble import VotingClassifier

# Split the train and test samples
test_samples = 100

X_train, Y_train = predictors[:-test_samples], target[:-test_samples]
X_test, Y_test = predictors[-test_samples:], target[-test_samples:]

# Instantiate the learners (classifiers)
learner_1 = neighbors.KNeighborsClassifier(n_neighbors=5)
learner_2 = linear_model.Perceptron(tol=1e-2, random_state=0)
learner_3 = svm.SVC(gamma=0.001)

# Instantiate the voting classifier
voting = VotingClassifier([('KNN', learner_1), 
                           ('Prc', learner_2), 
                           ('SVM', learner_3)])

# Fit classifier with the training data
voting.fit(X_train, Y_train)

# Predict the most voted class
hard_predictions = voting.predict(X_test)

# Accuracy of hard voting
print('Hard Voting:', accuracy_score(Y_test, hard_predictions)*100)

################## Soft Voting ###############
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
voting.fit(X_train, Y_train)
learner_4.fit(X_train, Y_train)
learner_5.fit(X_train, Y_train)
learner_6.fit(X_train, Y_train)

# Predict the most probable class
soft_predictions = voting.predict(X_test)

# Get the base learner predictions
predictions_4 = learner_4.predict(X_test)
predictions_5 = learner_5.predict(X_test)
predictions_6 = learner_6.predict(X_test)

# Accuracies of base learners
print('L4:', accuracy_score(Y_test,predictions_4)*100)
print('L5:', accuracy_score(Y_test,predictions_5)*100)
print('L6:', accuracy_score(Y_test,predictions_6)*100)

# Accuracy of Soft voting
print('Soft Voting:', accuracy_score(Y_test,soft_predictions)*100) 
