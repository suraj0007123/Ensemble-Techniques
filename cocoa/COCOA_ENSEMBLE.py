import pandas as pd

df=pd.read_excel(r"E:\DESKTOPFILES\suraj\assigments\ensemble technique\Datasets_ET\Coca_Rating_Ensemble.xlsx")

df.head()

df.info()

df.columns

df.drop(['REF','Review','Name','Company'],axis=1,inplace=True)

df.columns

df.dtypes

from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()

df['Company_Location']=lb.fit_transform(df['Company_Location'])

df['Bean_Type']=lb.fit_transform(df['Bean_Type'])

df['Origin']=lb.fit_transform(df['Origin'])

df.dtypes

df['Ratings']=pd.cut(df['Rating'], bins=[min(df.Rating) -1,
                                         df.Rating.mean(), max(df.Rating)], labels=["Low","High"])

df.columns

df.drop(['Rating'],axis=1, inplace=True)

df.columns

lb=LabelEncoder()
df['Ratings']=lb.fit_transform(df['Ratings'])

df.columns

df.dtypes

def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return(x)

predictors=df.loc[:,df.columns!='Ratings']
type(predictors)
predictors1=norm_func(predictors)

target=df['Ratings']
type(target)

################ Bagging ############# 
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test =train_test_split(predictors1, target, test_size=0.2, random_state=0) 

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

tree=DecisionTreeClassifier()
bag_clf=BaggingClassifier(base_estimator=tree, n_estimators=1500, random_state=42)

bag_clf.fit(x_train, y_train)

from sklearn.metrics import accuracy_score, confusion_matrix

##### Evaluation on Test Data ####### 
confusion_matrix(y_test, bag_clf.predict(x_test))
accuracy_score(y_test, bag_clf.predict(x_test))

##### Evaluation on Train Data ####### 
confusion_matrix(y_train, bag_clf.predict(x_train))
accuracy_score(y_train, bag_clf.predict(x_train))

########## AdaBoost ######### 
from sklearn.ensemble import AdaBoostClassifier

ada_clf=AdaBoostClassifier(learning_rate=0.03, n_estimators=50)

ada_clf.fit(x_train, y_train)

##### Evaluation on Test Data ####### 
confusion_matrix(y_test, ada_clf.predict(x_test))
accuracy_score(y_test, ada_clf.predict(x_test))

##### Evaluation on Train Data ####### 
confusion_matrix(y_train, ada_clf.predict(x_train))
accuracy_score(y_train, ada_clf.predict(x_train))

######### GradientBoosting ##########
from sklearn.ensemble import GradientBoostingClassifier

graboost_clf=GradientBoostingClassifier(n_estimators=100, random_state=42)

graboost_clf.fit(x_train, y_train)

##### Evaluation on Test Data ####### 
confusion_matrix(y_test, graboost_clf.predict(x_test))
accuracy_score(y_test, graboost_clf.predict(x_test))

##### Evaluation on Train Data ####### 
confusion_matrix(y_train, graboost_clf.predict(x_train))
accuracy_score(y_train, graboost_clf.predict(x_train))

########## XGBoost ######## 
import xgboost as xgb

xgb_clf=xgb.XGBClassifier(max_depth=3, n_estimators=400, learning_rate=0.3, n_jobs=0)

xgb_clf.fit(x_train, y_train)

##### Evaluation on Test Data ####### 
confusion_matrix(y_test, xgb_clf.predict(x_test))
accuracy_score(y_test, xgb_clf.predict(x_test))

##### Evaluation on Train Data ####### 
confusion_matrix(y_train, xgb_clf.predict(x_train))
accuracy_score(y_train, xgb_clf.predict(x_train))


xgb.plot_importance(xgb_clf)

xgb_clf = xgb.XGBClassifier(n_estimators = 500, learning_rate = 0.1,random_state = 42)

param_test1 = {'max_depth': range(3,10,2), 'gamma': [0.1, 0.2, 0.3], 'subsample': [0.8, 0.9],
               'colsample_bytree': [0.8, 0,9],'rag_alpha': [1e-2, 0.1, 1]}

# ___Grid Search
from sklearn.model_selection import GridSearchCV

grid_search = GridSearchCV(xgb_clf, param_test1, n_jobs = -1, cv = 5, scoring= 'accuracy')

grid_search.fit(x_train,y_train)

cv_xg_clf = grid_search.best_estimator_

# Evaluation on Testing Data with model with hyperparameter
accuracy_score(y_test, cv_xg_clf.predict(x_test))

grid_search.best_params_

########## Voting Hard and Soft ############

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

################## Soft Voting ################## 
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
print('L4:', accuracy_score(Y_test, predictions_4)*100)
print('L5:', accuracy_score(Y_test, predictions_5)*100)
print('L6:', accuracy_score(Y_test, predictions_6)*100)

# Accuracy of Soft voting
print('Soft Voting:', accuracy_score(Y_test, soft_predictions)*100) 


############# Stacking ###########
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
from mlxtend.classifier import StackingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

### Loading the dataset
bc=pd.read_excel(r"E:\DESKTOPFILES\suraj\assigments\ensemble technique\Datasets_ET\Coca_Rating_Ensemble.xlsx")

bc.head()

bc.info()

bc.columns

bc.drop(['REF','Review','Name','Company'],axis=1,inplace=True)

bc.columns

bc.dtypes

######### LabelEncoding ########### 
from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()

bc['Company_Location']=lb.fit_transform(bc['Company_Location'])

bc['Bean_Type']=lb.fit_transform(bc['Bean_Type'])

bc['Origin']=lb.fit_transform(bc['Origin'])

bc.dtypes

bc['Ratings']=pd.cut(bc['Rating'], bins=[min(bc.Rating) -1,
                                         bc.Rating.mean(), max(bc.Rating)], labels=["Low","High"])

bc.columns

bc.drop(['Rating'],axis=1, inplace=True)

bc.columns

lb=LabelEncoder()
bc['Ratings']=lb.fit_transform(bc['Ratings'])

x=bc.loc[:,bc.columns!='Ratings']
y=bc['Ratings']

######### Splitting the data into train and test
from sklearn.model_selection import train_test_split

# 20 % training dataset is considered for testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)


####### Standardizing the data 
from sklearn.preprocessing import StandardScaler
# initializing sc object
sc = StandardScaler()

bc.columns

# variables that needed to be transformed
var_transform = ['Cocoa_Percent', 'Company_Location', 'Bean_Type', 'Origin']
x_train[var_transform] = sc.fit_transform(x_train[var_transform]) # standardizing training data
x_test[var_transform] = sc.transform(x_test[var_transform])		 # standardizing test data
print(x_train.head())

####### Building the First Layer Estimators
KNC = KNeighborsClassifier() # initialising KNeighbors Classifier
NB = GaussianNB()			 # initialising Naive Bayes

##### Training the KNeighborsClassifier
model_kNeighborsClassifier = KNC.fit(x_train, y_train) # fitting Training Set

pred_knc = model_kNeighborsClassifier.predict(x_test) # Predicting on test dataset

###### Evaluation of KNeighborsClassifier
from sklearn.metrics import accuracy_score
acc_knc = accuracy_score(y_test, pred_knc) # evaluating accuracy score

print('accuracy score of KNeighbors Classifier is:', acc_knc * 100)

#### Training the Naive_bayes Classifier
model_NaiveBayes = NB.fit(x_train, y_train)
pred_nb = model_NaiveBayes.predict(x_test)

##### Evaluation of Navie_bayes Classifier
acc_nb = accuracy_score(y_test, pred_nb)
print('Accuracy of Naive Bayes Classifier:', acc_nb * 100)

###### Implementing Stacking Classifier
lr = LogisticRegression() # defining meta-classifier
clf_stack = StackingClassifier(classifiers =[KNC, NB], meta_classifier = lr, use_probas = True, use_features_in_secondary = True)

####### Training Stacking Classifier
model_stack = clf_stack.fit(x_train, y_train) # training of stacked model
pred_stack = model_stack.predict(x_test)	 # predictions on test data using stacked model

###### Evaluation Stacking Classifier
acc_stack = accuracy_score(y_test, pred_stack) # evaluating accuracy
print('accuracy score of Stacked model:', acc_stack * 100)

model_stack = clf_stack.fit(x_train, y_train) # training of stacked model
pred_stack = model_stack.predict(x_test)	 # predictions on test data using stacked model

###### Evaluating The Stacked Classifier 
acc_stack = accuracy_score(y_test, pred_stack) # evaluating accuracy
print('accuracy score of Stacked model:', acc_stack * 100)

