# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 21:23:27 2022

@author: junaid ansar
"""


#import statment

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

#Loading the dataset

ds=pd.read_csv("D:\Minispyder\BMI.csv")
print(ds)





#Analyz Data

ds.info()
print("")
ds.describe()
print("")
ds.isnull().sum()




#Convert Gender to number

X = ds.iloc[:, :-2].values
y = ds.iloc[:, 2].values
labelEncoder_gender = LabelEncoder()
X[:,0 ] = labelEncoder_gender.fit_transform(X[:,0 ])

print(X)
print(y)




# Replace directly in dataframe

ds['Gender'].replace('Female',0 , inplace=True)
ds['Gender'].replace('Male',1 , inplace=True)
X = ds.iloc[:, :-2].values
y = ds.iloc[:, 2].values
print(X)
print(y)




#Split dataset into training set and test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(X_train)
print(X_test)


#scalling data
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
#print(w_test)
#print(w_train)

#prediction

from sklearn.linear_model import LinearRegression,LogisticRegression

#lin_reg = LinearRegression(copy_X= True,fit_intercept=True,n_jobs= None,normalize='deprecated',positive= True)
'''from sklearn.model_selection import GridSearchCV
import time
dual=[True,False]
max_iter=[100,110,120,130,140]
param_grid = dict(dual=dual,max_iter=max_iter)
lr = LogisticRegression(penalty='l2')
grid = GridSearchCV(estimator=lr, param_grid=param_grid, cv = 3, n_jobs=-1)

start_time = time.time()
lin_reg = grid.fit(X_train, y_train)'''
#lin_reg= LogisticRegression(penalty='l2',dual=False,max_iter=110)

lin_reg = LinearRegression()

lin_reg.fit(X_train, y_train)
from sklearn.model_selection import cross_val_score    
from sklearn import metrics
cross_val_score(lin_reg, X, y, cv=5, scoring='f1_macro')  
lin_pred = lin_reg.predict(X_test)
print(lin_pred)

from sklearn import metrics

print('R square = ', metrics.r2_score(y_test, lin_pred))
print('Mean squared Error = ', metrics.mean_squared_error(y_test, lin_pred))
print('Mean absolute Error = ', metrics.mean_absolute_error(y_test, lin_pred))

my_weight_pred = lin_reg.predict([[1, 168]])
print('My predicted weight = ', my_weight_pred)


pickle.dump(lin_reg, open('bmi.pkl', 'wb'))


#bmi starting

# new spliting
ds['Gender'].replace('Female',1 , inplace=True)
ds['Gender'].replace('Male',0 , inplace=True)
w = ds.drop('Index', axis=1)
z = ds['Index']
w_train, w_test, z_train, z_test= train_test_split(w,z, test_size=0.20, random_state=123, stratify=z)
print(w)
print(z)


# scalling data

from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
w_train=sc. fit_transform(w_train)
w_test=sc. transform(w_test)
# print(w_test)
# print(w_train)




from sklearn.linear_model import LogisticRegression

# logic regresion
#le5=0
#logic= LogisticRegression(multi_class='multinominal',max_iter= 2500,C=le5, class_weight ='balanced')
# clf=logic
# kf=StratifiedKFold(shuffle=True,n_splits=10)
# scores=cross_val_score(clf,w_train,z_train,cv=kf,scoring='accuracy')
# print(scores)



#lG

#cv = KFold(n_splits=10, random_state=1, shuffle=True)
# create model
model = LogisticRegression()
# evaluate model
#scores = cross_val_score(model, w, z, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
#print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))

model.fit(w_train,z_train)
z_pred=model. predict (w_test)

# clasification report
from sklearn.metrics import confusion_matrix
cm= confusion_matrix (z_test,z_pred)

import matplotlib.pyplot as plt
import seaborn as sn
#%matplotlib inline
plt.figure(figsize=(10,7))
sn.heatmap( cm,annot=True)

plt .xlabel('predicted')
plt.ylabel('Truth')




#Random forest

# from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier

# rf=RandomForestClassifier(n_estimators=100,random_state=13,bootstrap=True,max_depth=10,min_samples_leaf=3,min_samples_split=2,max_features=3)
# clf=rf
# kf=KFold(shuffle=True,n_splits=10)
# scores=cross_val_score(clf,w_train,z_train,cv=kf,scoring='accuracy')
# scores=cross_val_score(clf,w_train,z_train)
# print(scores)
model1= RandomForestClassifier(n_estimators=100)
model1.fit(w_train,z_train)

model1. score(w_test,z_test)
z_pred=model1.predict ( w_test)

# clasification report
from sklearn.metrics import confusion_matrix
cm= confusion_matrix( z_test,z_pred)
cm

#% matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sn
plt.figure(

figsize=(10,7))
sn.heatmap( cm,annot=True)
plt. xlabel('predicted')
plt.ylabel('Truth')


#SVM

# from sklearn.svm import SVC
# model=SVC()
# model.fit(w_train,z_train)
# model.score(w_test,z_test)


from sklearn import svm
clf = svm.SVC(kernel='linear', C=1).fit(w_train, z_train)
clf.score(w_test, z_test)

# Use of SVC with kernal='rbf'
from sklearn.svm import SVC
clf1 = SVC(kernel='rbf', C=1).fit(w_train, z_train)
clf1.score(w_test, z_test)
# its accuracy is lessthan use the kernal model linear so avoid it

z_pred=clf.predict( w_test)

# clasification report
from sklearn.metrics import confusion_matrix
cm= confusion_matrix( z_test,z_pred)
cm

#% matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sn
plt.figure(figsize=(10,7))
sn.heatmap(cm, annot=True)
plt. xlabel('predicted')
plt.ylabel('Truth')



pickle.dump(clf,open('svc.pkl','wb'))

#Gradient boosting classifiers

from numpy import mean
from numpy import std
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold

model2 =GradientBoostingClassifier()
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, w, z, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
# fit the model on the whole dataset
model2 = GradientBoostingClassifier()
model2.fit(w_train, z_train)

# model = GradientBoostingClassifier(random_state=13,max_depth=110,max_features=3,min_weight_fraction_leaf=5,min_samples_split=10,n_estimators=500)
# clf=model
# kf=KFold(shuffle=True,n_splits=12)
# scores=cross_val_score(clf,w_train,z_train,cv=kf,scoring='accuracy')
# print(scores)
# print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))



z_pred=model2.predict( w_test)

# clasification report
from sklearn.metrics import confusion_matrix
cm= confusion_matrix( z_test,z_pred)


#% matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sn
plt.figure(figsize=(10,7))
sn.heatmap(cm, annot=True)
plt. xlabel('predicted')
plt.ylabel('Truth')


#test my own data

wnew=[[0,181,72]]
wnew =sc. transform(wnew )
znew=clf.predict(wnew)
# znew
wnew





#BMR Calculation function

def bmr_calcu():
    wgt=my_weight_pred
    H=int(input( "enter your height:"))
    A=int(input( "enter your Age:"))
    g=input("male or female"). lower()
    if g=='m':
        bmr=(10*wgt) +( 6.25*H)-(5*A)+5
    elif g == 'f' :
        bmr=(10*wgt )+ ( 6.25*H)-(5*A )-161
    print( "Your basel metabolic rate is:"+ str( bmr)+".")
    return bmr



#daily calories need
def daily_calory( bmrs):
    print("1->sedentary \n 2->Execise 1 - 3 times a week \n 3->Execise 4 - 5 times a week \n 4->Daily Exercise or intensive exercise 3-4 time a week \n 5->intence Exercise 6 time a week")
    al=int(input("enter the your activity level:"))
    if al==1:
        ALI=1.2
        # Activity Level Index ALI=1.2 
    elif al ==2:
        ALI=1.375
    elif al==3:
        ALI=1.46
    elif al==4:
        ALI=1.725
    elif al==5 :
        ALI=1.9
 # daily calories needed
    DCN= int (bmrs * ALI)
    print( "To maintain your current weight you need"+" "+str(DCN)+"calories a day.")
    return DCN


bmrs=bmr_calcu()
bmrs
DNC=daily_calory(bmrs)
DNC










