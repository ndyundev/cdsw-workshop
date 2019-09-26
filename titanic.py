#install -> joblib-0.13.2 scikit-learn-0.21.3 sklearn-0.0
#!pip3 install sklearn
#!pip install joblib


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# # Reading the 'titanic_train.csv' into a pandas dataframe
# Comments in a code file in CDSW can include
# [Markdown](https://daringfireball.net/projects/markdown/syntax).


train = pd.read_csv('titanic.csv')

train.head()

#We can use seaborn to create a simple heatmap to see where we are missing data!

sns.heatmap(train.isnull(),cbar=False)

#Parts of Age are missing, but could be probably derived.
#the Cabin column is almost useless though, we should remove it

#Plotting survivals
sns.countplot(x='Survived',data=train)
sns.countplot(x='Survived',hue='Sex',data=train)
sns.countplot(x='Survived',hue='Pclass',data=train)

# # Cleaning data!
train.drop('Cabin',axis=1,inplace=True)
train.drop('Age',axis=1,inplace=True)

# verify results
sns.heatmap(train.isnull(),cbar=False)

#remove rest
train.dropna(inplace=True)
sns.heatmap(train.isnull(),cbar=False)

#Aall data is complete now!

# # Now let us make all data numerical!
#Instead of having sex to be male/femail -> make it a boolean variable
sex = pd.get_dummies(train['Sex'],drop_first=True)

# Use same procedure to 
embark = pd.get_dummies(train['Embarked'],drop_first=True)

train.drop(['PassengerId','Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
#Pclass - passenger class
#SibSp - siblings or spouses onboard
#Parch - parents or children onboard

train = pd.concat([train,sex,embark],axis=1)

train.head()


# # Let's do ML



# \n Train/Test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1), 
                                                    train['Survived'], test_size=0.3, random_state=65)

#\n Training the model

from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()

#All is in the next line..
logmodel.fit(X_train,y_train)
#We have a fitted model! Yaay! :P

#Now let's use our new baby/model to make some guesses on the test dataset
predictions = logmodel.predict(X_test)

from sklearn.metrics import classification_report

#Compare our model's estimations against reallity:

print(classification_report(y_test,predictions))

train.head()
X_test.head()
y_test.head()
help(logmodel)

for index, row in X_test.iterrows():
  if(logmodel.predict(pd.DataFrame.from_records([row]))==[1]):
    print("\nfound one ",str(row))

testEntry = pd.DataFrame.from_records([{'Pclass': 3.0,'': 1.0,'Parch': 0.0,'Fare': 7.2292,'male':1.0,'Q':0.0,'S':0.0},
                                       {'Pclass': 1.0,'': 0.0,'Parch': 0.0,'Fare': 7.2292,'male':0.0,'Q':0.0,'S':0.0}
                                      ])
testEntry.dtypes
testEst=logmodel.predict(testEntry.astype(np.float64))
print(testEst)

help(logmodel)
import joblib
joblib.dump(logmodel, 'my_model.pkl', compress=9)

train.to_csv(r'titanic_clean.csv')

!hdfs dfs -mkdir titanic
!hdfs dfs -put titanic_clean.csv titanic/