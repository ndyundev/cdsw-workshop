import json
import pandas as pd
import numpy as np
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
#Instead of having male/femail -> make it a boolean variable
sex = pd.get_dummies(train['Sex'],drop_first=True)

# Use same procedure for the Embarked column 
embark = pd.get_dummies(train['Embarked'],drop_first=True)

#add newly generated columns
train = pd.concat([train,sex,embark],axis=1)
#and drop the exisintg/old ones
train.drop(['Sex','Embarked'],axis=1,inplace=True)

# store the dataset as 'titanic_clean.csv'
train.to_csv('titanic_clean.csv',index=False)

# remove unwanted columns for training
train.drop(['PassengerId','Name','Ticket'],axis=1,inplace=True)

#Pclass - passenger class
#SibSp - siblings or spouses onboard
#Parch - parents or children onboard

train.head()

#
# # Let's do ML
#

!pip install sklearn
# Train/Test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1), 
                                                    train['Survived'], test_size=0.3, random_state=65)

# Training the model

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

testEntry = pd.DataFrame.from_records([{'Pclass': 3.0,'SibSp': 1.0,'Parch': 0.0,'Fare': 7.2292,'male':1.0,'Q':0.0,'S':0.0},
                                       {'Pclass': 1.0,'SibSp': 0.0,'Parch': 0.0,'Fare': 7.2292,'male':0.0,'Q':0.0,'S':0.0},
                                       {'Pclass': 3.0,'SibSp': 1.0,'Parch': 0.0,'Fare': 16.1,'male':0.0,'Q':0.0,'S':1.0},
                                       {'Pclass': 3.0,'SibSp': 0.0,'Parch': 0.0,'Fare': 8.0292,'male':0.0,'Q':0.0,'S':0.0}
                                      ],columns=['Pclass', 'SibSp', 'Parch','Fare','male','Q','S'])

testEntry = testEntry.astype(np.float64)

testEst=logmodel.predict(testEntry)
print(testEst)


serialize = json.dumps
data = {}
data['init_params'] = logmodel.get_params()
data['model_params'] = mp = {}
for p in ('coef_', 'intercept_','classes_', 'n_iter_'):
    mp[p] = getattr(logmodel, p).tolist()
print("model in json format -> ",serialize(data))

#train.to_csv(r'titanic_clean.csv')
##train.to_parquet('titanic_clean.pqt')
#help(train)
!hdfs dfs -mkdir titanic
!hdfs dfs -rm titanic/titanic_clean.csv
!hdfs dfs -put titanic_clean.csv titanic/