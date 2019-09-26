import joblib
from sklearn.linear_model import *
import pandas as pd
import numpy as np


def predict_survival(args):
  logmodel = joblib.load('my_model.pkl')

  testEntry = pd.DataFrame.from_records([{'Pclass': args['Pclass'],'SibSp': args['SibSp'],'Parch': args['Parch'],'Fare': args['Fare'],'male':args['male'],'Q':args['Q'],'S':args['S']}
                                        ])

  testEst=logmodel.predict(testEntry.astype(np.float64))
  print(testEst.show())
  return {"Survived":testEst[0]}
  
  
#print(predict_survival({'Pclass': 3.0,'SibSp': 1.0,'Parch': 0.0,'Fare': 7.2292,'male':1.0,'Q':0.0,'S':0.0}))
print(predict_survival({'Pclass': 1.0,'SibSp': 0.0,'Parch': 0.0,'Fare': 7.2292,'male':0.0,'Q':0.0,'S':0.0}))

#```
#{'Pclass': 3.0,'SibSp': 1.0,'Parch': 0.0,'Fare': 7.2292,'male':1.0,'Q':0.0,'S':0.0}
#```

# Example output:

#```
#{"Survived": 0}
#```