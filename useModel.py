import joblib
from sklearn.linear_model import *
import pandas as pd
import numpy as np



logmodel = joblib.load('my_model.pkl')


testEntry = pd.DataFrame.from_records([{'Pclass': 3.0,'': 1.0,'Parch': 0.0,'Fare': 7.2292,'male':1.0,'Q':0.0,'S':0.0},
                                       {'Pclass': 1.0,'': 0.0,'Parch': 0.0,'Fare': 7.2292,'male':0.0,'Q':0.0,'S':0.0}
                                      ])

testEst=logmodel.predict(testEntry.astype(np.float64))
print(testEst)