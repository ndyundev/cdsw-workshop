from sklearn.linear_model import *
import pandas as pd
import numpy as np
import json

def predict_survival(args):
  model_json='{"model_params": {"classes_": [0, 1], "intercept_": [2.4167963327687887], "coef_": [[-0.5628169594733219, -0.2751165426284487, 0.080291494742051, 0.00783722463348259, -2.2771380590191654, -0.22835805927318106, -0.4681571338192308]], "n_iter_": [11]}, "init_params": {"warm_start": false, "C": 1.0, "n_jobs": null, "verbose": 0, "intercept_scaling": 1, "fit_intercept": true, "max_iter": 100, "penalty": "l2", "multi_class": "warn", "random_state": null, "dual": false, "tol": 0.0001, "solver": "warn", "class_weight": null}}'
  logmodel = logistic_regression_from_json(model_json)
  testEntry = pd.DataFrame.from_records([{'Pclass': args['Pclass'],'SibSp': args['SibSp'],'Parch': args['Parch'],'Fare': args['Fare'],'male':args['male'],'Q':args['Q'],'S':args['S']}
                                        ],columns=['Pclass', 'SibSp', 'Parch','Fare','male','Q','S'])

  testEst=logmodel.predict(testEntry.astype(np.float64))
  return {"Survived":testEst[0]}


def logistic_regression_from_json(jstring):
    data = json.loads(jstring)
    model = LogisticRegression(**data['init_params'])
    for name, p in data['model_params'].items():
        setattr(model, name, np.array(p))
    return model
  
print(predict_survival({"Pclass": 3,"SibSp": 1,"Parch": 0,"Fare": 7.2292,"male":1,"Q":0,"S":0 }))
#print(predict_survival({'Pclass': 1.0,'SibSp': 0.0,'Parch': 0.0,'Fare': 7.2292,'male':0.0,'Q':0.0,'S':0.0}))

#```
#{"Pclass": 3,"SibSp": 1,"Parch": 0,"Fare": 7.2292,"male":1,"Q":0,"S":0 }
#```

# Example output:

#```
#{"Survived": 0}
#```



# Example Input 2:
#```
#{"Pclass":1,"SibSp":0,"Parch":0,"Fare":7.2292,"male":0,"Q":0,"S":0}}
#```

#Example Output 2:
#```
#{"Survived": 1}
#```
