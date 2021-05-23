"""
Exercise description
--------------------

In the context of Mercadolibre's Marketplace an algorithm is needed to
predict if an item listed in the markeplace is new or used.

Your task to design a machine learning model to predict if an item is new or
used and then evaluate the model over held-out test data.

To assist in that task a dataset is provided in `MLA_100k.jsonlines` and a
function to read that dataset in `build_dataset`.

For the evaluation you will have to choose an appropiate metric and also
elaborate an argument on why that metric was chosen.

The deliverables are:
    - This file including all the code needed to define and evaluate a model.
    - A text file with a short explanation on the criteria applied to choose
      the metric and the performance achieved on that metric.


"""

import json
import pandas as pd
from sklearn import preprocessing 
from sklearn.linear_model import LogisticRegression


# You can safely assume that `build_dataset` is correctly implemented
def build_dataset():
    data = [json.loads(x) for x in open("C:/Users/mmaccari/OneDrive - everis/Documentos/mercado/ml_evaluation/MLA_100k.jsonlines")]
    target = lambda x: x.get("condition")
    N = -10000
    X_train = data[:N]
    X_test = data[N:]
    y_train = [target(x) for x in X_train]
    y_test = [target(x) for x in X_test]
    #df = pd.DataFrame.from_dict(data, dtype=None, columns=None)
    #df = pd.DataFrame.from_dict(data)
    #print(df)
    for x in X_test:
        del x["condition"]
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    print("Loading dataset...")
    # Train and test data following sklearn naming conventions
    # X_train (X_test too) is a list of dicts with information about each item.
    # y_train (y_test too) contains the labels to be predicted (new or used).
    # The label of X_train[i] is y_train[i].
    # The label of X_test[i] is y_test[i].
    X_train, X_test, y_train, y_test = build_dataset()

    # Insert your code below this line:
    # ...


#print(X_train[0])
FIELDS = ["state_id","state_name","city_id","city_name"]
df = pd.json_normalize(X_train["address"])
df[FIELDS]
