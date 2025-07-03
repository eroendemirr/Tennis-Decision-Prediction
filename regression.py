import pandas as pd
veriler =pd.read_csv("veriler.csv")

X = veriler.drop("play", axis=1)

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

X["windy"]=le.fit_transform(X["windy"])
y=veriler["play"].map({"yes":1,"no":0})


print(X)
print("-------------------------------------------------")
print(y)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer(
    transformers=[("encoder",OneHotEncoder(drop="first"), ["outlook"])],
    remainder="passthrough"
)

X_encoded=ct.fit_transform(X)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(X_encoded,y,test_size=0.33,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train, y_train)

y_pred=regressor.predict(x_test)
print("tahminler:\n",y_pred)

import numpy as np
import statsmodels.api as sm

X_opt=np.append(arr=np.ones((X_encoded.shape[0],1)).astype(int),values=X_encoded, axis=1)
y_array = y.values

def backward_elimination(X,y,sl=0.05):
    while True:
        model=sm.OLS(y,X).fit()
        max_p=max(model.pvalues)
        if max_p>sl:
            max_index=np.argmax(model.pvalues)
            print(f"Elendi: sütun {max_index}, p-değeri: {max_p:.4f}")
            X=np.delete(X,max_index ,axis=1)
        else:
            break
    print("\nFinal model OLS özeti:\n")
    print(model.summary())
    return X
X_selected=backward_elimination(X_opt,y_array)
print("-------------------------------------")
print(X_selected)

