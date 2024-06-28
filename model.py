import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pickle

import warnings
warnings.filterwarnings('ignore')

mental_health_data=pd.read_csv('cleaned_data.csv')

X = mental_health_data.drop('Mental_Health',axis= 1)
y = mental_health_data['Mental_Health']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=42)


sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
knnc = KNeighborsClassifier()
knnc.fit(X_train, y_train)


y_pred = knnc.predict(X_test)


pickle.dump(knnc, open('knnc.pkl','wb'))
model=pickle.load(open('knnc.pkl','rb'))
print(y_pred)