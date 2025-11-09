#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from pathlib import Path

# In[2]:

DATA_FILE = Path(__file__).resolve().parents[1] / "Data" / "StudentsPerform.csv"
df = pd.read_csv(DATA_FILE, na_values=["?"])
df.info()


# In[3]:


X = df.drop("race_code", axis=1).astype("float64")
y = df["race_code"]


# In[4]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)


# In[5]:


X_test.shape , y_test.shape


# In[6]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[7]:


from sklearn.svm import SVC

model = SVC(kernel="linear")
model.fit(X_train, y_train)


# In[8]:


y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

print("Train Accuracy:", round(model.score(X_train, y_train), 3))
print("Test Accuracy :", round(model.score(X_test, y_test), 3))


# In[9]:


model_2 = SVC(kernel="rbf")
model_2.fit(X_train, y_train)


# In[10]:


y_pred_train = model_2.predict(X_train)
y_pred_test = model_2.predict(X_test)

print("Train Accuracy:", round(model.score(X_train, y_train), 3))
print("Test Accuracy :", round(model.score(X_test, y_test), 3))


# In[ ]:



import pickle

with open('Svc.pkl', 'wb') as file:
    pickle.dump(model, file)
