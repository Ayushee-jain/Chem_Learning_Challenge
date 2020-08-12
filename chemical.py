#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data=pd.read_excel(r'C:\Users\LENOVO\Downloads\CLC_train.xlsx')


# In[3]:


data.head()


# In[4]:


data.isnull().sum()


# In[5]:


data.dtypes


# In[6]:


data['Date_year']=data['Date'].dt.year
data['Date_month'] = data['Date'].dt.month
data['Date_week'] = data['Date'].dt.week
data['Date_day'] = data['Date'].dt.day


# In[7]:


data['Date'] = data['Date'].dt.date


# In[8]:


data.dtypes


# In[9]:


data.head()


# In[10]:


data=data.drop(columns=['Date'])


# In[11]:


data.shape


# In[12]:


data['Time'].unique()


# In[13]:


data['Split']=data['Time'].astype(str).str.split(':')


# In[14]:


data.head()


# In[15]:


data['Time_hour']=data['Split'].map(lambda x:x[0].strip())


# In[16]:


data.head()


# In[17]:


data['Time_min']=data['Split'].map(lambda x:x[1].strip())


# In[18]:


data['Time_sec']=data['Split'].map(lambda x:x[2].strip())


# In[19]:


data=data.drop(columns=['Split'])


# In[20]:


data.head()


# In[21]:


import seaborn as sns
sns.countplot(x="CO_level", data=data)


# In[22]:


x=data['CO_level']
print(x)


# In[23]:


for i in range(len(x)):
    if(x[i]=='Very High'):
        x[i]='High'
    elif(x[i]=='Moderate'):
        x[i]='Low'
    elif(x[i]=='Very low'):
        x[i]='Low'


# In[24]:


print(x)


# In[25]:


data['CO_level']=x


# In[26]:


sns.countplot(x="CO_level", data=data)


# In[27]:


data.dtypes


# In[28]:


data=data.drop(columns=['Time'])


# In[29]:


data['Time_min'].unique()


# In[30]:


data.isnull().sum()


# In[31]:


data=data.apply(pd.to_numeric,errors="ignore")
data.dtypes


# In[32]:


data=data.drop(columns=['Time_min','Time_sec'])


# In[33]:


from sklearn.preprocessing import LabelEncoder
a=LabelEncoder()
data['CO_level']=a.fit_transform(data['CO_level'])


# In[34]:


data.head()


# In[35]:


fig = plt.figure(figsize=(8,6))
data.groupby('CO_level')["CO_GT"].count().plot.bar(ylim=0)
plt.show()


# In[36]:


data['CO_level'].unique()


# In[37]:


fig = plt.figure(figsize=(8,6))
data.groupby('CO_level')["NMHC_GT"].count().plot.bar(ylim=0)
plt.show()


# In[38]:


data.columns


# In[39]:


sns.violinplot(x=data["CO_GT"])


# In[40]:


sns.catplot(x="CO_level",y="CO_GT",data=data)


# In[41]:


sns.catplot(x="CO_level",y="PT08_S1_CO",data=data)


# In[42]:


sns.catplot(x="CO_level",y="NMHC_GT",data=data)


# In[43]:


sns.catplot(x="CO_level",y="C6H6_GT",data=data)


# In[44]:


train=data.drop(columns=['CO_level'])
test=data['CO_level']


# In[45]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(train,test,test_size=0.2)


# In[46]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)


# In[51]:


from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


# In[52]:


model = DecisionTreeClassifier()
param = {'max_depth':[2,4,6,9,10,15],}
gsc = GridSearchCV(estimator=model, param_grid=param, cv=5, scoring='accuracy', verbose=0, n_jobs=-1)
grid_result = gsc.fit(x_train, y_train)
print('Best Param', grid_result.best_params_)


# In[53]:


y_pred = grid_result.best_estimator_.predict(x_test)
print(y_pred.shape)
print('Accuracy', accuracy_score(y_test, y_pred))
print("classification Report:\n",classification_report(y_test,y_pred))
print('Confusion Matrix \n' , confusion_matrix(y_test, y_pred))


# In[54]:


model = RandomForestClassifier()
param = {'n_estimators':[2,4,6,9,10,15,20],}
gsc = GridSearchCV(estimator=model, param_grid=param, cv=5, scoring='accuracy', verbose=0, n_jobs=-1)
grid_result = gsc.fit(x_train, y_train)
print('Best Param', grid_result.best_params_)


# In[55]:


y_pred = grid_result.best_estimator_.predict(x_test)
print(y_pred.shape)
print('Accuracy', accuracy_score(y_test, y_pred))
print("classification Report:\n",classification_report(y_test,y_pred))
print('Confusion Matrix \n' , confusion_matrix(y_test, y_pred))


# In[56]:


model = SVC()
param = {'kernel':['rbf'], 'C':[1,5,10,15, 20], 'coef0':[0.001, 0.01,0.1, 0.5, 1]}
gsc = GridSearchCV(estimator=model,param_grid=param, cv=5, scoring='accuracy', verbose=0, n_jobs=-1)
grid_result = gsc.fit(x_train, y_train)
print('Best Param', grid_result.best_params_)


# In[58]:


y_pred = grid_result.best_estimator_.predict(x_test)
print(y_pred.shape)
print('Accuracy', accuracy_score(y_test, y_pred))
print("classification Report:\n",classification_report(y_test,y_pred))
print('Confusion Matrix \n' , confusion_matrix(y_test, y_pred))


# In[ ]:




