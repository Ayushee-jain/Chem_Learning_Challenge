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


data.shape


# In[5]:


data.isnull().sum()


# In[6]:


data['CO_GT'].value_counts()


# In[7]:


data['PT08_S1_CO'].value_counts()


# In[8]:


data['NMHC_GT'].value_counts()


# In[9]:


data['C6H6_GT'].value_counts()


# In[10]:


data['PT08_S2_NMHC'].value_counts()


# In[11]:


data['Nox_GT'].value_counts()


# In[12]:


data['PT08_S3_Nox'].value_counts()


# In[13]:


data['NO2_GT'].value_counts()


# In[14]:


data['PT08_S4_NO2'].value_counts()


# In[15]:


data['PT08_S5_O3'].value_counts()


# In[16]:


data['T'].value_counts()


# In[17]:


data['RH'].value_counts()


# In[18]:


data['AH'].value_counts()


# In[19]:


data=data.drop(columns=['NMHC_GT'])


# In[20]:


data=data.drop(data[data['AH']==-200].index)


# In[21]:


data.head()


# In[22]:


data['CO_GT'].value_counts()


# In[23]:


data['PT08_S1_CO'].value_counts()


# In[24]:


data['C6H6_GT'].value_counts()


# In[25]:


data['PT08_S2_NMHC'].value_counts()


# In[26]:


data['Nox_GT'].value_counts()


# In[27]:


data['PT08_S3_Nox'].value_counts()


# In[28]:


data['NO2_GT'].value_counts()


# In[29]:


data['PT08_S4_NO2'].value_counts()


# In[30]:


data['PT08_S5_O3'].value_counts()


# In[31]:


data['T'].value_counts()


# In[32]:


data['RH'].value_counts()


# In[33]:


sns.countplot(x="CO_level", data=data)


# In[34]:


#fig = plt.figure(figsize=(8,6))
data.groupby('CO_level')["CO_GT"].count().plot.bar(ylim=0)
plt.show()


# In[35]:


data.groupby('CO_level')["Nox_GT"].count().plot.bar(ylim=0)
plt.show()


# In[36]:


sns.violinplot(x=data["CO_GT"])


# In[37]:


x=data['CO_GT']
x=x.drop(x[x==-200].index)
a=x.mean()
data['CO_GT']=data['CO_GT'].replace(to_replace=-200, value =a) 
data['CO_GT'].value_counts()


# In[38]:


#fig = plt.figure(figsize=(8,6))
data.groupby('CO_level')["CO_GT"].count().plot.bar(ylim=0)
plt.show()


# In[39]:


x=data['Nox_GT']
x=x.drop(x[x==-200].index)
a=x.mean()
data['Nox_GT']=data['Nox_GT'].replace(to_replace=-200, value =a) 
data['Nox_GT'].value_counts()


# In[40]:


x=data['NO2_GT']
x=x.drop(x[x==-200].index)
a=x.mean()
data['NO2_GT']=data['NO2_GT'].replace(to_replace=-200, value =a) 
data['NO2_GT'].value_counts()


# In[41]:


data.head(10)


# In[42]:


data.dtypes


# In[43]:


data['Date_year']=data['Date'].dt.year
data['Date_month'] = data['Date'].dt.month
data['Date_week'] = data['Date'].dt.week
data['Date_day'] = data['Date'].dt.day


# In[44]:


data['Date'] = data['Date'].dt.date


# In[45]:


data.dtypes


# In[46]:


data.head()


# In[47]:


data=data.drop(columns=['Date'])


# In[48]:


data.shape


# In[49]:


data['Time'].unique()


# In[50]:


data['Split']=data['Time'].astype(str).str.split(':')


# In[51]:


data.head()


# In[52]:


data['Time_hour']=data['Split'].map(lambda x:x[0].strip())


# In[53]:


data.head()


# In[54]:


data['Time_min']=data['Split'].map(lambda x:x[1].strip())


# In[55]:


data['Time_sec']=data['Split'].map(lambda x:x[2].strip())


# In[56]:


data=data.drop(columns=['Split'])


# In[57]:


data.head()


# In[58]:


import seaborn as sns
sns.countplot(x="CO_level", data=data)


# In[59]:


x=data['CO_level']
print(x)


# In[63]:


data['CO_level']=data['CO_level'].replace(to_replace='Very High', value ='High')
data['CO_level']=data['CO_level'].replace(to_replace='Moderate', value ='Low')
data['CO_level']=data['CO_level'].replace(to_replace='Very low', value ='Low')


# In[64]:


sns.countplot(x="CO_level", data=data)


# In[65]:


data.dtypes


# In[66]:


data=data.drop(columns=['Time'])


# In[67]:


data['Time_min'].unique()


# In[70]:


data=data.apply(pd.to_numeric,errors="ignore")
data.dtypes


# In[71]:


data=data.drop(columns=['Time_min','Time_sec'])


# In[72]:


from sklearn.preprocessing import LabelEncoder
a=LabelEncoder()
data['CO_level']=a.fit_transform(data['CO_level'])


# In[73]:


data.head()


# In[74]:


fig = plt.figure(figsize=(8,6))
data.groupby('CO_level')["CO_GT"].count().plot.bar(ylim=0)
plt.show()


# In[75]:


data['CO_level'].unique()


# In[76]:


fig = plt.figure(figsize=(8,6))
data.groupby('CO_level')["C6H6_GT"].count().plot.bar(ylim=0)
plt.show()


# In[77]:


data.columns


# In[78]:


sns.violinplot(x=data["CO_GT"])


# In[79]:


sns.catplot(x="CO_level",y="CO_GT",data=data)


# In[80]:


sns.catplot(x="CO_level",y="PT08_S1_CO",data=data)


# In[81]:


sns.catplot(x="CO_level",y="C6H6_GT",data=data)


# In[82]:


sns.catplot(x="CO_level",y="PT08_S2_NMHC",data=data)


# In[83]:


train=data.drop(columns=['CO_level'])
test=data['CO_level']


# In[84]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(train,test,test_size=0.2)


# In[85]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)


# In[86]:


from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


# In[87]:


model = DecisionTreeClassifier()
param = {'max_depth':[2,4,6,9,10,15],}
gsc = GridSearchCV(estimator=model, param_grid=param, cv=5, scoring='accuracy', verbose=0, n_jobs=-1)
grid_result = gsc.fit(x_train, y_train)
print('Best Param', grid_result.best_params_)


# In[88]:


y_pred = grid_result.best_estimator_.predict(x_test)
print(y_pred.shape)
print('Accuracy', accuracy_score(y_test, y_pred))
print("classification Report:\n",classification_report(y_test,y_pred))
print('Confusion Matrix \n' , confusion_matrix(y_test, y_pred))


# In[89]:


model = RandomForestClassifier()
param = {'n_estimators':[2,4,6,9,10,15,20],}
gsc = GridSearchCV(estimator=model, param_grid=param, cv=5, scoring='accuracy', verbose=0, n_jobs=-1)
grid_result = gsc.fit(x_train, y_train)
print('Best Param', grid_result.best_params_)


# In[90]:


y_pred = grid_result.best_estimator_.predict(x_test)
print(y_pred.shape)
print('Accuracy', accuracy_score(y_test, y_pred))
print("classification Report:\n",classification_report(y_test,y_pred))
print('Confusion Matrix \n' , confusion_matrix(y_test, y_pred))


# In[91]:


model = SVC()
param = {'kernel':['rbf'], 'C':[1,5,10,15, 20], 'coef0':[0.001, 0.01,0.1, 0.5, 1]}
gsc = GridSearchCV(estimator=model,param_grid=param, cv=5, scoring='accuracy', verbose=0, n_jobs=-1)
grid_result = gsc.fit(x_train, y_train)
print('Best Param', grid_result.best_params_)


# In[92]:


y_pred = grid_result.best_estimator_.predict(x_test)
print(y_pred.shape)
print('Accuracy', accuracy_score(y_test, y_pred))
print("classification Report:\n",classification_report(y_test,y_pred))
print('Confusion Matrix \n' , confusion_matrix(y_test, y_pred))


# In[ ]:




