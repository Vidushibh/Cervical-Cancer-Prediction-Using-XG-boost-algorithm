#!/usr/bin/env python
# coding: utf-8

# ## UNDERSTAND THE PROBLEM STATEMENT
# 
# 
Every year Cervical cancer kills about 4,000 women in the U.S. and about 300,000 women worlwide. 
The death rate can be reduced if we can increase the medical screening. So, the goal of this project is to predict 
cervical cancer in 858 patients based on the input features like Age, Number of Pregnancies,their smoking habits, 
IUD, STDs etc. 
This dataset was collected at 'Hospital Universitario de Caracas' in Caracas, Venezuela and 
contains demographic information, habits and historic medical records of 858 patients.
# ## IMPORT DATASET AND LIBRARIES

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import zipfile

get_ipython().system('pip install plotly')
import plotly.express as px


# In[4]:


# import the csv files using pandas 
cancer_df = pd.read_csv("cervical_cancer.csv")


# In[5]:


# Let's explore the dataframe
cancer_df


# In[6]:


cancer_df.tail(20)


# ## PERFORM EXPLORATORY DATA ANALYSIS

# In[11]:


# Get data frame info
cancer_df.info()


# In[12]:


# Get the statistics of the data frame
cancer_df.describe()


# In[13]:


# Notice many question marks indicating missing values
cancer_df = cancer_df.replace('?', np.nan)
cancer_df


# In[15]:


# count na values 
cancer_df.isna().sum()


# In[16]:


# Plot heatmap
plt.figure(figsize = (20,20))
sns.heatmap(cancer_df.isnull())


# In[17]:


# Get data frame info
cancer_df.info()


# In[18]:


# Since STDs: Time since first diagnosis  and STDs: Time since last diagnosis have more than 80% missing values 
# we can drop them
cancer_df = cancer_df.drop(['STDs: Time since first diagnosis','STDs: Time since last diagnosis'], axis = 1)
cancer_df


# In[21]:


# Since most of the column types are object, we are not able to get the statistics of the dataframe.
# Convert them to numeric type

cancer_df = cancer_df.apply(pd.to_numeric)
cancer_df.info()


# In[22]:


# Get the statistics of the dataframe
cancer_df.describe()


# In[23]:


cancer_df.mean()


# In[24]:


# Replace null values with mean
cancer_df = cancer_df.fillna(cancer_df.mean())
cancer_df


# In[26]:


# Nan heatmap
sns.heatmap(cancer_df.isnull())


# In[41]:


print('Age range is between', cancer_df['Age'].min() ,'and', cancer_df['Age'].max())
cancer_df[cancer_df['Age'] == 84]['Biopsy']


# ## PERFORM DATA VISUALIZATION

# In[43]:


# Get the correlation matrix
corr_matrix = cancer_df.corr()
corr_matrix


# In[45]:


# Plot the correlation matrix
plt.figure(figsize = (20,20))
sns.heatmap(cancer_df.corr(), annot = True)


# In[51]:


cancer_df.hist(bins = 10, figsize = (30,30), color = 'w')


# ## PREPARE THE DATA BEFORE TRAINING

# In[52]:


target_df = cancer_df['Biopsy']
input_df = cancer_df.drop('Biopsy',axis = 1)


# In[54]:


target_df.shape


# In[55]:


input_df.shape


# In[56]:


X = np.array(input_df).astype('float32')
y = np.array(target_df).astype('float32')


# In[57]:


# reshaping the array from (421570,) to (421570, 1)
# y = y.reshape(-1,1)
y.shape


# In[58]:


# scaling the data before feeding the model
from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)


# In[59]:


X


# In[62]:


# spliting the data in to test and train sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2)
X_test, X_val, y_test, y_val = train_test_split(X,y,test_size = 0.2)


# ## TRAIN AND EVALUATE XGBOOST CLASSIFIER 

# In[30]:


get_ipython().system('pip install xgboost')


# In[66]:


# Train an XGBoost classifier model 
import xgboost as xgb

model = xgb.XGBClassifier(learning_rate = 0.1, max_depth = 5, n_estimators = 10)
model.fit(X_train, y_train)


# In[67]:


result_train = model.score(X_train, y_train)
result_train


# In[68]:


# predict the score of the trained model using the testing dataset
result_test = model.score(X_test, y_test)
result_test


# In[69]:


# make predictions on the test data
y_predict = model.predict(X_test)
y_predict


# In[70]:


from sklearn.metrics import confusion_matrix, classification_report
print(classification_report(y_test, y_predict))


# In[72]:


cm = confusion_matrix(y_predict, y_test)
sns.heatmap(cm)


# In[ ]:




