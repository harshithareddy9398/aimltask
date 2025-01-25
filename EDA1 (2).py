#!/usr/bin/env python
# coding: utf-8

# In[39]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[40]:


data = pd.read_csv("data_clean.csv")
print(data)


# In[41]:


#printing the information
data.info()


# In[42]:


data1 = data.drop(['Unnamed: 0',"Temp C"], axis =1)
data1


# In[43]:


data1['Month']=pd.to_numeric(data['Month'],errors='coerce')
data1.info()


# In[44]:


#change column names(Rename the columns)
data1.rename({'Solar.R': 'Solar'}, axis=1, inplace = True)
data1


# In[45]:


#Display the data1 info()
data1.info()


# In[46]:


#display data1 missing values count in each column using isnull(),sum()
data1.isnull().sum()


# In[47]:


#visualize the missing values using heat app


cols = data1.columns
colors = ['black', 'yellow']
sns.heatmap(data1[cols].isnull(),cmap=sns.color_palette(colors),cbar = True)


# In[51]:


#find the mean and median values of each
#Imputation of missing value with median
median_ozone = data1["Ozone"].median()
mean_ozone = data1["Ozone"].mean()
print("Median of Ozone: ", median_ozone)
print("Mean of Ozone: ", mean_ozone)


# In[52]:


#replace the ozone missing values with median values
data1['Ozone'] = data1['Ozone'].fillna(median_ozone)
data1.isnull().sum()


# In[ ]:





# In[ ]:





# In[ ]:




