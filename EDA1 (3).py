#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


data = pd.read_csv("data_clean.csv")
print(data)


# In[4]:


#printing the information
data.info()


# In[5]:


data1 = data.drop(['Unnamed: 0',"Temp C"], axis =1)
data1


# In[6]:


data1['Month']=pd.to_numeric(data['Month'],errors='coerce')
data1.info()


# In[7]:


#change column names(Rename the columns)
data1.rename({'Solar.R': 'Solar'}, axis=1, inplace = True)
data1


# In[8]:


#Display the data1 info()
data1.info()


# In[9]:


#display data1 missing values count in each column using isnull(),sum()
data1.isnull().sum()


# In[10]:


#visualize the missing values using heat app


cols = data1.columns
colors = ['black', 'yellow']
sns.heatmap(data1[cols].isnull(),cmap=sns.color_palette(colors),cbar = True)


# In[11]:


#find the mean and median values of each
#Imputation of missing value with median
median_ozone = data1["Ozone"].median()
mean_ozone = data1["Ozone"].mean()
print("Median of Ozone: ", median_ozone)
print("Mean of Ozone: ", mean_ozone)


# In[12]:


#replace the ozone missing values with median values
data1['Ozone'] = data1['Ozone'].fillna(median_ozone)
data1.isnull().sum()


# In[17]:


fig, axes = plt.subplots(2, 1, figsize=(8,6), gridspec_kw={'height_ratios': [1,3]})
sns.boxplot(data=data1["Ozone"], ax=axes[0], color='skyblue', width=0.5, orient= 'h')
axes[0].set_title("Boxplot")
axes[0].set_xlabel("Ozone Levels")
sns.histplot(data1["Ozone"], kde=True, ax=axes[1], color='purple', bins=30)
axes[1].set_title("Histogram with KDE")
axes[1].set_xlabel("Ozone levels")
axes[1].set_ylabel("Frequency")
plt.tight_layout()
plt.show


# In[19]:


data = {
    'solar': [3.2, 4.5, 5.6, 6.1, 7.0, 5.5, 6.3, 7.4, 3.8, 6.9, 4.8, 5.3, 6.0, 4.9, 5.7]
} 
df = pd.DataFrame(data)
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
sns.boxplot(data=df, x='solar',ax=axes[0])
axes[0].set_title('Boxplot of Solar Data')
axes[0].set_xlabel('Solar Energy')
sns.histplot(df['solar'], bins=10, kde=True, ax=axes[1])
axes[1].set_title('Histogram of Solar Data')
axes[1].set_xlabe;('Solar Energy')
axes[1].set_ylabel('Frequency')
plt.tight_layout()
plt.show()


# In[ ]:


t

