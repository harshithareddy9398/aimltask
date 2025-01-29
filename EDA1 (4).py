#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[9]:


data = pd.read_csv("data_clean.csv")
print(data)


# In[10]:


#printing the information
data.info()


# In[11]:


data1 = data.drop(['Unnamed: 0',"Temp C"], axis =1)
data1


# In[12]:


data1['Month']=pd.to_numeric(data['Month'],errors='coerce')
data1.info()


# In[13]:


#change column names(Rename the columns)
data1.rename({'Solar.R': 'Solar'}, axis=1, inplace = True)
data1


# In[14]:


#Display the data1 info()
data1.info()


# In[15]:


#display data1 missing values count in each column using isnull(),sum()
data1.isnull().sum()


# In[16]:


#visualize the missing values using heat app


cols = data1.columns
colors = ['black', 'yellow']
sns.heatmap(data1[cols].isnull(),cmap=sns.color_palette(colors),cbar = True)


# In[17]:


#find the mean and median values of each
#Imputation of missing value with median
median_ozone = data1["Ozone"].median()
mean_ozone = data1["Ozone"].mean()
print("Median of Ozone: ", median_ozone)
print("Mean of Ozone: ", mean_ozone)


# In[18]:


#replace the ozone missing values with median values
data1['Ozone'] = data1['Ozone'].fillna(median_ozone)
data1.isnull().sum()


# In[19]:


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


# In[22]:


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
axes[1].set_xlabel('Solar Energy')
axes[1].set_ylabel('Frequency')
plt.tight_layout()
plt.show()


# In[29]:


plt.figure(figsize=(6,2))
boxplot_data = plt.boxplot(data1["Ozone"], vert=False)
[item.get_xdata()for item in boxplot_data['fliers']]


# In[34]:


import scipy.stats as stats
plt.figure(figsize=(8,6))
stats.probplot(data1["Ozone"], dist="norm", plot=plt)
plt.title("Q-Q plot for outlier Dectection", fontsize=14)
plt.xlabel("Theoretical Quantile", fontsize=12)


# In[23]:


#create a figure  for violin plot
sns.violinplot(data=data1["Ozone"], color='lightgreen')
plt.title("Vilinplot")
#show the plot
plt.show()


# In[27]:


sns.swarmplot(data=data1, x = "Weather", y = "Ozone",color="orange",palette="Set2", size=6)


# In[30]:


sns.kdeplot(data=data1["Ozone"], fill=True, color="blue")
sns.rugplot(data=data1["Ozone"], color="black")


# In[32]:


#Category wise boxplot for ozone
sns.boxplot(data = data1, x = "Weather", y = "Ozone")


# In[35]:


plt.scatter(data1["Wind"], data1["Temp"])


# In[36]:


#Compute pearson correlation corfficient
data1["Wind"].corr(data1["Temp"])


# In[37]:


data1_numeric = data1.iloc[:,[0,1,2,6]]
data1_numeric


# In[ ]:




