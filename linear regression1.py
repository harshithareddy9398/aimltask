#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


# In[3]:


data1 = pd.read_csv("NewspaperData.csv")
data1


# In[4]:


data1.info()
data1


# In[5]:


data1.describe()


# In[6]:


#Boxplot for daily column
plt.figure(figsize=(6,3))
plt.title("Box plot for Daily Sales")
plt.boxplot(data1["daily"], vert = False)
plt.show()


# In[7]:


sns.histplot(data1['daily'], kde = True,stat= 'density',)
plt.show()


# In[8]:


sns.histplot(data1['sunday'], kde = True,stat= 'density',)
plt.show()


# In[ ]:


###observations
. There are no missing values
. There daily coloumn values appears to be right-skewed
. The sunday coloumn values also appear to be right-skewed
. There are two outliers in both daily column and also observed from the 


# In[10]:


data1["x= data1["daily"]
y= data1["sunday"]
plt.scatter(data1["daily"], data1["sunday"])
plt.xlim(0, max(x) + 100)
plt.ylim(0, max(y) + 100)
plt.show()


# In[11]:


data1["daily"].corr(data1["sunday"])


# In[12]:


data1[["daily","sunday"]].corr()


# In[ ]:


###observations
. the relationship between x(daily) and y(sunday) is seen to be linear as seen to be linear as seen from scatter plot
. the correlation is strong and positive with pearson's correlation of 0.958154


# In[14]:


#Build regression model
import statsmodels.formula.api as smf
model1 = smf.ols("sunday~daily",data = data1).fit()
model1.summary()


# In[ ]:




