#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np


# In[3]:


df = pd.read_csv("Universities.csv")
df


# In[4]:


np.mean(df["SAT"])


# In[5]:


np.median(df["SAT"])


# In[6]:


np.var(df["SFRatio"])


# In[7]:


df.describe()


# In[8]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[9]:


plt.hist(df["GradRate"])


# In[10]:


plt.figure()
plt.title("graduation Rate")
plt.hist(df["GradRate"])


# In[13]:


s = [20,15,10,25,30,45,28,55,65,70]
scores = pd.Series(s)
scores


# In[17]:


plt.boxplot(scores,vert=False)


# In[22]:


s = [120,]
scores = pd.Series(s)
scores


# In[23]:


plt.boxplot(scores,vert=False)


# In[24]:


df = pd.read_csv("Universities.csv")
df


# In[25]:


plt.boxplot(df["SAT"])


# In[26]:


plt.figure(figsize=(6,2))
plt.title("Box plot for SAT Score")
plt.boxplot(df["SAT"],vert = False)


# In[ ]:





# In[ ]:




