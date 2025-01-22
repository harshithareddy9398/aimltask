#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
import numpy as np


# In[13]:


df = pd.read_csv("Universities.csv")
df


# In[14]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[15]:


plt.figure(figsize=(4,6))
plt.title("Graduation Rate")
plt.hist(df["GradRate"])


# In[ ]:




