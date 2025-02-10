#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans


# In[2]:


Univ = pd.read_csv("Universities.csv")
Univ


# In[3]:


Univ.describe()


# In[4]:


#Read all numeric columns in to Univ1
Univ1 = Univ.iloc[:,1:]
Univ1


# In[7]:


cols = Univ1.columns


# In[8]:


#Standardisation function
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_Univ_df = pd.DataFrame(scaler.fit_transform(Univ1),columns = cols )
scaled_Univ_df
#scaler.fit_transform(Univ1)


# In[ ]:




