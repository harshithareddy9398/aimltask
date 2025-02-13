#!/usr/bin/env python
# coding: utf-8

# In[6]:


#Install mlxtend library
get_ipython().system('pip install mlxtend')


# In[6]:


#Import necessary libraries
import pandas as pd
import mlxtend
from mlxtend.frequent_patterns import apriori,association_rules
import matplotlib.pyplot as plt


# In[7]:


# from google.colab import files
#uploaded = files.upload()


# In[8]:


#print the dataframe
titanic = pd.read_csv("Titanic.csv")
titanic


# In[9]:


titanic.info()


# In[10]:


###observations 
- All columns are object data type and categprical in nature
- There are no null values
- As the cloumns are categorical, we can adopt one-hot-encoding


# In[11]:


# Plot a bar chrt to visualize the category of people on the ship
counts = titanic['Class'].value_counts()
plt.bar(counts.index,counts.values)


# In[12]:


counts = titanic['Age'].value_counts()
plt.bar(counts.index,counts.values)


# In[13]:


counts = titanic['Survived'].value_counts()
plt.bar(counts.index,counts.values)


# In[14]:


counts = titanic['Gender'].value_counts()
plt.bar(counts.index,counts.values)


# In[15]:


#Perform onehot encoding on categoical coumns
df=pd.get_dummies(titanic,dtype=int)
df.head()


# In[16]:


df.info()


# In[17]:


##Apply Apriori algorithm to get itemset combinations
frequent_itemsets = apriori(df, min_support = 0.05,use_colnames=True,max_len=None)
frequent_itemsets


# In[19]:


frequent_itemsets.info()


# In[21]:


rules = association_rules(frequent_itemsets,metric="lift", min_threshold=1.0)
rules


# In[22]:


rules.sort_values(by='lift', ascending =True)


# In[26]:


import matplotlib.pyplot as plt
rules[['support','confidence','lift']].hist(figsize=(15,7))
plt.show()


# In[ ]:




