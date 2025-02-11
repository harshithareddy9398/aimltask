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


# In[5]:


cols = Univ1.columns


# In[6]:


#Standardisation function
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_Univ_df = pd.DataFrame(scaler.fit_transform(Univ1),columns = cols )
scaled_Univ_df
#scaler.fit_transform(Univ1)


# In[9]:


#Build 3 clusters using KMeans Cluster algorithm
from sklearn.cluster import KMeans
clusters_new = KMeans(3, random_state=0) #specify 3 clusters
clusters_new.fit(scaled_Univ_df)


# In[11]:


#print the cluster labels
clusters_new.labels_


# In[12]:


set(clusters_new.labels_)


# In[13]:


#Assign clusters to the Univ data set
Univ['clusterid_new'] = clusters_new.labels_
Univ


# In[14]:


Univ.sort_values(by = "clusterid_new")


# In[16]:


#use groupby() to find aggregated (mean) values in each cluster
Univ.iloc[:,1:].groupby("clusterid_new").mean()


# In[ ]:


###Observations:
.Custer 2 appears to be the top rated universities cluster as the cutt off,top10,SFRatio parameter mean values are highest
.Cluster 1 appears to occupy the middle level rated universties
.Cluster 0 comes as the lower level rated universities


# In[20]:


wcss = []
for i in range(1, 20):
    kmeans = KMeans(n_clusters=i,random_state=0)
    kmeans.fit(scaled_Univ_df)
    wcss.append(kmeans.inertia_)
print(wcss)
plt.plot(range(1, 20), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[22]:


#Build 3 clusters using KMeans cluster algorithm
from sklearn.cluster import KMeans
clusters_new = KMeans(3,random_state=0)
clusters_new.fit(scaled_Univ_df)


# In[ ]:




