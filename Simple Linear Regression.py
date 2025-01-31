#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


# In[3]:


data1 = pd.read_csv("NewspaperData.csv")
data1


# In[9]:


data1.info()
data1


# In[10]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Generate random data
data = np.random.randn(1000) 

# Create a figure and axes
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 1. Boxplot
sns.boxplot(data=data, ax=axes[0], color='lightblue')
axes[0].set_title('Boxplot')
axes[0].set_xlabel('Data')
axes[0].set_ylabel('Values')

# 2. Histogram
sns.histplot(data, kde=False, ax=axes[1], color='orange', bins=30)
axes[1].set_title('Histogram')
axes[1].set_xlabel('Data')
axes[1].set_ylabel('Frequency')

# 3. KDE plot
sns.kdeplot(data, ax=axes[2], color='green', shade=True)
axes[2].set_title('KDE (Kernel Density Estimate)')
axes[2].set_xlabel('Data')
axes[2].set_ylabel('Density')

# Add a title to the whole figure
fig.suptitle('Boxplot, Histogram, and KDE', fontsize=16)

# Show the plots
plt.tight_layout()
plt.subplots_adjust(top=0.85)  # Adjust title to not overlap
plt.show()


# In[18]:


df = sns.load_dataset('tips')
plt.figure(figsize=(10,6))
sns.scatterplot(x='day',y='total_bill', data=df)
plt.title('scatter of total Bill by Day')
plt.show()


# In[19]:


#import numpy as np
# x = np.arange(10)
# plt.plot(2 + 3 *x)
#plt.show()


# In[ ]:




