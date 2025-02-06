#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.graphics.regressionplots import influence_plot
import numpy as np


# In[2]:


# read the data from csv file
cars = pd.read_csv("Cars.csv")
cars.head()


# In[3]:


# Rearrange the columns
cars = pd.DataFrame(cars, columns=["HP","VOL","SP","WT","MPG"])
cars.head()


# In[4]:


cars.isna().sum()


# In[5]:


fig,(ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15,.85)})
sns.boxplot(data=cars, x='HP', ax=ax_box, orient='h')
ax_box.set(xlabel='')
sns.histplot(data=cars, x='HP', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')
plt.tight_layout()
plt.show()


# In[6]:


cars[cars.duplicated()]


# In[10]:


sns.set_style(style='darkgrid')
sns.pairplot(cars)


# In[ ]:


### Observations from correlation plots and coefficients
.Between x and y, all the x variables are showing moderate to high correlation strength, hihest being between hp and mpg
.Therefore this dataset qualifies for building a multiple linear regression model to predict MPG
.Among x columns (x1,x2,x3 andx4),some very high correlation strengths are observed between SP vsHP, VOL vs WT
.The high correlation among x columns is not desirable as it might lead to multicollinearity problem


# In[9]:


#Building model
#import statsmodels.formula.api as smf
model1 = smf.ols('MPG~WT+VOL+SP+HP',data=cars).fit()
model1.summary()


# In[ ]:


##Observations from model summary
.The R-squared and adjusted R-squared values are good and about 75%of variability in y is explained by x columns
.The probability value with respect to f-statistic is close to zero,indicating that all or someof X columns are signficant
>The p-values for VOL and WT are higher than 5% indicating some interaction issue among themselves,which need to be further explored


# In[13]:


#Find the performance metrices
#create a data frame with actual y and predicted y columns
df1 = pd.DataFrame()
df1["actal_y1"] = cars["MPG"]
df1.head()


# In[ ]:




