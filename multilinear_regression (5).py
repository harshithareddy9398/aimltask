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


# In[7]:


sns.set_style(style='darkgrid')
sns.pairplot(cars)


# In[8]:


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


# In[10]:


##Observations from model summary
.The R-squared and adjusted R-squared values are good and about 75%of variability in y is explained by x columns
.The probability value with respect to f-statistic is close to zero,indicating that all or someof X columns are signficant
>The p-values for VOL and WT are higher than 5% indicating some interaction issue among themselves,which need to be further explored


# In[11]:


#Find the performance metrices
#create a data frame with actual y and predicted y columns
df1 = pd.DataFrame()
df1["actual_y1"] = cars["MPG"]
df1.head()


# In[12]:


#predict for the given X data columns
pred_y1 = model1.predict(cars.iloc[:,0:4])
df1["pred_y1"] = pred_y1
df1.head()


# In[13]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df1["actual_y1"],df1["pred_y1"])
print("MSE :",mse)
print("RMSE :",np.sqrt(mse))


# In[14]:


#cars.head()
# Compute VIF values
rsq_hp = smf.ols('HP~WT+VOL+SP',data=cars).fit().rsquared
vif_hp = 1/(1-rsq_hp)

rsq_wt = smf.ols('WT~HP+VOL+SP',data=cars).fit().rsquared  
vif_wt = 1/(1-rsq_wt) 

rsq_vol = smf.ols('VOL~WT+SP+HP',data=cars).fit().rsquared  
vif_vol = 1/(1-rsq_vol) 

rsq_sp = smf.ols('SP~WT+VOL+HP',data=cars).fit().rsquared  
vif_sp = 1/(1-rsq_sp) 

# Storing vif values in a data frame
d1 = {'Variables':['Hp','WT','VOL','SP'],'VIF':[vif_hp,vif_wt,vif_vol,vif_sp]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame


# In[15]:


#Observations for VIF values:
.The ideal range of VIF values shall be between 0to10.However slightly hiher values can be tolerated
.As seen from the very high VIF values for VOL and WT,it is clear that they are prone to multicollinearity problem.
.Hence it is decided to drop one of the columns(either VOL or WT) to overcome the multicollinearity.
.It is decided to drop WT and retain VOL column in further models


# In[16]:


cars1 = cars.drop("WT", axis=1)
cars1.head()


# In[17]:


#Build model2 on cars1 dataset
import statsmodels.formula.api as smf
model2=smf.ols('MPG~HP+VOL+SP', data=cars).fit()
model2.summary()


# In[18]:


#Observations from model2 summary()
.The adjusted R-suared value improved slightly to 0.76
.All the p-values for model parameters are less than 5% hence they are significant
.Therefore the HP,VOL,SP columns are finalized as the significant predictor for the MPG response variable
.There is no implrovement in MSE value


# In[19]:


#### Leverage (Hat Values):
Leverage values diagnose if a data point has an extreme value in terms of the independent variables. A point with high leverage has a great ability to influence the regre


# In[20]:


#Define variables and assign values
k = 3 # no of x-columns in cars1
n = 81 # no of observations (rows)
leverage_cutoff = 3*((k+1))/n
leverage_cutoff


# In[21]:


from statsmodels.graphics.regressionplots import influence_plot

influence_plot(model2,alpha=.05)

y=[i for i in range(-2,8)]
x=[leverage_cutoff for i in range(10)]
plt.plot(x,y,'r+')

plt.show()


# In[25]:


cars1[cars1.index.isin([65,70,76,78,79,80])]


# In[33]:


cars2=cars1.drop(cars1.index[[65,70,76,78,79,80]],axis=0).reset_index(drop=True)


# In[34]:


#Rebuild the model model
model3= smf.ols('MPG~VOL+SP+HP',data = cars2).fit()
model3.summary()


# In[35]:


#PERFORMANCE METRICES FOR MODEL3
df3= pd.DataFrame()
df3["actual_y3"] =cars2["MPG"]
df3.head()


# In[36]:


#predict on all X data columns
pred_y3 = model3.predict(cars2.iloc[:,0:3])
df3["pred_y3"] = pred_y3
df3.head()


# In[37]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df3["actual_y3"], df3["pred_y3"])
print("MSE :", mse )
print("RMSE :",np.sqrt(mse))


# In[39]:


### Comparison of models
                     

| Metric         | Model 1 | Model 2 | Model 3 |
|----------------|---------|---------|---------|
| R-squared      | 0.771   | 0.770   | 0.885   |
| Adj. R-squared | 0.758   | 0.761   | 0.880   |
| MSE            | 18.89   | 18.91   | 8.68    |
| RMSE           | 4.34    | 4.34    | 2.94    |


- **From the above comparison table it is observed that model3 is the best among all with superior performance metrics**


# In[ ]:




