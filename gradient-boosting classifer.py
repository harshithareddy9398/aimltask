#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler


# In[4]:


# Load dataset
df = pd.read_csv('diabetes.csv')
df


# In[5]:


X = df.drop('class', axis=1)
y = df['class']
# standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled


# In[6]:


X_train, X_test, y_train, y_test = train_test_split(X_scaled,y, test_size = 0.8, random_state =42)


# In[7]:


bc = GradientBoostingClassifier(random_state=42)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'subsample': [0.8, 1.0]
}

# Grid search with cross-validation
grid_search = GridSearchCV(estimator=gbc,
                          param_grid=param_grid,
                          cv=kfold,
                          scoring='recall',
                          n_jobs=-1,
                          verbose=1)

# Assuming you have your training data in X_train and y_train
grid_search.fit(X_train, y_train)

# Access the best hyperparameters and best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print(f"Best Parameters: {best_params}")
print(f"Best Recall Score: {best_score}")
Key Points:


# In[8]:


bc = GradientBoostingClassifier(random_state=42)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'subsample': [0.8, 1.0]
}

# Grid search with cross-validation
grid_search = GridSearchCV(estimator=gbc,
                          param_grid=param_grid,
                          cv=kfold,
                          scoring='recall',
                          n_jobs=-1,
                          verbose=1)


# In[9]:


best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print("\nConfusion Matrix:\n", confuion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




