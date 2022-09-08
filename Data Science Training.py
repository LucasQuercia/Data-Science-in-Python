#!/usr/bin/env python
# coding: utf-8

# In[124]:


import pandas
from pandas import DataFrame
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression


# In[125]:


data = pandas.read_csv('valor_2.csv')


# In[126]:


data.describe()


# In[ ]:


X = DataFrame(data, columns=['production_budget_usd'])
y = DataFrame(data, columns=['worldwide_gross_usd'])


# In[141]:


plt.figure(figsize=(10,6))
plt.scatter(X, y, alpha=0.3)
plt.title('Film Cost vs Global Revenue')
plt.xlabel('Production Budget $')
plt.ylabel('Worldwide Gross $')
plt.ylim(0, 3000000000)
plt.xlim(0, 450000000)
plt.show()


# In[1]:


import life as hitchhikersGuide


# In[2]:


import math


# In[4]:


result = hitchhikersGuide.square_root(63.14)
print(result)


# In[ ]:





# In[ ]:




