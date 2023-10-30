#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# In[2]:


df= pd.read_csv('bottle.csv')


# In[3]:


print(df.shape)


# In[4]:


print(df.info())


# In[35]:


print(df.describe())


# In[5]:


numeric_df = df.select_dtypes(include='number')
corr_matrix = numeric_df.corr()
print(corr_matrix)
sns.heatmap(corr_matrix, cmap="viridis")
plt.show()


# In[6]:


df1 = df[[ 'Cst_Cnt', 'Depthm', 'T_degC', 'Salnty', 'O2ml_L', 'STheta']]
df1.head()
cor = df1.corr()
sns.heatmap(cor, cmap="YlGnBu", annot=True)
plt.show()


# In[7]:


df1.isnull().sum()
df1.dropna(axis=0, inplace=True)
print(df1.shape)


# In[8]:


df1=df1.sample(1000)
print(df1.shape)
print(df1.describe())


# In[9]:


plt.scatter(df1['T_degC'], df1['Salnty'], alpha=0.8)
plt.xlabel('Water Temp')
plt.ylabel('Water Salinity')
plt.show()


# In[10]:


plt.scatter(df['Depthm'], df['T_degC'], alpha=0.8)
plt.xlabel("Depth")
plt.ylabel("Temperature")
plt.title("Temperature vs Depth")
plt.show()


# In[11]:


variable_of_interest = 'T_degC'
plt.figure(figsize=(10, 6))  # Set the figure size
plt.hist(df[variable_of_interest], bins=20, color='skyblue', edgecolor='black')  # Customize bin count and colors
plt.title(f'Histogram of {variable_of_interest}')  # Set the title
plt.xlabel(variable_of_interest)  # Label the x-axis
plt.ylabel('Frequency')  # Label the y-axis
plt.grid(axis='y', alpha=0.75)  # Add grid lines
plt.show()


# In[12]:




# Assuming you have loaded your DataFrame 'df1'

X = df1['T_degC'].values
Y = df1['Salnty'].values

# Create histograms for 'T_degC' and 'Salnty'
plt.figure(figsize=(10, 5))  # Set the figure size

# Histogram for 'T_degC'
plt.hist(X, bins=30, color='blue', alpha=0.5, label='T_degC')
# Histogram for 'Salnty'
plt.hist(Y, bins=30, color='red', alpha=0.5, label='Salnty')

# Add labels and a legend
plt.xlabel('Temperature (°C) or Salinity')
plt.ylabel('Frequency')
plt.legend()

# Show the plot
plt.show()


# In[13]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# In[14]:


# Linear regression from scratch using Gd
# Let's assume the dataset contains two columns 'X' and 'Y', and we want to predict 'Y' based on 'X'.
# Extract features and target
#X = df['T_degC'].values
#Y = df['Salnty'].values


# In[15]:


# calculation of r squared and mean sq error
target_column = "T_degC"
independent_column = "Salnty"

X = df[independent_column].values.reshape(-1, 1)
y = df[target_column].values.reshape(-1, 1)
X = df[independent_column].values.reshape(-1, 1)
y = df[target_column].values.reshape(-1, 1)
# Create a Linear Regression model
model = LinearRegression()
X_train = X_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)

# Fit the model to the training data
model.fit(X_train, Y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)
# Calculate R-squared
r2 = r2_score(Y_test, y_pred)
print(f"R-squared: {r2}")

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(Y_test, y_pred)
print(f"Mean Squared Error: {mse}")


# # Multiple Linear Regression

# In[18]:


# calculation of r squared and mean sq error
target_column = "Salnty"
independent_column = ['Depthm', 'T_degC']

X = df[independent_column].values.reshape(-1, 1)
y = df[target_column].values.reshape(-1, 1)
X = df[independent_column].values.reshape(-1, 1)
y = df[target_column].values.reshape(-1, 1)
# Create a Linear Regression model
model = LinearRegression()
X_train = X_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)

# Fit the model to the training data
model.fit(X_train, Y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)
# Calculate R-squared
r2 = r2_score(Y_test, y_pred)
print(f"R-squared: {r2}")

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(Y_test, y_pred)
print(f"Mean Squared Error: {mse}")


# In[23]:


# check shapes
print("X shape:", X.shape)
print("y shape:", Y.shape)


# # Sklearn Linear Regression

# In[24]:


model = LinearRegression()


# In[26]:


model.fit(X_train, Y_train)


# In[27]:


y_pred = model.predict(X_test)


# In[31]:


mse = mean_squared_error(Y_test, y_pred)
r2 = r2_score(Y_test, y_pred)
print("Mean Squared Error:", mse)
print("R-squared:", r2)


# In[32]:


intercept = model.intercept_
coefficients = model.coef_
print("Intercept (b0):", intercept)
print("Coefficients (b1, b2, ...):", coefficients)


# In[34]:


plt.scatter(X_train, Y_train, alpha=0.8)
plt.plot(X_test, y_pred, c='orange')
plt.xlabel('Water Temp')
plt.ylabel('Water Salnity')
plt.title('Linear Regression')
plt.show()


# In[ ]:




