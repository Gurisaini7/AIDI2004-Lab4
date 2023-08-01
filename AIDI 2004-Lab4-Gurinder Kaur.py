#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install gradio')


# In[68]:


import pandas as pd
import gradio as gr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[69]:


# Load the data
df = pd.read_csv('./Fish.csv')


# In[70]:


df.head()


# In[71]:


df.dropna()


# In[72]:


print(df.isnull().sum())


# In[73]:


# Drop the "Species" column from the DataFrame
df.drop("Species", axis=1, inplace=True)


# In[74]:


# Now fill missing values with the mean of the remaining columns
df.fillna(df.mean(), inplace=True)


# In[75]:


# Drop rows with NaN values
df.dropna(inplace=True)


# In[76]:


# Split the data into features and target
X = df[['Length1','Length2','Length3', 'Height', 'Width']]
y = df['Weight']


# In[77]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[78]:


# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[79]:


# Initialize the random forest regression model
model = RandomForestRegressor(n_estimators=100, random_state=42)


# In[80]:


# Train the model
model.fit(X_train, y_train)


# In[81]:


# Function to make predictions using the model
def predict_fish_weight(length1, length2, length3, height, width):
    data = {
        'Length1': [length1],
        'Length2':[length2],
        'Length3':[length3],
        'Height': [height],
        'Width': [width]
    }
    df = pd.DataFrame(data)
    prediction = model.predict(df)[0]
    return prediction


# # In[82]:


# # Define the input components for the Gradio interface
# inputs = [
#     gr.inputs.Number(label='Length1'),
#     gr.inputs.Number(label='Length2'),
#     gr.inputs.Number(label='Length3'),
#     gr.inputs.Number(label='Height'),
#     gr.inputs.Number(label='Width'),
# ]


# # In[83]:


# # Define the output component for the Gradio interface
# output = gr.outputs.Label()


# # In[84]:


# # Create the Gradio interface
# interface = gr.Interface(fn=predict_fish_weight, inputs=inputs, outputs=output, live=True)


# # In[85]:


# # Launch the interface with share=True to get a public link
# interface.launch(share=True)


# # In[ ]:




