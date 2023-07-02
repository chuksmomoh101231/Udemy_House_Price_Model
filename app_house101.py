#!/usr/bin/env python
# coding: utf-8

# In[1]:


from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import uvicorn

# Load pre-trained model
model = joblib.load("house_prediction_model-MAIN_1.pkl")

# Initialize FastAPI app
app = FastAPI()

# Define the request body for batch prediction
class PredictionInput(BaseModel):
    data: list

@app.post("/batch_predict")
async def batch_predict(input: PredictionInput):
    # Convert the incoming json data into a pandas dataframe
    df = pd.DataFrame(input.data)

    # Do the prediction
    predictions = model.predict(df)

    # Convert numpy array of predictions to a pandas dataframe
    prediction_df = pd.DataFrame(predictions, columns=['Prediction'])

    # Concatenate the original dataframe with the prediction dataframe
    result = pd.concat([df, prediction_df], axis=1)

    # Convert the resulting dataframe into a dictionary and return it as JSON
    return result.to_dict(orient='records')


# In[ ]:




