import pickle
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import numpy as np

class Iris(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

app = FastAPI()
file = open("model.pkl","rb")
classifier=pickle.load(file)
@app.post('/predict')
def predict_iris(request:Iris):
    request = request.dict()
    sepal_length = request['sepal_length']
    sepal_width = request['sepal_width']
    petal_length = request['petal_length']
    petal_width = request['petal_width']
    prediction = classifier.predict([[sepal_length,sepal_width,petal_length,petal_width]])
    output = prediction[0]
    if (output == 0):
        output = 'Setosa'
    elif (output == 1):
        output = 'Versicolor'
    else:
        output = 'Virginica'
    return { 'Data Recieved':request,'prediction': output}


if __name__=='__main__':
    uvicorn.run(app,host='127.0.0.1', port=8000)