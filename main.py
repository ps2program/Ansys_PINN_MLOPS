import onnxruntime as ort
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

# Create FastAPI instance
app = FastAPI()

# Define the input data structure
class InputData(BaseModel):
    input: List[float]

# Load the ONNX model
onnx_model = ort.InferenceSession("pinn_model.onnx")

@app.post("/predict/")
async def predict(data: InputData):
    # Convert input data to numpy array
    input_data = np.array(data.input, dtype=np.float32).reshape(1, -1)
    
    # Get the input name for the model (assuming it's the first input)
    input_name = onnx_model.get_inputs()[0].name
    
    # Run the model to get the output
    output = onnx_model.run(None, {input_name: input_data})

    # Return the output
    return {"output": output[0].tolist()}

'''
curl -X 'POST' \
  'http://127.0.0.1:8000/predict/' \
  -H 'Content-Type: application/json' \
  -d '{
  "input": [0.5, -0.2, 0.1]
}'

'''


'''

http://127.0.0.1:8000/docs#/default/predict_predict__post

'''