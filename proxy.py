# proxy.py
import os
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load Hugging Face token from environment variable
HF_API_KEY = os.getenv("HF_API_KEY")

# Model to use
model_name = "mosaicml/mpt-7b-chat"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map={"": "cpu"},    # Force CPU usage
    use_auth_token=HF_API_KEY
)

app = FastAPI()

# Define request body structure
class InputData(BaseModel):
    inputs: str

# POST endpoint
@app.post("/")
async def generate(data: InputData):
    # Tokenize input and move to CPU
    inputs = tokenizer(data.inputs, return_tensors="pt").to("cpu")
    
    # Generate output
    outputs = model.generate(**inputs, max_new_tokens=150)
    
    # Decode text
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return {"generated_text": text}
