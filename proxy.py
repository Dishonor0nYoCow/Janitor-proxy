# proxy.py
import os
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load Hugging Face token from environment variable
HF_API_KEY = os.getenv("HF_API_KEY")

# Choose a smaller, fast model that works well on free tiers
model_name = "mosaicml/mpt-7b-chat"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    use_auth_token=HF_API_KEY
)

app = FastAPI()

# Define request body
class InputData(BaseModel):
    inputs: str

# POST endpoint for generating text
@app.post("/")
async def generate(data: InputData):
    # Tokenize the input
    inputs = tokenizer(data.inputs, return_tensors="pt").to("cuda")
    
    # Generate output
    outputs = model.generate(**inputs, max_new_tokens=150)
    
    # Decode to text
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return {"generated_text": text}
