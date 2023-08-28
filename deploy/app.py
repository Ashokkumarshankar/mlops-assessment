import mlflow
import mlflow.pyfunc
from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = FastAPI()

class ModelWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        # Load the model from MLflow
        model_path = context.artifacts['model_path']
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def preprocess_input(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
        return inputs

    def predict(self, context, model_input):
        inputs = self.preprocess_input(model_input['input'])
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probabilities = torch.softmax(logits, dim=1)
        return probabilities.tolist()

model = ModelWrapper()

@app.post("/predict")
async def predict(input_data: dict):
    model_output = model.predict(None, input_data)
    return {"output": model_output}
