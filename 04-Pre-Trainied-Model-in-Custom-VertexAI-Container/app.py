from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List
import joblib

# Load trained model
model = joblib.load("model.joblib")

app = FastAPI()

class RequestBody(BaseModel):
    instances: List[str]

# Health endpoints
@app.get("/")
@app.get("/health")
def health_check():
    return {"status": "healthy"}

# Fallback GET handler so Vertex AI health checks don't fail on odd paths
@app.get("/{full_path:path}")
def catch_all(full_path: str, request: Request):
    return {"status": "ok", "path": full_path}
    
# Custom predict (good for local testing)
@app.post("/predict")
def predict(request: RequestBody):
    preds = model.predict(request.instances)
    return {"predictions": preds.tolist()}

# Vertex AI route (official prediction path)
@app.post("/v1/models/{model}:predict")
def predict_vertex_ai(model: str, request: RequestBody):
    preds = model.predict(request.instances)
    return {"predictions": preds.tolist()}

@app.post("/v1/endpoints/{endpoint}/deployedModels/{deployed_model}:predict")
def predict_endpoint(endpoint: str, deployed_model: str, request: RequestBody):
    preds = model.predict(request.instances)
    return {"predictions": preds.tolist()}