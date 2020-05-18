import json
import torch
from commons import get_model, get_tensor


classes = ["10", "100", "20", "200", "2000", "5", "50", "500", "none"]

model=get_model()
def get_denomination_name(image_bytes):
    tensor=get_tensor(image_bytes)
    outputs=model(tensor)
    _,prediction=outputs.max(1)
    category=prediction.item()
    denomination_name=classes[category]

    return denomination_name