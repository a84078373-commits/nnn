---
language: en
library_name: transformers
tags:
- image-classification
- resnet
- asl
- sign-language
license: mit
datasets:
- grassknoted/asl-alphabet
metrics:
- accuracy
model-index:
- name: asl-sign-language-classifier
  results:
  - task:
      type: image-classification
      name: Image Classification
    dataset:
      name: ASL Alphabet Dataset
      type: image
      split: test
    metrics:
    - name: Accuracy
      type: accuracy
      value: 0.9999
---

# ASL Sign Language Classification Model

This model is trained to recognize **American Sign Language (ASL)** alphabets using the [ASL Alphabet Dataset](https://www.kaggle.com/grassknoted/asl-alphabet).  
It uses a ResNet50 backbone for image classification.

## Model Details

- **Base Architecture**: ResNet50  
- **Number of Classes**: 29  
- **Test Accuracy**: 0.9999  
- **Dataset**: ASL Alphabet (A–Z, space, delete, nothing)

## Usage

```python
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch

# Load model and processor
model = AutoModelForImageClassification.from_pretrained("Abuzaid01/asl-sign-language-classifier")
processor = AutoImageProcessor.from_pretrained("Abuzaid01/asl-sign-language-classifier")

# Load an image
image = Image.open("asl_sample.jpg")

# Preprocess
inputs = processor(images=image, return_tensors="pt")

# Predict
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = logits.argmax(-1).item()

print("Predicted class:", model.config.id2label[predicted_class])
