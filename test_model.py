"""
سكريبت اختبار بسيط للتحقق من عمل النموذج
"""
import torch
import json
from model import ASLResNet
from PIL import Image
import numpy as np

def test_model():
    print("Testing model loading...")
    
    try:
        # Load config
        with open('config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        print(f"Config loaded: {config['num_classes']} classes")
        
        # Initialize model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {device}")
        
        model = ASLResNet(num_classes=config['num_classes'])
        state_dict = torch.load('pytorch_model.bin', map_location=device)
        
        # Handle different model save formats
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            model.load_state_dict(state_dict['model_state_dict'])
        elif isinstance(state_dict, dict) and 'state_dict' in state_dict:
            model.load_state_dict(state_dict['state_dict'])
        else:
            model.load_state_dict(state_dict)
        
        model.to(device)
        model.eval()
        
        print("Model loaded successfully!")
        
        # Test prediction
        print("\nTesting prediction...")
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        
        with torch.no_grad():
            output = model(dummy_input)
            predicted = output.argmax(-1).item()
        
        class_names = config['class_names']
        predicted_class = class_names[predicted]
        
        print(f"Prediction works! Result: {predicted_class}")
        print(f"Number of classes: {len(class_names)}")
        print(f"Classes: {', '.join(class_names[:10])}...")
        
        return True
        
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_model()
    if success:
        print("\nEverything works correctly!")
    else:
        print("\nPlease check files and settings")

