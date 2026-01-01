from flask import Flask, render_template, request, jsonify, Response
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image, ImageEnhance
import json
import io
import base64
import numpy as np
import cv2
from model import ASLResNet
import os

app = Flask(__name__)

# Load model configuration
with open('config.json', 'r') as f:
    config = json.load(f)

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"📱 استخدام الجهاز: {device}")

try:
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
    print("✅ تم تحميل النموذج بنجاح!")
except Exception as e:
    print(f"❌ خطأ في تحميل النموذج: {e}")
    print("⚠️ سيتم استخدام النموذج بدون تحميل الوزنات المدربة")
    model = ASLResNet(num_classes=config['num_classes'])
    model.to(device)
    model.eval()

# Class names mapping
class_names = config['class_names']
id2label = {i: name for i, name in enumerate(class_names)}

# Hand detection using MediaPipe (if available) or OpenCV
try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    USE_MEDIAPIPE = True
    print("✅ MediaPipe متاح - سيتم استخدامه لكشف اليد")
except ImportError:
    USE_MEDIAPIPE = False
    print("⚠️ MediaPipe غير متاح - سيتم استخدام OpenCV لكشف اليد")

def detect_hand_mediapipe(image_np):
    """Detect hand using MediaPipe"""
    if not USE_MEDIAPIPE:
        return None
    
    try:
        results = hands.process(cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
        if results.multi_hand_landmarks:
            # Get hand bounding box
            h, w = image_np.shape[:2]
            x_coords = [landmark.x * w for hand_landmarks in results.multi_hand_landmarks 
                       for landmark in hand_landmarks.landmark]
            y_coords = [landmark.y * h for hand_landmarks in results.multi_hand_landmarks 
                       for landmark in hand_landmarks.landmark]
            
            if x_coords and y_coords:
                x_min, x_max = int(min(x_coords)), int(max(x_coords))
                y_min, y_max = int(min(y_coords)), int(max(y_coords))
                
                # Add padding
                padding = 30
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(w, x_max + padding)
                y_max = min(h, y_max + padding)
                
                return (x_min, y_min, x_max, y_max)
    except Exception as e:
        print(f"خطأ في MediaPipe: {e}")
    
    return None

def detect_hand_opencv(image_np):
    """Detect hand using OpenCV skin detection"""
    try:
        # Convert to HSV
        hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
        
        # Define skin color range
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Create mask
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Apply morphological operations
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get largest contour (likely the hand)
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            # Filter by area (hand should be reasonably large)
            h, w = image_np.shape[:2]
            min_area = (h * w) * 0.02  # At least 2% of image
            
            if area > min_area:
                x, y, w_rect, h_rect = cv2.boundingRect(largest_contour)
                
                # Add padding
                padding = 20
                x_min = max(0, x - padding)
                y_min = max(0, y - padding)
                x_max = min(w, x + w_rect + padding)
                y_max = min(h, y + h_rect + padding)
                
                return (x_min, y_min, x_max, y_max)
    except Exception as e:
        print(f"خطأ في OpenCV hand detection: {e}")
    
    return None

def enhance_image(image):
    """Enhance image quality (brightness, contrast, sharpness)"""
    # Convert to numpy for processing
    img_array = np.array(image)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for better contrast
    if len(img_array.shape) == 3:
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge channels
        lab = cv2.merge([l, a, b])
        img_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    # Convert back to PIL Image
    enhanced_image = Image.fromarray(img_array)
    
    # Additional enhancements using PIL
    enhancer = ImageEnhance.Contrast(enhanced_image)
    enhanced_image = enhancer.enhance(1.1)  # Slight contrast boost
    
    enhancer = ImageEnhance.Brightness(enhanced_image)
    enhanced_image = enhancer.enhance(1.05)  # Slight brightness boost
    
    enhancer = ImageEnhance.Sharpness(enhanced_image)
    enhanced_image = enhancer.enhance(1.1)  # Slight sharpness boost
    
    return enhanced_image

def extract_hand_region(image):
    """Extract hand region from image"""
    image_np = np.array(image)
    
    # Try MediaPipe first
    bbox = detect_hand_mediapipe(image_np)
    
    # Fallback to OpenCV
    if bbox is None:
        bbox = detect_hand_opencv(image_np)
    
    # If hand detected, crop it
    if bbox:
        x_min, y_min, x_max, y_max = bbox
        # Ensure valid bounding box
        w, h = image.size
        x_min = max(0, min(x_min, w-1))
        y_min = max(0, min(y_min, h-1))
        x_max = max(x_min+1, min(x_max, w))
        y_max = max(y_min+1, min(y_max, h))
        
        hand_crop = image.crop((x_min, y_min, x_max, y_max))
        return hand_crop, True
    
    # If no hand detected, return center crop (assume hand is in center)
    w, h = image.size
    center_x, center_y = w // 2, h // 2
    crop_size = min(w, h) * 0.75  # 75% of smaller dimension for better coverage
    x_min = max(0, int(center_x - crop_size // 2))
    y_min = max(0, int(center_y - crop_size // 2))
    x_max = min(w, int(center_x + crop_size // 2))
    y_max = min(h, int(center_y + crop_size // 2))
    
    hand_crop = image.crop((x_min, y_min, x_max, y_max))
    return hand_crop, False

# Image preprocessing with better handling and hand detection
def preprocess_image(image, detect_hand=True, enhance=True):
    """Preprocess image for model input with hand detection and enhancement"""
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Enhance image quality
    if enhance:
        image = enhance_image(image)
    
    # Extract hand region if enabled
    hand_detected = False
    if detect_hand:
        image, hand_detected = extract_hand_region(image)
    
    # Resize to 224x224 using high-quality resampling
    image = image.resize((224, 224), Image.Resampling.LANCZOS)
    
    # Convert to numpy array and normalize
    image_array = np.array(image).astype(np.float32) / 255.0
    
    # Normalize with ImageNet stats
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_array = (image_array - mean) / std
    
    # Convert to tensor (C, H, W)
    image_tensor = torch.from_numpy(image_array.transpose(2, 0, 1)).float()
    
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor.to(device), hand_detected

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Predict sign language from uploaded image"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        image = Image.open(io.BytesIO(file.read()))
        
        # Preprocess with enhancement
        image_tensor, _ = preprocess_image(image, detect_hand=False, enhance=True)
        
        # Predict
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = outputs.argmax(-1).item()
            confidence = probabilities[0][predicted_class].item()
        
        predicted_letter = id2label[predicted_class]
        
        return jsonify({
            'predicted_letter': predicted_letter,
            'confidence': float(confidence),
            'all_probabilities': {
                id2label[i]: float(prob) 
                for i, prob in enumerate(probabilities[0])
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_base64', methods=['POST'])
def predict_base64():
    """Predict sign language from base64 encoded image"""
    try:
        data = request.json
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        # Decode base64 image
        image_data = data['image'].split(',')[1] if ',' in data['image'] else data['image']
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Preprocess with hand detection and enhancement
        image_tensor, hand_detected = preprocess_image(image, detect_hand=True, enhance=True)
        
        # Predict
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = outputs.argmax(-1).item()
            confidence = probabilities[0][predicted_class].item()
        
        predicted_letter = id2label[predicted_class]
        
        # Improved filtering logic - lower thresholds for better accuracy
        # Base threshold is lower to allow more predictions through
        min_confidence = 0.25 if hand_detected else 0.35
        
        # Get top 3 predictions for better analysis
        top3_probs, top3_indices = torch.topk(probabilities[0], 3)
        top3_letters = [id2label[idx.item()] for idx in top3_indices]
        
        # Calculate confidence ratio between top 2 predictions
        confidence_ratio = top3_probs[0].item() / (top3_probs[1].item() + 1e-6)
        
        # If top prediction is clearly better (ratio > 1.5), accept it even with lower confidence
        # Only require higher confidence if predictions are very close
        if confidence_ratio < 1.3:  # Very close predictions
            min_confidence = max(min_confidence, 0.4)
        elif confidence_ratio > 2.0:  # Clear winner
            min_confidence = max(min_confidence - 0.1, 0.15)  # Lower threshold
        
        # Only filter out 'nothing' class if confidence is very low
        # Allow other predictions with reasonable confidence
        if predicted_letter == 'nothing' and confidence > 0.3:
            # If 'nothing' has high confidence but other classes are close, prefer second best
            if top3_probs[1].item() > 0.25 and top3_letters[1] != 'nothing':
                predicted_letter = top3_letters[1]
                confidence = top3_probs[1].item()
        elif confidence < min_confidence:
            # Very low confidence - return nothing
            predicted_letter = 'nothing'
            confidence = 0.0
        
        return jsonify({
            'predicted_letter': predicted_letter,
            'confidence': float(confidence),
            'hand_detected': hand_detected,
            'top3_predictions': [
                {'letter': top3_letters[i], 'confidence': float(top3_probs[i].item())}
                for i in range(len(top3_letters))
            ]
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/translate_word', methods=['POST'])
def translate_word():
    """Convert word to sign language letters"""
    try:
        data = request.json
        word = data.get('word', '').upper().strip()
        
        if not word:
            return jsonify({'error': 'No word provided'}), 400
        
        # Filter only letters
        letters = [char for char in word if char.isalpha()]
        
        return jsonify({
            'word': word,
            'letters': letters,
            'count': len(letters)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_test_letter', methods=['GET'])
def get_test_letter():
    """Get a random letter for testing"""
    import random
    # Get only alphabet letters (A-Z)
    letters = [letter for letter in class_names if letter.isalpha() and len(letter) == 1]
    random_letter = random.choice(letters)
    
    return jsonify({
        'letter': random_letter,
        'all_letters': letters
    })

@app.route('/check_answer', methods=['POST'])
def check_answer():
    """Check if the predicted sign matches the test letter"""
    try:
        data = request.json
        predicted_letter = data.get('predicted_letter', '').upper()
        correct_letter = data.get('correct_letter', '').upper()
        
        if not predicted_letter or not correct_letter:
            return jsonify({'error': 'Missing parameters'}), 400
        
        is_correct = predicted_letter == correct_letter
        
        return jsonify({
            'is_correct': is_correct,
            'predicted': predicted_letter,
            'correct': correct_letter
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("\n" + "="*50)
    print("🚀 بدء تشغيل الخادم...")
    print(f"📍 العنوان المحلي: http://127.0.0.1:{port}")
    print(f"📍 العنوان المحلي: http://localhost:{port}")
    print("="*50 + "\n")
    app.run(debug=False, host='0.0.0.0', port=port, threaded=True)

