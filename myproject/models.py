from django.db import models
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model as keras_load_model
from tensorflow.keras.models import model_from_json

import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
import os
import base64
from io import BytesIO

# Create your models here.
model = None

# 커스텀 손실 함수 및 메트릭 정의
def load_model():
    global model
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_path = os.path.join(base_dir, 'myproject', 'models', 'model_ver1.json')
    weights_path = os.path.join(base_dir, 'myproject', 'models', 'model_weights_ver1.h5')

    if not os.path.exists(json_path) or not os.path.exists(weights_path):
        print(f"Model file does not exist at: {json_path} or {weights_path}")
        return None

    try:
        print(f"Loading model from: {json_path} and {weights_path}")
        with open(json_path, 'r') as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json)
        model.load_weights(weights_path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None



# def preprocess_input(image):
#     image = image.resize((224, 224))
#     image = np.array(image) / 255.0
#     image = np.expand_dims(image, axis=0)
#     return image

# def preprocess_input(image_path, target_size):
#     image = Image.open(image_path).resize(target_size)
#     image = img_to_array(image) / 255.0
#     image = np.expand_dims(image, axis=0)
#     return image


def preprocess_input(image, target_size=(224, 224)):
    image = image.resize(target_size)
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image
def predict(front_image, back_image):
    global model
    if model is None:
        print("Model is None, loading model...")  # 디버깅 로그 추가
        load_model()
    if model is None:
        print("Model could not be loaded after attempting to load.")  # 디버깅 로그 추가
        raise ValueError("Model could not be loaded.")
    print(f"Model loaded: {model is not None}")  # 디버깅 로그 추가

    front_input = preprocess_input(front_image, target_size=(224,224))
    back_input = preprocess_input(back_image, target_size=(224,224))
    print("Predicting...")  # 디버깅 로그 추가
    predictions = model.predict([front_input, back_input])
    print(f"Predictions: {predictions}")  # 디버깅 로그 추가
    return predictions


def decode_base64_image(base64_string):
    try:
        if isinstance(base64_string, str):
            # Base64 문자열에서 메타데이터 제거
            if ',' in base64_string:
                base64_string = base64_string.split(',')[1]

            # Base64 문자열의 길이를 4의 배수로 만들어 패딩 문제를 해결
            base64_string = base64_string + '=' * (-len(base64_string) % 4)
            base64_bytes = base64.b64decode(base64_string)
            image = Image.open(BytesIO(base64_bytes))
        else:
            # InMemoryUploadedFile 타입 처리
            image = Image.open(base64_string)
        return image
    except Exception as e:
        print(f"Error decoding base64 image: {e}")
        return None