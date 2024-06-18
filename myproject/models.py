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

# Create your model here.

# 모델을 전역 변수로 선언
model = None

# 모델 로드하는 함수
def load_model():
    global model

    # 프로젝트 디렉토리와 모델 파일 경로 설정
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_path = os.path.join(base_dir, 'myproject', 'models', 'model.json')
    weights_path = os.path.join(base_dir, 'myproject', 'models', 'model_weights.h5')

    # 모델 JSON 파일과 가중치 파일이 존재하는지 확인
    if not os.path.exists(json_path) or not os.path.exists(weights_path):
        print(f"Model file does not exist at: {json_path} or {weights_path}")
        return None

    try:
        # 모델 JSON 파일과 가중치 파일을 읽어서 모델을 로드
        print(f"Loading model from: {json_path} and {weights_path}")
        with open(json_path, 'r') as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json)
        model.load_weights(weights_path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None

# 이미지를 전처리하는 함수
def preprocess_input(image, target_size=(224, 224)):
    # 이미지를 지정된 크기로 조정
    image = image.resize(target_size)
    # 이미지를 배열로 변환하고 정규화
    image = img_to_array(image) / 255.0
    # 배치 차원을 추가하여 모델 입력 형식에 맞춤
    image = np.expand_dims(image, axis=0)
    return image

# 예측을 수행하는 함수
def predict(front_image, back_image):
    global model
    if model is None:
        load_model()
    if model is None:
        raise ValueError("Model could not be loaded.")

    # 전면 및 후면 이미지를 전처리
    front_input = preprocess_input(front_image, target_size=(224,224))
    back_input = preprocess_input(back_image, target_size=(224,224))
    # 모델을 사용하여 예측 수행
    predictions = model.predict([front_input, back_input])
    return predictions

# Base64 문자열을 이미지로 디코딩하는 함수
def decode_base64_image(base64_string):
    try:
        if isinstance(base64_string, str):
            # Base64 문자열에서 메타데이터 제거
            if ',' in base64_string:
                base64_string = base64_string.split(',')[1]

            # Base64 문자열의 길이를 4의 배수로 만들어 패딩 문제를 해결
            base64_string = base64_string + '=' * (-len(base64_string) % 4)
            # Base64 문자열을 바이트로 디코딩
            base64_bytes = base64.b64decode(base64_string)
            # 바이트 데이터를 이미지로 변환
            image = Image.open(BytesIO(base64_bytes))
        else:
            # InMemoryUploadedFile 타입 처리
            image = Image.open(base64_string)
        return image
    except Exception as e:
        print(f"Error decoding base64 image: {e}")
        return None