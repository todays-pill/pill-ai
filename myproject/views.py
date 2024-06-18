from django.shortcuts import render
from django.http import JsonResponse
from rest_framework.parsers import MultiPartParser
from rest_framework.views import APIView
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from .models import predict, preprocess_input, decode_base64_image
from .forms import ImageUploadForm
from PIL import Image

import json
import os
import numpy as np
import base64
from io import BytesIO

# Create your views here.

# JSON 파일에서 라벨 맵을 로드하는 함수
def load_label_map():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    label_map_path = os.path.join(base_dir, 'myproject', 'models', 'label_map_ver2.json')

    with open(label_map_path, 'r') as file:
        label_map = json.load(file)

    return label_map

# 홈페이지를 렌더링하는 함수
def home_view(request):
    return render(request, 'myproject/home.html')

# 예측 요청을 처리하는 뷰
@csrf_exempt
def predict_view(request):
    if request.method == 'POST':
        try:
            # 요청에서 전면 및 후면 이미지 파일을 가져옴
            front_image = request.FILES.get('front')
            back_image = request.FILES.get('back')

            # Base64 인코딩된 이미지를 디코딩
            front_image = decode_base64_image(front_image)
            back_image = decode_base64_image(back_image)

            # 이미지가 유효하지 않으면 에러 반환
            if front_image is None or back_image is None:
                return JsonResponse({'error': 'Invalid image data'}, status=400)

            # 이미지 예측 수행
            predictions = predict(front_image, back_image)

            # 가장 높은 확률 값을 가진 클래스 인덱스를 찾음
            predicted_class_index = int(np.argmax(predictions[0]))

            # 클래스 인덱스를 클래스 이름으로 변환
            label_map = load_label_map()
            predicted_class_name = label_map[str(predicted_class_index)]

            # 예측 결과를 JSON으로 반환
            return JsonResponse({
                'predictions': predictions.tolist(),
                'predicted_class_index': predicted_class_index,
                'predicted_class_name': predicted_class_name

            })
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=405)

# 이미지 업로드 요청을 처리하는 뷰
@csrf_exempt
def upload_image_view(request):
    if request.method == 'POST':
        # 업로드된 이미지가 있는지 확인
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            front_image = form.cleaned_data['front_image']
            back_image = form.cleaned_data['back_image']

            # 이미지를 임시 디렉토리에 저장
            front_image_path = default_storage.save('tmp/' + front_image.name, ContentFile(front_image.read()))
            back_image_path = default_storage.save('tmp/' + back_image.name, ContentFile(back_image.read()))

            full_front_image_path = os.path.join(default_storage.location, front_image_path)
            full_back_image_path = os.path.join(default_storage.location, back_image_path)

            # 저장된 이미지를 열어 예측 수행
            with open(full_front_image_path, 'rb') as front_image_file, open(full_back_image_path,
                                                                             'rb') as back_image_file:
                front_img = Image.open(front_image_file)
                back_img = Image.open(back_image_file)
                predictions = predict(front_img, back_img)

            # 예측 결과를 바탕으로 메시지 생성
            result = "This is a pill." if predictions[0][0] > 0.5 else "This is not a pill."

            # 결과 페이지 렌더링
            return render(request, 'myproject/result.html',
                          {'result': result, 'front_image_url': default_storage.url(front_image_path),
                           'back_image_url': default_storage.url(back_image_path)})
    else:
        form = ImageUploadForm()

    # 이미지 업로드 페이지 렌더링
    return render(request, 'myproject/upload.html', {'form': form})