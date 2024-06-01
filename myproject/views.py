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

def load_label_map():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    label_map_path = os.path.join(base_dir, 'myproject', 'models', 'label_map_ver2.json')

    with open(label_map_path, 'r') as file:
        label_map = json.load(file)

    return label_map


def home_view(request):
    return render(request, 'myproject/home.html')


@csrf_exempt
def predict_view(request):
    if request.method == 'POST':
        try:
            front_image = request.FILES.get('front')
            back_image = request.FILES.get('back')

            # Base64 인코딩된 이미지를 디코딩
            front_image = decode_base64_image(front_image)
            back_image = decode_base64_image(back_image)

            if front_image is None or back_image is None:
                return JsonResponse({'error': 'Invalid image data'}, status=400)

            predictions = predict(front_image, back_image)

            # 가장 높은 확률 값을 가진 클래스 인덱스를 찾음
            predicted_class_index = int(np.argmax(predictions[0]))

            # 클래스 인덱스를 클래스 이름으로 변환
            label_map = load_label_map()
            predicted_class_name = label_map[str(predicted_class_index)]

            return JsonResponse({
                'predictions': predictions.tolist(),
                'predicted_class_index': predicted_class_index,
                'predicted_class_name': predicted_class_name

            })
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=405)

@csrf_exempt
def upload_image_view(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            front_image = form.cleaned_data['front_image']
            back_image = form.cleaned_data['back_image']
            front_image_path = default_storage.save('tmp/' + front_image.name, ContentFile(front_image.read()))
            back_image_path = default_storage.save('tmp/' + back_image.name, ContentFile(back_image.read()))

            full_front_image_path = os.path.join(default_storage.location, front_image_path)
            full_back_image_path = os.path.join(default_storage.location, back_image_path)

            with open(full_front_image_path, 'rb') as front_image_file, open(full_back_image_path,
                                                                             'rb') as back_image_file:
                front_img = Image.open(front_image_file)
                back_img = Image.open(back_image_file)
                predictions = predict(front_img, back_img)

            result = "This is a pill." if predictions[0][0] > 0.5 else "This is not a pill."

            return render(request, 'myproject/result.html',
                          {'result': result, 'front_image_url': default_storage.url(front_image_path),
                           'back_image_url': default_storage.url(back_image_path)})
    else:
        form = ImageUploadForm()

    return render(request, 'myproject/upload.html', {'form': form})