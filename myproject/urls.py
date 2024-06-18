from django.urls import path
from .views import upload_image_view, predict_view, home_view

from . import views

# URL 패턴을 정의합니다.
urlpatterns = [
    # 홈 페이지 URL 패턴
    path('', home_view, name='home'),

    # 이미지 업로드 페이지 URL 패턴
    path('upload/', upload_image_view, name='upload_image'),

    # 예측 요청을 처리하는 URL 패턴
    path('predict/', predict_view, name='predict_view'),
]
