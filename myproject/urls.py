from django.urls import path
from .views import upload_image_view, predict_view, home_view

from . import views

urlpatterns = [
    path('', home_view, name='home'),
    path('upload/', upload_image_view, name='upload_image'),
    path('predict/', predict_view, name='predict_view'),
]