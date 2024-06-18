from django.apps import AppConfig
import tensorflow as tf
import os

# Django 앱의 설정을 정의하는 클래스
class DlappConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'  # 기본 자동 필드 설정
    name = 'myproject'  # 앱의 이름 설정

    # 앱이 준비되었을 때 실행되는 함수
    def ready(self):
        from myproject.models import load_model  # 모델 로드 함수를 임포트
        load_model()  # 모델 로드 함수 호출
