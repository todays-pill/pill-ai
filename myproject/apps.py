from django.apps import AppConfig
import tensorflow as tf
import os

class DlappConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'myproject'

    def ready(self):
        from myproject.models import load_model
        load_model()