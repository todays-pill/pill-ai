from django import forms

class ImageUploadForm(forms.Form):
    front_image = forms.ImageField()
    back_image = forms.ImageField()