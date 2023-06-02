"""gender_recognition URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.urls import path
from recognizer.views import gender_prediction
from recognizer.views import about
from recognizer.views import gender_voice

urlpatterns = [
    path('gender_prediction/', gender_prediction, name='gender_prediction'),
    path('gender_prediction/gender_prediction', gender_prediction, name='gender_prediction'),
    path('gender_voice/', gender_voice, name='gender_voice'),
    path('gender_voice/gender_voice', gender_voice, name='gender_voice'),
    path('gender_voice/gender_prediction', gender_prediction, name='gender_prediction'),
    path('gender_prediction/gender_voice', gender_voice, name='gender_voice'),
    path('about',about, name='about'),
    path('',about, name='about'),
]