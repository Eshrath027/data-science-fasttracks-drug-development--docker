from django.contrib import admin
from django.urls import path, include
from drugapp import settings
from . import views
from .views import IndexView, ChatbotView

# from rest_framework import routers
# router= routers.SimpleRouter(trailing_slash=True)

# router.register(r'medilib', views.IndexView, basename='index')
# urlpatterns = []
# urlpatterns += router.urls



urlpatterns = [
    path('medilib/', IndexView, name='index'),
    path('medilib/chat/', ChatbotView.as_view(), name='chat'),
]
