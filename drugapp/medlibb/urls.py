
from django.urls import path
from . import views

import medlibb

urlpatterns = [
    path('',views.index,name='index'),
    path('base/', views.base, name='base'),
    # path('search-results/', views.search_results, name='search_results'),
    path('login/', views.login_user, name='login'),
    path('logout/', views.logout_user, name='logout'),
    path('news/', views.news_articles_view, name='news_articles_view'),
    path('chatbot/', views.chatbot, name='chatbot'),
    path('analysis/', views.analysis, name='analysis'),
    path('ddi/', views.ddi, name='ddi'),
    path("register/", views.register, name="register"),

]
