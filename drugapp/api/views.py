from django.shortcuts import render
from django.http import JsonResponse
from django.views import View
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.contrib.auth.models import User, auth
from django.contrib.auth import login as auth_login
from django.contrib import messages
from django.urls import reverse
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import *
from .serializers import UserChatSerializer
from datetime import datetime
import json
import os
from dotenv import load_dotenv
# from langchain_community.chat_models import ChatAnthropic
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

load_dotenv()

# Create your views here.
def IndexView(request):
    return render(request, 'index.html', {})

# Option 1: Using Django's View
@method_decorator(csrf_exempt, name='dispatch')
class ChatbotView(View):
    def get(self, request):
        """
        Render the chatbot HTML page.
        """
        return render(request, 'chatbot.html', {'page_name': 'Chatbot'})

    def post(self, request):
        """
        Handle AJAX POST request to generate and save a chatbot response.
        """
        try:
            data = json.loads(request.body)
            question = data.get('question', '')

            if not question:
                return JsonResponse({'error': 'No question provided'}, status=400)

            # Get chatbot response
            llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")
            response = llm.invoke([HumanMessage(content=question)])
            response_text = response.content

            # Serialize and save
            chat_data = {
                "question": question,
                "response": response_text,
                "timestamp": datetime.now()
            }

            serializer = UserChatSerializer(data=chat_data)
            if serializer.is_valid():
                serializer.save()
                return JsonResponse(serializer.data, status=201)
            else:
                return JsonResponse(serializer.errors, status=400)

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

# # Option 2: Using DRF's APIView (alternative approach)
# class ChatbotAPIView(APIView):
#     def get(self, request):
#         """
#         Render the chatbot HTML page.
#         """
#         return render(request, 'chatbot.html', {'page_name': 'Chatbot'})

#     def post(self, request):
#         """
#         Handle API POST request to generate and save a chatbot response.
#         """
#         try:
#             question = request.data.get('question', '')

#             if not question:
#                 return Response({'error': 'No question provided'}, status=status.HTTP_400_BAD_REQUEST)

#             # Get chatbot response
#             llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")
#             response = llm.invoke([HumanMessage(content=question)])
#             response_text = response.content

#             # Serialize and save
#             chat_data = {
#                 "question": question,
#                 "response": response_text,
#                 "timestamp": datetime.now()
#             }

#             serializer = UserChatSerializer(data=chat_data)
#             if serializer.is_valid():
#                 serializer.save()
#                 return Response(serializer.data, status=status.HTTP_201_CREATED)
#             else:
#                 return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

#         except Exception as e:
#             return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
