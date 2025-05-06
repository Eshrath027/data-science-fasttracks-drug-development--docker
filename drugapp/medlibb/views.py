import json
import sys
import os
from .models import *
from django.contrib.auth.models import User, auth
from django.contrib.auth import login as auth_login
from django.contrib import messages
from django.urls import reverse
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from django.views.decorators.csrf import ensure_csrf_cookie,csrf_exempt

# Get the current directory of the Django project
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# SRC_DIR = os.path.join(BASE_DIR, 'src')
# sys.path.append(SRC_DIR)



import sys
from django.http import JsonResponse
import google

# sys.path.append("mixtral-offloading")
# import torch
# from torch.nn import functional as F
# from src.quantize import BaseQuantizeConfig
# from huggingface_hub import snapshot_download
# from IPython.display import clear_output
# from tqdm.auto import trange
# from transformers import AutoConfig, AutoTokenizer
# from transformers.utils import logging as hf_logging
# import torch
# from transformers import TextStreamer
# import numpy
# from huggingface_hub import snapshot_download
# from IPython.display import clear_output
# from tqdm.auto import trange
# from transformers.utils import logging as hf_logging

# from src.build_model import OffloadConfig, QuantConfig, build_model
import openai
import medlibb.creds.creds as creds
from django.shortcuts import render, redirect
from GoogleNews import GoogleNews
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from .models import UserQuery
from django.shortcuts import render
import requests
from newspaper import Article
from transformers import BartTokenizer, BartForConditionalGeneration
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from heapq import nlargest
import numpy
from IPython.display import clear_output
import os
from django.shortcuts import render

import google.generativeai as genai
from django.views.decorators.csrf import csrf_exempt


import nltk
nltk.download('stopwords')
nltk.download('punkt')

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Now you can access the environment variables like this
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
API_URL = os.getenv("API_URL")
API_KEY = os.getenv("API_KEY")

# from googlesearch import search


# def search_latest_medical_articles(query, num_articles=10):
#     print("abcd",query)
#     googlenews = GoogleNews(lang='en', period='7d', encode='utf-8')
#     googlenews.search(query)
#     news_items = googlenews.result()[:num_articles]
#     return news_items



def calculate_sentence_scores(text, stopwords):
    word_frequencies = Counter()
    sentence_scores = {}

    # Tokenize the text into sentences and words
    sentences = sent_tokenize(text)
    for sentence in sentences:
        words = word_tokenize(sentence.lower())
        for word in words:
            if word not in stopwords:
                word_frequencies[word] += 1

    # Calculate sentence scores based on word frequencies
    for sentence in sentences:
        words = word_tokenize(sentence.lower())
        for word in words:
            if word in word_frequencies:
                if len(sentence.split(' ')) < 30:
                    if sentence not in sentence_scores:
                        sentence_scores[sentence] = word_frequencies[word]
                    else:
                        sentence_scores[sentence] += word_frequencies[word]

    return sentence_scores

def generate_summary(text, num_sentences):
    stop_words = set(stopwords.words('english'))
    sentence_scores = calculate_sentence_scores(text, stop_words)
    summarized_sentences = nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
    return ' '.join(summarized_sentences)

def summarize_url_content(url, num_sentences=3):
    try:
        # Extract text content from the URL using newspaper3k
        article = Article(url)
        article.download()
        article.parse()
        text = article.text

        # Generate summary
        summary = generate_summary(text, num_sentences)
        return summary
    except Exception as e:
        return f"Error: {str(e)}"



def fetch_latest_news_articles(query, num_articles):
    try:

        full_api_url = f"{API_URL}?apiKey={API_KEY}"

        # Make a GET request to the API endpoint
        response = requests.get(full_api_url, params={'q': query, 'pageSize': num_articles})

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Extract the response data (assuming JSON response)
            data = response.json()
            # Return the articles
            return data['articles']
        else:
            # Print an error message if the request was not successful
            print(f"Error: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        # Handle exceptions if any
        print(f"Error: {e}")
        return None

def news_articles_view(request):
    # query = 'medical articles of brain'
    query = request.GET.get('query', '')
    # query = f"recent medical articles about {query_param}" if query_param else None
    num_articles = 10

    if query:
        user_query = UserQuery(query_text=query)
        user_query.save()
    # Fetch latest news articles based on the query and number of articles
    articles = fetch_latest_news_articles(query, num_articles)


    for article in articles:
        summary=  summarize_url_content(article['url'], 5)
        article["summary"]= summary

    # Pass the articles to the template for rendering
    return render(request, 'base.html', {'articles': articles})

# chatbot code

import google.generativeai as genai


# Configure the API key

genai.configure(api_key=GOOGLE_API_KEY)


#original chatbotcode
# def chatbot(request):
#     page_name = 'Chatbot'
#     if request.method == 'POST':
#         # question = request.POST.get('question')
#         data = request.body
#         data = data.decode('utf-8')
#         data = json.loads(data)
#         question= data['question']

#         response = request.POST.get('response')

#         if question:
#             try:
#                 model = genai.GenerativeModel('gemini-pro')
#                 chat = model.start_chat(history=[])
#                 response = chat.send_message(question)
#                 response_text = response.text
#                 return JsonResponse({'response': response_text})  # Return JSON response
#             except Exception as e:
#                 # Handle any errors and return an error response
#                 return JsonResponse({'error': str(e)}, status=500)
#     # return render(request, 'chatbot.html')
#     return render(request, 'chatbot.html', {'page_name': page_name})


# @ensure_csrf_cookie
@csrf_exempt
def chatbot(request):
    page_name = 'Chatbot'
    print("Request method:", request.method)

    # Handle POST request (API call)
    if request.method == 'POST':
        try:
            data = json.loads(request.body.decode('utf-8'))
            question = data.get('question')

            if not question:
                return JsonResponse({'error': 'No question provided'}, status=400)

            # Process with Anthropic Claude
            llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")
            response = llm.invoke([HumanMessage(content=question)])
            response_text = response.content
            print("Response:", response_text)

            # Save the chat conversation to the database
            user_chat = UserChat(question=question, response=response_text)
            user_chat.save()

            return JsonResponse({'response': response_text})

        except Exception as e:
            print("Error:", str(e))
            return JsonResponse({'error': str(e)}, status=500)

    # For GET requests, just render the template
    return render(request, 'chatbot.html', {'page_name': page_name})




def index(request):
    return render(request, 'index.html', {})

def base(request):
    return render(request, 'base.html', {})

# def analysis(request):
#     return render(request, 'analysis.html', {})

from django.http import HttpResponse

# def ddi(request):
#     if request.method == 'POST':
#         drug1 = request.POST.get('drug1')
#         drug2 = request.POST.get('drug2')

#         # Initialize Gemini model and start chat
#         model = genai.GenerativeModel('gemini-pro')
#         chat = model.start_chat(history=[])

#         # Send the question to the model and get the response
#         text = f"Can {drug1} and {drug2} be taken together?"
#         response = chat.send_message(text)

#         # Return the response to the template
#         return render(request, 'ddi.html', {'response': response.text})

#     return render(request, 'ddi.html')

from django.shortcuts import render

def ddi(request):
    drug1 = ''  # Initialize drug names as empty strings
    drug2 = ''
    response_text = None  # Initialize response text as None initially

    if request.method == 'POST':
        drug1 = request.POST.get('drug1')
        drug2 = request.POST.get('drug2')
        text = f"Can {drug1} and {drug2} be taken together?"

        if drug1 and drug2:
            drug_interaction = DrugInteraction(drug1=drug1, drug2=drug2)
            drug_interaction.save()

        model = genai.GenerativeModel('gemini-pro')
        chat = model.start_chat(history=[])
        response = chat.send_message(text)
        response_text = response.text  # Assign response text if there's a response
        response_text = response.text.replace('*', '')

    return render(request, 'ddi.html', {'drug1': drug1, 'drug2': drug2, 'response_text': response_text})




# def chatbot(request):
#     return render(request, 'chatbot.html', {})

# def search_results(request):
#     if request.method == 'GET':
#         query = request.GET.get('query', '')
#         num_articles = 10
#         medical_articles = search_latest_medical_articles(query, num_articles)

#         context = {
#             'query': query,
#             'medical_articles': medical_articles,
#         }

#         return render(request, 'base.html', context)

#     return render(request, 'base.html', {})


# analysis
from django.shortcuts import render
import pandas as pd


# Load the DataFrame
df = pd.read_csv(r"/home/eshrath/data-science-fasttracks-drug-development--docker/drugapp/medlibb/drugsenti.csv")
# df = pd.read_csv(r"/djangodrugapp/medlibb/drugsenti.csv")

def find_unique_drugs_for_condition(user_condition, df):
    user_condition = user_condition.lower()
    useful_drugs = []
    non_useful_drugs = []
    for index, row in df.iterrows():
        if not pd.isna(row['condition']):
            conditions = row['condition'].lower()
            if user_condition in conditions:
                drug_name = row['drugName']
                label = row['label']
                if label == 1:
                    useful_drugs.append(drug_name)
                else:
                    non_useful_drugs.append(drug_name)
    return  useful_drugs, non_useful_drugs

def find_top_five_drugs_for_condition(user_condition, df):
    user_condition = user_condition.lower()
    unique_drugs_label_0 = {}
    unique_drugs_label_1 = {}
    for index, row in df.iterrows():
        if not pd.isna(row['condition']):
            conditions = row['condition'].lower()
            if user_condition in conditions:
                drug_name = row['drugName']
                label = row['label']
                useful_count = row['usefulCount']
                if label == 0:
                    if drug_name not in unique_drugs_label_0 or unique_drugs_label_0[drug_name] < useful_count:
                        unique_drugs_label_0[drug_name] = useful_count
                elif label == 1:
                    if drug_name not in unique_drugs_label_1 or unique_drugs_label_1[drug_name] < useful_count:
                        unique_drugs_label_1[drug_name] = useful_count
    sorted_unique_drugs_label_0 = sorted(unique_drugs_label_0.items(), key=lambda x: x[1], reverse=True)[:5]
    sorted_unique_drugs_label_1 = sorted(unique_drugs_label_1.items(), key=lambda x: x[1], reverse=True)[:5]
    return sorted_unique_drugs_label_0, sorted_unique_drugs_label_1

def analysis(request):
    if request.method == 'POST':
        user_condition = request.POST.get('condition')
        selection = request.POST.get('selection')

        if selection == 'drug':
            useful_drugs,non_useful_drugs = find_unique_drugs_for_condition(user_condition, df)
            return render(request, 'analysis.html', {'unique_drugs': useful_drugs, 'non_useful_drugs': non_useful_drugs})
        elif selection == 'general':
            top_five_drugs_label_0, top_five_drugs_label_1 = find_top_five_drugs_for_condition(user_condition, df)
            return render(request, 'analysis.html', {'top_five_drugs_label_0': top_five_drugs_label_0, 'top_five_drugs_label_1': top_five_drugs_label_1})

    return render(request, 'analysis.html')






# def login_user(request):
#     if request.method== "POST":
#         username=request.POST['username']
#         password=request.POST['password']
#         user=authenticate(request,username=username,password=password)
#         if user is not None:
#             login(request,user)
#             messages.success(request,("You have been logged in"))
#             return redirect('index')
#         else:
#             messages.error(request,("Invalid username or password"))
#             return redirect('login')
#     else:
#         return render(request, 'login.html', {})

def login_user(request):
    if request.method== 'POST':
        username = request.POST['username']
        password = request.POST['password']

        user = auth.authenticate(username=username,password=password)

        if user is not None:
            auth.login(request, user)
            # return redirect("/medilib/")
            return redirect(reverse('index'))
        else:
            messages.info(request,'Invalid credentials')
            return redirect('/medilib/login/')


    else:
        return render(request,'login.html')



def register(request):

    if request.method == 'POST':
        first_name = request.POST['first_name']
        last_name = request.POST['last_name']
        username = request.POST['username']
        password1 = request.POST['password1']
        password2 = request.POST['password2']
        email = request.POST['email']

        if password1==password2:
            if User.objects.filter(username=username).exists():
                messages.info(request,'Username Taken')
                return redirect('register')
            elif User.objects.filter(email=email).exists():
                messages.info(request,'Email Taken')
                return redirect('register')
            else:
                user = User.objects.create_user(username=username, password=password1, email=email,first_name=first_name,last_name=last_name)
                user.save();
                print('user created')
                return redirect('login')

        else:
            messages.info(request,'password not matching..')
            return redirect('register')
        return redirect('/')

    else:
        return render(request,'register.html')

def logout_user(request):
    # logout(request)
    # messages.success(request,("You have been logged out"))
    # return redirect('index')
    auth.logout(request)
    return redirect('/medilib/')
