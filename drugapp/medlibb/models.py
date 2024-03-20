from django.db import models
from django.contrib.auth.models import User
import datetime

# Create your models here.




class UserQuery(models.Model):
    query_text = models.CharField(max_length=255)
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.query_text
    
class UserChat(models.Model):
    question = models.TextField()
    response = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f'{self.question} - {self.timestamp}'
    
class DrugInteraction(models.Model):
    drug1 = models.CharField(max_length=100)
    drug2 = models.CharField(max_length=100)
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f'{self.drug1} and {self.drug2} - {self.timestamp}'
