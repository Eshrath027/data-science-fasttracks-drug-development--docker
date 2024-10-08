from django.contrib import admin
from .models import *

# Register your models here.
admin.site.register(UserQuery)
admin.site.register(DrugInteraction)
admin.site.register(UserChat)
