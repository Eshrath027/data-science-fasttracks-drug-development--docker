from rest_framework import serializers
from medlibb.models import *

class UserChatSerializer(serializers.Serializer):
    question = serializers.CharField(max_length=255)
    response = serializers.CharField()
    timestamp = serializers.DateTimeField()

    def create(self, validated_data):
        return UserChat.objects.create(**validated_data)

    # def update(self, instance, validated_data):
    #     instance.question = validated_data.get('question', instance.question)
    #     instance.response = validated_data.get('response', instance.response)
    #     instance.timestamp = validated_data.get('timestamp', instance.timestamp)
    #     instance.save()
    #     return instance

    # class Meta:
    #     model = UserChat
    #     # fields = ['question', 'response', 'timestamp']
    #     fields = '__all__'
