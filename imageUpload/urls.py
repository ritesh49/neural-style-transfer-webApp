from django.urls import path,include
from .views import uploadView

urlpatterns = [
    path('upload',uploadView,name = 'upload_view')
]