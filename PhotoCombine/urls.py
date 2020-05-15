from django.contrib import admin
from django.urls import path,include
from imageUpload.views import index

urlpatterns = [
    path('',index,name = 'index_view'),
    path('api/',include('imageUpload.urls')),
    path('admin/', admin.site.urls),
]