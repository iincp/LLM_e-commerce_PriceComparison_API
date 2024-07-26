#map the urls to view function 

from django.urls import path 
from . import views 

#this variable should be exact this name, because this we the variable where the django looking for
#which should be the array of URL pattern objs 
#called URLConf, which every app has its own url configuration
#but now we need to import this url configuration into the main configuration for this project 
urlpatterns = [              
    path('', views.search_prompt),#path is the function to creat URLPattern obj
    path('form', views.form)
]