# core/urls.py

from django.urls import path
from . import views
from django.contrib.auth.decorators import login_required


urlpatterns = [
    path(
        'map/',
        login_required(views.map_view),           # or TemplateView.as_view(...) if it’s CBV
        name='map'
    ),
]
