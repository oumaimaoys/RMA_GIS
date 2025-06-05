# core/urls.py

from django.urls import path
from . import views
from django.contrib.auth.decorators import login_required

app_name = "frontend"
urlpatterns = [
    path(
        'map/',
        login_required(views.map_view),           # or TemplateView.as_view(...) if it’s CBV
        name='map'
    ),
    path("notifications/", views.notif_list, name="list"),
    path("notifications/<int:pk>/", views.notif_read, name="read_and_redirect"),
    path("notifications/mark-all/", views.mark_all_read, name="mark_all_read"),
    path("notifications/toggle/<int:pk>/", views.toggle_read, name="toggle"),
]
