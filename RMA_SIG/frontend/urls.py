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

    path('dashboard', views.dashboard_view, name='main_dashboard'),
    
    # Detailed views
    path('area/<int:area_id>/', views.area_detail_view, name='area_detail'),
    path('market-analysis/', views.market_analysis_view, name='market_analysis'),
    path('rma-performance/', views.rma_performance_view, name='rma_performance'),
    
    # API endpoints for AJAX requests
    path('api/coverage-data/', views.api_coverage_data, name='api_coverage_data'),
    path('api/area-stats/<int:area_id>/', views.api_area_stats, name='api_area_stats'),
]
