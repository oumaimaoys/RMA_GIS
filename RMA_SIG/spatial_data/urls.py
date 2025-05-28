# spatial_data/urls.py

from django.urls import path
from . import views

urlpatterns = [
    path("api/provinces.geojson", views.provinces_geojson, name="provinces-geojson"),
    path("api/communes.geojson",  views.communes_geojson,  name="communes-geojson"),
    path("api/competitor.geojson",  views.competitor_geojson,  name="competitor-geojson"),
    path("api/rma.geojson",  views.rma_office_geojson,  name="rma-geojson"),
    path('api/run-score/', views.run_score_view, name='run-score'),
    path('api/area/<int:pk>/metrics/', views.area_metrics, name='area-metrics'),




]