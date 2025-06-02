# spatial_data/urls.py

from django.urls import path
from . import views

urlpatterns = [
    path("api/provinces.geojson", views.provinces_geojson, name="provinces-geojson"),
    path("api/communes.geojson",  views.communes_geojson,  name="communes-geojson"),
    path("api/competitor.geojson",  views.competitor_geojson,  name="competitor-geojson"),
    path("api/rma.geojson",  views.rma_office_geojson,  name="rma-geojson"),
    path("api/coverage-scores.geojson",  views.coverage_scores_geojson,  name="coverage-scores-geojson"),
    path('api/run-score/', views.run_score_view, name='run-score'),
    path('api/export/pdf/', views.export_pdf, name='export-pdf'),
    path("api/province-scores.geojson",  views.province_scores_geojson, name="province-scores-geojson"),
    path("api/commune-scores.geojson",   views.commune_scores_geojson, name="commune-scores-geojson"),
    path("api/simulate-score/", views.simulate_score, name="simulate-score"),
    path('api/run-stats/', views.run_stats, name='run-stats'),







]