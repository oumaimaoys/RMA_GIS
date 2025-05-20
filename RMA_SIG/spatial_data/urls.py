# spatial_data/urls.py

from django.urls import path
from . import views

urlpatterns = [
    path("api/provinces.geojson", views.provinces_geojson, name="provinces-geojson"),
    path("api/communes.geojson",  views.communes_geojson,  name="communes-geojson"),
    path("api/acaps_data.geojson",  views.acaps_data_geojson,  name="acaps_data-geojson"),

]