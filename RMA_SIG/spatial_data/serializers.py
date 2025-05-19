from rest_framework import serializers
from rest_framework_gis.serializers import GeoFeatureModelSerializer
from .models import Province, Commune, Area

class AreaSerializer(GeoFeatureModelSerializer):
    class Meta:
        model = Area
        geo_field = "boundary"
        fields = ('id', 'name', 'population', 'estimated_vehicles')

class ProvinceSerializer(GeoFeatureModelSerializer):
    class Meta:
        model = Province
        geo_field = "boundary"
        fields = ('id', 'name', 'population', 'estimated_vehicles')

class CommuneSerializer(GeoFeatureModelSerializer):
    class Meta:
        model = Commune
        geo_field = "boundary"
        fields = ('id', 'name', 'population', 'estimated_vehicles')