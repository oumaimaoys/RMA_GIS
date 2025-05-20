from rest_framework import serializers
from rest_framework_gis.serializers import GeoFeatureModelSerializer
from .models import Province, Commune, Area, Competitor, RMAOffice

class AreaSerializer(GeoFeatureModelSerializer):
    class Meta:
        model = Area
        geo_field = "boundary"
        fields = ('id', 'name', 'population', 'insured_population','estimated_vehicles')

class ProvinceSerializer(GeoFeatureModelSerializer):
    class Meta:
        model = Province
        geo_field = "boundary"
        fields = ('id', 'name', 'population','insured_population', 'estimated_vehicles')

class CommuneSerializer(GeoFeatureModelSerializer):
    class Meta:
        model = Commune
        geo_field = "boundary"
        fields = ('id', 'name', 'population', 'insured_population', 'estimated_vehicles')

class CompetitorSerializer(GeoFeatureModelSerializer):
    class Meta:
        model = Competitor
        geo_field = "location"
        fields = ('id', 'code_ACAPS', 'company_name', 'mandante', 'competitor_type', 'adaress','city')


class RMAOfficeSerializer(GeoFeatureModelSerializer):
    class Meta:
        model = RMAOffice
        geo_field = "location"
        fields = ('id', 'name', 'address', 'city')