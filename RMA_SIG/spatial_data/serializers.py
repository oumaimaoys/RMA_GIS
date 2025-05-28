from rest_framework import serializers
from rest_framework_gis.serializers import GeoFeatureModelSerializer
from .models import Province, Commune, Area, Competitor, RMAOffice, CoverageScore

class AreaSerializer(GeoFeatureModelSerializer):
    class Meta:
        model = Area
        geo_field = "boundary"
        fields = ('id', 'name', 'population', 'insured_population','estimated_vehicles')


class ProvinceSerializer(GeoFeatureModelSerializer):
    coverage_score     = serializers.SerializerMethodField()
    coverage_potential = serializers.SerializerMethodField()

    class Meta:
        model     = Province
        geo_field = "boundary"
        fields    = (
            "id", "name", "population", "insured_population",
            "estimated_vehicles", "coverage_score", "coverage_potential"
        )

    def get_coverage_score(self, obj):
        cs = getattr(obj, "_latest_cs", None) or \
             obj.coverage_scores.order_by("-calculation_date").first()
        return cs.score if cs else None

    def get_coverage_potential(self, obj):
        cs = getattr(obj, "_latest_cs", None) or \
             obj.coverage_scores.order_by("-calculation_date").first()
        return cs.potential if cs else None


class CommuneSerializer(ProvinceSerializer):   # same fields
    class Meta(ProvinceSerializer.Meta):
        model = Commune

class CompetitorSerializer(GeoFeatureModelSerializer):
    class Meta:
        model = Competitor
        geo_field = "location"
        fields = ('id', 'code_ACAPS', 'company_name', 'mandante', 'competitor_type', 'address','city')


class RMAOfficeSerializer(GeoFeatureModelSerializer):
    class Meta:
        model = RMAOffice
        geo_field = "location"
        fields = ('id', 'name', 'address', 'city')