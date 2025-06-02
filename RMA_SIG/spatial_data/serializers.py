import json
from rest_framework import serializers
from rest_framework_gis.serializers import GeoFeatureModelSerializer

from .models import Province, Commune, Area, Competitor, RMAOffice, CoverageScore


class AreaSerializer(GeoFeatureModelSerializer):
    class Meta:
        model = Area
        geo_field = "boundary"
        fields = (
            "id",
            "name",
            "population",
            "insured_population",
            "estimated_vehicles",
        )


class ProvinceSerializer(GeoFeatureModelSerializer):
    # Base fields from Area
    name = serializers.CharField(source="area_ptr.name", read_only=True)
    population = serializers.IntegerField(
        source="area_ptr.population", read_only=True
    )
    insured_population = serializers.IntegerField(
        source="area_ptr.insured_population", read_only=True
    )
    estimated_vehicles = serializers.IntegerField(
        source="area_ptr.estimated_vehicles", read_only=True
    )

    # Flattened coverage‐score fields:
    score = serializers.SerializerMethodField()
    potential = serializers.SerializerMethodField()
    calculation_date = serializers.SerializerMethodField()
    demand_score = serializers.SerializerMethodField()
    competition_score = serializers.SerializerMethodField()
    economic_score = serializers.SerializerMethodField()
    accessibility_score = serializers.SerializerMethodField()
    risk_score = serializers.SerializerMethodField()
    travel_time_to_centroid_minutes = serializers.SerializerMethodField()

    # Return geometry as before (we already fixed this in the previous step)
    boundary_from_area_parent = serializers.SerializerMethodField()

    class Meta:
        model = Province
        geo_field = "boundary_from_area_parent"
        fields = (
            "id",
            "name",
            "population",
            "insured_population",
            "estimated_vehicles",
            # Now flattened:
            "score",
            "potential",
            "calculation_date",
            "demand_score",
            "competition_score",
            "economic_score",
            "accessibility_score",
            "risk_score",
            "travel_time_to_centroid_minutes",
        )

    def _latest(self, obj):
        """
        Helper to return the latest CoverageScore instance (or None).
        The view should have set `obj.latest_score_object`.
        """
        return getattr(obj, "latest_score_object", None)

    def get_score(self, obj):
        latest = self._latest(obj)
        return latest.score if latest is not None else None

    def get_potential(self, obj):
        latest = self._latest(obj)
        return latest.potential if latest is not None else None

    def get_calculation_date(self, obj):
        latest = self._latest(obj)
        # Return ISO8601 string
        return latest.calculation_date.isoformat() if latest is not None else None

    def get_demand_score(self, obj):
        latest = self._latest(obj)
        return latest.demand_score if latest is not None else None

    def get_competition_score(self, obj):
        latest = self._latest(obj)
        return latest.competition_score if latest is not None else None

    def get_economic_score(self, obj):
        latest = self._latest(obj)
        return latest.economic_score if latest is not None else None

    def get_accessibility_score(self, obj):
        latest = self._latest(obj)
        return latest.accessibility_score if latest is not None else None

    def get_risk_score(self, obj):
        latest = self._latest(obj)
        return latest.risk_score if latest is not None else None

    def get_travel_time_to_centroid_minutes(self, obj):
        latest = self._latest(obj)
        return latest.travel_time_to_centroid_minutes if latest is not None else None

    def get_boundary_from_area_parent(self, obj):
        """
        Return a GeoJSON‐compatible dict instead of a GEOSGeometry.
        """
        geom = None
        if hasattr(obj, "area_ptr") and obj.area_ptr and obj.area_ptr.boundary:
            geom = obj.area_ptr.boundary
        if geom:
            return json.loads(geom.geojson)
        return None


class CommuneSerializer(GeoFeatureModelSerializer):
    name = serializers.CharField(source="area_ptr.name", read_only=True)
    population = serializers.IntegerField(
        source="area_ptr.population", read_only=True
    )
    insured_population = serializers.IntegerField(
        source="area_ptr.insured_population", read_only=True
    )
    estimated_vehicles = serializers.IntegerField(
        source="area_ptr.estimated_vehicles", read_only=True
    )

    # Flattened coverage‐score fields (exact same pattern):
    score = serializers.SerializerMethodField()
    potential = serializers.SerializerMethodField()
    calculation_date = serializers.SerializerMethodField()
    demand_score = serializers.SerializerMethodField()
    competition_score = serializers.SerializerMethodField()
    economic_score = serializers.SerializerMethodField()
    accessibility_score = serializers.SerializerMethodField()
    risk_score = serializers.SerializerMethodField()
    travel_time_to_centroid_minutes = serializers.SerializerMethodField()

    boundary_from_area_parent = serializers.SerializerMethodField()

    class Meta:
        model = Commune
        geo_field = "boundary_from_area_parent"
        fields = (
            "id",
            "name",
            "population",
            "insured_population",
            "estimated_vehicles",
            # Flattened:
            "score",
            "potential",
            "calculation_date",
            "demand_score",
            "competition_score",
            "economic_score",
            "accessibility_score",
            "risk_score",
            "travel_time_to_centroid_minutes",
        )

    def _latest(self, obj):
        return getattr(obj, "latest_score_object", None)

    def get_score(self, obj):
        latest = self._latest(obj)
        return latest.score if latest is not None else None

    def get_potential(self, obj):
        latest = self._latest(obj)
        return latest.potential if latest is not None else None

    def get_calculation_date(self, obj):
        latest = self._latest(obj)
        return latest.calculation_date.isoformat() if latest is not None else None

    def get_demand_score(self, obj):
        latest = self._latest(obj)
        return latest.demand_score if latest is not None else None

    def get_competition_score(self, obj):
        latest = self._latest(obj)
        return latest.competition_score if latest is not None else None

    def get_economic_score(self, obj):
        latest = self._latest(obj)
        return latest.economic_score if latest is not None else None

    def get_accessibility_score(self, obj):
        latest = self._latest(obj)
        return latest.accessibility_score if latest is not None else None

    def get_risk_score(self, obj):
        latest = self._latest(obj)
        return latest.risk_score if latest is not None else None

    def get_travel_time_to_centroid_minutes(self, obj):
        latest = self._latest(obj)
        return latest.travel_time_to_centroid_minutes if latest is not None else None

    def get_boundary_from_area_parent(self, obj):
        geom = None
        if hasattr(obj, "area_ptr") and obj.area_ptr and obj.area_ptr.boundary:
            geom = obj.area_ptr.boundary
        if geom:
            return json.loads(geom.geojson)
        return None


class CompetitorSerializer(GeoFeatureModelSerializer):
    class Meta:
        model = Competitor
        geo_field = "location"
        fields = (
            "id",
            "code_ACAPS",
            "agency_name",
            "mandante",
            "competitor_type",
            "address",
            "city",
        )


class RMAOfficeSerializer(GeoFeatureModelSerializer):
    class Meta:
        model = RMAOffice
        geo_field = "location"
        fields = ("id", "name", "address", "city")


class CoverageScoreSerializer(GeoFeatureModelSerializer):
    area_name = serializers.CharField(source="area.name", read_only=True)

    class Meta:
        model = CoverageScore
        geo_field = "geom"
        fields = (
            "id",
            "score",
            "potential",
            "area_name",
            "demand_score",
            "competition_score",
            "economic_score",
            "accessibility_score",
            "risk_score",
            "calculation_date",
            "travel_time_to_centroid_minutes",
        )
        extra_kwargs = {
            field: {"allow_null": True, "required": False}
            for field in [
                "demand_score",
                "competition_score",
                "economic_score",
                "accessibility_score",
                "risk_score",
                "travel_time_to_centroid_minutes",
            ]
        }
