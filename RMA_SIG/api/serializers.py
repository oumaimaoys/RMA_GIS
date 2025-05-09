from rest_framework import serializers
from spatial_data.models import (
    RMAOffice, RMABGD, RMAAgent, Bank, Competitor,
    PopulationArea, CoverageScore
)

# Base serializer for location-based models
class BaseLocationSerializer(serializers.ModelSerializer):
    class Meta:
        fields = ['id', 'name', 'latitude', 'longitude', 'date_added', 'last_updated', 'is_active']
        abstract = True


class RMAOfficeSerializer(serializers.ModelSerializer):
    class Meta:
        model = RMAOffice
        fields = [
            'id', 'name', 'longitude', 'latitude', 'date_added', 'last_updated', 'is_active',
            'code_ACAPS', 'code_RMA', 'address', 'city', 'code_tel', 'tel', 'tel_GSM'
        ]


class RMABGDSerializer(serializers.ModelSerializer):
    class Meta:
        model = RMABGD
        fields = RMAOfficeSerializer.Meta.fields + [
            'type_BGD', 'Partenaire', 'date_creation', 'RMA_BGD_state'
        ]


class RMAAgentSerializer(serializers.ModelSerializer):
    class Meta:
        model = RMAAgent
        fields = RMAOfficeSerializer.Meta.fields


class BankSerializer(serializers.ModelSerializer):
    class Meta:
        model = Bank
        fields = [
            'id', 'name', 'longitude', 'latitude', 'date_added', 'last_updated', 'is_active',
            'bank_id', 'institution_name'
        ]


class CompetitorSerializer(serializers.ModelSerializer):
    competitor_type_display = serializers.CharField(source='get_competitor_type_display', read_only=True)

    class Meta:
        model = Competitor
        fields = [
            'id', 'name', 'longitude', 'latitude', 'date_added', 'last_updated', 'is_active',
            'code_ACAPS', 'company_name', 'address', 'city', 'competitor_type', 'competitor_type_display'
        ]


class PopulationAreaSerializer(serializers.ModelSerializer):
    class Meta:
        model = PopulationArea
        fields = [
            'id', 'name', 'ZIP_code', 'boundary', 'total_population', 'total_vihicules', 'date_updated'
        ]


class CoverageScoreSerializer(serializers.ModelSerializer):
    area = PopulationAreaSerializer(read_only=True)
    nearest_office = RMAOfficeSerializer(read_only=True)

    class Meta:
        model = CoverageScore
        fields = [
            'id', 'area', 'score', 'population_covered', 'coverage_percentage',
            'nearest_office', 'calculation_date',
            'competitor_factor', 'bank_partnership_factor'
        ]
