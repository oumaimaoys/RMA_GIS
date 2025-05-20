from django.http import HttpResponse
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models import Province, Commune, RMAOffice, Competitor
from .serializers import ProvinceSerializer, CommuneSerializer, CompetitorSerializer, RMAOfficeSerializer

@api_view(['GET'])
def provinces_geojson(request):
    qs = Province.objects.all()
    serializer = ProvinceSerializer(qs, many=True)
    return Response(serializer.data)

@api_view(['GET'])
def communes_geojson(request):
    qs = Commune.objects.all()
    serializer = CommuneSerializer(qs, many=True)
    return Response(serializer.data)

@api_view(['GET'])
def acaps_data_geojson(request):
    """
    Returns a single GeoJSON FeatureCollection combining
    all Competitor and RMAOffice records.
    """
    comps = Competitor.objects.all()
    rmas  = RMAOffice.objects.all()

    comp_ser = CompetitorSerializer(comps, many=True)
    rma_ser  = RMAOfficeSerializer(rmas, many=True)

    # Each serializer.data is already a GeoJSON Feature
    features = comp_ser.data + rma_ser.data

    return Response({
        "type": "FeatureCollection",
        "features": features
    })


