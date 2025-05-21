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
def competitor_geojson(request):
    comps = Competitor.objects.all()
    comp_ser = CompetitorSerializer(comps, many=True)
    return Response(comp_ser.data)

@api_view(['GET'])
def rma_office_geojson(request):
    rmas  = RMAOffice.objects.all()
    rma_ser  = RMAOfficeSerializer(rmas, many=True)
    return Response(rma_ser.data)

