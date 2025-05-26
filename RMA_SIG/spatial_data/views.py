from django.http import HttpResponse
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated  # or AllowAny if local only
from rest_framework import status
from django.core.management import call_command
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


@api_view(['POST'])
def run_score_view(request):
    print("🔥 Received POST to run-score")
    print("Headers:", request.headers)
    print("Body:", request.body)

    try:
        call_command('calculate_scores')
        return Response({"message": "Score calculation triggered."}, status=200)
    except Exception as e:
        return Response({"error": str(e)}, status=500)

