from django.http import HttpResponse
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models import Province, Commune
from .serializers import ProvinceSerializer, CommuneSerializer

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

# If you still want to use Django's built-in serializer, keep this function as an alternative
def provinces_geojson_django(request):
    from django.core.serializers import serialize
    qs = Province.objects.all()
    geojson = serialize(
        "geojson", 
        qs,
        geometry_field="boundary",
        fields=("name", "population", "estimated_vehicles"),
    )
    return HttpResponse(geojson, content_type="application/json")