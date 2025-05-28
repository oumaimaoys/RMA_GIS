from django.http import HttpResponse
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated  # or AllowAny if local only
from rest_framework import status
from django.core.management import call_command
from rest_framework.response import Response
from .models import Province, Commune, RMAOffice, Competitor, CoverageScore, LossRatio, Area
from django.db.models import Avg
from django.db.models import Prefetch
from .serializers import ProvinceSerializer, CommuneSerializer, CompetitorSerializer, RMAOfficeSerializer

@api_view(["GET"])
def provinces_geojson(request):
    latest_qs = CoverageScore.objects.order_by("-calculation_date")
    qs = (
        Province.objects.all()
        .prefetch_related(
            Prefetch("coverage_scores", queryset=latest_qs, to_attr="_cs")  # pull *all*
        )
    )
    # Mark the first (latest) result for O(1) access in the serializer
    for p in qs:
        p._latest_cs = p._cs[0] if p._cs else None
    return Response(ProvinceSerializer(qs, many=True).data)


@api_view(["GET"])
def communes_geojson(request):
    latest_qs = CoverageScore.objects.order_by("-calculation_date")
    qs = (
        Commune.objects.all()
        .prefetch_related(
            Prefetch("coverage_scores", queryset=latest_qs, to_attr="_cs")
        )
    )
    for c in qs:
        c._latest_cs = c._cs[0] if c._cs else None
    return Response(CommuneSerializer(qs, many=True).data)

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

@api_view(['GET'])
def area_metrics(request, pk):
    area = Area.objects.get(pk=pk)
    cov = area.coverage_scores.order_by('-calculation_date').first()
    lr  = (LossRatio.objects.filter(commune=area) if hasattr(area,'commune') else
           LossRatio.objects.filter(province=area) if hasattr(area,'province') else
           LossRatio.objects.none()).aggregate(avg=Avg('loss_ratio'))['avg']
    return Response({
        "score":      cov.score if cov else None,
        "potential":  cov.potential if cov else None,
        "loss_ratio": lr,
        "rma_count":  area.competition_count,          # pre-computed
        # … add premium volume, claim freq when you have them …
    })
