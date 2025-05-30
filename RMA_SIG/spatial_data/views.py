from django.http import HttpResponse
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated  # or AllowAny if local only
from rest_framework import status
from django.core.management import call_command
from rest_framework.response import Response
from .models import Province, Commune, RMAOffice, Competitor, CoverageScore, LossRatio, Area, CoverageStats
from django.db.models import Avg
from django.db.models import Prefetch
from .serializers import ProvinceSerializer, CommuneSerializer, CompetitorSerializer, RMAOfficeSerializer, AreaSerializer, CoverageScoreSerializer
import base64, io
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from rest_framework.decorators import api_view
from rest_framework.parsers import JSONParser
from django.http import FileResponse
import json
from django.http import JsonResponse
from django.db.models import F
import math

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

@api_view(["GET"])
def coverage_scores_geojson(request):
    # ▸ dernier calcul par zone (PostgreSQL)
    qs = (CoverageScore.objects
            .order_by("area_id", "-calculation_date")
            .distinct("area_id")
            .select_related("area")
            .annotate(geom=F("area__boundary")))    # <- copie la géométrie

    serializer = CoverageScoreSerializer(qs, many=True)
    return Response(serializer.data)    
@api_view(["GET"])
def province_scores_geojson(request):
    """
    Returns the latest coverage scores for each province.
    """
    latest_qs = CoverageScore.objects.order_by("-calculation_date")
    qs = (
        Province.objects.all()
        .prefetch_related(
            Prefetch("coverage_scores", queryset=latest_qs, to_attr="_cs")
        )
    )
    for p in qs:
        p._latest_cs = p._cs[0] if p._cs else None
    return Response(ProvinceSerializer(qs, many=True).data)

@api_view(["GET"])
def commune_scores_geojson(request):
    """
    Returns the latest coverage scores for each commune.
    """
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

from reportlab.lib.utils import ImageReader  # Add this import

# views.py  – revised export_pdf
# views.py
import base64, io
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from rest_framework.decorators import api_view
from django.http import FileResponse


@api_view(["POST"])
def export_pdf(request):
    """
    Accepts JSON  {img: dataURL, title: str, kpis: {...}}
    Returns a generated PDF file.
    """
    data = request.data          # ✅ DRF already parsed the JSON

    # -- decode the JPEG/PNG sent by the browser -----------------------
    img_b64   = data["img"].split(",", 1)[1]
    img_bytes = base64.b64decode(img_b64)

    # -- build the PDF -------------------------------------------------
    buffer = io.BytesIO()
    pdf    = canvas.Canvas(buffer, pagesize=A4)

    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(40, 800, f"RMA GIS Report — {data['title']}")

    img_reader = ImageReader(io.BytesIO(img_bytes))   # tells reportlab the format
    pdf.drawImage(img_reader, 40, 380, width=515, height=400)

    y = 340
    pdf.setFont("Helvetica", 11)
    for label, val in data["kpis"].items():
        pdf.drawString(40, y, f"{label.capitalize():<15}: {val}")
        y -= 18

    pdf.save()
    buffer.seek(0)
    return FileResponse(buffer, as_attachment=True,
                        filename="RMA_report.pdf")

# spatial_data/views.py
from math import exp
from statistics import fmean, pstdev

from django.contrib.gis.geos               import Point
from django.contrib.gis.measure            import D
from django.contrib.gis.db.models.functions import Distance
from django.db.models                      import Avg
from rest_framework.decorators             import api_view
from rest_framework.response               import Response
from rest_framework import status

from .models import (
    Commune, Province, Area,
    Competitor, Bank, LossRatio, CoverageScore      # Agency = RMA
)

# ----------  constants : keep them in sync with the batch -------------
BETA             = -1.5
LOSS_MID         = 0.65
LOSS_STEEPNESS   = 10
COMP_RADIUS_KM   = 30
PROJ_SRID        = 3857
# ----------------------------------------------------------------------


@api_view(['POST'])
def simulate_score(request):
    """
    Accepts {lat, lon}. Returns *freshly recomputed* score & potential
    for the area (Commune or, if none, Province) containing the point.

    The formula, weights and distance-decay are identical to the nightly
    `calculate_scores` batch, so you get a perfect preview of what the
    score would be if you inserted a **new RMA agency right here**.
    """
    stats = CoverageStats.objects.order_by('-calc_date').first()

    try:
        lat = float(request.data['lat'])
        lon = float(request.data['lon'])
    except (KeyError, ValueError):
        return Response({"detail": "Invalid lat/lon"},
                        status=status.HTTP_400_BAD_REQUEST)

    pnt = Point(lon, lat, srid=4326)

    #–––– 1. which admin area?  ----------------------------------------
    try:
        area = Commune.objects.get(boundary__intersects=pnt)
    except Commune.DoesNotExist:
        try:
            area = Province.objects.get(boundary__intersects=pnt)
        except Province.DoesNotExist:
            return Response({"detail": "Point outside known areas"},
                            status=status.HTTP_404_NOT_FOUND)

    #–––– 2. Collect raw variables for *all* areas in the same class ----
    cls        = Commune if isinstance(area, Commune) else Province
    all_areas  = list(cls.objects.all())

    pop            = [a.population or 0               for a in all_areas]
    insured        = [a.insured_population or 0       for a in all_areas]
    demand_gap     = [max(p - i, 0)                   for p, i in zip(pop, insured)]
    veh            = [a.estimated_vehicles or 0       for a in all_areas]
    bank_density   = [(a.bank_count or 0) / (p or 1) * 1000.0
                      for a, p in zip(all_areas, pop)]

    # loss ratio
    loss_ratio = []
    for a in all_areas:
        qs = (LossRatio.objects
                            .filter(commune=a) if isinstance(a, Commune)
              else LossRatio.objects
                            .filter(province=a))
        loss_ratio.append(qs.aggregate(avg=Avg('loss_ratio'))['avg'] or 0)

    # competition intensity
    comp_intensity = [competition_intensity(a) for a in all_areas]

    #–––– 3. z-scores ---------------------------------------------------
    pop_z, gap_z, veh_z, bank_z, comp_z = map(
        _zscores, [pop, demand_gap, veh, bank_density, comp_intensity]
    )

    idx = all_areas.index(area)

    demand       = 0.4 * pop_z[idx] + 0.6 * gap_z[idx]
    competition  = -comp_z[idx]                         # lower is better
    economic     = bank_z[idx]
    accessibility = 0                                   # still 0
    risk         = _logistic(loss_ratio[idx])

    raw = (0.35 * demand   +
           0.20 * competition +
           0.15 * economic +
           0.10 * accessibility +
           0.20 * risk)

    # rescale within the class (commune *or* province) to 0-100
    score_100    = round((raw - stats.raw_min) / (stats.raw_max - stats.raw_min) * 100, 2)
    potential    = ('HIGH' if score_100 >= 70 else
                    'MEDIUM' if score_100 >= 40 else
                    'LOW')

    return Response({
        "area_id"  : area.id,
        "area_name": area.name,
        "coverage" : score_100,
        "potential": potential
    })


# ----------------------------------------------------------------------
# helpers – verbatim copies of the ones in calculate_scores.py
# ----------------------------------------------------------------------
def _zscores(vals):
    μ = fmean(vals)
    σ = pstdev(vals) or 1.0
    return [(v - μ) / σ for v in vals]

def _logistic(x, mid=LOSS_MID, k=LOSS_STEEPNESS):
    return 1.0 / (1.0 + exp(k * (x - mid)))

def competition_intensity(area):
    centroid = area.boundary.centroid
    qs = (
      Competitor.objects
        .filter(location__distance_lte=(area.boundary, D(km=COMP_RADIUS_KM)))
        .annotate(d_km=Distance('location', centroid)/1000.0)
        .values_list('d_km', flat=True)
    )
    return sum(math.exp(BETA * d) for d in qs)
