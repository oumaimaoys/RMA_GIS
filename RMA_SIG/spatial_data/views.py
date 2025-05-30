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

# spatial_data/views.py

import logging
from django.core.management import call_command
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

logger = logging.getLogger(__name__)

@api_view(['POST'])
def run_score_view(request):
    """
    Trigger the nightly AHP‐weighted coverage‐score recalculation.
    """
    try:
        logger.info("Received POST to run-score; running calculate_scores command")
        # this will call spatial_data/management/commands/calculate_scores.py
        call_command('calculate_scores')
        logger.info("calculate_scores completed successfully")
        return Response({'status': 'ok'}, status=status.HTTP_200_OK)

    except Exception as e:
        # log full traceback
        logger.exception("Error running calculate_scores")
        # return JSON error message
        return Response(
            {
                'status': 'error',
                'detail': str(e)
            },
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


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

import math
import logging
import requests

from statistics import fmean, pstdev
from django.conf import settings
from django.db.models import Avg, F, Q, Sum
from django.contrib.gis.geos import Point
from django.contrib.gis.measure import D
from django.contrib.gis.db.models.functions import Distance, Centroid
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

from .models import (
    Commune, Province,
    Bank, Competitor, LossRatio,
    CoverageStats
)

# -----------------------------------------------------------------------------
# Constants (must match your batch job)
# -----------------------------------------------------------------------------
BETA             = -1.5
LOSS_MID         = 0.65
LOSS_STEEPNESS   = 10
COMP_RADIUS_KM   = 30  # km
PROJ_SRID        = 3857
ORS_DIRECTIONS_URL = "https://api.openrouteservice.org/v2/directions/driving-car"
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def get_drive_time(lat_from, lon_from, lat_to, lon_to):
    """
    Query OpenRouteService for driving time (minutes) between two WGS84 points.
    Supports both GeoJSON and ORS v2 formats.
    """
    api_key = settings.ORS_API_KEY
    resp = requests.post(
        ORS_DIRECTIONS_URL,
        headers={
            "Authorization": api_key,
            "Content-Type": "application/json"
        },
        json={"coordinates": [[lon_from, lat_from], [lon_to, lat_to]]},
        timeout=5
    )
    resp.raise_for_status()
    data = resp.json()

    # GeoJSON-style "features"
    feats = data.get("features")
    if isinstance(feats, list) and feats:
        segs = feats[0].get("properties", {}).get("segments", [])
        if segs and segs[0].get("duration") is not None:
            return segs[0]["duration"] / 60.0

    # ORS v2-style "routes"
    routes = data.get("routes")
    if isinstance(routes, list) and routes:
        segs = routes[0].get("segments", [])
        if segs and segs[0].get("duration") is not None:
            return segs[0]["duration"] / 60.0

    logging.error("ORS response had no duration: %r", data)
    raise ValueError("No route duration returned from OpenRouteService")


def competition_intensity_py(area):
    """
    Sum exp(BETA·d_km) for all Competitor.location within COMP_RADIUS_KM.
    """
    # project centroid for accurate metres-based distance
    centroid_m = area.boundary.centroid.transform(PROJ_SRID, clone=True)
    nearby = Competitor.objects.filter(
        location__distance_lte=(area.boundary, D(km=COMP_RADIUS_KM))
    )
    total = 0.0
    for comp in nearby:
        d_m = centroid_m.distance(comp.location.transform(PROJ_SRID, clone=True))
        d_km = d_m / 1000.0
        total += math.exp(BETA * d_km)
    return total


def bank_intensity_py(area):
    """
    Sum exp(BETA·d_km) for all Bank.location within COMP_RADIUS_KM.
    """
    centroid_m = area.boundary.centroid.transform(PROJ_SRID, clone=True)
    nearby = Bank.objects.filter(
        location__distance_lte=(area.boundary, D(km=COMP_RADIUS_KM))
    )
    total = 0.0
    for b in nearby:
        d_m = centroid_m.distance(b.location.transform(PROJ_SRID, clone=True))
        d_km = d_m / 1000.0
        total += math.exp(BETA * d_km)
    return total


def _zscores(vals):
    μ = fmean(vals)
    σ = pstdev(vals) or 1.0
    return [(v - μ) / σ for v in vals]


def _logistic(x, mid=LOSS_MID, k=LOSS_STEEPNESS):
    return 1.0 / (1.0 + math.exp(k * (x - mid)))


# -----------------------------------------------------------------------------
# ORM-based annotation for loss‐ratio (others done in Python)
# -----------------------------------------------------------------------------
def get_area_queryset(cls):
    """
    Annotate each Area (Commune/Province) with:
      - id, name, population, insured_population, estimated_vehicles
      - loss_ratio_avg = Avg('lossratio__loss_ratio')
    """
    return (
        cls.objects
           .values('id', 'name', 'population', 'insured_population', 'estimated_vehicles')
           .annotate(loss_ratio_avg=Avg('lossratio__loss_ratio'))
    )


# -----------------------------------------------------------------------------
# Main endpoint
# -----------------------------------------------------------------------------
@api_view(['POST'])
def simulate_score(request):
    # load latest stats
    stats = CoverageStats.objects.latest('calc_date')

    # parse input
    try:
        lat = float(request.data['lat'])
        lon = float(request.data['lon'])
    except (KeyError, ValueError):
        return Response({"detail": "Invalid lat/lon"}, status=status.HTTP_400_BAD_REQUEST)
    pnt = Point(lon, lat, srid=4326)

    # find intersection in Communes, else Provinces
    area_dict = (
        get_area_queryset(Commune)
        .filter(boundary__intersects=pnt)
        .order_by('id')
        .first()
    )
    Model = Commune
    if not area_dict:
        area_dict = (
            get_area_queryset(Province)
            .filter(boundary__intersects=pnt)
            .order_by('id')
            .first()
        )
        Model = Province
    if not area_dict:
        return Response({"detail": "Outside known areas"}, status=status.HTTP_404_NOT_FOUND)

    # load geometry for Python‐based intensities & routing
    area_obj = Model.objects.only('boundary').get(pk=area_dict['id'])

    # build raw lists
    all_areas = list(get_area_queryset(Model))
    instances = Model.objects.in_bulk([a['id'] for a in all_areas])

    pops   = [a['population'] for a in all_areas]
    gaps   = [max(a['population'] - a['insured_population'], 0) for a in all_areas]
    vehs   = [a['estimated_vehicles'] for a in all_areas]
    banks  = [bank_intensity_py(instances[a['id']]) for a in all_areas]
    comps  = [competition_intensity_py(instances[a['id']]) for a in all_areas]
    losses = [a['loss_ratio_avg'] or 0 for a in all_areas]

    # z-score normalization
    def z(v, mean, std): return (v - mean) / std if std else 0
    pop_z  = [z(v, stats.pop_mean,  stats.pop_std)  for v in pops]
    gap_z  = [z(v, stats.gap_mean,  stats.gap_std)  for v in gaps]
    veh_z  = [z(v, stats.veh_mean,  stats.veh_std)  for v in vehs]
    bank_z = [z(v, stats.bank_mean, stats.bank_std) for v in banks]
    comp_z = [z(v, stats.comp_mean, stats.comp_std) for v in comps]

    # find our area’s index
    idx = next(i for i,a in enumerate(all_areas) if a['id'] == area_dict['id'])

    # compute components
    demand      = 0.35 * pop_z[idx]   + 0.35 * gap_z[idx] + 0.30 * veh_z[idx]
    competition = -comp_z[idx]
    economic    = bank_z[idx]

    # compute routing‐driven accessibility
    cent = area_obj.boundary.centroid.clone()
    cent.transform(4326)
    try:
        travel_time = get_drive_time(lat, lon, cent.y, cent.x)
    except ValueError as e:
        return Response({"detail": str(e)}, status=status.HTTP_502_BAD_GATEWAY)
    accessibility = (stats.access_mean - travel_time) / stats.access_std

    risk = _logistic(losses[idx])

    # raw weighted score
    raw = (
        0.35 * demand +
        0.20 * competition +
        0.15 * economic +
        0.10 * accessibility +
        0.20 * risk
    )

    # final 0–100
    score = round((raw - stats.raw_min) / (stats.raw_max - stats.raw_min) * 100, 2)
    potential = 'HIGH' if score >= 70 else 'MEDIUM' if score >= 40 else 'LOW'

    return Response({
        "area_id":   area_dict['id'],
        "area_name": area_dict['name'],
        "coverage":  score,
        "potential": potential
    })
