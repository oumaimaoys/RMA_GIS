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
from functools import lru_cache

from statistics import fmean, pstdev
from django.conf import settings
from django.db.models import Avg, F, Q, Sum
from django.contrib.gis.geos import Point
from django.contrib.gis.measure import D
from django.contrib.gis.db.models.functions import Distance, Centroid
from django.core.cache import cache
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

from .models import (
    Commune, Province,
    Bank, Competitor, LossRatio,
    CoverageStats
)

# -----------------------------------------------------------------------------
# Constants (keeping your original constants for compatibility)
# -----------------------------------------------------------------------------
BETA             = -1.5
LOSS_MID         = 0.65
LOSS_STEEPNESS   = 10
COMP_RADIUS_KM   = 30  # km
PROJ_SRID        = 3857
ORS_DIRECTIONS_URL = "https://api.openrouteservice.org/v2/directions/driving-car"

# Cache settings
CACHE_TIMEOUT = 1800  # 30 minutes
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Cached helper functions
# -----------------------------------------------------------------------------
@lru_cache(maxsize=1000)
def cached_exp_calculation(distance_km):
    """Cache expensive exponential calculations."""
    return math.exp(BETA * max(distance_km, 0.1))

def get_drive_time(lat_from, lon_from, lat_to, lon_to):
    """
    Optimized routing with caching and fallback.
    """
    # Create cache key
    cache_key = f"route_{lat_from:.4f}_{lon_from:.4f}_{lat_to:.4f}_{lon_to:.4f}"
    
    # Try cache first
    try:
        cached_time = cache.get(cache_key)
        if cached_time is not None:
            return cached_time
    except:
        pass  # Cache might not be configured
    
    try:
        api_key = settings.ORS_API_KEY
        resp = requests.post(
            ORS_DIRECTIONS_URL,
            headers={
                "Authorization": api_key,
                "Content-Type": "application/json"
            },
            json={"coordinates": [[lon_from, lat_from], [lon_to, lat_to]]},
            timeout=3
        )
        resp.raise_for_status()
        data = resp.json()

        # GeoJSON-style "features"
        feats = data.get("features")
        if isinstance(feats, list) and feats:
            segs = feats[0].get("properties", {}).get("segments", [])
            if segs and segs[0].get("duration") is not None:
                travel_time = segs[0]["duration"] / 60.0
                try:
                    cache.set(cache_key, travel_time, CACHE_TIMEOUT)
                except:
                    pass
                return travel_time

        # ORS v2-style "routes"
        routes = data.get("routes")
        if isinstance(routes, list) and routes:
            segs = routes[0].get("segments", [])
            if segs and segs[0].get("duration") is not None:
                travel_time = segs[0]["duration"] / 60.0
                try:
                    cache.set(cache_key, travel_time, CACHE_TIMEOUT)
                except:
                    pass
                return travel_time

        # If no duration found, fall back to distance estimation
        raise ValueError("No route duration returned")
        
    except Exception as e:
        logging.warning(f"Routing API failed, using distance estimate: {e}")
        
        # Fallback: estimate based on straight-line distance
        lat1, lon1, lat2, lon2 = map(math.radians, [lat_from, lon_from, lat_to, lon_to])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        distance_km = 2 * 6371 * math.asin(math.sqrt(a))
        
        # Estimate driving time (assume average 50 km/h with routing factor)
        estimated_time = distance_km / 50 * 1.3
        
        try:
            cache.set(cache_key, estimated_time, 300)  # Cache for 5 minutes
        except:
            pass
        
        return estimated_time


def competition_intensity_py(area):
    """
    Optimized competition intensity calculation.
    """
    cache_key = f"comp_intensity_{area.id}"
    try:
        cached_result = cache.get(cache_key)
        if cached_result is not None:
            return cached_result
    except:
        pass
    
    # project centroid for accurate metres-based distance
    centroid_m = area.boundary.centroid.transform(PROJ_SRID, clone=True)
    nearby = Competitor.objects.filter(
        location__distance_lte=(area.boundary, D(km=COMP_RADIUS_KM))
    )
    total = 0.0
    for comp in nearby:
        d_m = centroid_m.distance(comp.location.transform(PROJ_SRID, clone=True))
        d_km = d_m / 1000.0
        total += cached_exp_calculation(d_km)
    
    try:
        cache.set(cache_key, total, CACHE_TIMEOUT)
    except:
        pass
    
    return total


def bank_intensity_py(area):
    """
    Optimized bank intensity calculation.
    """
    cache_key = f"bank_intensity_{area.id}"
    try:
        cached_result = cache.get(cache_key)
        if cached_result is not None:
            return cached_result
    except:
        pass
    
    centroid_m = area.boundary.centroid.transform(PROJ_SRID, clone=True)
    nearby = Bank.objects.filter(
        location__distance_lte=(area.boundary, D(km=COMP_RADIUS_KM))
    )
    total = 0.0
    for b in nearby:
        d_m = centroid_m.distance(b.location.transform(PROJ_SRID, clone=True))
        d_km = d_m / 1000.0
        total += cached_exp_calculation(d_km)
    
    try:
        cache.set(cache_key, total, CACHE_TIMEOUT)
    except:
        pass
    
    return total


def _zscores(vals):
    μ = fmean(vals)
    σ = pstdev(vals) or 1.0
    return [(v - μ) / σ for v in vals]


def _logistic(x, mid=LOSS_MID, k=LOSS_STEEPNESS):
    if x is None or x == 0:
        return 0.5
    return 1.0 / (1.0 + math.exp(k * (x - mid)))


# -----------------------------------------------------------------------------
# ORM-based annotation for loss‐ratio
# -----------------------------------------------------------------------------
def get_area_queryset(cls):
    """
    Annotate each Area (Commune/Province) with required fields.
    """
    return (
        cls.objects
           .values('id', 'name', 'population', 'insured_population', 'estimated_vehicles')
           .annotate(loss_ratio_avg=Avg('lossratio__loss_ratio'))
    )


def get_cached_stats():
    """Get coverage stats with caching."""
    cache_key = 'coverage_stats_latest'
    try:
        stats = cache.get(cache_key)
        if stats is not None:
            return stats
    except:
        pass
    
    stats = CoverageStats.objects.latest('calc_date')
    
    try:
        cache.set(cache_key, stats, CACHE_TIMEOUT)
    except:
        pass
    
    return stats


# -----------------------------------------------------------------------------
# Main optimized endpoint
# -----------------------------------------------------------------------------
@api_view(['POST'])
def simulate_score(request):
    # Load stats (with caching)
    try:
        stats = get_cached_stats()
    except CoverageStats.DoesNotExist:
        return Response({"detail": "Coverage stats not found"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    # Parse input
    try:
        lat = float(request.data['lat'])
        lon = float(request.data['lon'])
    except (KeyError, ValueError):
        return Response({"detail": "Invalid lat/lon"}, status=status.HTTP_400_BAD_REQUEST)
    
    pnt = Point(lon, lat, srid=4326)

    # Find intersection in Communes, else Provinces
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

    # Load geometry for Python‐based intensities & routing
    area_obj = Model.objects.only('boundary').get(pk=area_dict['id'])

    # Check cache for preprocessed data
    cache_key = f"areas_data_{Model.__name__.lower()}"
    try:
        cached_areas_data = cache.get(cache_key)
    except:
        cached_areas_data = None
    
    if cached_areas_data is None:
        # Build raw lists (only if not cached)
        all_areas = list(get_area_queryset(Model))
        instances = Model.objects.in_bulk([a['id'] for a in all_areas])

        pops   = [a['population'] for a in all_areas]
        gaps   = [max(a['population'] - a['insured_population'], 0) for a in all_areas]
        vehs   = [a['estimated_vehicles'] for a in all_areas]
        banks  = [bank_intensity_py(instances[a['id']]) for a in all_areas]
        comps  = [competition_intensity_py(instances[a['id']]) for a in all_areas]
        losses = [a['loss_ratio_avg'] or 0 for a in all_areas]
        
        areas_data = {
            'all_areas': all_areas,
            'pops': pops,
            'gaps': gaps, 
            'vehs': vehs,
            'banks': banks,
            'comps': comps,
            'losses': losses
        }
        
        try:
            cache.set(cache_key, areas_data, CACHE_TIMEOUT)
        except:
            pass
    else:
        areas_data = cached_areas_data

    # Extract data
    all_areas = areas_data['all_areas']
    pops = areas_data['pops']
    gaps = areas_data['gaps']
    vehs = areas_data['vehs']
    banks = areas_data['banks']
    comps = areas_data['comps']
    losses = areas_data['losses']

    # z-score normalization
    def z(v, mean, std): return (v - mean) / std if std else 0
    pop_z  = [z(v, stats.pop_mean,  stats.pop_std)  for v in pops]
    gap_z  = [z(v, stats.gap_mean,  stats.gap_std)  for v in gaps]
    veh_z  = [z(v, stats.veh_mean,  stats.veh_std)  for v in vehs]
    bank_z = [z(v, stats.bank_mean, stats.bank_std) for v in banks]
    comp_z = [z(v, stats.comp_mean, stats.comp_std) for v in comps]

    # find our area's index
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
        accessibility = (stats.access_mean - travel_time) / stats.access_std if stats.access_std > 0 else 0
    except Exception as e:
        logging.error(f"Travel time calculation failed: {e}")
        accessibility = 0
        travel_time = None

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
    score_range = stats.raw_max - stats.raw_min
    if score_range > 0:
        score = round((raw - stats.raw_min) / score_range * 100, 2)
    else:
        score = 50.0  # Default middle score if no range
    
    # Ensure score is within bounds
    score = max(0, min(100, score))
    
    potential = 'HIGH' if score >= 70 else 'MEDIUM' if score >= 40 else 'LOW'

    return Response({
        "area_id":   area_dict['id'],
        "area_name": area_dict['name'],
        "coverage":  score,
        "potential": potential,
        "travel_time_minutes": round(travel_time, 1) if travel_time else None
    })