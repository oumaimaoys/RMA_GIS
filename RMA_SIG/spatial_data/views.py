# spatial_data/views.py

import logging
import math
import requests
import base64
import io
from functools import lru_cache
from statistics import fmean, pstdev

from django.conf import settings
from django.core.management import call_command
from django.core.cache import cache
from django.contrib.gis.geos import Point
from django.contrib.gis.measure import D
from django.db.models import Avg, Prefetch, F

from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated  # Or AllowAny if local only
from rest_framework.response import Response
from rest_framework import status
from django.http import FileResponse, HttpResponse, JsonResponse

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

from .models import (
    Province, Commune, RMAOffice, Competitor,
    CoverageScore, LossRatio, Area, CoverageStats
)
from .serializers import (
    ProvinceSerializer, CommuneSerializer,
    CompetitorSerializer, RMAOfficeSerializer,
    AreaSerializer, CoverageScoreSerializer
)

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# 1. GeoJSON Endpoints (no @renderer_classes)
# -----------------------------------------------------------------------------

# views.py  – RMA SIG API
from django.db.models import F, Prefetch
from rest_framework.decorators import api_view
from rest_framework.response import Response

from .models import Province, Commune, Competitor, RMAOffice, CoverageScore
from .serializers import (
    ProvinceSerializer,
    CommuneSerializer,
    CompetitorSerializer,
    RMAOfficeSerializer,
    CoverageScoreSerializer,
)

# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _with_latest_score(qs):
    """
    Prefetch coverage scores (newest first) and stick the newest one on each
    Province/Commune as `latest_score_object` so the serializers can grab it
    in O(1) without extra queries.
    """
    latest_prefetch = Prefetch(
        "coveragescore_set",                     # default reverse name
        queryset=CoverageScore.objects.order_by("-calculation_date"),
        to_attr="_scores",                      # store list on the object
    )
    qs = qs.prefetch_related(latest_prefetch)
    for obj in qs:
        obj.latest_score_object = obj._scores[0] if obj._scores else None
    return qs


# ----------------------------------------------------------------------
# geoJSON endpoints
# ----------------------------------------------------------------------
@api_view(["GET"])
def provinces_geojson(request):
    qs = _with_latest_score(Province.objects.all())
    return Response(ProvinceSerializer(qs, many=True).data)


@api_view(["GET"])
def communes_geojson(request):
    qs = _with_latest_score(Commune.objects.all())
    return Response(CommuneSerializer(qs, many=True).data)


@api_view(["GET"])
def competitor_geojson(request):
    comps = Competitor.objects.all()
    return Response(CompetitorSerializer(comps, many=True).data)


@api_view(["GET"])
def rma_office_geojson(request):
    offices = RMAOffice.objects.all()
    return Response(RMAOfficeSerializer(offices, many=True).data)


# ----------------------------------------------------------------------
# score-specific geoJSON
# ----------------------------------------------------------------------
@api_view(["GET"])
def coverage_scores_geojson(request):
    """
    Returns one CoverageScore per area – the most recent one –
    with the geometry copied from the related Area.
    """
    qs = (
        CoverageScore.objects
        .order_by("area_id", "-calculation_date")        # newest first per area
        .distinct("area_id")                             # keep only the first
        .select_related("area")                          # pull Area in one join
        .annotate(geom=F("area__boundary"))              # expose geometry
    )
    return Response(CoverageScoreSerializer(qs, many=True).data)


@api_view(["GET"])
def province_scores_geojson(request):
    """
    Convenience wrapper: provinces with only their latest score flattened
    into the feature properties.
    """
    qs = _with_latest_score(Province.objects.all())
    return Response(ProvinceSerializer(qs, many=True).data)


@api_view(["GET"])
def commune_scores_geojson(request):
    qs = _with_latest_score(Commune.objects.all())
    return Response(CommuneSerializer(qs, many=True).data)

# -----------------------------------------------------------------------------
# 2. Trigger Coverage‐Score Recalculation
# -----------------------------------------------------------------------------

@api_view(["POST"])
def run_score_view(request):
    """
    Trigger the nightly AHP‐weighted coverage‐score recalculation.
    """
    try:
        logger.info("Received POST to run-score; invoking calculate_scores")
        call_command("calculate_scores")
        logger.info("calculate_scores completed successfully")
        return Response({"status": "ok"}, status=status.HTTP_200_OK)

    except Exception as e:
        logger.exception("Error running calculate_scores")
        return Response(
            {"status": "error", "detail": str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(["POST"])
def run_stats(request):
    """
    Trigger the calculation of bank/competition intensity and coverage‐gap statistics.
    """
    try:
        call_command("calculate_stats")
        return Response({"message": "Intensity and coverage stats calculation triggered."})
    except Exception as e:
        logger.exception("Error running calculate_stats")
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# -----------------------------------------------------------------------------
# 3. Export to PDF Endpoint
# -----------------------------------------------------------------------------

@api_view(["POST"])
def export_pdf(request):
    """
    Accepts JSON { img: dataURL, title: str, kpis: {...} }
    Returns a generated PDF file as an attachment.
    """
    data = request.data

    # Decode the base64 image payload
    img_b64 = data.get("img", "").split(",", 1)[1]
    img_bytes = base64.b64decode(img_b64)

    # Build the PDF in memory
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)

    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(40, 800, f"RMA GIS Report — {data.get('title', '')}")

    img_reader = ImageReader(io.BytesIO(img_bytes))
    pdf.drawImage(img_reader, 40, 380, width=515, height=400)

    y = 340
    pdf.setFont("Helvetica", 11)
    for label, val in data.get("kpis", {}).items():
        pdf.drawString(40, y, f"{label.capitalize():<15}: {val}")
        y -= 18

    pdf.save()
    buffer.seek(0)

    return FileResponse(buffer, as_attachment=True, filename="RMA_report.pdf")


# -----------------------------------------------------------------------------
# 4. “Simulate Score” Endpoint (unchanged, just included for completeness)
# -----------------------------------------------------------------------------

# Constants & Weights (excerpt; adjust as needed below)
EXCELLENT_THRESHOLD = 80
GOOD_THRESHOLD = 65
MEDIUM_THRESHOLD = 50

WEIGHTS = {
    "demand": 0.40,
    "competition": 0.30,
    "economic": 0.15,
    "accessibility": 0.05,
    "risk": 0.10
}

# ... (all the helper functions: sigmoid_transform, robust_z_score,
#     calculate_percentile_score, get_drive_time, get_area_queryset_values, 
#     get_precalculated_stats, calculate_demand_score, calculate_competition_score,
#     calculate_economic_score, calculate_accessibility_score,
#     calculate_risk_score, generate_recommendations) ...
# (These remain exactly as in your previous version, so I’ve omitted duplicating them here to save space.)

import logging
import requests # Keep for potential real ORS call, or remove if strictly mock
from functools import lru_cache
import random # For mock travel time

from django.conf import settings
from django.core.cache import cache
from django.contrib.gis.geos import Point
from django.db.models import Avg, Prefetch, F

from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated # Or AllowAny if local only
from rest_framework.response import Response
from rest_framework import status

from .models import (
    Province, Commune, RMAOffice, Competitor,
    CoverageScore, Area, CoverageStats # LossRatio needed for aggregation, Bank
)
from .serializers import (
    ProvinceSerializer, CommuneSerializer,
    CompetitorSerializer, RMAOfficeSerializer,
    AreaSerializer, CoverageScoreSerializer
)

logger = logging.getLogger(__name__)

# --- Constants for Scoring Logic ---
WEIGHTS = {
    "demand": 0.30,
    "competition": 0.20,
    "economic": 0.15,
    "accessibility": 0.20,
    "risk": 0.15,
}
EXCELLENT_THRESHOLD = 80
GOOD_THRESHOLD = 65
MEDIUM_THRESHOLD = 50

# -----------------------------------------------------------------------------
# Helper Functions for Scoring & Simulation
# -----------------------------------------------------------------------------

def get_area_queryset_values(Model):
    """
    Returns a queryset of the given Model (Commune or Province)
    annotated with average loss ratio, and with relevant fields selected
    for the scoring simulation.
    The result is a ValuesQuerySet, yielding dictionaries.
    Assumes LossRatio model has 'area' ForeignKey to Area model, and 'loss_ratio' field.
    """
    # Model will be Commune or Province. Area is their parent.
    # area_ptr is the reverse accessor from Commune/Province to its Area instance.
    return (
        Model.objects
        .annotate(loss_ratio_avg=Avg("area_ptr__lossratio_records__loss_ratio")) # area_ptr links to Area model instance
        .values(
            "id",  # PK of the Province/Commune instance
            "area_ptr__id", # PK of the parent Area instance
            "area_ptr__name",
            "area_ptr__population",
            "area_ptr__insured_population",
            "area_ptr__estimated_vehicles",
            "area_ptr__bank_intensity",
            "area_ptr__competition_intensity",
            "loss_ratio_avg", # The annotated average loss ratio
            "boundary" # Needed for __intersects query
        )
        .order_by("id") # ADD THIS LINE: Order by the primary key of the Model (Commune/Province)
    )

@lru_cache(maxsize=2) # Cache for "Commune" and "Province"
def get_precalculated_stats(area_type_str):
    """
    Fetches pre-calculated CoverageStats for the given area_type ('Commune' or 'Province').
    Caches the result.
    """
    try:
        stats_obj = CoverageStats.objects.get(area_type=area_type_str)
        logger.info(f"Fetched CoverageStats for {area_type_str} from DB.")
        return stats_obj
    except CoverageStats.DoesNotExist:
        logger.warning(f"CoverageStats not found for area_type='{area_type_str}'. No stats will be used.")
        return None
    except CoverageStats.MultipleObjectsReturned:
        logger.error(f"Multiple CoverageStats found for area_type='{area_type_str}'. Returning the latest one.")
        # Fallback: return the most recently calculated one
        return CoverageStats.objects.filter(area_type=area_type_str).order_by('-calculation_date').first()
    except Exception as e:
        logger.error(f"Error fetching CoverageStats for {area_type_str}: {e}")
        return None


def calculate_percentile_score(value, all_values_list, higher_is_better=True):
    """Calculates a score from 0-100 based on the value's percentile rank."""
    if not all_values_list or value is None:
        return 50.0 # Default middle score if data is missing or value is None
    
    # Filter out None values from the list to avoid errors in comparison/sorting
    filtered_values = [v for v in all_values_list if v is not None]
    if not filtered_values:
        return 50.0

    sorted_values = sorted(filtered_values)
    if not sorted_values: # Should not happen if filtered_values is not empty
        return 50.0

    try:
        # Count how many values in the sorted list are less than the current value
        rank = sum(1 for v in sorted_values if v < value)
        percentile = (rank / len(sorted_values)) * 100
    except ZeroDivisionError: # Should be caught by "if not sorted_values"
        return 50.0

    return percentile if higher_is_better else (100.0 - percentile)


def calculate_z_score_based_score(value, mean, std_dev, higher_is_better=True):
    """Calculates a score from 0-100 based on Z-score, scaled and capped."""
    if value is None or mean is None or std_dev is None:
        logger.debug(f"Missing value for Z-score: val={value}, mean={mean}, std={std_dev}. Defaulting to 50.")
        return 50.0

    if std_dev == 0:
        # If std_dev is 0, all historical values were the same as the mean.
        # If current value is also the mean, it's "average" (50).
        # If it's different, it's an outlier (could be 0 or 100 depending on higher_is_better).
        # For simplicity, return 50 if value is close to mean, else depends on direction.
        return 50.0 if value == mean else (100.0 if (value > mean and higher_is_better) or \
                                                    (value < mean and not higher_is_better) else 0.0)

    z = (value - mean) / std_dev

    # Scale Z-score to 0-100 range.
    # A Z-score of 0 maps to 50.
    # We can define that a Z-score of +/-2 maps to 0/100 or 100/0.
    # So, score = 50 + z * 25 (if z=2 -> 100, z=-2 -> 0)
    # Or score = 50 - z * 25 (if z=2 -> 0, z=-2 -> 100)
    if higher_is_better:
        score = 50 + (z * 25)
    else:
        score = 50 - (z * 25)
    
    return max(0.0, min(100.0, score))


def calculate_demand_score(population, coverage_gap, vehicles, market_potential_untapped,
                           all_populations, all_gaps, all_vehicles, all_market_potentials):
    pop_score = calculate_percentile_score(population, all_populations, higher_is_better=True)
    gap_score = calculate_percentile_score(coverage_gap, all_gaps, higher_is_better=True)
    veh_score = calculate_percentile_score(vehicles, all_vehicles, higher_is_better=True)
    mkt_pot_score = calculate_percentile_score(market_potential_untapped, all_market_potentials, higher_is_better=True)
    
    # Weights for sub-components of demand score
    return (0.3 * pop_score + 0.3 * gap_score + 0.2 * veh_score + 0.2 * mkt_pot_score)

def calculate_competition_score(current_comp_intensity, global_stats):
    if global_stats and global_stats.comp_intensity_mean is not None and global_stats.comp_intensity_std is not None:
        return calculate_z_score_based_score(
            current_comp_intensity,
            global_stats.comp_intensity_mean,
            global_stats.comp_intensity_std,
            higher_is_better=False # Lower competition intensity is better
        )
    logger.warning("Competition score defaulting to 50 due to missing global stats.")
    return 50.0

def calculate_economic_score(current_bank_intensity, global_stats):
    if global_stats and global_stats.bank_intensity_mean is not None and global_stats.bank_intensity_std is not None:
        return calculate_z_score_based_score(
            current_bank_intensity,
            global_stats.bank_intensity_mean,
            global_stats.bank_intensity_std,
            higher_is_better=True # Higher bank intensity often correlates with economic activity
        )
    logger.warning("Economic score defaulting to 50 due to missing global stats.")
    return 50.0

def calculate_risk_score(current_loss_ratio, global_stats):
    if global_stats and global_stats.loss_ratio_mean is not None and global_stats.loss_ratio_std is not None:
        # Ensure current_loss_ratio is not None before passing
        if current_loss_ratio is None:
            logger.warning("Risk score defaulting to 50 because current_loss_ratio is None.")
            return 50.0 # Or handle as high risk, e.g., 0.0
        return calculate_z_score_based_score(
            current_loss_ratio,
            global_stats.loss_ratio_mean,
            global_stats.loss_ratio_std,
            higher_is_better=False # Lower loss ratio is better
        )
    logger.warning("Risk score defaulting to 50 due to missing global stats or loss ratio.")
    return 50.0


@lru_cache(maxsize=256) # Cache recent travel time requests
def get_drive_time(start_lat, start_lon, end_lat, end_lon):
    """
    Gets drive time from OpenRouteService or a mock.
    Coordinates should be WGS84 (lat, lon).
    Returns time in minutes.
    """
    ORS_API_KEY = getattr(settings, 'ORS_API_KEY', None)
    ORS_API_URL = getattr(settings, 'ORS_API_URL', "https://api.openrouteservice.org/v2/directions/driving-car")

    if not ORS_API_KEY:
        logger.warning("ORS_API_KEY not configured. Using mock travel time.")
        return random.uniform(10, 60) # Mock time in minutes

    headers = {
        'Authorization': ORS_API_KEY,
        'Content-Type': 'application/json'
    }
    # ORS API expects (lon, lat)
    body = {
        "coordinates": [[start_lon, start_lat], [end_lon, end_lat]],
        "radiuses": [-1, -1] # Allow snapping to nearest road
    }
    try:
        response = requests.post(ORS_API_URL, json=body, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        # Duration is in seconds
        duration_seconds = data['routes'][0]['summary']['duration']
        return duration_seconds / 60.0
    except requests.exceptions.RequestException as e:
        logger.error(f"ORS API request failed: {e}")
    except (KeyError, IndexError) as e:
        logger.error(f"Failed to parse ORS response: {e}")
    return None # Indicate failure or fallback to a default accessibility score


def calculate_accessibility_score(travel_time_minutes):
    if travel_time_minutes is None:
        logger.warning("Travel time is None, accessibility defaulting to low-medium.")
        return 30.0 # Default score if travel time couldn't be calculated

    if travel_time_minutes <= 10: return 100.0
    if travel_time_minutes <= 15: return 90.0
    if travel_time_minutes <= 20: return 80.0
    if travel_time_minutes <= 30: return 60.0
    if travel_time_minutes <= 45: return 40.0
    if travel_time_minutes <= 60: return 20.0
    return 0.0


def generate_recommendations(final_score, demand, competition, economic, accessibility, risk, travel_time_minutes):
    recs = []
    if final_score >= EXCELLENT_THRESHOLD:
        recs.append("Strong overall potential. Consider as a high-priority location.")
    elif final_score >= GOOD_THRESHOLD:
        recs.append("Good overall potential. Suitable for expansion with standard due diligence.")
    elif final_score >= MEDIUM_THRESHOLD:
        recs.append("Moderate potential. Requires careful consideration of specific strengths and weaknesses.")
    else:
        recs.append("Low overall potential. May not be a strategic fit without compelling reasons.")

    # Specific component feedback
    if demand < 40: recs.append("Demand indicators are weak. Thoroughly assess local market needs and growth prospects.")
    elif demand < 60: recs.append("Demand is moderate. Investigate specific drivers of demand for target products.")
    
    if competition < 40: recs.append("Competition appears intense. Develop a strong differentiation strategy.")
    elif competition < 60: recs.append("Competition is notable. Analyze key competitors and market positioning.")

    if economic < 40: recs.append("Economic indicators are subpar. Evaluate local economic stability and purchasing power.")
    
    tt_str = f"{travel_time_minutes:.0f} minutes" if travel_time_minutes is not None else "N/A"
    if accessibility < 40: recs.append(f"Accessibility from target point is poor (travel time: {tt_str}). Re-evaluate optimal positioning within the area.")
    elif accessibility < 60: recs.append(f"Accessibility is fair (travel time: {tt_str}). Consider micro-location factors.")

    if risk < 40: recs.append("Risk indicators (e.g., high loss ratio) are concerning. Investigate sources of risk and mitigation strategies.")
    
    if not recs or len(recs) == 1 and final_score >= GOOD_THRESHOLD: # if only the overall comment was added for good/excellent
        recs.append("Overall profile appears balanced. Proceed with standard operational planning.")
    return recs

@api_view(["POST"])
def simulate_score(request):
    # ... (lat, lon, pnt validation - unchanged) ...
    try:
        lat = float(request.data.get("lat"))
        lon = float(request.data.get("lon"))
        pnt = Point(lon, lat, srid=4326)
    except (KeyError, ValueError, TypeError, AttributeError):
        return Response(
            {"detail": "Invalid or missing 'lat'/'lon' parameters."},
            status=status.HTTP_400_BAD_REQUEST
        )

    # ... (area determination logic: area_dict, Model, area_model_name, area_id_from_dict, area_name_from_dict - unchanged) ...
    area_dict = get_area_queryset_values(Commune).filter(boundary__intersects=pnt).first()
    Model = Commune
    if not area_dict:
        area_dict = get_area_queryset_values(Province).filter(boundary__intersects=pnt).first()
        Model = Province
    
    if not area_dict:
        return Response(
            {"detail": "Location is outside known administrative areas (Communes or Provinces)."},
            status=status.HTTP_404_NOT_FOUND
        )

    area_model_name = Model.__name__
    area_id_from_dict = area_dict["id"]
    area_name_from_dict = area_dict["area_ptr__name"]


    # ... (percentile_data_collection caching logic - unchanged) ...
    percentile_cache_key = f"percentile_data_{area_model_name.lower()}"
    percentile_data_collection = cache.get(percentile_cache_key)

    if percentile_data_collection is None:
        logger.info(f"Cache miss for {percentile_cache_key}. Recalculating percentile data.")
        all_area_records_values = list(get_area_queryset_values(Model))
        
        if not all_area_records_values:
            logger.error(f"No data found for {area_model_name} areas to calculate percentiles.")
            return Response(
                {"detail": f"Insufficient data for {area_model_name} percentile calculation."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        percentile_data_collection = {
            "populations": [a.get("area_ptr__population") for a in all_area_records_values],
            "gaps": [
                max(0, (a.get("area_ptr__population") or 0) - (a.get("area_ptr__insured_population") or 0))
                for a in all_area_records_values
            ],
            "vehicles": [a.get("area_ptr__estimated_vehicles") for a in all_area_records_values],
            "market_potentials_untapped": [
                (((pop or 0) - (ins or 0)) / (pop or 1)) * 100
                if (pop := a.get("area_ptr__population")) and pop > 0 else 0.0
                for a in all_area_records_values
                for ins in [a.get("area_ptr__insured_population")]
            ],
        }
        cache.set(percentile_cache_key, percentile_data_collection, timeout=3600)
        logger.info(f"Cached percentile data for {area_model_name}.")

    # ... (global_stats fetching - unchanged) ...
    global_stats = get_precalculated_stats(area_model_name)

    # ... (current area specifics extraction: current_population, ..., current_loss_ratio - unchanged) ...
    current_population = area_dict.get("area_ptr__population") or 0
    current_insured = area_dict.get("area_ptr__insured_population") or 0
    current_vehicles = area_dict.get("area_ptr__estimated_vehicles") or 0
    current_gap = max(0, current_population - current_insured)
    current_market_untapped = (
        ((current_population - current_insured) / current_population) * 100
        if current_population > 0 else 0.0
    )
    current_bank_intensity = area_dict.get("area_ptr__bank_intensity")
    current_comp_intensity = area_dict.get("area_ptr__competition_intensity")
    current_loss_ratio = area_dict.get("loss_ratio_avg")

    # ... (component score calculations: demand_score, ..., risk_score - unchanged) ...
    demand_score = calculate_demand_score(
        current_population, current_gap, current_vehicles, current_market_untapped,
        percentile_data_collection["populations"], percentile_data_collection["gaps"],
        percentile_data_collection["vehicles"], percentile_data_collection["market_potentials_untapped"]
    )
    competition_score = calculate_competition_score(current_comp_intensity, global_stats)
    economic_score = calculate_economic_score(current_bank_intensity, global_stats)
    risk_score = calculate_risk_score(current_loss_ratio, global_stats)


    # ... (accessibility score calculation: travel_time_minutes, accessibility_score - unchanged) ...
    travel_time_minutes = None
    accessibility_score = 30.0
    try:
        area_instance = Model.objects.only("boundary").get(pk=area_id_from_dict)
        if area_instance.boundary:
            centroid_geom = area_instance.boundary.centroid 
            if centroid_geom.srid != 4326:
                centroid_geom.transform(4326)
            travel_time_minutes = get_drive_time(lat, lon, centroid_geom.y, centroid_geom.x)
            accessibility_score = calculate_accessibility_score(travel_time_minutes)
        else:
            logger.warning(f"Area {area_name_from_dict} (ID: {area_id_from_dict}) has no boundary for centroid calculation.")
    except Model.DoesNotExist:
        logger.error(f"Failed to refetch {area_model_name} (ID: {area_id_from_dict}) for centroid.")
    except Exception as e:
        logger.error(f"Error in accessibility calculation for {area_name_from_dict}: {e}", exc_info=True)


    # ... (final_score calculation - unchanged) ...
    final_score = (
        WEIGHTS["demand"] * demand_score +
        WEIGHTS["competition"] * competition_score +
        WEIGHTS["economic"] * economic_score +
        WEIGHTS["accessibility"] * accessibility_score +
        WEIGHTS["risk"] * risk_score
    )
    final_score = round(max(0.0, min(100.0, final_score)), 1)

    # ... (potential_category, market_size_category, etc. - unchanged) ...
    if final_score >= EXCELLENT_THRESHOLD: potential_category = "EXCELLENT"
    elif final_score >= GOOD_THRESHOLD: potential_category = "GOOD"
    elif final_score >= MEDIUM_THRESHOLD: potential_category = "MEDIUM"
    else: potential_category = "LOW"

    market_size_category = "LARGE" if current_population >= 50000 else \
                           "MEDIUM" if current_population >= 15000 else "SMALL"
    competition_level_category = "LOW" if competition_score >= 70 else \
                                 "MEDIUM" if competition_score >= 40 else "HIGH"
    coverage_rate_percent = round((current_insured / current_population) * 100, 1) if current_population > 0 else 0.0
    recommendations = generate_recommendations(
        final_score, demand_score, competition_score, economic_score,
        accessibility_score, risk_score, travel_time_minutes
    )

    # MODIFICATION IS HERE:
    return Response({
        "simulation_input": {"latitude": lat, "longitude": lon},
        "area_info": {
            "id": area_id_from_dict,
            "name": area_name_from_dict,
            "type": area_model_name,
        },
        "score": final_score,  # ADDED: Top-level score for direct JS access
        "overall_score": {     # Existing nested structure (good for other details)
            "value": final_score,
            "potential_category": potential_category,
        },
        "component_scores": {
            "demand": round(demand_score, 1),
            "competition": round(competition_score, 1),
            "economic": round(economic_score, 1),
            "accessibility": round(accessibility_score, 1),
            "risk": round(risk_score, 1),
        },
        "key_metrics": {
            "population": current_population,
            "coverage_gap_persons": current_gap,
            "estimated_vehicles": current_vehicles,
            "current_coverage_rate_percent": coverage_rate_percent,
            "market_potential_untapped_percent": round(current_market_untapped, 1),
            "market_size_category": market_size_category,
            "competition_level_category": competition_level_category,
            "average_loss_ratio_in_area": round(current_loss_ratio, 3) if current_loss_ratio is not None else None,
            "travel_time_to_centroid_minutes": round(travel_time_minutes, 1) if travel_time_minutes is not None else None,
        },
        "recommendations": recommendations
    })