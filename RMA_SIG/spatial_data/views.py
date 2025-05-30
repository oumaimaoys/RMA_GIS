from django.http import HttpResponse
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated  # or AllowAny if local only
from rest_framework import status
from django.core.management import call_command
from rest_framework.response import Response
from .models import Province, Commune, RMAOffice, Competitor, CoverageScore, LossRatio, Area
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
