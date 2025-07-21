from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect, get_object_or_404
from .models import Notification
from django.http import JsonResponse, HttpResponseRedirect
from django.shortcuts import get_object_or_404, resolve_url

def map_view(request):
    return render(request, "frontend/map.html")

# frontend/views.py


@login_required
def notif_list(request):
    qs = request.user.notifications.all()
    return render(request, "notifications/list.html", {"notifications": qs})

@login_required
def notif_read(request, pk):
    n = get_object_or_404(Notification, pk=pk, recipient=request.user)
    n.mark_read()
    return redirect(n.url or "notifications:list")


from django.shortcuts import get_object_or_404, redirect, resolve_url
from django.contrib.auth.decorators import login_required

@login_required
def read_and_redirect_notification(request, pk):
    notification = get_object_or_404(Notification, pk=pk, user=request.user)
    if not notification.read:
        notification.read = True
        notification.save(update_fields=['read'])

    # Assuming your Notification model has a 'target_url' or similar
    # or can derive it from related objects
    if hasattr(notification, 'get_absolute_url_for_target'): # Example method
        redirect_url = notification.get_absolute_url_for_target()
    elif notification.target_url: # If you store a URL directly
         redirect_url = notification.target_url
    else:
        # Fallback to the main notification list or a default page
        redirect_url = resolve_url('frontend:list')

    return redirect(redirect_url)
@login_required
def mark_all_read(request):
    if request.method == "POST":
        request.user.notifications.filter(read=False).update(read=True)
        if request.headers.get("Hx-Request") or request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return JsonResponse({"ok": True})      # AJAX
    return HttpResponseRedirect(request.META.get("HTTP_REFERER", resolve_url("frontend:list")))

@login_required
def toggle_read(request, pk):
    """
    Flip the read flag for one notif and return JSON with the new state
    """
    n = get_object_or_404(Notification, pk=pk, recipient=request.user)
    n.read = not n.read
    n.save(update_fields=["read"])
    return JsonResponse({"read": n.read})

# views.py
from django.shortcuts import render
from django.db.models import Avg, Count, Sum, Max, Min, Q
from django.contrib.gis.db.models.functions import Distance
from django.contrib.gis.geos import Point
from django.http import JsonResponse
from django.core.paginator import Paginator
from django.utils import timezone
from decimal import Decimal
import json

from spatial_data.models import (
    Area, Commune, Province, RMAOffice, RMABGD, RMAAgent, 
    Competitor, Bank, LossRatio, CoverageScore, CoverageStats,
    CA, OSMDiscoveredCompetitor, Variables
)

def dashboard_view(request):
    """Main dashboard view with overview statistics and key metrics"""
    # ─── Filters ─────────────────────────────────────────────────────────
    province_filter  = request.GET.get('province')
    potential_filter = request.GET.get('potential')

    communes  = Commune.objects.all()
    provinces = Province.objects.all()

    if province_filter:
        communes = communes.filter(name__icontains=province_filter)

    # ─── Overview Statistics ────────────────────────────────────────────
    total_communes           = communes.count()
    total_provinces          = provinces.count()
    total_rma_offices        = RMAOffice.objects.count()
    total_competitors        = Competitor.objects.count()
    total_banks              = Bank.objects.count()
    total_population         = communes.aggregate(t=Sum('population'))['t'] or 0
    total_estimated_vehicles = communes.aggregate(t=Sum('estimated_vehicles'))['t'] or 0
    total_insured_population = communes.aggregate(t=Sum('insured_population'))['t'] or 0

    # ─── Coverage Distribution (for chart & template) ────────────────
    coverage_qs = (
        CoverageScore.objects
        .values('potential')
        .annotate(count=Count('id'))
        .order_by('potential')
    )
    coverage_distribution = list(coverage_qs)  # for json
    # template can also use coverage_qs if you needed it

    # ─── Top Opportunities ─────────────────────────────────────────────
    coverage_scores = CoverageScore.objects.select_related('area')
    if potential_filter:
        coverage_scores = coverage_scores.filter(potential=potential_filter)
    top_opportunities = coverage_scores.order_by('-score')[:10]

    # ─── Competition & Market Gaps ─────────────────────────────────────
    high_competition_areas = communes.filter(
        competition_intensity__gt=0
    ).order_by('-competition_intensity')[:10]

    market_gaps = communes.filter(
        population__gt=10000,
        competition_count__lt=3
    ).order_by('-population')[:10]

    # ─── Recent Loss Ratios (template) & JSON for chart ───────────────
    recent_loss_qs = (
        LossRatio.objects
        .select_related('area', 'commune', 'province', 'RMA_office')
        .order_by('-id')[:10]
    )
    # build plain list for JSON
    recent_loss_ratios_json = []
    for lr in recent_loss_qs:
        if lr.area:
            label = lr.area.name
        elif lr.commune:
            label = lr.commune.name
        elif lr.province:
            label = lr.province.name
        else:
            label = ""
        recent_loss_ratios_json.append({
            'label':      label,
            'loss_ratio': lr.loss_ratio,
        })

    # ─── RMA Performance ───────────────────────────────────────────────
    rma_performance_list = []
    for Model, office_type in ((RMABGD, 'RMABGD'), (RMAAgent, 'RMAAgent')):
        offices = Model.objects.all()
        ca_stats = CA.objects.filter(agency__in=offices).aggregate(
            avg_ca=Avg('CA_value'),
            total_ca=Sum('CA_value'),
            count=Count('id')
        )
        rma_performance_list.append({
            'type':     office_type,
            'count':    offices.count(),
            'avg_ca':   ca_stats['avg_ca'] or 0,
            'total_ca': ca_stats['total_ca'] or 0,
        })

    # ─── Provinces Data for Chart ──────────────────────────────────────
    provinces_data_json = []
    for province in provinces[:10]:
        if province.boundary:
            # Sum up population of all Communes whose boundary lies within this Province
            pop = communes \
                .filter(boundary__within=province.boundary) \
                .aggregate(t=Sum('population'))['t'] or 0

            offices_count = RMAOffice.objects.filter(
                location__within=province.boundary
            ).count()

            comp_count = Competitor.objects.filter(
                location__within=province.boundary
            ).count()

            avg_score = CoverageScore.objects.filter(
                area__boundary__within=province.boundary
            ).aggregate(a=Avg('score'))['a'] or 0
        else:
            pop = offices_count = comp_count = avg_score = 0

        provinces_data_json.append({
            'name':        province.name,
            'population':  pop,
            'rma_offices': offices_count,
            'competitors': comp_count,
            'avg_score':   round(avg_score, 1),
        })

    # ─── High Risk Areas ───────────────────────────────────────────────
    high_risk_areas = (
        LossRatio.objects
        .filter(loss_ratio__gt=0.7)
        .select_related('area')
        .order_by('-loss_ratio')[:10]
    )

    # ─── Build Context ──────────────────────────────────────────────────
    context = {
        # overview
        'total_communes':            total_communes,
        'total_provinces':           total_provinces,
        'total_rma_offices':         total_rma_offices,
        'total_competitors':         total_competitors,
        'total_banks':               total_banks,
        'total_population':          total_population,
        'total_estimated_vehicles':  total_estimated_vehicles,
        'total_insured_population':  total_insured_population,

        # template data
        'coverage_distribution':     coverage_qs,
        'top_opportunities':         top_opportunities,
        'high_competition_areas':    high_competition_areas,
        'market_gaps':               market_gaps,
        'recent_loss_ratios':        recent_loss_qs,
        'rma_performance':           rma_performance_list,
        'provinces_data':            provinces,              # for any table loops
        'high_risk_areas':           high_risk_areas,

        # filters
        'current_province_filter':   province_filter,
        'current_potential_filter':  potential_filter,
        'potential_choices':         ['EXCELLENT','GOOD','MEDIUM','LOW'],

        # JSON‐serializable lists for Chart.js via json_script
        'coverage_chart_data':       coverage_distribution,
        'provinces_data_json':       provinces_data_json,
        'recent_loss_ratios_json':   recent_loss_ratios_json,
        'rma_performance_json':      rma_performance_list,
    }

    return render(request, 'dashboard/main_dashboard.html', context)


def area_detail_view(request, area_id):
    """Detailed view for a specific area"""
    try:
        area = Area.objects.get(id=area_id)
    except Area.DoesNotExist:
        return render(request, '404.html')
    
    # Get coverage score for this area
    try:
        coverage_score = CoverageScore.objects.get(area=area)
    except CoverageScore.DoesNotExist:
        coverage_score = None
    
    # Get competitors in the area
    competitors_in_area = []
    if area.boundary:
        competitors_in_area = Competitor.objects.filter(
            location__within=area.boundary
        )
    
    # Get banks in the area
    banks_in_area = []
    if area.boundary:
        banks_in_area = Bank.objects.filter(
            location__within=area.boundary
        )
    
    # Get nearby RMA offices
    nearby_rma_offices = []
    if area.boundary:
        # Get centroid of the area for distance calculation
        centroid = area.boundary.centroid
        nearby_rma_offices = RMAOffice.objects.annotate(
            distance=Distance('location', centroid)
        ).order_by('distance')[:5]
    
    # Get loss ratio history
    loss_ratio_history = LossRatio.objects.filter(
        Q(area=area) | Q(commune=area) | Q(province=area)
    ).order_by('-id')[:10]
    
    context = {
        'area': area,
        'coverage_score': coverage_score,
        'competitors_in_area': competitors_in_area,
        'banks_in_area': banks_in_area,
        'nearby_rma_offices': nearby_rma_offices,
        'loss_ratio_history': loss_ratio_history,
    }
    
    return render(request, 'dashboard/area_detail.html', context)


def api_coverage_data(request):
    """API endpoint for coverage data (for AJAX requests)"""
    coverage_scores = CoverageScore.objects.select_related('area').all()
    
    data = []
    for score in coverage_scores:
        data.append({
            'area_id': score.area.id,
            'area_name': score.area.name,
            'score': score.score,
            'potential': score.potential,
            'population': score.area.population,
            'competition_count': score.area.competition_count,
            'bank_count': score.area.bank_count,
            'demand_score': score.demand_score,
            'competition_score': score.competition_score,
            'economic_score': score.economic_score,
            'accessibility_score': score.accessibility_score,
            'risk_score': score.risk_score,
        })
    
    return JsonResponse({'data': data})


def market_analysis_view(request):
    """Market analysis focused view"""
    
    # Market potential analysis
    high_potential_low_competition = (
        Commune.objects
        .filter(population__gt=5000, competition_count__lte=2)
        .order_by('-population')[:20]
    )
    
    # Underserved markets
    underserved_markets = (
        Commune.objects
        .filter(
            population__gt=10000,
            competition_intensity__lt=5.0  # Less than 5 competitors per 10k population
        )
        .order_by('-population')[:15]
    )
    
    # Oversaturated markets
    oversaturated_markets = (
        Commune.objects
        .filter(competition_intensity__gt=20.0)  # More than 20 competitors per 10k population
        .order_by('-competition_intensity')[:15]
    )
    
    # Market penetration analysis
    penetration_analysis = []
    provinces = Province.objects.all()[:10]
    
    for province in provinces:
        # Calculate market penetration metrics
        total_pop = province.population
        insured_pop = province.insured_population
        penetration_rate = (insured_pop / total_pop * 100) if total_pop > 0 else 0
        
        penetration_analysis.append({
            'province': province.name,
            'total_population': total_pop,
            'insured_population': insured_pop,
            'penetration_rate': round(penetration_rate, 2),
            'estimated_vehicles': province.estimated_vehicles,
            'competition_count': province.competition_count,
        })
    
    context = {
        'high_potential_low_competition': high_potential_low_competition,
        'underserved_markets': underserved_markets,
        'oversaturated_markets': oversaturated_markets,
        'penetration_analysis': penetration_analysis,
    }
    
    return render(request, 'dashboard/market_analysis.html', context)


def rma_performance_view(request):
    """RMA office performance analysis"""
    
    # Get performance data for different office types
    bgd_offices = RMABGD.objects.select_related().all()
    agent_offices = RMAAgent.objects.select_related().all()
    
    # Performance metrics by office
    office_performance = []
    
    for office in RMAOffice.objects.all()[:20]:  # Limit for performance
        # Get CA data
        ca_data = CA.objects.filter(agency=office).aggregate(
            total_ca=Sum('CA_value'),
            avg_ca=Avg('CA_value'),
            years_count=Count('year', distinct=True)
        )
        
        # Get nearby competition
        if office.location:
            # Create a buffer around the office (e.g., 5km radius)
            from django.contrib.gis.measure import D
            nearby_competitors = Competitor.objects.filter(
                location__distance_lte=(office.location, D(km=5))
            ).count()
        else:
            nearby_competitors = 0
        
        office_performance.append({
            'office': office,
            'total_ca': ca_data['total_ca'] or 0,
            'avg_ca': ca_data['avg_ca'] or 0,
            'years_active': ca_data['years_count'] or 0,
            'nearby_competitors': nearby_competitors,
            'office_type': 'BGD' if hasattr(office, 'rmabgd') else 'Agent',
        })
    
    # Sort by total CA
    office_performance.sort(key=lambda x: x['total_ca'], reverse=True)
    
    # BGD specific analysis
    bgd_analysis = []
    for bgd in bgd_offices[:15]:
        bgd_analysis.append({
            'bgd': bgd,
            'type_bgd': bgd.type_BGD,
            'partenaire': bgd.Partenaire,
            'state': bgd.RMA_BGD_state,
            'creation_date': bgd.formatted_date,
        })
    
    context = {
        'office_performance': office_performance[:15],
        'bgd_analysis': bgd_analysis,
        'total_bgd': bgd_offices.count(),
        'total_agents': agent_offices.count(),
    }
    
    return render(request, 'dashboard/rma_performance.html', context)

def api_area_stats(request, area_id):
    """API endpoint for detailed area statistics"""
    try:
        area = Area.objects.get(id=area_id)
        
        # Get coverage score
        coverage_score = None
        try:
            coverage_score = CoverageScore.objects.get(area=area)
        except CoverageScore.DoesNotExist:
            pass
        
        # Calculate additional metrics
        data = {
            'area_id': area.id,
            'name': area.name,
            'population': area.population,
            'insured_population': area.insured_population,
            'estimated_vehicles': area.estimated_vehicles,
            'population_density': area.population_density,
            'vehicle_density': area.vehicle_density,
            'competition_count': area.competition_count,
            'competition_intensity': area.competition_intensity,
            'bank_count': area.bank_count,
            'bank_intensity': area.bank_intensity,
            'coverage_score': {
                'score': coverage_score.score if coverage_score else None,
                'potential': coverage_score.potential if coverage_score else None,
                'demand_score': coverage_score.demand_score if coverage_score else None,
                'competition_score': coverage_score.competition_score if coverage_score else None,
                'economic_score': coverage_score.economic_score if coverage_score else None,
                'accessibility_score': coverage_score.accessibility_score if coverage_score else None,
                'risk_score': coverage_score.risk_score if coverage_score else None,
            } if coverage_score else None
        }
        
        return JsonResponse(data)
        
    except Area.DoesNotExist:
        return JsonResponse({'error': 'Area not found'}, status=404)
    
from django.shortcuts import render
from django.db.models import Q, Avg, Sum
from django.contrib.gis.geos import MultiPolygon
from spatial_data.models import (
    Area, Commune, Province, RegionAdministrative, 
    RMAOffice, RMABGD, RMAAgent, Competitor, Bank, 
    LossRatio, CoverageScore, CA
)

def coverage_analysis_view(request):
    area_type = request.GET.get('area_type', 'commune')
    search_query = request.GET.get('search', '')

    # Choose base queryset
    if area_type == 'commune':
        queryset = Commune.objects.all()
        model_name = 'Commune'
    elif area_type == 'province':
        queryset = Province.objects.all()
        model_name = 'Province'
    elif area_type == 'region':
        queryset = RegionAdministrative.objects.all()
        model_name = 'Administrative Region'
    else:
        queryset = Area.objects.all()
        model_name = 'Area'

    if search_query:
        queryset = queryset.filter(name__icontains=search_query)

    if area_type == 'region':
        regions_data = []

        for region in queryset:
            provinces_in_region = Province.objects.filter(id__in=region.provinces.values_list('id', flat=True))

            total_population = sum(p.population for p in provinces_in_region)
            total_estimated_vehicles = sum(p.estimated_vehicles for p in provinces_in_region)
            total_insured_population = sum(p.insured_population for p in provinces_in_region)
            total_competition_count = sum(p.competition_count for p in provinces_in_region)
            total_bank_count = sum(p.bank_count for p in provinces_in_region)

            avg_competition_intensity = (
                sum(p.competition_intensity for p in provinces_in_region) / len(provinces_in_region)
                if provinces_in_region else 0
            )
            avg_bank_intensity = (
                sum(p.bank_intensity for p in provinces_in_region) / len(provinces_in_region)
                if provinces_in_region else 0
            )

            # --- Optimization: Use merged MultiPolygon for spatial query
            merged_boundary = MultiPolygon(*[p.boundary for p in provinces_in_region if p.boundary])
            rma_offices = RMAOffice.objects.filter(location__within=merged_boundary) if merged_boundary else []

            coverage_scores = CoverageScore.objects.filter(area__in=provinces_in_region)
            avg_score = coverage_scores.aggregate(Avg('score'))['score__avg'] or 0

            loss_ratios = LossRatio.objects.filter(
                Q(province__in=provinces_in_region) |
                Q(commune__province__in=provinces_in_region) |
                Q(area__in=provinces_in_region)
            )
            avg_loss_ratio = loss_ratios.aggregate(Avg('loss_ratio'))['loss_ratio__avg'] or 0

            regions_data.append({
                'area': region,
                'area_id': region.id,
                'population': total_population,
                'estimated_vehicles': total_estimated_vehicles,
                'insured_population': total_insured_population,
                'competition_count': total_competition_count,
                'bank_count': total_bank_count,
                'competition_intensity': avg_competition_intensity,
                'bank_intensity': avg_bank_intensity,
                'rma_offices': rma_offices,
                'coverage_score': avg_score,
                'avg_loss_ratio': avg_loss_ratio,
                'provinces_count': len(provinces_in_region),
            })

        context = {
            'areas_data': regions_data,
            'area_type': area_type,
            'model_name': model_name,
            'search_query': search_query,
        }

    else:
        areas_data = []
        for area in queryset:
            rma_offices = RMAOffice.objects.filter(location__within=area.boundary)
            rma_bgd = RMABGD.objects.filter(location__within=area.boundary)
            rma_agents = RMAAgent.objects.filter(location__within=area.boundary)

            coverage_score = CoverageScore.objects.filter(area=area).order_by('-calculation_date').first()

            loss_ratio_query = Q(area=area)
            if isinstance(area, Commune):
                loss_ratio_query |= Q(commune=area)
            elif isinstance(area, Province):
                loss_ratio_query |= Q(province=area)

            avg_loss_ratio = LossRatio.objects.filter(loss_ratio_query).aggregate(Avg('loss_ratio'))['loss_ratio__avg'] or 0

            total_ca = CA.objects.filter(agency__location__within=area.boundary).aggregate(Sum('CA_value'))['CA_value__sum'] or 0

            areas_data.append({
                'area': area,
                'area_id': area.id,
                'rma_offices': rma_offices,
                'rma_bgd': rma_bgd,
                'rma_agents': rma_agents,
                'coverage_score': coverage_score,
                'avg_loss_ratio': avg_loss_ratio,
                'total_ca': total_ca,
            })

        context = {
            'areas_data': areas_data,
            'area_type': area_type,
            'model_name': model_name,
            'search_query': search_query,
        }

    return render(request, 'dashboard/table_summary.html', context)



def get_area_details(request, area_id):
    """
    AJAX endpoint to get detailed information about a specific area
    """
    area_type = request.GET.get('area_type', 'commune')
    
    try:
        if area_type == 'commune':
            area = Commune.objects.get(id=area_id)
        elif area_type == 'province':
            area = Province.objects.get(id=area_id)
        elif area_type == 'region':
            area = RegionAdministrative.objects.get(id=area_id)
        else:
            area = Area.objects.get(id=area_id)
        
        # Get detailed data
        if area_type == 'region':
            # For regions, aggregate data from provinces
            provinces_in_region = area.provinces.all()
            
            rma_offices = []
            competitors = []
            banks = []
            
            for province in provinces_in_region:
                rma_offices.extend(RMAOffice.objects.filter(location__within=province.boundary))
                competitors.extend(Competitor.objects.filter(location__within=province.boundary))
                banks.extend(Bank.objects.filter(location__within=province.boundary))
            
            # Remove duplicates
            rma_offices = list(set(rma_offices))
            competitors = list(set(competitors))
            banks = list(set(banks))
            
            total_population = sum(p.population for p in provinces_in_region)
            total_estimated_vehicles = sum(p.estimated_vehicles for p in provinces_in_region)
        else:
            rma_offices = RMAOffice.objects.filter(location__within=area.boundary)
            competitors = Competitor.objects.filter(location__within=area.boundary)
            banks = Bank.objects.filter(location__within=area.boundary)
            total_population = area.population
            total_estimated_vehicles = area.estimated_vehicles
        
        data = {
            'name': area.name,
            'population': total_population,
            'estimated_vehicles': total_estimated_vehicles,
            'competition_count': len(competitors),
            'bank_count': len(banks),
            'rma_offices_count': len(rma_offices),
            'competitors': [{'name': c.agency_name, 'type': c.competitor_type} for c in competitors[:10]],
            'banks': [{'name': b.institution_name} for b in banks[:10]],
            'rma_offices': [{'name': o.name, 'type': o.__class__.__name__} for o in rma_offices],
        }
        
        return JsonResponse(data)
    
    except (Area.DoesNotExist, Commune.DoesNotExist, Province.DoesNotExist, RegionAdministrative.DoesNotExist):
        return JsonResponse({'error': 'Area not found'}, status=404)