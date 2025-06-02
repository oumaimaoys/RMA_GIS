# management/commands/calculate_intensity_stats.py

import math
from django.core.management.base import BaseCommand
from django.utils import timezone
from django.db.models import Avg
from spatial_data.models import Area, Commune, Province, Bank, Competitor, CoverageStats
from django.contrib.gis.measure import D
from django.contrib.gis.geos import Point

BETA = -1.5
COMP_RADIUS_KM = 30
PROJ_SRID = 3857  # Web Mercator projection


def calculate_intensity(area, model_cls):
    """
    Calculate intensity based on distance-weighted nearby points
    """
    try:
        # Get the centroid of the area boundary and transform to projected CRS
        centroid = area.boundary.centroid
        if centroid.srid != PROJ_SRID:
            centroid.transform(PROJ_SRID)
        
        # Debug: Print centroid info
        print(f"Area centroid: {centroid.x}, {centroid.y} (SRID: {centroid.srid})")
        
        # Find nearby points within radius
        # Use the original geometry for spatial query (don't transform the filter geometry)
        nearby = model_cls.objects.filter(
            location__distance_lte=(area.boundary, D(km=COMP_RADIUS_KM))
        )
        
        nearby_count = nearby.count()
        print(f"Found {nearby_count} nearby {model_cls.__name__} objects within {COMP_RADIUS_KM}km")
        
        if nearby_count == 0:
            return 0.0
            
        total_intensity = 0.0
        
        for obj in nearby:
            try:
                # Transform point to same projected CRS as centroid
                point = obj.location
                if point.srid != PROJ_SRID:
                    point = point.transform(PROJ_SRID, clone=True)
                
                # Calculate distance in meters
                distance_m = centroid.distance(point)
                distance_km = distance_m / 1000.0
                
                # Ensure minimum distance to avoid division issues
                distance_km = max(distance_km, 0.1)
                
                # Calculate intensity contribution
                intensity_contribution = math.exp(BETA * distance_km)
                total_intensity += intensity_contribution
                
                print(f"  {model_cls.__name__} at distance {distance_km:.2f}km, contribution: {intensity_contribution:.4f}")
                
            except Exception as e:
                print(f"Error processing {model_cls.__name__} object {obj.id}: {e}")
                continue
        
        print(f"Total {model_cls.__name__} intensity: {total_intensity:.4f}")
        return total_intensity
        
    except Exception as e:
        print(f"Error calculating intensity for area {area.id}: {e}")
        return 0.0


class Command(BaseCommand):
    help = "Calculate and store bank/competition intensities and coverage stats for all Areas."

    def add_arguments(self, parser):
        parser.add_argument(
            '--debug',
            action='store_true',
            help='Enable debug output',
        )
        parser.add_argument(
            '--limit',
            type=int,
            help='Limit processing to N areas for testing',
        )

    def handle(self, *args, **options):
        debug = options.get('debug', False)
        limit = options.get('limit')
        
        # Get areas with proper related data
        areas_query = Area.objects.select_related('commune', 'province').all()
        
        if limit:
            areas_query = areas_query[:limit]
            
        all_areas = list(areas_query)
        
        if not all_areas:
            self.stdout.write(self.style.ERROR("No areas found!"))
            return
            
        print(f"Calculating intensities for {len(all_areas)} areas...")
        
        # Debug: Check if we have any banks/competitors
        bank_count = Bank.objects.count()
        competitor_count = Competitor.objects.count()
        print(f"Total banks in database: {bank_count}")
        print(f"Total competitors in database: {competitor_count}")
        
        if bank_count == 0:
            self.stdout.write(self.style.WARNING("No banks found in database!"))
        if competitor_count == 0:
            self.stdout.write(self.style.WARNING("No competitors found in database!"))

        pops, gaps, vehs, banks, comps = [], [], [], [], []

        for i, area in enumerate(all_areas):
            if debug:
                print(f"\nProcessing area {i+1}/{len(all_areas)}: {area.id}")
                print(f"Area boundary: {area.boundary}")
            
            # Get population data
            pop = max(area.population or 0, 1)
            ins = area.insured_population or 0
            veh = area.estimated_vehicles or 0
            gap = max(pop - ins, 0)

            # Calculate intensities
            if debug:
                print(f"Calculating bank intensity...")
            bank_intensity = calculate_intensity(area, Bank)
            
            if debug:
                print(f"Calculating competitor intensity...")
            comp_intensity = calculate_intensity(area, Competitor)

            # Update area with calculated intensities
            area.bank_intensity = bank_intensity
            area.competition_intensity = comp_intensity
            area.save(update_fields=['bank_intensity', 'competition_intensity'])

            # Collect stats
            pops.append(pop)
            gaps.append(gap)
            vehs.append(veh)
            banks.append(bank_intensity)
            comps.append(comp_intensity)
            
            if debug:
                print(f"Area {area.id}: bank_intensity={bank_intensity:.4f}, comp_intensity={comp_intensity:.4f}")

        def mean_std(arr):
            if not arr:
                return 0.0, 0.0
            avg = sum(arr) / len(arr)
            if len(arr) == 1:
                std = 0.0
            else:
                std = (sum((x - avg) ** 2 for x in arr) / len(arr)) ** 0.5
            return round(avg, 2), round(std, 2)

        # Calculate summary statistics
        pop_mean, pop_std = mean_std(pops)
        gap_mean, gap_std = mean_std(gaps)
        veh_mean, veh_std = mean_std(vehs)
        bank_mean, bank_std = mean_std(banks)
        comp_mean, comp_std = mean_std(comps)

        # Create coverage stats record
        stats = CoverageStats.objects.create(
            calc_date=timezone.now(),
            pop_mean=pop_mean,
            pop_std=pop_std,
            gap_mean=gap_mean,
            gap_std=gap_std,
            veh_mean=veh_mean,
            veh_std=veh_std,
            bank_mean=bank_mean,
            bank_std=bank_std,
            comp_mean=comp_mean,
            comp_std=comp_std
        )

        # Output summary
        self.stdout.write(self.style.SUCCESS(
            f"Successfully calculated intensities and stored CoverageStats ID {stats.id}"
        ))
        
        print(f"\nSummary Statistics:")
        print(f"Bank intensity - Mean: {bank_mean}, Std: {bank_std}")
        print(f"Competition intensity - Mean: {comp_mean}, Std: {comp_std}")
        print(f"Population - Mean: {pop_mean}, Std: {pop_std}")
        print(f"Coverage gap - Mean: {gap_mean}, Std: {gap_std}")
        print(f"Vehicles - Mean: {veh_mean}, Std: {veh_std}")