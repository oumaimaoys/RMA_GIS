import math
import statistics
from typing import List, Dict, Tuple
from django.core.management.base import BaseCommand
from django.utils import timezone
from django.contrib.gis.measure import D
from django.contrib.gis.db.models.functions import Distance
from django.conf import settings
import requests
from django.db.models import Avg, Prefetch
from django.contrib.gis.geos import Point
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from functools import lru_cache


from spatial_data.models import (
    Area, Commune, Province,
    CoverageScore, CoverageStats, LossRatio,
    Competitor, Bank
)

# ----------------------- parameters ----------------------------------
BETA             = -1.5    # distance-decay exponent
LOSS_MID         = 0.65    # logistic midpoint for loss ratio
LOSS_STEEPNESS   = 10      # steepness of logistic for loss ratio
COMP_RADIUS_KM   = 30      # km radius for banks/competitors
PROJ_SRID        = 3857    # for distance calcs in metres
ORS_DIRECTIONS_URL = "https://api.openrouteservice.org/v2/directions/driving-car"
MAX_WORKERS      = 4       # for parallel processing
BATCH_SIZE       = 100     # for bulk operations
# Updated scoring weights for better balance
WEIGHTS = {
    'demand': 0.30,
    'competition': 0.25,
    'economic': 0.20,
    'accessibility': 0.15,
    'risk': 0.10
}
# ---------------------------------------------------------------------


class OptimizedCoverageCalculator:
    def __init__(self):
        self.areas_data = {}
        self.competitor_locations = []
        self.bank_locations = []
        
    def preload_spatial_data(self):
        """Preload all spatial data to minimize database queries"""
        # Load competitors and banks with their transformed coordinates
        competitors = Competitor.objects.select_related().all()
        banks = Bank.objects.select_related().all()
        
        self.competitor_locations = [
            (comp.location.transform(PROJ_SRID, clone=True), comp.id) 
            for comp in competitors
        ]
        self.bank_locations = [
            (bank.location.transform(PROJ_SRID, clone=True), bank.id) 
            for bank in banks
        ]
    
    @lru_cache(maxsize=1000)
    def _cached_exp_decay(self, distance_km: float) -> float:
        """Cached exponential decay calculation"""
        return math.exp(BETA * distance_km)
    
    def calculate_intensity_vectorized(self, area_centroid, locations: List[Tuple]) -> float:
        """Vectorized intensity calculation using numpy for better performance"""
        if not locations:
            return 0.0
            
        distances = []
        for location, _ in locations:
            try:
                d_m = area_centroid.distance(location)
                distances.append(d_m / 1000.0)  # Convert to km
            except Exception:
                continue
                
        if not distances:
            return 0.0
            
        # Vectorized calculation
        distances_array = np.array(distances)
        # Only consider locations within radius
        within_radius = distances_array <= COMP_RADIUS_KM
        if not np.any(within_radius):
            return 0.0
            
        valid_distances = distances_array[within_radius]
        intensities = np.exp(BETA * valid_distances)
        return float(np.sum(intensities))

    def batch_calculate_intensities(self, areas: List) -> Tuple[List[float], List[float]]:
        """Calculate all intensities in batch for better performance"""
        bank_intensities = []
        comp_intensities = []
        
        for area in areas:
            # Transform centroid once per area
            centroid_m = area.boundary.centroid.transform(PROJ_SRID, clone=True)
            
            # Calculate intensities
            bank_intensity = self.calculate_intensity_vectorized(centroid_m, self.bank_locations)
            comp_intensity = self.calculate_intensity_vectorized(centroid_m, self.competitor_locations)
            
            bank_intensities.append(bank_intensity)
            comp_intensities.append(comp_intensity)
            
        return bank_intensities, comp_intensities

    def batch_get_loss_ratios(self, areas: List) -> List[float]:
        """Batch fetch loss ratios with optimized queries"""
        # Separate areas by type for optimized querying
        communes = [a for a in areas if isinstance(a, Commune)]
        provinces = [a for a in areas if isinstance(a, Province)]
        
        # Bulk fetch loss ratios
        commune_losses = {}
        if communes:
            commune_loss_qs = LossRatio.objects.filter(
                commune__in=communes
            ).values('commune_id').annotate(avg_loss=Avg('loss_ratio'))
            commune_losses = {item['commune_id']: item['avg_loss'] or 0 for item in commune_loss_qs}
        
        province_losses = {}
        if provinces:
            province_loss_qs = LossRatio.objects.filter(
                province__in=provinces
            ).values('province_id').annotate(avg_loss=Avg('loss_ratio'))
            province_losses = {item['province_id']: item['avg_loss'] or 0 for item in province_loss_qs}
        
        # Map back to original area order
        losses = []
        for area in areas:
            if isinstance(area, Commune):
                losses.append(commune_losses.get(area.id, 0))
            elif isinstance(area, Province):
                losses.append(province_losses.get(area.id, 0))
            else:
                losses.append(0)
                
        return losses

    def improved_logistic(self, x: float, mid: float = LOSS_MID, k: float = LOSS_STEEPNESS) -> float:
        """
        Improved logistic function with better numerical stability
        """
        # Clamp extreme values to prevent overflow
        arg = k * (x - mid)
        if arg > 500:  # exp(500) would overflow
            return 0.0
        elif arg < -500:  # exp(-500) would underflow
            return 1.0
        return 1.0 / (1.0 + math.exp(arg))

    def robust_zscore_transform(self, values: List[float]) -> List[float]:
        """
        Robust z-score transformation using median and MAD for outlier resistance
        """
        if not values or len(values) < 2:
            return [0.0] * len(values)
            
        # Use numpy for efficient calculation
        vals_array = np.array(values)
        
        # Use median and MAD for robust standardization
        median = np.median(vals_array)
        mad = np.median(np.abs(vals_array - median))
        
        # If MAD is 0, fall back to standard deviation
        if mad == 0:
            std = np.std(vals_array)
            if std == 0:
                return [0.0] * len(values)
            return ((vals_array - np.mean(vals_array)) / std).tolist()
        
        # Robust z-score = (x - median) / (1.4826 * MAD)
        # 1.4826 is the constant to make MAD consistent with standard deviation for normal distributions
        robust_scale = 1.4826 * mad
        return ((vals_array - median) / robust_scale).tolist()

    def calculate_composite_scores(self, areas: List, bank_intensities: List[float], 
                                 comp_intensities: List[float], losses: List[float]) -> List[float]:
        """
        Improved composite scoring with better feature engineering
        """
        # Extract basic features
        pops = [a.population or 0 for a in areas]
        insured = [a.insured_population or 0 for a in areas]
        vehs = [a.estimated_vehicles or 0 for a in areas]
        
        # Calculate derived features
        gaps = [max(p - i, 0) for p, i in zip(pops, insured)]
        penetration_rates = [i / p if p > 0 else 0 for p, i in zip(pops, insured)]
        vehicle_density = [v / p if p > 0 else 0 for p, v in zip(pops, vehs)]
        
        # Robust standardization
        pop_z = self.robust_zscore_transform(pops)
        gap_z = self.robust_zscore_transform(gaps)
        veh_z = self.robust_zscore_transform(vehs)
        penetration_z = self.robust_zscore_transform(penetration_rates)
        density_z = self.robust_zscore_transform(vehicle_density)
        bank_z = self.robust_zscore_transform(bank_intensities)
        comp_z = self.robust_zscore_transform(comp_intensities)
        
        # Calculate composite scores
        raw_scores = []
        for i in range(len(areas)):
            # Improved demand component (considers multiple factors)
            demand = (0.4 * pop_z[i] + 0.3 * gap_z[i] + 0.2 * veh_z[i] + 
                     0.1 * density_z[i] - 0.1 * penetration_z[i])  # Lower penetration = higher opportunity
            
            # Competition component (negative impact)
            competition = -comp_z[i]
            
            # Economic accessibility
            economic = bank_z[i]
            
            # Risk component (using improved logistic)
            risk = self.improved_logistic(losses[i])
            
            # Placeholder for accessibility (to be calculated separately)
            accessibility = 0.0
            
            # Weighted composite score
            raw_score = (WEIGHTS['demand'] * demand + 
                        WEIGHTS['competition'] * competition +
                        WEIGHTS['economic'] * economic + 
                        WEIGHTS['accessibility'] * accessibility + 
                        WEIGHTS['risk'] * risk)
            
            raw_scores.append(raw_score)
            
        return raw_scores

    def bulk_update_scores(self, areas: List, raw_scores: List[float]):
        """Bulk update coverage scores for better database performance"""
        # Rescale to 0-100
        if not raw_scores:
            return
            
        s_min, s_max = min(raw_scores), max(raw_scores)
        span = s_max - s_min if s_max != s_min else 1.0
        
        # Use update_or_create in batches for compatibility
        batch_size = 100
        for i in range(0, len(areas), batch_size):
            batch_areas = areas[i:i + batch_size]
            batch_scores = raw_scores[i:i + batch_size]
            
            # Process batch
            for j, area in enumerate(batch_areas):
                score_100 = round((batch_scores[j] - s_min) / span * 100, 2)
                
                # Improved potential classification with more nuanced thresholds
                if score_100 >= 75:
                    potential = 'HIGH'
                elif score_100 >= 50:
                    potential = 'MEDIUM'
                elif score_100 >= 25:
                    potential = 'LOW'
                else:
                    potential = 'VERY_LOW'
                
                CoverageScore.objects.update_or_create(
                    area=area,
                    defaults={
                        'score': score_100,
                        'potential': potential,
                        'calculation_date': timezone.now()
                    }
                )

    def bulk_update_scores_alternative(self, areas: List, raw_scores: List[float]):
        """Alternative bulk update method using raw SQL for better performance"""
        # Rescale to 0-100
        if not raw_scores:
            return
            
        s_min, s_max = min(raw_scores), max(raw_scores)
        span = s_max - s_min if s_max != s_min else 1.0
        
        # Delete existing scores for these areas first
        area_ids = [area.id for area in areas]
        CoverageScore.objects.filter(area_id__in=area_ids).delete()
        
        # Prepare bulk create data
        coverage_scores = []
        for i, area in enumerate(areas):
            score_100 = round((raw_scores[i] - s_min) / span * 100, 2)
            
            # Improved potential classification with more nuanced thresholds
            if score_100 >= 75:
                potential = 'HIGH'
            elif score_100 >= 50:
                potential = 'MEDIUM'
            elif score_100 >= 25:
                potential = 'LOW'
            else:
                potential = 'VERY_LOW'
            
            coverage_scores.append(CoverageScore(
                area=area,
                score=score_100,
                potential=potential,
                calculation_date=timezone.now()
            ))
        
        # Bulk create new scores
        CoverageScore.objects.bulk_create(coverage_scores, batch_size=100)

    @lru_cache(maxsize=100)
    def get_drive_time_cached(self, lat_from: float, lon_from: float, 
                            lat_to: float, lon_to: float) -> float:
        """Cached version of drive time calculation"""
        return self._get_drive_time_uncached(lat_from, lon_from, lat_to, lon_to)
    
    def _get_drive_time_uncached(self, lat_from: float, lon_from: float, 
                               lat_to: float, lon_to: float) -> float:
        """
        Returns driving time in minutes between two WGS84 points via OpenRouteService.
        """
        try:
            api_key = settings.ORS_API_KEY
            resp = requests.post(
                ORS_DIRECTIONS_URL,
                headers={"Authorization": api_key, "Content-Type": "application/json"},
                json={"coordinates": [[lon_from, lat_from], [lon_to, lat_to]]},
                timeout=10  # Increased timeout
            )
            resp.raise_for_status()
            data = resp.json()

            # Try different response formats
            # GeoJSON "features" format
            feats = data.get("features")
            if isinstance(feats, list) and feats:
                segs = feats[0].get("properties", {}).get("segments", [])
                if segs and segs[0].get("duration") is not None:
                    return segs[0]["duration"] / 60.0
            
            # ORS v2 "routes" format
            routes = data.get("routes")
            if isinstance(routes, list) and routes:
                segs = routes[0].get("segments", [])
                if segs and segs[0].get("duration") is not None:
                    return segs[0]["duration"] / 60.0

            return 0.0  # Return 0 instead of raising error for batch processing
        except Exception:
            return 0.0  # Return 0 for failed requests


def calculate_accessibility_parallel(areas: List, calculator: OptimizedCoverageCalculator) -> List[float]:
    """Calculate accessibility times in parallel for better performance"""
    def get_area_accessibility(area):
        try:
            cent = area.boundary.centroid.clone()
            cent.transform(4326)
            # For now, return 0 as in original - this can be expanded for real accessibility calculation
            return 0.0
        except Exception:
            return 0.0
    
    # Use ThreadPoolExecutor for I/O bound operations
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(get_area_accessibility, area): area for area in areas}
        access_times = []
        
        for future in as_completed(futures):
            try:
                result = future.result(timeout=30)
                access_times.append(result)
            except Exception:
                access_times.append(0.0)
    
    return access_times


class Command(BaseCommand):
    help = "Optimized coverage scores + stats calculation for all areas."

    def add_arguments(self, parser):
        parser.add_argument(
            '--batch-size',
            type=int,
            default=BATCH_SIZE,
            help='Number of areas to process in each batch'
        )
        parser.add_argument(
            '--parallel',
            action='store_true',
            help='Enable parallel processing for intensive calculations'
        )

    def handle(self, *args, **kwargs):
        batch_size = kwargs.get('batch_size', BATCH_SIZE)
        use_parallel = kwargs.get('parallel', False)
        
        # Initialize calculator
        calculator = OptimizedCoverageCalculator()
        
        self.stdout.write("Loading spatial data...")
        calculator.preload_spatial_data()
        
        # Optimized area loading with select_related/prefetch_related
        areas = list(Area.objects.select_related().all())
        if not areas:
            self.stdout.write(self.style.WARNING("No Area records found."))
            return

        self.stdout.write(f"Processing {len(areas)} areas...")

        # 1️⃣ Batch calculate intensities (major optimization)
        self.stdout.write("Calculating bank and competitor intensities...")
        bank_intensities, comp_intensities = calculator.batch_calculate_intensities(areas)

        # 2️⃣ Batch get loss ratios (optimized queries)
        self.stdout.write("Fetching loss ratios...")
        losses = calculator.batch_get_loss_ratios(areas)

        # 3️⃣ Calculate composite scores with improved methodology
        self.stdout.write("Computing composite scores...")
        raw_scores = calculator.calculate_composite_scores(areas, bank_intensities, comp_intensities, losses)

        # 4️⃣ Bulk update scores
        self.stdout.write("Updating coverage scores...")
        # Choose one of these methods:
        calculator.bulk_update_scores(areas, raw_scores)  # Compatible but slower
        # calculator.bulk_update_scores_alternative(areas, raw_scores)  # Faster alternative

        # 5️⃣ Calculate accessibility (can be parallelized if needed)
        self.stdout.write("Calculating accessibility metrics...")
        if use_parallel:
            access_times = calculate_accessibility_parallel(areas, calculator)
        else:
            access_times = [0.0] * len(areas)  # Placeholder as in original

        # 6️⃣ Persist global stats with improved statistics
        self.stdout.write("Updating global statistics...")
        if raw_scores:
            s_min, s_max = min(raw_scores), max(raw_scores)
            raw_mean = statistics.mean(raw_scores)
            raw_std = statistics.pstdev(raw_scores)
            
            # Additional robust statistics
            raw_median = statistics.median(raw_scores)
            
            # Calculate other statistics
            pops = [a.population or 0 for a in areas]
            gaps = [max((a.population or 0) - (a.insured_population or 0), 0) for a in areas]
            vehs = [a.estimated_vehicles or 0 for a in areas]
            
            access_mean = statistics.mean(access_times) if access_times else 0
            access_std = statistics.pstdev(access_times) if access_times else 0
            
            CoverageStats.objects.update_or_create(
                pk=1,
                defaults={
                    'raw_min': s_min,
                    'raw_max': s_max,
                    'raw_mean': raw_mean,
                    'raw_stddev': raw_std,
                    'pop_mean': statistics.mean(pops),
                    'pop_std': statistics.pstdev(pops) or 1.0,
                    'gap_mean': statistics.mean(gaps),
                    'gap_std': statistics.pstdev(gaps) or 1.0,
                    'veh_mean': statistics.mean(vehs),
                    'veh_std': statistics.pstdev(vehs) or 1.0,
                    'bank_mean': statistics.mean(bank_intensities),
                    'bank_std': statistics.pstdev(bank_intensities) or 1.0,
                    'comp_mean': statistics.mean(comp_intensities),
                    'comp_std': statistics.pstdev(comp_intensities) or 1.0,
                    'access_mean': access_mean,
                    'access_std': access_std,
                    'calc_date': timezone.now(),
                }
            )

        self.stdout.write(self.style.SUCCESS(
            f"✅ Optimized coverage calculation completed for {len(areas)} areas!"
        ))
        self.stdout.write(self.style.SUCCESS(
            f"⚡ Performance improvements: batch processing, vectorized calculations, "
            f"robust statistics, and improved scoring methodology applied."
        ))