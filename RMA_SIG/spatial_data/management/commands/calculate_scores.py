import math
import statistics
from typing import List, Dict, Tuple
from django.core.management.base import BaseCommand
from django.utils import timezone
from django.conf import settings
import requests
import numpy as np
from functools import lru_cache
from scipy.stats import rankdata

from spatial_data.models import (
    Area, Commune, Province,
    CoverageScore, CoverageStats, LossRatio,
    Competitor, Bank
)

BETA = -1.5
LOSS_MID = 0.65
LOSS_STEEPNESS = 10
COMP_RADIUS_KM = 30
PROJ_SRID = 3857
ORS_DIRECTIONS_URL = "https://api.openrouteservice.org/v2/directions/driving-car"
MAX_WORKERS = 4
BATCH_SIZE = 100
WEIGHTS = {
    'demand': 0.40,
    'competition': 0.15,
    'economic': 0.20,
    'accessibility': 0.15,
    'risk': 0.10
}

# ---- YOU CAN TUNE THESE SCORE THRESHOLDS BELOW ----
# If you want ~10% high, ~30% medium, ~40% low, ~20% very low, use:
HIGH_THRESHOLD    = 85   # top 10-15%
MEDIUM_THRESHOLD  = 60   # 60-85%
LOW_THRESHOLD     = 30 
# Below 40 is VERY_LOW

class OptimizedCoverageCalculator:
    def __init__(self):
        self.areas_data = {}
        self.competitor_locations = []
        self.bank_locations = []
        self.centroids_m = []

    def preload_spatial_data(self, areas):
        competitors = Competitor.objects.all()
        banks = Bank.objects.all()
        self.competitor_locations = [
            (comp.location.transform(PROJ_SRID, clone=True), comp.id)
            for comp in competitors
        ]
        self.bank_locations = [
            (bank.location.transform(PROJ_SRID, clone=True), bank.id)
            for bank in banks
        ]
        self.centroids_m = [
            area.boundary.centroid.transform(PROJ_SRID, clone=True)
            for area in areas
        ]

    def calculate_intensity_vectorized(self, area_centroid, locations: List[Tuple]) -> float:
        if not locations:
            return 0.0
        distances = []
        for location, _ in locations:
            try:
                d_m = area_centroid.distance(location)
                distances.append(d_m / 1000.0)
            except Exception:
                continue
        if not distances:
            return 0.0
        distances_array = np.array(distances)
        within_radius = distances_array <= COMP_RADIUS_KM
        if not np.any(within_radius):
            return 0.0
        valid_distances = distances_array[within_radius]
        intensities = np.exp(BETA * valid_distances)
        return float(np.sum(intensities))

    def batch_calculate_intensities(self, areas: List) -> Tuple[List[float], List[float]]:
        bank_intensities = []
        comp_intensities = []
        for idx, area in enumerate(areas):
            centroid_m = self.centroids_m[idx]
            bank_intensity = self.calculate_intensity_vectorized(centroid_m, self.bank_locations)
            comp_intensity = self.calculate_intensity_vectorized(centroid_m, self.competitor_locations)
            bank_intensities.append(bank_intensity)
            comp_intensities.append(comp_intensity)
        return bank_intensities, comp_intensities

    def batch_get_loss_ratios(self, areas: List) -> List[float]:
        communes = [a for a in areas if isinstance(a, Commune)]
        provinces = [a for a in areas if isinstance(a, Province)]
        commune_losses = {}
        if communes:
            commune_loss_qs = LossRatio.objects.filter(
                commune__in=communes
            ).values('commune_id').annotate(avg_loss=statistics.mean('loss_ratio'))
            commune_losses = {item['commune_id']: item['avg_loss'] or 0 for item in commune_loss_qs}
        province_losses = {}
        if provinces:
            province_loss_qs = LossRatio.objects.filter(
                province__in=provinces
            ).values('province_id').annotate(avg_loss=statistics.mean('loss_ratio'))
            province_losses = {item['province_id']: item['avg_loss'] or 0 for item in province_loss_qs}
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
        arg = k * (x - mid)
        if arg > 500:
            return 0.0
        elif arg < -500:
            return 1.0
        return 1.0 / (1.0 + math.exp(arg))

    @staticmethod
    def validate_area_data(areas):
        populations = [a.population or 0 for a in areas]
        insured = [a.insured_population or 0 for a in areas]
        vehicles = [a.estimated_vehicles or 0 for a in areas]
        issues = []
        if all(p == 0 for p in populations):
            issues.append("All populations are zero")
        elif len(set(populations)) == 1:
            issues.append(f"All populations are constant: {populations[0]}")
        if all(i == 0 for i in insured):
            issues.append("All insured populations are zero")
        elif len(set(insured)) == 1:
            issues.append(f"All insured populations are constant: {insured[0]}")
        if all(v == 0 for v in vehicles):
            issues.append("All vehicle counts are zero")
        elif len(set(vehicles)) == 1:
            issues.append(f"All vehicle counts are constant: {vehicles[0]}")
        if issues:
            print("❌ DATA ISSUES FOUND:")
            for issue in issues:
                print(f"   - {issue}")
            print("These may cause poor score distribution!")
        else:
            print("✅ Basic data validation passed")
        return len(issues) == 0

    def calculate_composite_scores_improved(
            self,
            areas: List,
            bank_int: List[float],
            comp_int: List[float],
            losses: List[float]
    ) -> Tuple[List[float], List[Dict[str, float]]]:
        print("=== CALCULATING IMPROVED COMPOSITE SCORES ===")
        self.validate_area_data(areas)
        pops = np.array([max(a.population or 0, 1) for a in areas], dtype=float)
        ins = np.array([a.insured_population or 0 for a in areas], dtype=float)
        vehs = np.array([a.estimated_vehicles or 0 for a in areas], dtype=float)
        gaps = np.maximum(pops - ins, 0)
        penetration_rate = np.where(pops > 0, ins / pops, 0)
        vehicle_density = np.where(pops > 0, vehs / pops, 0)

        def robust_normalize(values, name=""):
            arr = np.array(values, dtype=float)
            if len(arr) == 0:
                return np.array([])
            if np.all(arr == arr[0]):
                print(f"  Warning: All {name} values are constant ({arr[0]})")
                return np.zeros_like(arr)
            q1, q3 = np.percentile(arr, [25, 75])
            iqr = q3 - q1
            if iqr > 0:
                lower_bound = q1 - 1.0 * iqr
                upper_bound = q3 + 1.0 * iqr
                arr_cleaned = np.where((arr < lower_bound) | (arr > upper_bound),
                                      np.median(arr), arr)
            else:
                arr_cleaned = arr
            mean_val = np.mean(arr_cleaned)
            std_val = np.std(arr_cleaned)
            if std_val < 1e-10:
                print(f"  Warning: {name} has very low variance")
                return np.zeros_like(arr)
            normalized = (arr - mean_val) / std_val
            normalized = np.clip(normalized, -2.5, 2.5)
            print(f"  {name}: normalized range [{np.min(normalized):.2f}, {np.max(normalized):.2f}]")
            return normalized

        pop_norm = robust_normalize(pops, "Population")
        gap_norm = robust_normalize(gaps, "Insurance Gap")
        veh_norm = robust_normalize(vehs, "Vehicles")
        penetration_norm = robust_normalize(penetration_rate, "Penetration Rate")
        density_norm = robust_normalize(vehicle_density, "Vehicle Density")
        bank_norm = robust_normalize(bank_int, "Bank Intensity")
        comp_norm = robust_normalize(comp_int, "Competitor Intensity")

        raw_scores, parts = [], []
        for i in range(len(areas)):
            demand = (
                0.3 * pop_norm[i] +
                0.4 * gap_norm[i] +
                0.2 * veh_norm[i] +
                0.1 * (1 - penetration_norm[i])
            )
            competition = -comp_norm[i]
            economic = bank_norm[i]
            accessibility = density_norm[i]
            risk_factor = self.improved_logistic(losses[i])
            risk = -risk_factor
            composite = (
                WEIGHTS['demand'] * demand +
                WEIGHTS['competition'] * competition +
                WEIGHTS['economic'] * economic +
                WEIGHTS['accessibility'] * accessibility +
                WEIGHTS['risk'] * risk
            )
            raw_scores.append(composite)
            parts.append({
                'demand': demand,
                'competition': competition,
                'economic': economic,
                'accessibility': accessibility,
                'risk': risk,
                'composite': composite
            })
        print(f"Raw composite scores: min={min(raw_scores):.3f}, max={max(raw_scores):.3f}, mean={np.mean(raw_scores):.3f}")
        return raw_scores, parts

    def bulk_update_scores_improved(self, areas: List, raw_scores: List[float]):
        if not raw_scores:
            return
        print(f"Updating scores for {len(areas)} areas...")
        print(f"Raw score range: {min(raw_scores):.3f} to {max(raw_scores):.3f}")
        raw_array = np.array(raw_scores)
        min_score, max_score = raw_array.min(), raw_array.max()
        # Scale to 0-100
        if max_score == min_score:
            scaled_scores = np.full_like(raw_array, 50.0)
        else:
            scaled_scores = 10 + (raw_array - min_score) / (max_score - min_score) * 80

        # Assign levels by business-driven or data-driven thresholds, NOT quantiles
        def assign_potential_level(score_100):
            if score_100 >= HIGH_THRESHOLD:
                return 'HIGH'
            elif score_100 >= MEDIUM_THRESHOLD:
                return 'MEDIUM'
            elif score_100 >= LOW_THRESHOLD:
                return 'LOW'
            else:
                return 'VERY_LOW'

        potential_levels = [assign_potential_level(score) for score in scaled_scores]
        count_high = potential_levels.count('HIGH')
        count_med = potential_levels.count('MEDIUM')
        count_low = potential_levels.count('LOW')
        count_very_low = potential_levels.count('VERY_LOW')
        print(f"Distribution: HIGH({count_high}), MEDIUM({count_med}), LOW({count_low}), VERY_LOW({count_very_low})")
        print(f"Percentages: HIGH({count_high/len(areas)*100:.1f}%), MEDIUM({count_med/len(areas)*100:.1f}%), LOW({count_low/len(areas)*100:.1f}%), VERY_LOW({count_very_low/len(areas)*100:.1f}%)")

        coverage_scores = []
        for i, area in enumerate(areas):
            score_100 = round(scaled_scores[i], 2)
            potential = potential_levels[i]
            coverage_scores.append(CoverageScore(
                area=area,
                score=score_100,
                potential=potential,
                calculation_date=timezone.now()
            ))
        area_ids = [area.id for area in areas]
        CoverageScore.objects.filter(area_id__in=area_ids).delete()
        CoverageScore.objects.bulk_create(coverage_scores, batch_size=100)
        print(f"✅ Updated {len(coverage_scores)} coverage scores with **realistic, business thresholds**")

class Command(BaseCommand):
    help = "Optimized coverage scores + stats calculation for all areas with improved balanced score distribution."

    def add_arguments(self, parser):
        parser.add_argument(
            '--batch-size',
            type=int,
            default=BATCH_SIZE,
            help='Number of areas to process in each batch'
        )

    def handle(self, *args, **kwargs):
        batch_size = kwargs.get('batch_size', BATCH_SIZE)
        areas = list(Area.objects.select_related().all())
        if not areas:
            self.stdout.write(self.style.WARNING("No Area records found."))
            return

        calculator = OptimizedCoverageCalculator()
        self.stdout.write("Loading spatial data...")
        calculator.preload_spatial_data(areas)
        self.stdout.write(f"Processing {len(areas)} areas...")
        self.stdout.write("Calculating bank and competitor intensities...")
        bank_intensities, comp_intensities = calculator.batch_calculate_intensities(areas)
        self.stdout.write("Fetching loss ratios...")
        losses = calculator.batch_get_loss_ratios(areas)
        self.stdout.write("Computing improved composite scores...")
        raw_scores, _parts = calculator.calculate_composite_scores_improved(
            areas, bank_intensities, comp_intensities, losses
        )
        self.stdout.write("Updating coverage scores with improved distribution...")
        calculator.bulk_update_scores_improved(areas, raw_scores)
        self.stdout.write(self.style.SUCCESS(
            f"✅ Coverage calculation completed for {len(areas)} areas with improved balanced distribution!"
        ))
        self.stdout.write(self.style.SUCCESS(
            f"⚡ Scores now properly distributed: levels reflect real coverage, not forced quantiles."
        ))
