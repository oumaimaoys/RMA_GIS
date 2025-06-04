import math
import statistics
from typing import List, Dict, Tuple, Optional # Added Optional
from django.core.management.base import BaseCommand
from django.utils import timezone
from django.conf import settings
import requests
import numpy as np
from functools import lru_cache # Keep if used, though not directly in this snippet
# from scipy.stats import rankdata # Keep if used, though not directly in this snippet
from django.db.models import Avg, Subquery, OuterRef, F, Value # For CA subquery
from django.db.models.functions import Coalesce # For CA subquery
from django.contrib.gis.geos import Point

from spatial_data.models import (
    Area, Commune, Province,
    CoverageScore, CoverageStats, LossRatio,
    Competitor, Bank,
    RMAOffice, CA # Your actual models
)

BETA = -1.5
LOSS_MID = 0.65
LOSS_STEEPNESS = 10
COMP_RADIUS_KM = 30
PROJ_SRID = 3857 # Assuming this is your target projection for distance calcs
# ORS_DIRECTIONS_URL = "https://api.openrouteservice.org/v2/directions/driving-car" # Not used in this batch script
# MAX_WORKERS = 4 # Not used in this script
BATCH_SIZE = 100

# --- NEW CONSTANTS FOR RMA OFFICE INFLUENCE ---
RMA_OFFICE_INFLUENCE_RADIUS_KM = 40 # How far an RMA office's performance influences
RMA_OFFICE_BETA_DECAY = -1.0 # Decay factor for RMA office influence
NUM_YEARS_CA_FOR_AVG = 3 # Number of recent years of CA to average

WEIGHTS = {
    'demand': 0.35,
    'competition': 0.15,
    'economic': 0.15,
    'accessibility': 0.15, # Still placeholder for batch, calculated live in simulate_score
    'risk': 0.10,
    'rma_office_performance': 0.10 # New component for RMA office CA influence
}
# Sum: 0.35 + 0.15 + 0.15 + 0.15 + 0.10 + 0.10 = 1.0

# ---- SCORE THRESHOLDS ----
HIGH_THRESHOLD    = 85
MEDIUM_THRESHOLD  = 60
LOW_THRESHOLD     = 30
# Below LOW_THRESHOLD is VERY_LOW (implied)

# Update CoverageScore potential choices to match the script's output if they differ
# Assuming your CoverageScore.potential choices are 'HIGH', 'MEDIUM', 'LOW', 'VERY_LOW'
# If they are 'EXCELLENT', 'GOOD', 'MEDIUM', 'LOW', adjust thresholds or mapping function.
# For now, I will assume 'HIGH', 'MEDIUM', 'LOW', 'VERY_LOW' based on thresholds.
POTENTIAL_LEVEL_MAPPING = {
    'HIGH': 'EXCELLENT', # Or keep 'HIGH' if model matches
    'MEDIUM': 'GOOD',    # Or keep 'MEDIUM'
    'LOW': 'MEDIUM',     # Or keep 'LOW' (this mapping might seem odd, adjust as per your needs)
    'VERY_LOW': 'LOW'    # Or keep 'VERY_LOW'
}
# Let's simplify and assume the script output labels directly match the model or
# a direct mapping is used if CoverageScore.potential has fewer/different levels.
# For this implementation, I will use HIGH, MEDIUM, LOW, VERY_LOW as categories
# and assume your model's choices can accommodate these or you'll adjust `assign_potential_level`.


class OptimizedCoverageCalculator:
    def __init__(self):
        self.competitor_locations: List[Tuple[Point, int]] = []
        self.bank_locations: List[Tuple[Point, int]] = []
        self.rma_office_locations_revenue: List[Tuple[Point, float]] = []

    def preload_spatial_data(self, areas: List[Area]): # areas param not strictly needed here anymore
        competitors = Competitor.objects.filter(location__isnull=False)
        banks = Bank.objects.filter(location__isnull=False)
        
        # Fetch RMAOffices with their average CA of the last N years
        # This is a bit complex to do efficiently in one go for all offices.
        # Strategy:
        # 1. Get all RMAOffices.
        # 2. For each office, get its recent CA.
        # This can be N+1 queries. If performance is an issue for many offices,
        # consider a more complex single query with window functions or raw SQL if necessary.
        
        rma_offices = RMAOffice.objects.filter(location__isnull=False)

        self.competitor_locations = [
            (comp.location.transform(PROJ_SRID, clone=True), comp.id)
            for comp in competitors
        ]
        self.bank_locations = [
            (bank.location.transform(PROJ_SRID, clone=True), bank.id)
            for bank in banks
        ]
        
        self.rma_office_locations_revenue = []
        current_year = timezone.now().year
        years_to_consider = list(range(current_year - NUM_YEARS_CA_FOR_AVG, current_year)) # e.g. [2021, 2022, 2023] for 2024

        for office in rma_offices:
            # Fetch CA for the specific office for the last N years
            recent_ca_values = CA.objects.filter(
                agency=office,
                year__in=years_to_consider
            ).order_by('-year').values_list('CA_value', flat=True)[:NUM_YEARS_CA_FOR_AVG]
            
            # Ensure we have float values and handle cases where CA might be None or 0
            valid_ca_values = [float(ca) for ca in recent_ca_values if ca is not None and float(ca) > 0]

            if valid_ca_values:
                avg_revenue = sum(valid_ca_values) / len(valid_ca_values)
                if avg_revenue > 0:
                    self.rma_office_locations_revenue.append(
                        (office.location.transform(PROJ_SRID, clone=True), avg_revenue)
                    )
            # else: No valid CA data for this office in the specified period.

        print(f"Loaded {len(self.competitor_locations)} competitors, {len(self.bank_locations)} banks.")
        print(f"Loaded {len(self.rma_office_locations_revenue)} RMA offices with recent average CA data.")


    def _calculate_proximity_influence(
        self,
        area_centroid_projected: Point,
        points_data: List[Tuple[Point, float]], # List of (projected_point, attribute_value)
        radius_km: float,
        beta_decay: float
    ) -> float:
        if not points_data:
            return 0.0
        
        influence_sum = 0.0
        for point_projected, attribute_value in points_data:
            try:
                # Ensure point_projected is a Point object
                if not isinstance(point_projected, Point):
                    # print(f"Warning: Skipping invalid point data: {point_projected}") # Debug
                    continue

                d_m = area_centroid_projected.distance(point_projected)
                distance_km = d_m / 1000.0
                
                if distance_km <= radius_km:
                    # Ensure attribute_value is positive
                    if attribute_value > 0: # Important for revenue
                         influence_sum += attribute_value * math.exp(beta_decay * distance_km)
            except Exception as e:
                # print(f"Error in proximity influence calculation: {e}") # Optional debug
                continue
        
        return float(influence_sum)

    def calculate_intensity_unweighted(self, area_centroid_projected: Point, locations: List[Tuple[Point, int]], radius_km: float, beta_decay: float) -> float:
        # For banks, competitors where each point has a weight of 1 for intensity sum
        if not locations:
            return 0.0
        
        adapted_locations_with_unit_weight = [(loc_point, 1.0) for loc_point, _ in locations if isinstance(loc_point, Point)]

        return self._calculate_proximity_influence(
            area_centroid_projected,
            adapted_locations_with_unit_weight,
            radius_km,
            beta_decay
        )

    def batch_calculate_all_influences(self, areas: List[Area]) -> Tuple[List[float], List[float], List[float]]:
        bank_intensities = []
        comp_intensities = []
        rma_office_ca_influences = []
        
        projected_centroids = [
            area.boundary.centroid.transform(PROJ_SRID, clone=True) 
            for area in areas if area.boundary # Ensure boundary exists
        ]
        
        # Handle areas without boundaries if any (though Area model requires it)
        # For simplicity, assume all areas passed have valid boundaries here.

        for centroid_m_proj in projected_centroids: # Iterate over successfully projected centroids
            bank_intensity = self.calculate_intensity_unweighted(
                centroid_m_proj, self.bank_locations, COMP_RADIUS_KM, BETA
            )
            comp_intensity = self.calculate_intensity_unweighted(
                centroid_m_proj, self.competitor_locations, COMP_RADIUS_KM, BETA
            )
            
            rma_influence = self._calculate_proximity_influence(
                centroid_m_proj,
                self.rma_office_locations_revenue, # List of (Point, avg_revenue)
                RMA_OFFICE_INFLUENCE_RADIUS_KM,
                RMA_OFFICE_BETA_DECAY
            )
            
            bank_intensities.append(bank_intensity)
            comp_intensities.append(comp_intensity)
            rma_office_ca_influences.append(rma_influence)
            
        # Ensure output lists match the number of areas successfully processed
        # If some areas had no boundary, the lists might be shorter.
        # This needs careful handling if areas without boundaries are possible and should still get a score.
        # For now, assuming all areas passed to this function will have centroids.
        # If len(projected_centroids) != len(areas), this indicates an issue.
        if len(projected_centroids) != len(areas):
            print(f"Warning: Number of projected centroids ({len(projected_centroids)}) does not match number of areas ({len(areas)}). This might indicate areas without boundaries.")
            # Pad with zeros or handle appropriately. For now, this will lead to shorter lists.
            # A robust way is to map results back to original areas by ID or index.

        return bank_intensities, comp_intensities, rma_office_ca_influences


    def batch_get_loss_ratios(self, areas: List[Area]) -> List[float]:
        area_ids = [area.id for area in areas]
        # Assuming LossRatio.area links directly to the Area PK
        loss_ratios_qs = LossRatio.objects.filter(area_id__in=area_ids)\
                                          .values('area_id')\
                                          .annotate(avg_loss=Avg('loss_ratio'))
        losses_map = {item['area_id']: item['avg_loss'] for item in loss_ratios_qs if item['avg_loss'] is not None}
        
        final_losses = [losses_map.get(area.id, LOSS_MID) for area in areas] # Default to LOSS_MID if no LR found
        # Using LOSS_MID as default is a common practice to avoid extreme risk scores for areas without data.
        return final_losses


    def improved_logistic(self, x: float, mid: float = LOSS_MID, k: float = LOSS_STEEPNESS) -> float:
        # Higher x (loss_ratio) -> lower score (higher risk_factor implicitly) -> lower logistic value
        # This function returns a "goodness" factor (higher is better/lower risk)
        arg = k * (x - mid)
        if arg > 500: return 0.0
        elif arg < -500: return 1.0
        try:
            return 1.0 / (1.0 + math.exp(arg))
        except OverflowError:
            return 0.0 if arg > 0 else 1.0


    @staticmethod
    def validate_area_data(areas: List[Area]):
        # (Keep existing validation as is)
        populations = [a.population or 0 for a in areas]
        insured = [a.insured_population or 0 for a in areas]
        vehicles = [a.estimated_vehicles or 0 for a in areas]
        issues = []
        if not populations:
            issues.append("No areas provided for validation.")
            print("❌ DATA ISSUES FOUND:\n   - No areas provided for validation.")
            return False

        if all(p == 0 for p in populations): issues.append("All populations are zero")
        elif len(set(populations)) == 1 and len(populations) > 1: issues.append(f"All populations are constant: {populations[0]}")
        
        if all(i == 0 for i in insured): issues.append("All insured populations are zero")
        elif len(set(insured)) == 1 and len(insured) > 1: issues.append(f"All insured populations are constant: {insured[0]}")
        
        if all(v == 0 for v in vehicles): issues.append("All vehicle counts are zero")
        elif len(set(vehicles)) == 1 and len(vehicles) > 1: issues.append(f"All vehicle counts are constant: {vehicles[0]}")
        
        if issues:
            print("❌ DATA ISSUES FOUND:")
            for issue in issues: print(f"   - {issue}")
            print("These may cause poor score distribution!")
            return False
        else:
            print("✅ Basic data validation passed")
            return True

    def calculate_composite_scores_improved(
            self,
            areas: List[Area],
            bank_int_list: List[float], # Renamed for clarity
            comp_int_list: List[float], # Renamed
            losses_list: List[float],   # Renamed
            rma_office_influences: List[float]
    ) -> Tuple[List[float], List[Dict[str, float]]]:
        print("=== CALCULATING IMPROVED COMPOSITE SCORES (WITH RMA OFFICE PERFORMANCE) ===")
        if not self.validate_area_data(areas):
            print("Aborting score calculation due to data validation issues.")
            return [], []
        
        num_areas = len(areas)
        if not all(len(lst) == num_areas for lst in [bank_int_list, comp_int_list, losses_list, rma_office_influences]):
            print("Error: Mismatch in lengths of input lists for composite score calculation.")
            print(f"Areas: {num_areas}, Banks: {len(bank_int_list)}, Comps: {len(comp_int_list)}, Losses: {len(losses_list)}, RMA: {len(rma_office_influences)}")
            # This can happen if batch_calculate_all_influences had issues with centroids
            # Or if data fetching itself had issues.
            # Decide on a strategy: error out, or try to align (difficult without IDs here).
            # For now, erroring out is safer.
            return [], []


        pops = np.array([max(a.population or 0, 1) for a in areas], dtype=float)
        ins = np.array([a.insured_population or 0 for a in areas], dtype=float)
        vehs = np.array([a.estimated_vehicles or 0 for a in areas], dtype=float)
        
        gaps = np.maximum(pops - ins, 0)
        penetration_rate = np.where(pops > 0, ins / pops, 0.0)
        market_opportunity_via_penetration = 1.0 - penetration_rate

        def robust_normalize(values, name=""):
            # (Keep existing robust_normalize as is)
            arr = np.array(values, dtype=float)
            if len(arr) == 0: return np.array([])
            if np.all(arr == arr[0]):
                # print(f"  Warning: All '{name}' values are constant ({arr[0]}). Normalizing to 0.")
                # For constant values, a Z-score is technically undefined or 0. Returning 0 is reasonable.
                return np.zeros_like(arr, dtype=float)
            q1, q3 = np.percentile(arr, [25, 75])
            iqr = q3 - q1
            median_val = np.median(arr) 
            if iqr > 1e-9: # Check if IQR is non-zero
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                arr_cleaned = np.where((arr < lower_bound) | (arr > upper_bound), median_val, arr)
            else: # IQR is zero, data is highly concentrated or constant
                arr_cleaned = arr
            
            mean_cleaned = np.mean(arr_cleaned)
            std_cleaned = np.std(arr_cleaned)
            
            if std_cleaned < 1e-9: # Std dev is very small (effectively zero)
                # print(f"  Warning: '{name}' has very low variance after cleaning. Normalizing to 0.")
                return np.zeros_like(arr, dtype=float)
                
            normalized = (arr - mean_cleaned) / std_cleaned # Use original arr for normalization using robust mean/std
            normalized = np.clip(normalized, -3.0, 3.0) # Clip Z-scores
            # print(f"  '{name}': raw range [{np.min(arr):.2f}, {np.max(arr):.2f}], norm range [{np.min(normalized):.2f}, {np.max(normalized):.2f}]")
            return normalized

        pop_norm = robust_normalize(pops, "Population")
        gap_norm = robust_normalize(gaps, "Insurance Gap")
        veh_norm = robust_normalize(vehs, "Vehicles")
        market_opp_norm = robust_normalize(market_opportunity_via_penetration, "Market Opportunity (1-Penetration)")
        
        bank_norm = robust_normalize(bank_int_list, "Bank Proximity Influence")
        comp_norm = robust_normalize(comp_int_list, "Competitor Proximity Influence")
        rma_perf_norm = robust_normalize(rma_office_influences, "RMA Office CA Influence")

        raw_composite_scores = []
        component_parts_list = []

        for i in range(num_areas):
            demand_component = (
                0.3 * pop_norm[i] +
                0.4 * gap_norm[i] +
                0.2 * veh_norm[i] +
                0.1 * market_opp_norm[i]
            )
            competition_component = -comp_norm[i] # Higher comp_norm (more competition) is bad
            economic_component = bank_norm[i]    # Higher bank_norm (more banks nearby) is good
            accessibility_component = 0.0 # Batch placeholder
            
            loss_goodness_factor = self.improved_logistic(losses_list[i]) # losses_list[i] is the loss ratio for area i
            risk_component = loss_goodness_factor # Higher factor is better (lower risk)

            rma_office_performance_component = rma_perf_norm[i] # Higher normalized influence is better

            current_composite = (
                WEIGHTS['demand'] * demand_component +
                WEIGHTS['competition'] * competition_component +
                WEIGHTS['economic'] * economic_component +
                WEIGHTS['accessibility'] * accessibility_component +
                WEIGHTS['risk'] * risk_component +
                WEIGHTS['rma_office_performance'] * rma_office_performance_component
            )
            
            raw_composite_scores.append(current_composite)
            component_parts_list.append({
                'demand_norm_factor': demand_component,
                'competition_norm_factor': competition_component,
                'economic_norm_factor': economic_component,
                'accessibility_norm_factor': accessibility_component,
                'risk_norm_factor': risk_component, # This is the 0-1 goodness factor
                'rma_office_performance_norm_factor': rma_office_performance_component # This is Z-like
            })
        
        if raw_composite_scores:
            mean_raw_score = np.mean(raw_composite_scores)
            min_raw_score = np.min(raw_composite_scores)
            max_raw_score = np.max(raw_composite_scores)
            print(f"Raw composite scores: min={min_raw_score:.3f}, max={max_raw_score:.3f}, mean={mean_raw_score:.3f}")
        else:
            print("No raw composite scores generated.")

        return raw_composite_scores, component_parts_list

    def bulk_update_scores_improved(self, areas: List[Area], raw_scores: List[float], component_parts_list: List[Dict[str, float]]):
        if not raw_scores or not areas or not component_parts_list:
            print("No raw scores, areas, or component parts. Skipping update.")
            return
        if not (len(areas) == len(raw_scores) == len(component_parts_list)):
            print(f"Error: Mismatch in lengths for bulk update. Skipping update.")
            return

        print(f"Updating scores for {len(areas)} areas...")
        raw_array = np.array(raw_scores, dtype=float)
        
        if len(raw_array) == 0: # Should be caught by the first check, but good for robustness
            print("Raw array is empty, cannot scale scores.")
            return

        min_r, max_r = raw_array.min(), raw_array.max()

        if abs(max_r - min_r) < 1e-9: # All raw scores are practically the same
            scaled_final_scores = np.full_like(raw_array, 50.0)
        else:
            # Scale to 10-90 range
            scaled_final_scores = 10.0 + ((raw_array - min_r) / (max_r - min_r)) * 80.0
        scaled_final_scores = np.clip(scaled_final_scores, 0, 100) # Ensure strictly within 0-100

        def assign_potential_level_from_score(score_0_100: float) -> str:
            # This function maps the numeric score to the categorical potential levels.
            # Ensure the returned strings match CoverageScore.potential.choices.
            # Your CoverageScore model has: ('EXCELLENT', 'GOOD', 'MEDIUM', 'LOW')
            # The script uses HIGH, MEDIUM, LOW, VERY_LOW internally.
            # We need to map:
            # HIGH (>=85) -> EXCELLENT
            # MEDIUM (60-84) -> GOOD
            # LOW (30-59) -> MEDIUM
            # VERY_LOW (<30) -> LOW
            if score_0_100 >= HIGH_THRESHOLD:    return 'EXCELLENT' # Was 'HIGH'
            elif score_0_100 >= MEDIUM_THRESHOLD: return 'GOOD'      # Was 'MEDIUM'
            elif score_0_100 >= LOW_THRESHOLD:    return 'MEDIUM'    # Was 'LOW'
            else:                               return 'LOW'       # Was 'VERY_LOW'

        potential_levels_for_model = [assign_potential_level_from_score(s) for s in scaled_final_scores]
        
        # --- Logging distribution based on internal categories before mapping ---
        internal_levels = []
        for s in scaled_final_scores:
            if s >= HIGH_THRESHOLD: internal_levels.append('HIGH')
            elif s >= MEDIUM_THRESHOLD: internal_levels.append('MEDIUM')
            elif s >= LOW_THRESHOLD: internal_levels.append('LOW')
            else: internal_levels.append('VERY_LOW')
        
        counts = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0, 'VERY_LOW': 0}
        for level in internal_levels: counts[level] +=1
        total_areas = len(areas)
        print("Score Distribution (Internal Categories based on Thresholds):")
        for level, count in counts.items():
            percentage = (count / total_areas) * 100 if total_areas > 0 else 0
            print(f"  {level}: {count} areas ({percentage:.1f}%)")
        # --- End Logging ---

        coverage_score_objects_to_create = []
        for i, area in enumerate(areas):
            final_score_val = round(scaled_final_scores[i], 2)
            potential_val_for_model = potential_levels_for_model[i] # Use the mapped value
            
            area_components = component_parts_list[i]

            def scale_z_to_100(z_score: Optional[float]) -> Optional[float]:
                if z_score is None: return None
                return round(max(0, min(100, (z_score + 3) / 6 * 100)), 1)

            demand_s = scale_z_to_100(area_components['demand_norm_factor'])
            competition_s = scale_z_to_100(area_components['competition_norm_factor'])
            economic_s = scale_z_to_100(area_components['economic_norm_factor'])
            accessibility_s = scale_z_to_100(area_components['accessibility_norm_factor']) # Will be ~50
            
            # Risk factor is 0-1 (goodness), scale to 0-100
            risk_norm_factor = area_components.get('risk_norm_factor') # Is 0-1 goodness
            risk_s = round(max(0, min(100, risk_norm_factor * 100)), 1) if risk_norm_factor is not None else None
            
            # RMA office performance factor is Z-like, scale it
            rma_office_perf_s = scale_z_to_100(area_components['rma_office_performance_norm_factor'])

            # Ensure you have a field named 'rma_office_performance_score' in CoverageScore model
            # If not, this line will cause an error.
            coverage_score_objects_to_create.append(CoverageScore(
                area=area,
                score=final_score_val,
                potential=potential_val_for_model, # Use mapped value
                calculation_date=timezone.now(),
                demand_score=demand_s,
                competition_score=competition_s,
                economic_score=economic_s,
                accessibility_score=accessibility_s,
                risk_score=risk_s,
                rma_office_performance_score=rma_office_perf_s, # Field must exist in model
                # travel_time_to_centroid_minutes is not calculated here
            ))
        
        area_ids_to_update = [area.id for area in areas]
        # Delete old scores for these specific areas first
        CoverageScore.objects.filter(area_id__in=area_ids_to_update).delete()
        CoverageScore.objects.bulk_create(coverage_score_objects_to_create, batch_size=min(100, BATCH_SIZE))
        
        print(f"✅ Updated {len(coverage_score_objects_to_create)} coverage scores with RMA office CA influence and component details.")


class Command(BaseCommand):
    help = "Calculates and updates coverage scores for areas, incorporating RMA office CA performance."

    def add_arguments(self, parser):
        parser.add_argument(
            '--batch-size', type=int, default=BATCH_SIZE,
            help='Number of areas to process in each batch for bulk operations (default: %(default)s)'
        )
        parser.add_argument(
            '--area-ids', type=str, default=None,
            help='Comma-separated list of Area IDs to process (e.g., "1,2,3"). Processes all if not specified.'
        )
        parser.add_argument(
            '--area-type', type=str, default='Commune', choices=['Commune', 'Province'],
            help="Specify whether to process 'Commune' or 'Province' areas (default: %(default)s)."
        )

    def handle(self, *args, **kwargs):
        # batch_size_arg = kwargs['batch_size'] # BATCH_SIZE is global, not instance/class var
        area_ids_str = kwargs.get('area_ids')
        area_type_str = kwargs.get('area_type')

        TargetModel: type[Area] # Type hint for clarity
        if area_type_str == 'Commune':
            TargetModel = Commune
        elif area_type_str == 'Province':
            TargetModel = Province
        else:
            self.stdout.write(self.style.ERROR(f"Invalid area type: {area_type_str}."))
            return

        self.stdout.write(f"Attempting to process {TargetModel._meta.verbose_name_plural}.")

        areas_qs = TargetModel.objects.select_related().filter(boundary__isnull=False) # Base queryset

        if area_ids_str:
            try:
                target_area_ids = [int(id_str.strip()) for id_str in area_ids_str.split(',')]
                # Filter the specific model's queryset by IDs.
                # These IDs should be PKs of Commune or Province records, not general Area PKs
                # unless Commune/Province PKs are the same as their Area PKs (which is typical with Django model inheritance).
                areas_qs = areas_qs.filter(id__in=target_area_ids)
                self.stdout.write(f"Processing specific {TargetModel._meta.verbose_name_plural} with IDs: {target_area_ids}")
            except ValueError:
                self.stdout.write(self.style.ERROR("Invalid --area-ids format. Use comma-separated integers."))
                return
        else:
            self.stdout.write(f"Processing all {TargetModel._meta.verbose_name_plural} with boundaries.")

        areas = list(areas_qs) # Execute the queryset

        if not areas:
            self.stdout.write(self.style.WARNING(
                f"No {TargetModel._meta.verbose_name_plural} found to process with the given criteria."
            ))
            return
        
        self.stdout.write(f"Found {len(areas)} {TargetModel._meta.verbose_name_plural} to process.")


        calculator = OptimizedCoverageCalculator()
        
        self.stdout.write("Loading spatial data for banks, competitors, and RMA offices...")
        # Pass the actual list of Commune or Province objects
        calculator.preload_spatial_data(areas) 
        
        self.stdout.write(f"Calculating proximity influences for {len(areas)} {TargetModel._meta.verbose_name_plural}...")
        bank_influences, comp_influences, rma_office_ca_influences = calculator.batch_calculate_all_influences(areas)
        
        if not (len(areas) == len(bank_influences) == len(comp_influences) == len(rma_office_ca_influences)):
            self.stdout.write(self.style.ERROR(
                f"Mismatch in number of areas ({len(areas)}) and calculated influences. Aborting. "
                f"(Banks: {len(bank_influences)}, Comps: {len(comp_influences)}, RMA: {len(rma_office_ca_influences)})"
            ))
            return

        self.stdout.write("Fetching loss ratios...")
        losses = calculator.batch_get_loss_ratios(areas)
        
        self.stdout.write("Computing improved composite scores (with RMA office CA performance)...")
        raw_scores, component_parts = calculator.calculate_composite_scores_improved(
            areas, bank_influences, comp_influences, losses, rma_office_ca_influences
        )
        
        if not raw_scores:
            self.stdout.write(self.style.ERROR("Score calculation aborted or yielded no scores. No scores to update."))
            return

        self.stdout.write(f"Updating coverage scores for {len(areas)} {TargetModel._meta.verbose_name_plural}...")
        calculator.bulk_update_scores_improved(areas, raw_scores, component_parts)
        
        self.stdout.write(self.style.SUCCESS(
            f"✅ Coverage calculation (with RMA office CA performance) completed for {len(areas)} {TargetModel._meta.verbose_name_plural}."
        ))