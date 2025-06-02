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
from django.db.models import Avg

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

# ... (Keep all existing imports and the rest of the OptimizedCoverageCalculator class as is) ...
# ... (BETA, LOSS_MID, etc. constants remain the same) ...
# ... (WEIGHTS and THRESHOLDS remain the same) ...

class OptimizedCoverageCalculator:
    def __init__(self):
        self.areas_data = {}
        self.competitor_locations = []
        self.bank_locations = []
        # self.centroids_m = [] # This was defined but not used in the provided snippet for batch_calculate_intensities
                              # It's better to calculate centroids on the fly or pass areas directly if needed
                              # For batch_calculate_intensities, it seems to expect area_centroid passed in.
                              # Let's refine preload_spatial_data if centroids_m is truly needed globally by it.
                              # For now, I'll assume calculate_intensity_vectorized gets the centroid it needs.

    def preload_spatial_data(self, areas: List[Area]): # Added Type Hint for areas
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
        # If self.centroids_m is used by other methods that expect it to be pre-populated:
        # self.centroids_m = [
        #     area.boundary.centroid.transform(PROJ_SRID, clone=True)
        #     for area in areas
        # ]

    def calculate_intensity_vectorized(self, area_centroid_projected, locations: List[Tuple]) -> float:
        # area_centroid_projected is assumed to be already in PROJ_SRID
        if not locations:
            return 0.0
        
        point_coords = np.array([loc[0].coords[0] for loc in locations]) # Assuming loc[0] is a Point object
        centroid_coord = np.array(area_centroid_projected.coords[0])

        # Calculate squared distances first to avoid sqrt until necessary
        # This is a more direct way to get distances with NumPy if locations are Point objects
        # However, Point.distance is usually optimized. Sticking to original loop for now for clarity
        # with direct Point.distance unless performance is a major bottleneck here.

        distances = []
        for location_point, _ in locations: # location_point is already projected
            try:
                # location_point is already a GEOSGeometry object (Point)
                d_m = area_centroid_projected.distance(location_point)
                distances.append(d_m / 1000.0) # convert to km
            except Exception as e:
                # print(f"Debug: Error in distance calc: {e}") # Optional debug
                continue
        
        if not distances:
            return 0.0
        
        distances_array = np.array(distances, dtype=float)
        
        # Filter by radius
        within_radius_mask = distances_array <= COMP_RADIUS_KM
        
        if not np.any(within_radius_mask):
            return 0.0
            
        valid_distances = distances_array[within_radius_mask]
        
        # Calculate intensity contributions
        # Ensure BETA is defined correctly (e.g., -1.5)
        intensities = np.exp(BETA * valid_distances) # BETA should be negative for decay
        
        return float(np.sum(intensities))

    def batch_calculate_intensities(self, areas: List[Area]) -> Tuple[List[float], List[float]]: # Added Type Hint for areas
        bank_intensities = []
        comp_intensities = []
        
        # Pre-transform centroids if not already done
        # Or do it on the fly if memory is a concern for very large number of areas
        projected_centroids = [area.boundary.centroid.transform(PROJ_SRID, clone=True) for area in areas]

        for idx, area in enumerate(areas):
            centroid_m_proj = projected_centroids[idx] # Use the pre-transformed centroid
            
            bank_intensity = self.calculate_intensity_vectorized(centroid_m_proj, self.bank_locations)
            comp_intensity = self.calculate_intensity_vectorized(centroid_m_proj, self.competitor_locations)
            
            bank_intensities.append(bank_intensity)
            comp_intensities.append(comp_intensity)
            
        return bank_intensities, comp_intensities

    def batch_get_loss_ratios(self, areas: List[Area]) -> List[float]: # Added Type Hint for areas
        # Your original logic for batch_get_loss_ratios seems reasonable.
        # Ensure LossRatio model has commune_id and province_id if you query like that,
        # or a direct FK to Area.
        # For simplicity, assuming LossRatio.area FK and related_name 'lossratio_records' or default 'lossratio_set'
        
        area_ids = [area.id for area in areas]
        # Fetch all relevant loss ratios in one go
        loss_ratios_qs = LossRatio.objects.filter(area_id__in=area_ids)\
                                          .values('area_id')\
                                          .annotate(avg_loss=Avg('loss_ratio')) # Use Avg for aggregation

        losses_map = {item['area_id']: item['avg_loss'] or 0.0 for item in loss_ratios_qs}
        
        final_losses = [losses_map.get(area.id, 0.0) for area in areas] # Default to 0.0 if no LR found
        return final_losses


    def improved_logistic(self, x: float, mid: float = LOSS_MID, k: float = LOSS_STEEPNESS) -> float:
        # This logistic function seems to be used for risk.
        # A higher x (loss_ratio) should give a lower score (higher risk_factor).
        # If k is positive (e.g. 10), and x > mid, then exp term is large, result is small (good).
        # So, this function as is, if x is loss_ratio, returns a "goodness" factor.
        # In calculate_composite_scores_improved, risk = -risk_factor, so higher goodness -> lower risk part -> better overall. This is correct.

        arg = k * (x - mid)
        if arg > 500: return 0.0
        elif arg < -500: return 1.0
        try:
            return 1.0 / (1.0 + math.exp(arg))
        except OverflowError:
            return 0.0 if arg > 0 else 1.0


    @staticmethod
    def validate_area_data(areas: List[Area]): # Added Type Hint for areas
        # This validation function is good.
        populations = [a.population or 0 for a in areas]
        insured = [a.insured_population or 0 for a in areas]
        vehicles = [a.estimated_vehicles or 0 for a in areas]
        issues = []
        if not populations: # Handle empty areas list
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
            areas: List[Area], # Added Type Hint for areas
            bank_int: List[float],
            comp_int: List[float],
            losses: List[float]
    ) -> Tuple[List[float], List[Dict[str, float]]]: # Return type hint
        print("=== CALCULATING IMPROVED COMPOSITE SCORES ===")
        if not self.validate_area_data(areas):
            # If validation fails, return empty or raise an error
            # This prevents further processing with bad data that might skew normalization
            print("Aborting score calculation due to data validation issues.")
            return [], []


        pops = np.array([max(a.population or 0, 1) for a in areas], dtype=float)
        ins = np.array([a.insured_population or 0 for a in areas], dtype=float)
        vehs = np.array([a.estimated_vehicles or 0 for a in areas], dtype=float)
        
        gaps = np.maximum(pops - ins, 0)
        # penetration_rate: higher is "less opportunity" for basic coverage
        # So, (1 - penetration_rate) is "opportunity".
        penetration_rate = np.where(pops > 0, ins / pops, 0.0)
        market_opportunity_via_penetration = 1.0 - penetration_rate # Higher means more opportunity

        # vehicle_density: This was used for accessibility.
        # Is it vehicles per population or vehicles per sq_km?
        # Assuming vehicles per population as a proxy for "saturation" or "motorization rate"
        # If it's for physical accessibility, then travel time is a more direct measure.
        # For now, let's assume it's vehicles per capita. Higher might mean more potential auto insurance.
        vehicles_per_capita = np.where(pops > 0, vehs / pops, 0.0)

        def robust_normalize(values, name=""):
            # This robust_normalize function is good.
            arr = np.array(values, dtype=float)
            if len(arr) == 0: return np.array([])
            
            # Handle case where all values are identical
            if np.all(arr == arr[0]):
                print(f"  Warning: All '{name}' values are constant ({arr[0]}). Normalizing to 0.")
                return np.zeros_like(arr, dtype=float)
            
            # Robust scaling using IQR, then Z-score
            q1, q3 = np.percentile(arr, [25, 75])
            iqr = q3 - q1
            
            # Median for clipping (more robust than mean for skewed data before Z-scoring)
            median_val = np.median(arr) 

            if iqr > 1e-9: # Check if IQR is non-zero to avoid issues with constant data segments
                lower_bound = q1 - 1.5 * iqr # Standard 1.5 IQR for outlier detection
                upper_bound = q3 + 1.5 * iqr
                # Clip outliers to the median before Z-scoring
                arr_cleaned = np.where((arr < lower_bound) | (arr > upper_bound), median_val, arr)
            else: # If IQR is zero (or very small), means data is highly concentrated or constant
                arr_cleaned = arr # No IQR-based clipping needed/possible
            
            mean_cleaned = np.mean(arr_cleaned)
            std_cleaned = np.std(arr_cleaned)
            
            if std_cleaned < 1e-9: # If std dev is still very small after cleaning
                print(f"  Warning: '{name}' has very low variance after cleaning. Normalizing to 0.")
                return np.zeros_like(arr, dtype=float)
                
            normalized = (arr - mean_cleaned) / std_cleaned # Use original arr for normalization after finding robust mean/std
            normalized = np.clip(normalized, -3.0, 3.0) # Clip Z-scores to a typical range
            print(f"  '{name}': raw range [{np.min(arr):.2f}, {np.max(arr):.2f}], norm range [{np.min(normalized):.2f}, {np.max(normalized):.2f}]")
            return normalized

        # Normalizing components for demand score
        pop_norm = robust_normalize(pops, "Population")
        gap_norm = robust_normalize(gaps, "Insurance Gap") # Higher is more demand
        veh_norm = robust_normalize(vehs, "Vehicles") # Higher is more demand
        market_opp_norm = robust_normalize(market_opportunity_via_penetration, "Market Opportunity (1-Penetration)") # Higher is more demand

        # Normalizing other factors
        bank_norm = robust_normalize(bank_int, "Bank Intensity") # Higher is better economic
        comp_norm = robust_normalize(comp_int, "Competitor Intensity") # Higher is more competition (bad)
        
        # Accessibility:
        # The original script used vehicle_density for accessibility.
        # This is an indirect measure. Travel time (calculated per request in simulate_score)
        # is more direct for a *specific point*.
        # For this batch command, we don't have a specific point.
        # If 'accessibility' in WEIGHTS refers to general area characteristics:
        # - High population density might imply better infrastructure.
        # - Or, it might mean more congestion.
        # Let's use population density for now if available, or remove accessibility from batch calc.
        # For now, I will remove the 'density_norm' and the 'accessibility' component from the batch calculation
        # as it's not clearly defined from available area-wide data without a target point.
        # The accessibility score should ideally be calculated in the `simulate_score` endpoint
        # based on travel time to the specific point of interest.
        # If you want a general area accessibility proxy, population_density could be one.
        # For now, let's set accessibility_component to 0 here and adjust weights.
        # Alternatively, use vehicle_per_capita if that's the intended proxy.
        
        # Using vehicle_per_capita as a proxy for local economic capacity / motorization
        # This could be part of 'demand' or 'economic'. Let's put it in demand for now.
        vehicle_per_capita_norm = robust_normalize(vehicles_per_capita, "Vehicles Per Capita")


        raw_composite_scores = []
        component_parts_list = []

        for i in range(len(areas)):
            # Demand Component
            # Original: 0.3*pop + 0.4*gap + 0.2*veh + 0.1*(1-pen)
            # Let's keep these weights for the normalized values.
            demand_component = (
                0.3 * pop_norm[i] +
                0.4 * gap_norm[i] +
                0.2 * veh_norm[i] +
                0.1 * market_opp_norm[i] 
                # Optional: Add vehicle_per_capita_norm with a small weight if desired
                # + 0.05 * vehicle_per_capita_norm[i] 
            )
            
            # Competition Component: Higher comp_norm is bad, so we want to penalize.
            competition_component = -comp_norm[i] # Negative because higher intensity is worse.
            
            # Economic Component: Higher bank_norm is good.
            economic_component = bank_norm[i]
            
            # Accessibility Component:
            # As discussed, this is tricky for batch calculation without a target point.
            # Setting to neutral (0) for now. The `simulate_score` API should calculate this properly.
            # If you insist on an area-wide proxy, you could use population_density or similar.
            accessibility_component = 0.0 # Placeholder, ideally calculated in simulate_score endpoint

            # Risk Component:
            # improved_logistic returns higher for "good" (low loss).
            # So, risk_factor is a "goodness" factor.
            # To make it a "risk" (where higher is bad), we can do (1 - goodness_factor)
            # or use it directly if WEIGHTS['risk'] is for "low-risk contribution"
            loss_goodness_factor = self.improved_logistic(losses[i]) # Higher = better (lower effective loss)
            # If WEIGHTS['risk'] is for "contribution of low risk", then risk_component = loss_goodness_factor
            # If WEIGHTS['risk'] is for "penalty of high risk", then risk_component = -loss_goodness_factor or (1-loss_goodness_factor) and then normalize.
            # The original script had risk = -risk_factor; assuming risk_factor was high for high loss.
            # Here, loss_goodness_factor is high for low loss. So, if WEIGHTS['risk'] is for "low risk premium":
            risk_component = loss_goodness_factor # Higher is better (contributes positively)

            # Summing up:
            # demand, economic, risk_component (as defined) are "higher is better"
            # competition_component is "higher is better" (because -comp_norm makes high raw competition a negative value)
            current_composite = (
                WEIGHTS['demand'] * demand_component +
                WEIGHTS['competition'] * competition_component + # comp_component is already signed appropriately
                WEIGHTS['economic'] * economic_component +
                WEIGHTS['accessibility'] * accessibility_component + # Currently 0
                WEIGHTS['risk'] * risk_component
            )
            
            raw_composite_scores.append(current_composite)
            component_parts_list.append({
                # Store the *scaled component contributions* before final weighting for clarity if needed
                # Or store the normalized Z-scores directly
                'demand_score_contribution': WEIGHTS['demand'] * demand_component, # example
                'competition_score_contribution': WEIGHTS['competition'] * competition_component,
                'economic_score_contribution': WEIGHTS['economic'] * economic_component,
                'accessibility_score_contribution': WEIGHTS['accessibility'] * accessibility_component,
                'risk_score_contribution': WEIGHTS['risk'] * risk_component,
                # Storing the individual normalized factors for CoverageScore model:
                'demand_norm_factor': demand_component, # This is the combined normalized demand score
                'competition_norm_factor': competition_component, # This is -comp_norm
                'economic_norm_factor': economic_component, # This is bank_norm
                'accessibility_norm_factor': accessibility_component, # Currently 0
                'risk_norm_factor': risk_component, # This is loss_goodness_factor
                # 'composite_raw': current_composite # Redundant if raw_composite_scores list is used
            })
        
        mean_raw_score = np.mean(raw_composite_scores) if raw_composite_scores else 0
        min_raw_score = min(raw_composite_scores) if raw_composite_scores else 0
        max_raw_score = max(raw_composite_scores) if raw_composite_scores else 0

        print(f"Raw composite scores: min={min_raw_score:.3f}, max={max_raw_score:.3f}, mean={mean_raw_score:.3f}")
        return raw_composite_scores, component_parts_list # Pass component_parts_list

    def bulk_update_scores_improved(self, areas: List[Area], raw_scores: List[float], component_parts_list: List[Dict[str, float]]): # Added component_parts_list
        if not raw_scores or not areas:
            print("No raw scores or areas provided to bulk_update_scores_improved. Skipping update.")
            return
        if len(areas) != len(raw_scores) or len(areas) != len(component_parts_list):
            print(f"Error: Mismatch in lengths of areas ({len(areas)}), raw_scores ({len(raw_scores)}), and component_parts ({len(component_parts_list)}). Skipping update.")
            return

        print(f"Updating scores for {len(areas)} areas...")
        print(f"Raw score range: {min(raw_scores):.3f} to {max(raw_scores):.3f}")
        
        raw_array = np.array(raw_scores, dtype=float)
        min_r, max_r = raw_array.min(), raw_array.max()

        # Scale raw composite scores to 0-100 for the final 'score' field
        # This scaling ensures the final score fits the 0-100 range, using the observed min/max of raw scores.
        # A more robust scaling might use predefined min/max if the theoretical range of raw_scores is known,
        # or use a sigmoid transformation on the raw_scores if they are Z-like.
        # For now, simple min-max scaling to 10-90 range (leaves room at ends).
        if abs(max_r - min_r) < 1e-9: # All raw scores are practically the same
            scaled_final_scores = np.full_like(raw_array, 50.0) # Assign a neutral score
        else:
            # Scale to 0-1, then to 10-90 (or 0-100)
            # score = k1 + (x - min_x) * (k2-k1) / (max_x - min_x)
            # To scale to 0-100:
            # scaled_final_scores = ((raw_array - min_r) / (max_r - min_r)) * 100.0
            # To scale to 10-90 (more centered distribution, avoids extreme 0s and 100s unless truly warranted):
            scaled_final_scores = 10.0 + ((raw_array - min_r) / (max_r - min_r)) * 80.0
        
        scaled_final_scores = np.clip(scaled_final_scores, 0, 100) # Ensure strictly within 0-100

        # Assign potential levels based on the scaled_final_scores (0-100)
        def assign_potential_level(score_0_100): # Changed param name for clarity
            if score_0_100 >= HIGH_THRESHOLD: return 'HIGH'
            elif score_0_100 >= MEDIUM_THRESHOLD: return 'MEDIUM'
            elif score_0_100 >= LOW_THRESHOLD: return 'LOW'
            else: return 'VERY_LOW' # Assuming VERY_LOW is the category for scores < LOW_THRESHOLD

        potential_levels = [assign_potential_level(s) for s in scaled_final_scores]
        
        # --- Logging distribution ---
        counts = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0, 'VERY_LOW': 0}
        for level in potential_levels: counts[level] +=1
        total_areas = len(areas)
        print("Score Distribution based on Thresholds:")
        for level, count in counts.items():
            percentage = (count / total_areas) * 100 if total_areas > 0 else 0
            print(f"  {level}: {count} areas ({percentage:.1f}%)")
        # --- End Logging ---

        coverage_score_objects_to_create = []
        for i, area in enumerate(areas):
            final_score_val = round(scaled_final_scores[i], 2)
            potential_val = potential_levels[i]
            
            # Get component scores for this area
            # These are the normalized factors, not yet scaled to 0-100 themselves.
            # If you want component scores also on 0-100, they need similar scaling.
            # For now, storing the normalized factors as they are.
            # The component_parts_list contains the *weighted* contributions.
            # We should store the *unweighted, normalized* component factors in CoverageScore.
            # So 'demand_norm_factor' from component_parts_list[i] is what we want for `demand_score` field.
            
            area_components = component_parts_list[i]

            # The `CoverageScore` model fields for components (demand_score, etc.)
            # should store scores that are interpretable, ideally on a 0-100 scale.
            # The `demand_component`, `competition_component`, etc. from the loop in
            # `calculate_composite_scores_improved` are normalized (Z-like).
            # We need to scale these to 0-100 for storage in `CoverageScore`.

            # Simple scaling for component Z-like scores (-3 to 3) to 0-100: (Z + 3) / 6 * 100
            # Demand (higher is better)
            demand_s = round(max(0, min(100, (area_components['demand_norm_factor'] + 3) / 6 * 100)), 1)
            # Competition (higher is better, as it's -comp_norm)
            competition_s = round(max(0, min(100, (area_components['competition_norm_factor'] + 3) / 6 * 100)), 1)
            # Economic (higher is better)
            economic_s = round(max(0, min(100, (area_components['economic_norm_factor'] + 3) / 6 * 100)), 1)
            # Accessibility (higher is better, currently 0, so will be 50)
            accessibility_s = round(max(0, min(100, (area_components['accessibility_norm_factor'] + 3) / 6 * 100)), 1)
            # Risk (higher is better, as it's loss_goodness_factor which is 0-1, scale to 0-100)
            # risk_norm_factor is already 0-1 (goodness factor). So just multiply by 100.
            risk_s = round(max(0, min(100, area_components['risk_norm_factor'] * 100)), 1)


            coverage_score_objects_to_create.append(CoverageScore(
                area=area,
                score=final_score_val,
                potential=potential_val,
                calculation_date=timezone.now(), # Ensure this is timezone-aware
                # Assign component scores
                demand_score=demand_s,
                competition_score=competition_s,
                economic_score=economic_s,
                accessibility_score=accessibility_s, # Will be ~50 if factor is 0
                risk_score=risk_s,
                # travel_time_to_centroid_minutes is not calculated in this batch command
            ))
        
        # Efficiently update: delete old scores for these areas, then bulk create new ones.
        area_ids_to_update = [area.id for area in areas]
        # Consider if you want to delete ALL old scores or just for the processed areas.
        # If processing all areas, deleting all CoverageScore records first might be simpler.
        # For now, deleting only for the areas being processed.
        CoverageScore.objects.filter(area_id__in=area_ids_to_update).delete()
        CoverageScore.objects.bulk_create(coverage_score_objects_to_create, batch_size=min(100, BATCH_SIZE)) # Use BATCH_SIZE from args
        
        print(f"✅ Updated {len(coverage_score_objects_to_create)} coverage scores with component details using realistic, business-driven thresholds.")


class Command(BaseCommand):
    help = "Optimized coverage scores + stats calculation for all areas with improved balanced score distribution and component scores."

    def add_arguments(self, parser):
        parser.add_argument(
            '--batch-size',
            type=int,
            default=BATCH_SIZE, # Use the global BATCH_SIZE constant
            help='Number of areas to process in each batch for bulk operations (default: %(default)s)'
        )
        parser.add_argument(
            '--area-ids',
            type=str,
            default=None,
            help='Comma-separated list of Area IDs to process specifically (e.g., "1,2,3"). Processes all if not specified.'
        )

    def handle(self, *args, **kwargs):
        batch_size_arg = kwargs.get('batch_size') # Get from command line args
        area_ids_str = kwargs.get('area_ids')

        if area_ids_str:
            try:
                target_area_ids = [int(id_str.strip()) for id_str in area_ids_str.split(',')]
                areas = list(Area.objects.filter(id__in=target_area_ids).select_related()) # select_related() is good if accessing related models often
                self.stdout.write(f"Processing specific Area IDs: {target_area_ids}")
            except ValueError:
                self.stdout.write(self.style.ERROR("Invalid --area-ids format. Use comma-separated integers."))
                return
        else:
            areas = list(Area.objects.select_related().all())
            self.stdout.write("Processing all Area records.")

        if not areas:
            self.stdout.write(self.style.WARNING("No Area records found to process."))
            return

        calculator = OptimizedCoverageCalculator()
        
        self.stdout.write("Loading spatial data for banks and competitors...")
        calculator.preload_spatial_data(areas) # Pass areas in case it's needed for centroids_m
        
        self.stdout.write(f"Calculating intensities for {len(areas)} areas...")
        bank_intensities, comp_intensities = calculator.batch_calculate_intensities(areas)
        
        self.stdout.write("Fetching loss ratios...")
        losses = calculator.batch_get_loss_ratios(areas)
        
        self.stdout.write("Computing improved composite scores and component factors...")
        raw_scores, component_parts = calculator.calculate_composite_scores_improved( # Capture component_parts
            areas, bank_intensities, comp_intensities, losses
        )
        
        if not raw_scores: # If calculation was aborted due to validation
            self.stdout.write(self.style.ERROR("Score calculation aborted. No scores to update."))
            return

        self.stdout.write("Updating coverage scores with improved distribution and component details...")
        calculator.bulk_update_scores_improved(areas, raw_scores, component_parts) # Pass component_parts
        
        self.stdout.write(self.style.SUCCESS(
            f"✅ Coverage calculation completed for {len(areas)} areas."
        ))