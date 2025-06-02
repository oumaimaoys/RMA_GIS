# management/commands/calculate_area_data_and_stats.py

import math
from django.core.management.base import BaseCommand
from django.utils import timezone
from django.db.models import Avg # Keep for potential future use
from statistics import fmean, pstdev # For more robust stats calculation

# Assuming your models are in an app named 'spatial_data' or similar
# Adjust the import path if your app name is different
from spatial_data.models import Area, Commune, Province, Bank, Competitor, CoverageStats, LossRatio
# from spatial_data.models import Area, Commune, Province, Bank, Competitor, CoverageStats, LossRatio # Example

from django.contrib.gis.measure import D
from django.contrib.gis.geos import Point # Keep for potential future use

# Constants for intensity calculation
BETA = -1.5  # Decay factor for distance weighting. Negative means influence decreases with distance.
COMP_RADIUS_KM = 30  # Search radius for banks/competitors.
PROJ_SRID = 3857  # Web Mercator projection SRID for accurate distance calculations.

# --- Helper function for intensity calculation ---
def calculate_gravity_intensity(area_obj, point_model_cls, debug=False):
    """
    Calculates a distance-weighted "gravity" intensity for an area based on nearby points.
    area_obj: The Area instance (Commune or Province).
    point_model_cls: The model class for points (Bank or Competitor).
    """
    try:
        area_centroid = area_obj.boundary.centroid
        if area_centroid.srid != PROJ_SRID:
            area_centroid.transform(PROJ_SRID)
        
        if debug:
            print(f"  Area '{area_obj.name}' (ID: {area_obj.id}) centroid (SRID {PROJ_SRID}): {area_centroid.x:.2f}, {area_centroid.y:.2f}")

        # Find nearby points within the radius using the original geometry SRID (usually 4326)
        # The 'location__distance_lte' query handles SRID transformation implicitly if needed.
        nearby_points = point_model_cls.objects.filter(
            location__dwithin=(area_obj.boundary.centroid, D(km=COMP_RADIUS_KM)) # dwithin is generally more efficient
        )
        # location__distance_lte=(area_obj.boundary.centroid, D(km=COMP_RADIUS_KM)) # Alternative

        nearby_count = nearby_points.count()
        if debug:
            print(f"  Found {nearby_count} nearby {point_model_cls.__name__} objects within {COMP_RADIUS_KM}km.")

        if nearby_count == 0:
            return 0.0
            
        total_intensity_score = 0.0
        
        for point_obj in nearby_points:
            try:
                point_location = point_obj.location.transform(PROJ_SRID, clone=True)
                
                distance_meters = area_centroid.distance(point_location)
                distance_km = distance_meters / 1000.0
                distance_km = max(distance_km, 0.01) # Ensure a minimum distance to avoid math issues (e.g. log(0) or 1/0) or extreme weights.
                                                    # 0.1 km = 100m. If a bank is right on centroid, 0.01km = 10m.
                
                # Intensity contribution using exponential decay
                # Lower distance_km -> higher contribution (closer to exp(0)=1)
                # Higher distance_km -> lower contribution (closer to 0)
                intensity_contribution = math.exp(BETA * distance_km) 
                total_intensity_score += intensity_contribution
                
                if debug:
                    print(f"    {point_model_cls.__name__} (ID: {point_obj.id}) at {distance_km:.2f}km, contribution: {intensity_contribution:.4f}")
            except Exception as e:
                if debug:
                    print(f"    Error processing {point_model_cls.__name__} object {point_obj.id}: {e}")
                continue # Skip this point if there's an issue
        
        if debug:
            print(f"  Total raw {point_model_cls.__name__} intensity for '{area_obj.name}': {total_intensity_score:.4f}")
        return total_intensity_score
        
    except Exception as e:
        if debug:
            print(f"  Error calculating intensity for area '{area_obj.name}' (ID: {area_obj.id}): {e}")
        return 0.0


class Command(BaseCommand):
    help = "Calculates and stores bank/competition intensities for Areas, and updates CoverageStats."

    def add_arguments(self, parser):
        parser.add_argument(
            '--debug',
            action='store_true',
            help='Enable detailed debug output during calculations.',
        )
        parser.add_argument(
            '--limit',
            type=int,
            help='Limit processing to N areas for each type (Commune, Province) for testing.',
        )
        parser.add_argument(
            '--area_ids',
            type=str,
            help='Comma-separated list of Area IDs to process specifically.',
        )
        parser.add_argument(
            '--skip_intensity',
            action='store_true',
            help='Skip recalculating intensities on Area objects (only calculate stats).',
        )

    def handle(self, *args, **options):
        debug = options.get('debug', False)
        limit = options.get('limit')
        area_ids_str = options.get('area_ids')
        skip_intensity_calc = options.get('skip_intensity', False)

        area_models_to_process = [Commune, Province]

        for AreaModel in area_models_to_process:
            area_model_name = AreaModel.__name__
            self.stdout.write(self.style.HTTP_INFO(f"Processing {area_model_name} areas..."))

            areas_query = AreaModel.objects.all()
            if area_ids_str:
                try:
                    selected_ids = [int(id_str.strip()) for id_str in area_ids_str.split(',')]
                    areas_query = areas_query.filter(id__in=selected_ids)
                    self.stdout.write(f"  Processing specific {area_model_name} IDs: {selected_ids}")
                except ValueError:
                    self.stdout.write(self.style.ERROR("Invalid area_ids format. Please use comma-separated integers."))
                    return
            
            if limit:
                areas_query = areas_query[:limit]
            
            all_areas_of_type = list(areas_query)
            
            if not all_areas_of_type:
                self.stdout.write(self.style.WARNING(f"No {area_model_name} areas found matching criteria!"))
                continue # Skip to next AreaModel type
            
            if debug:
                bank_count_db = Bank.objects.count()
                competitor_count_db = Competitor.objects.count()
                self.stdout.write(f"  Total Banks in DB: {bank_count_db}, Competitors: {competitor_count_db}")
                if bank_count_db == 0: self.stdout.write(self.style.WARNING("  No banks in DB! Bank intensity will be 0."))
                if competitor_count_db == 0: self.stdout.write(self.style.WARNING("  No competitors in DB! Competitor intensity will be 0."))

            # Lists to store data for calculating mean/std_dev for CoverageStats
            populations_list = []
            gaps_list = []
            vehicles_list = []
            bank_intensities_list = []
            comp_intensities_list = []
            market_potentials_list = []
            loss_ratios_list = [] # For area-level average loss ratios

            self.stdout.write(f"  Calculating data for {len(all_areas_of_type)} {area_model_name} areas...")
            for i, area in enumerate(all_areas_of_type):
                if debug or (i + 1) % 50 == 0 : # Print progress every 50 areas or if debug
                    self.stdout.write(f"    Processing {area_model_name} {i+1}/{len(all_areas_of_type)}: '{area.name}' (ID: {area.id})")

                # --- 1. Calculate and Update Area-Specific Intensities (if not skipped) ---
                if not skip_intensity_calc:
                    if debug: self.stdout.write(f"      Calculating bank intensity for '{area.name}'...")
                    calculated_bank_intensity = calculate_gravity_intensity(area, Bank, debug=debug)
                    
                    if debug: self.stdout.write(f"      Calculating competitor intensity for '{area.name}'...")
                    calculated_comp_intensity = calculate_gravity_intensity(area, Competitor, debug=debug)

                    area.bank_intensity = calculated_bank_intensity
                    area.competition_intensity = calculated_comp_intensity
                    # The area.save() will also trigger area.update_derived_fields()
                    # which calculates other things like density, simple counts, etc.
                    area.save(update_fields=['bank_intensity', 'competition_intensity']) 
                                            # Add other fields if this command is solely responsible for them
                    if debug:
                        self.stdout.write(f"      Saved {area_model_name} '{area.name}': bank_intensity={area.bank_intensity:.4f}, comp_intensity={area.competition_intensity:.4f}")
                else: # Reload area to get potentially updated values if intensities were calculated by other means
                    area.refresh_from_db()


                # --- 2. Collect Data for Aggregate Statistics ---
                populations_list.append(area.population or 0)
                gap = max(0, (area.population or 0) - (area.insured_population or 0))
                gaps_list.append(gap)
                vehicles_list.append(area.estimated_vehicles or 0)
                
                # Use the (potentially updated) intensities from the area object
                bank_intensities_list.append(area.bank_intensity or 0.0)
                comp_intensities_list.append(area.competition_intensity or 0.0)
                

                market_potential_untapped = 0
                if area.population and area.population > 0:
                    market_potential_untapped = (gap / area.population) * 100
                market_potentials_list.append(market_potential_untapped)

                # Calculate average loss ratio for this area
                # Assumes LossRatio model is linked to Area via 'lossratio_records' related_name
                # Or directly to Commune/Province models. Adjust if your relation is different.
                area_loss_ratios = LossRatio.objects.filter(area=area).aggregate(avg_lr=Avg('loss_ratio'))
                avg_lr_for_area = area_loss_ratios.get('avg_lr')
                if avg_lr_for_area is not None:
                    loss_ratios_list.append(avg_lr_for_area)
                # If LossRatio is linked to Commune/Province specifically:
                # if isinstance(area, Commune):
                #     lr_agg = area.commune_lossratios.aggregate(Avg('loss_ratio')) # if related_name is commune_lossratios
                # elif isinstance(area, Province):
                #     lr_agg = area.province_lossratios.aggregate(Avg('loss_ratio'))
                # ... append lr_agg.get('loss_ratio__avg') to loss_ratios_list ...


            # --- 3. Calculate Mean and Standard Deviation for each metric ---
            def calculate_mean_std(data_list, metric_name="data"):
                clean_data = [x for x in data_list if x is not None]
                if not clean_data:
                    if debug: self.stdout.write(f"    No valid data for {metric_name} to calculate mean/std.")
                    return 0.0, 0.0
                
                # Using statistics module for fmean (handles floats better) and pstdev (population std dev)
                # mean_val = sum(clean_data) / len(clean_data)
                mean_val = fmean(clean_data)
                if len(clean_data) < 2: # pstdev requires at least 2 data points for a meaningful std dev other than 0
                    std_dev_val = 0.0
                else:
                    std_dev_val = pstdev(clean_data) 
                
                if debug:
                    self.stdout.write(f"    Stats for {metric_name}: Count={len(clean_data)}, Mean={mean_val:.2f}, StdDev={std_dev_val:.2f}")
                return round(mean_val, 4), round(std_dev_val, 4) # Increased precision for stats

            pop_mean, pop_std = calculate_mean_std(populations_list, "Population")
            gap_mean, gap_std = calculate_mean_std(gaps_list, "Coverage Gap")
            veh_mean, veh_std = calculate_mean_std(vehicles_list, "Vehicles")
            bank_int_mean, bank_int_std = calculate_mean_std(bank_intensities_list, "Bank Intensity")
            comp_int_mean, comp_int_std = calculate_mean_std(comp_intensities_list, "Competition Intensity")
            market_pot_mean, market_pot_std = calculate_mean_std(market_potentials_list, "Market Potential Untapped")
            lr_mean, lr_std = calculate_mean_std(loss_ratios_list, "Average Loss Ratio")

            # --- 4. Create or Update CoverageStats record for this AreaModel type ---
            # Store one stats record per area_type per calculation run.
            # Or update the latest if you prefer. For simplicity, creating a new one.
            stats_obj = CoverageStats.objects.create(
                area_type=area_model_name,
                calculation_date=timezone.now(), # Time this specific stats calculation was done
                
                pop_mean=pop_mean, pop_std=pop_std,
                gap_mean=gap_mean, gap_std=gap_std,
                veh_mean=veh_mean, veh_std=veh_std,
                
                bank_intensity_mean=bank_int_mean, # Make sure model field name matches
                bank_intensity_std=bank_int_std,   # Make sure model field name matches
                comp_intensity_mean=comp_int_mean, # Make sure model field name matches
                comp_intensity_std=comp_int_std,   # Make sure model field name matches
                
                loss_ratio_mean=lr_mean, loss_ratio_std=lr_std,
                
                market_potential_mean=market_pot_mean, market_potential_std=market_pot_std,
            )

            self.stdout.write(self.style.SUCCESS(
                f"  Successfully processed {area_model_name} areas and stored CoverageStats ID {stats_obj.id}"
            ))
            if debug:
                self.stdout.write(f"    Population: Mean={pop_mean:.2f}, Std={pop_std:.2f}")
                self.stdout.write(f"    Coverage Gap: Mean={gap_mean:.2f}, Std={gap_std:.2f}")
                self.stdout.write(f"    Vehicles: Mean={veh_mean:.2f}, Std={veh_std:.2f}")
                self.stdout.write(f"    Bank Intensity: Mean={bank_int_mean:.4f}, Std={bank_int_std:.4f}")
                self.stdout.write(f"    Competition Intensity: Mean={comp_int_mean:.4f}, Std={comp_int_std:.4f}")
                self.stdout.write(f"    Market Potential Untapped: Mean={market_pot_mean or 0:.2f}%, Std={market_pot_std or 0:.2f}%")
                self.stdout.write(f"    Average Loss Ratio: Mean={lr_mean or 0:.3f}, Std={lr_std or 0:.3f}")

        self.stdout.write(self.style.SUCCESS("All processing finished."))