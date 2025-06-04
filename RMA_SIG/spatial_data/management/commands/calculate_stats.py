# management/commands/calculate_area_data_and_stats.py

import math
from django.core.management.base import BaseCommand
from django.utils import timezone
from django.db.models import Avg
from statistics import fmean, pstdev
from typing import Optional, List, Type, Tuple, Union

from spatial_data.models import (
    Area, Commune, Province, Bank, Competitor, CoverageStats, LossRatio,
    RMAOffice, CA
)

from django.contrib.gis.measure import D
from django.contrib.gis.geos import Point

BETA = -1.5
COMP_RADIUS_KM = 30
PROJ_SRID = 3857

RMA_OFFICE_INFLUENCE_RADIUS_KM = 40
RMA_OFFICE_BETA_DECAY = -1.0
NUM_YEARS_CA_FOR_AVG = 3


def calculate_gravity_intensity(area_obj: Area, point_model_cls: Union[Type[Bank], Type[Competitor]], debug=False) -> float:
    try:
        area_centroid = area_obj.boundary.centroid
        if area_centroid.srid != PROJ_SRID:
            area_centroid.transform(PROJ_SRID)
        
        nearby_points = point_model_cls.objects.filter(
            location__dwithin=(area_obj.boundary.centroid, D(km=COMP_RADIUS_KM))
        )
        
        if not nearby_points.exists():
            return 0.0
            
        total_intensity_score = 0.0
        for point_obj in nearby_points:
            try:
                point_location = point_obj.location.transform(PROJ_SRID, clone=True)
                distance_meters = area_centroid.distance(point_location)
                distance_km = max(distance_meters / 1000.0, 0.01)
                intensity_contribution = math.exp(BETA * distance_km) 
                total_intensity_score += intensity_contribution
            except Exception:
                continue
        return total_intensity_score
    except Exception:
        return 0.0


def get_rma_office_avg_ca_for_stats(office: RMAOffice, num_years: int) -> Optional[float]:
    current_year = timezone.now().year
    years_to_consider = list(range(current_year - num_years, current_year))

    recent_ca_values = CA.objects.filter(
        agency=office,
        year__in=years_to_consider,
        CA_value__gt=0
    ).order_by('-year').values_list('CA_value', flat=True)[:num_years]

    valid_ca_values = [float(ca) for ca in recent_ca_values if ca is not None]

    if valid_ca_values:
        return sum(valid_ca_values) / len(valid_ca_values)
    return None


def calculate_rma_office_influence(area_obj: Area, debug=False) -> float:
    try:
        area_centroid_orig_srid = area_obj.boundary.centroid
        area_centroid_proj = area_centroid_orig_srid.transform(PROJ_SRID, clone=True)
        
        nearby_rma_offices = RMAOffice.objects.filter(
            location__isnull=False,
            location__dwithin=(area_centroid_orig_srid, D(km=RMA_OFFICE_INFLUENCE_RADIUS_KM))
        )
        
        if not nearby_rma_offices.exists():
            return 0.0
            
        total_rma_influence_score = 0.0
        
        for rma_office in nearby_rma_offices:
            avg_ca = get_rma_office_avg_ca_for_stats(rma_office, NUM_YEARS_CA_FOR_AVG)
            if avg_ca and avg_ca > 0 and rma_office.location:
                try:
                    office_location_proj = rma_office.location.transform(PROJ_SRID, clone=True)
                    distance_meters = area_centroid_proj.distance(office_location_proj)
                    distance_km = max(distance_meters / 1000.0, 0.01)
                    influence_contribution = avg_ca * math.exp(RMA_OFFICE_BETA_DECAY * distance_km)
                    total_rma_influence_score += influence_contribution
                except Exception:
                    continue
        return total_rma_influence_score
    except Exception:
        return 0.0

def _calculate_mean_std(data_list: List[Optional[float]], metric_name: str = "data", is_debug: bool = False, stdout_writer=None) -> Tuple[float, float]:
    clean_data = [x for x in data_list if x is not None]
    if not clean_data:
        if is_debug and stdout_writer: stdout_writer.write(f"    No valid data for {metric_name} to calculate mean/std.")
        return 0.0, 0.0
    mean_val = fmean(clean_data)
    std_dev_val = pstdev(clean_data) if len(clean_data) >= 2 else 0.0
    if is_debug and stdout_writer:
        stdout_writer.write(f"    Stats for {metric_name}: Count={len(clean_data)}, Mean={mean_val:.6f}, StdDev={std_dev_val:.6f}")
    return round(mean_val, 6), round(std_dev_val, 6)


class Command(BaseCommand):
    help = "Calculates area intensities, RMA influence, and updates CoverageStats."

    def add_arguments(self, parser):
        parser.add_argument('--debug', action='store_true', help='Enable detailed debug output.')
        parser.add_argument('--limit', type=int, help='Limit processing to N areas for each type.')
        parser.add_argument('--area_ids', type=str, help='Comma-separated list of Area IDs to process.')
        parser.add_argument('--skip_intensity', action='store_true', help='Skip recalculating intensities/RMA influence on Area objects.')
        parser.add_argument('--area_type', type=str, choices=['Commune', 'Province', 'Both'], default='Both',
                            help="Specify area type to process: 'Commune', 'Province', or 'Both'. Default is 'Both'.")

    def handle(self, *args, **options):
        debug = options.get('debug', False)
        limit = options.get('limit')
        area_ids_str = options.get('area_ids')
        skip_recalculation = options.get('skip_intensity', False)
        area_type_option = options.get('area_type')

        area_models_to_process: List[Type[Area]] = []
        if area_type_option == 'Commune':
            area_models_to_process.append(Commune)
        elif area_type_option == 'Province':
            area_models_to_process.append(Province)
        elif area_type_option == 'Both':
            area_models_to_process.extend([Commune, Province])
        
        if not area_models_to_process:
            self.stdout.write(self.style.ERROR("No valid area type selected for processing."))
            return

        for AreaModel in area_models_to_process:
            area_model_name = AreaModel.__name__
            self.stdout.write(self.style.HTTP_INFO(f"Processing {area_model_name} areas..."))

            areas_query = AreaModel.objects.filter(boundary__isnull=False)
            if area_ids_str:
                try:
                    selected_ids = [int(id_str.strip()) for id_str in area_ids_str.split(',')]
                    areas_query = areas_query.filter(id__in=selected_ids)
                except ValueError:
                    self.stdout.write(self.style.ERROR("Invalid area_ids format."))
                    return
            
            if limit:
                areas_query = areas_query[:limit]
            
            all_areas_of_type = list(areas_query)
            
            if not all_areas_of_type:
                self.stdout.write(self.style.WARNING(f"No {area_model_name} areas found matching criteria!"))
                continue
            
            populations_list, gaps_list, vehicles_list = [], [], []
            bank_intensities_list, comp_intensities_list = [], []
            market_potentials_list, loss_ratios_list = [], []
            rma_office_influences_list = []

            self.stdout.write(f"  Calculating data for {len(all_areas_of_type)} {area_model_name} areas...")
            for i, area in enumerate(all_areas_of_type):
                if debug or (i + 1) % 25 == 0 :
                    self.stdout.write(f"    Processing {area_model_name} {i+1}/{len(all_areas_of_type)}: '{area.name}' (ID: {area.id})")

                if not skip_recalculation:
                    area.bank_intensity = calculate_gravity_intensity(area, Bank, debug=debug)
                    area.competition_intensity = calculate_gravity_intensity(area, Competitor, debug=debug)
                    area.save(update_fields=['bank_intensity', 'competition_intensity'])
                else:
                    area.refresh_from_db(fields=['bank_intensity', 'competition_intensity', 'population', 'insured_population', 'estimated_vehicles'])

                current_rma_office_influence = calculate_rma_office_influence(area, debug=debug)
                rma_office_influences_list.append(current_rma_office_influence)

                populations_list.append(area.population or 0)
                gap = max(0, (area.population or 0) - (area.insured_population or 0))
                gaps_list.append(gap)
                vehicles_list.append(area.estimated_vehicles or 0)
                
                bank_intensities_list.append(area.bank_intensity or 0.0)
                comp_intensities_list.append(area.competition_intensity or 0.0)
                
                market_potential_untapped = (gap / (area.population or 1)) * 100
                market_potentials_list.append(market_potential_untapped)

                area_loss_ratios = LossRatio.objects.filter(area_id=area.id).aggregate(avg_lr=Avg('loss_ratio'))
                avg_lr_for_area = area_loss_ratios.get('avg_lr')
                if avg_lr_for_area is not None:
                    loss_ratios_list.append(avg_lr_for_area)

            pop_mean, pop_std = _calculate_mean_std(populations_list, "Population", debug, self.stdout)
            gap_mean, gap_std = _calculate_mean_std(gaps_list, "Coverage Gap", debug, self.stdout)
            veh_mean, veh_std = _calculate_mean_std(vehicles_list, "Vehicles", debug, self.stdout)
            bank_int_mean, bank_int_std = _calculate_mean_std(bank_intensities_list, "Bank Intensity", debug, self.stdout)
            comp_int_mean, comp_int_std = _calculate_mean_std(comp_intensities_list, "Comp Intensity", debug, self.stdout)
            market_pot_mean, market_pot_std = _calculate_mean_std(market_potentials_list, "Market Potential", debug, self.stdout)
            lr_mean, lr_std = _calculate_mean_std(loss_ratios_list, "Avg Loss Ratio", debug, self.stdout)
            rma_inf_mean, rma_inf_std = _calculate_mean_std(rma_office_influences_list, "RMA Office Influence", debug, self.stdout)

            stats_obj, created = CoverageStats.objects.update_or_create(
                area_type=area_model_name,
                defaults={
                    'calculation_date': timezone.now(),
                    'pop_mean': pop_mean, 'pop_std': pop_std,
                    'gap_mean': gap_mean, 'gap_std': gap_std,
                    'veh_mean': veh_mean, 'veh_std': veh_std,
                    'bank_intensity_mean': bank_int_mean, 'bank_intensity_std': bank_int_std,
                    'comp_intensity_mean': comp_int_mean, 'comp_intensity_std': comp_int_std,
                    'loss_ratio_mean': lr_mean, 'loss_ratio_std': lr_std,
                    'market_potential_mean': market_pot_mean, 'market_potential_std': market_pot_std,
                    'rma_office_influence_mean': rma_inf_mean,
                    'rma_office_influence_std': rma_inf_std,
                }
            )
            action_str = "Created" if created else "Updated"
            self.stdout.write(self.style.SUCCESS(
                f"  Successfully {action_str} CoverageStats for {area_model_name} (ID {stats_obj.id})"
            ))
            if debug:
                 self.stdout.write(f"    RMA Office Influence: Mean={rma_inf_mean:.4f}, Std={rma_inf_std:.4f}")


        self.stdout.write(self.style.SUCCESS("All processing finished."))