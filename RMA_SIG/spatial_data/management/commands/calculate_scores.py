from django.core.management.base import BaseCommand
from django.db.models import Avg
from django.utils import timezone
from spatial_data.models import Area, LossRatio, CoverageScore

class Command(BaseCommand):
    help = "Calculate and save global coverage score for each area with loss ratio consideration"

    def handle(self, *args, **kwargs):
        areas = Area.objects.all()

        # Precompute loss ratios
        loss_ratios = {
            area.id: LossRatio.objects.filter(area=area).aggregate(avg=Avg('loss_ratio'))['avg'] or 0
            for area in areas
        }

        # Gather data for normalization
        pop_list = [a.population for a in areas]
        insured_list = [a.insured_population for a in areas]
        veh_list = [a.estimated_vehicles for a in areas]
        comp_list = [a.competition_count for a in areas]
        bank_list = [a.bank_count for a in areas]
        loss_ratio_list = [loss_ratios[a.id] for a in areas]

        def normalize(series):
            min_val = min(series)
            max_val = max(series)
            return [(v - min_val) / (max_val - min_val + 1e-9) for v in series]

        # Normalize fields
        pop_norm = normalize(pop_list)
        insured_norm = normalize(insured_list)
        veh_norm = normalize(veh_list)
        comp_norm = normalize(comp_list)
        bank_norm = normalize(bank_list)
        loss_norm = normalize(loss_ratio_list)

        for i, area in enumerate(areas):
            demand_score = 0.4 * pop_norm[i] + 0.4 * insured_norm[i] + 0.2 * veh_norm[i]
            competition_score = 10 * (1 - comp_norm[i])
            economic_score = 10 * bank_norm[i]
            loss_penalty = 10 * (1 - loss_norm[i])

            final_score_10pt = (
                0.35 * demand_score +
                0.25 * area.coverage_score +
                0.2 * competition_score +
                0.1 * economic_score +
                0.1 * loss_penalty
            )

            final_score_100pt = round(final_score_10pt * 10, 2)  # Convert to 0–100 scale

            # Determine potential level
            if final_score_100pt >= 70:
                potential = 'HIGH'
            elif final_score_100pt >= 40:
                potential = 'MEDIUM'
            else:
                potential = 'LOW'

            # Save or update CoverageScore
            CoverageScore.objects.update_or_create(
                area=area,
                calculation_date=timezone.now(),  # grouped by date
                defaults={
                    'score': final_score_100pt,
                    'potential': potential
                }
            )

        self.stdout.write(self.style.SUCCESS("Coverage scores calculated and saved for all areas."))
