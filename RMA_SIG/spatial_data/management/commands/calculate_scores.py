from django.core.management.base import BaseCommand
from django.db.models import Avg
from django.utils import timezone
from spatial_data.models import Area, LossRatio, CoverageScore, Commune, Province

class Command(BaseCommand):
    help = "Calculate and save global coverage score for each area with loss ratio consideration"

    def handle(self, *args, **kwargs):
        areas = Area.objects.all()

        # Precompute loss ratios
        loss_ratios = {}
        for area in areas:
            if isinstance(area, Commune):
                qs = LossRatio.objects.filter(commune=area)
            elif isinstance(area, Province):
                qs = LossRatio.objects.filter(province=area)
            else:
                qs = None

            if qs is not None:
                avg_loss = qs.aggregate(avg=Avg('loss_ratio'))['avg'] or 0
            else:
                avg_loss = 0

            loss_ratios[area.id] = avg_loss

        # Gather data for normalization
        pop_list = [a.population for a in areas]
        insured_list = [a.insured_population for a in areas]
        veh_list = [a.estimated_vehicles for a in areas]
        comp_list = [a.competition_count for a in areas]
        bank_list = [a.bank_count for a in areas]
        loss_ratio_list = [loss_ratios[a.id] for a in areas]

        def normalize(series):
            """Normalize to 0-1 range"""
            min_val = min(series)
            max_val = max(series)
            if max_val == min_val:  # Handle case where all values are the same
                return [0.5 for _ in series]  # Return middle value
            return [(v - min_val) / (max_val - min_val) for v in series]

        # Normalize all metrics to 0-1 scale
        pop_norm = normalize(pop_list)
        insured_norm = normalize(insured_list)
        veh_norm = normalize(veh_list)
        comp_norm = normalize(comp_list)
        bank_norm = normalize(bank_list)
        loss_norm = normalize(loss_ratio_list)

        # Debug: Print some normalized values to check variation
        self.stdout.write(f"Sample normalized values:")
        self.stdout.write(f"Population: {pop_norm[:5]}")
        self.stdout.write(f"Competition: {comp_norm[:5]}")
        self.stdout.write(f"Banks: {bank_norm[:5]}")

        for i, area in enumerate(areas):
            # Market Demand Score (0-1 scale)
            demand_score = 0.4 * pop_norm[i] + 0.4 * insured_norm[i] + 0.2 * veh_norm[i]
            
            # Competition Score (inverted - less competition = higher score)
            competition_score = 1 - comp_norm[i]
            
            # Economic Access Score
            economic_score = bank_norm[i]
            
            # Risk Score (inverted - lower loss ratio = higher score)
            risk_score = 1 - loss_norm[i]

            # Calculate weighted final score (0-1 scale)
            final_score_normalized = (
                0.40 * demand_score +      # Market potential
                0.30 * competition_score + # Competition advantage  
                0.20 * economic_score +    # Economic accessibility
                0.10 * risk_score          # Risk consideration
            )

            # Convert to 0-100 scale
            final_score_100pt = round(final_score_normalized * 100, 2)

            # Determine potential level
            if final_score_100pt >= 70:
                potential = 'HIGH'
            elif final_score_100pt >= 40:
                potential = 'MEDIUM'
            else:
                potential = 'LOW'

            # Debug: Print calculation for first few areas
            if i < 3:
                self.stdout.write(f"Area {area.name}:")
                self.stdout.write(f"  Demand: {demand_score:.3f}, Competition: {competition_score:.3f}")
                self.stdout.write(f"  Economic: {economic_score:.3f}, Risk: {risk_score:.3f}")
                self.stdout.write(f"  Final Score: {final_score_100pt}")

            # Save or update CoverageScore
            CoverageScore.objects.update_or_create(
                area=area,
                defaults={
                    'score': final_score_100pt,
                    'potential': potential,
                    'calculation_date': timezone.now(),
                }
            )

        self.stdout.write(self.style.SUCCESS("Coverage scores calculated and saved for all areas."))