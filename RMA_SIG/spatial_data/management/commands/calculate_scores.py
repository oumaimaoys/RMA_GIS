# spatial_data/management/commands/calculate_scores.py
"""
Compute an AHP-weighted coverage score for every Area (Commune / Province).

Score =
    0.35·Demand
  + 0.20·Competition          (distance-decay over Competitor locations)
  + 0.15·Economic             (bank density)
  + 0.10·Accessibility        (stub = 0 for now)
  + 0.20·Risk                 (logistic of loss ratio)

The raw score is min-max rescaled to 0-100, then bucketed into
HIGH / MEDIUM / LOW potential.

Works with: Area.boundary (MultiPolygon), Competitor.location, Bank.location
"""

import math
import statistics as stats
from typing import List

from django.core.management.base import BaseCommand
from django.db.models import Avg
from django.utils import timezone
from django.contrib.gis.measure import D
from spatial_data.models import (
    Area, Commune, Province,
    CoverageScore, LossRatio,
    Competitor, Bank
)

# ----------------------- parameters ----------------------------------
BETA             = -1.5    # distance-decay exponent for competition
LOSS_MID         = 0.65    # logistic midpoint for loss ratio
LOSS_STEEPNESS   = 10      # steeper drop above LOSS_MID
COMP_RADIUS_KM   = 30      # search radius around area centroid
PROJ_SRID        = 3857    # Web-Mercator (metres) for distance calcs
# ---------------------------------------------------------------------


# ----------------------- helpers -------------------------------------

def zscores(values: List[float]) -> List[float]:
    μ = stats.fmean(values)
    σ = stats.pstdev(values) or 1.0
    return [(v - μ) / σ for v in values]


def logistic(x: float, mid=LOSS_MID, k=LOSS_STEEPNESS) -> float:
    """S-curve in [0, 1]; higher x above *mid* → lower output."""
    return 1.0 / (1.0 + math.exp(k * (x - mid)))


def competition_intensity(anchor: Area,
                          radius_km: float = COMP_RADIUS_KM,
                          beta: float = BETA) -> float:
    """
    Sum exp(beta·d_km) for all Competitor.points within *radius_km*
    of the area’s centroid. Returns 0 if boundary is missing.
    """
    if not anchor.boundary:
        return 0.0

    # centroid as Point in metres
    centroid = anchor.boundary.centroid.transform(PROJ_SRID, clone=True)

    nearby = (
        Competitor.objects
        .filter(location__distance_lte=(anchor.boundary, D(km=radius_km)))
        .only('location')
    )

    intensity = 0.0
    for comp in nearby:
        loc = comp.location
        if not loc:
            continue
        d_m  = centroid.distance(loc.transform(PROJ_SRID, clone=True))
        d_km = d_m / 1000.0
        intensity += math.exp(beta * d_km)

    return intensity


# ----------------------- command -------------------------------------

class Command(BaseCommand):
    help = "Recalculate coverage scores for all areas."

    def handle(self, *args, **kwargs):
        areas = list(
            Area.objects.all()
            .select_related()     # pulls Commune / Province attrs
        )

        if not areas:
            self.stdout.write(self.style.WARNING("No Area records found."))
            return

        # ---- collect raw variables ----------------------------------
        pop            = [a.population or 0               for a in areas]
        insured        = [a.insured_population or 0       for a in areas]
        demand_gap     = [max(p - i, 0)                   for p, i in zip(pop, insured)]
        veh            = [a.estimated_vehicles or 0       for a in areas]
        bank_density   = [
            (a.bank_count or 0) / (p or 1) * 1000.0       # banks per 1 000 inhabitants
            for a, p in zip(areas, pop)
        ]

        # loss ratio per area
        loss_ratio = []
        for a in areas:
            qs = (
                LossRatio.objects.filter(commune=a) if isinstance(a, Commune)
                else LossRatio.objects.filter(province=a) if isinstance(a, Province)
                else LossRatio.objects.none()
            )
            loss_ratio.append(qs.aggregate(avg=Avg('loss_ratio'))['avg'] or 0)

        # distance-decay competition
        comp_intensity = [competition_intensity(a) for a in areas]

        # ---- z-standardise ------------------------------------------
        pop_z, gap_z, veh_z, bank_z, comp_z = map(
            zscores, [pop, demand_gap, veh, bank_density, comp_intensity]
        )

        # ---- build raw composite ------------------------------------
        raw_scores = []
        parts      = []   # for optional debugging

        for i, a in enumerate(areas):
            demand       = 0.4 * pop_z[i] + 0.6 * gap_z[i]
            competition  = -comp_z[i]                  # less intensity ⇒ higher score
            economic     = bank_z[i]
            accessibility = 0                          # TODO: add drive-time
            risk         = logistic(loss_ratio[i])

            final_raw = (
                0.35 * demand +
                0.20 * competition +
                0.15 * economic +
                0.10 * accessibility +
                0.20 * risk
            )

            raw_scores.append(final_raw)
            parts.append((demand, competition, economic, risk))

        # ---- rescale to 0-100 ---------------------------------------
        s_min, s_max = min(raw_scores), max(raw_scores)
        span         = s_max - s_min or 1.0

        for i, a in enumerate(areas):
            score_100 = round((raw_scores[i] - s_min) / span * 100, 2)
            potential = (
                'HIGH'   if score_100 >= 70 else
                'MEDIUM' if score_100 >= 40 else
                'LOW'
            )

            CoverageScore.objects.update_or_create(
                area=a,
                defaults=dict(
                    score=score_100,
                    potential=potential,
                    calculation_date=timezone.now(),
                )
            )

            # print details for first three areas
            if i < 3:
                dem, com, eco, risk = parts[i]
                self.stdout.write(f"{a.name:<25} "
                                  f"D={dem:6.2f}  C={com:6.2f}  "
                                  f"E={eco:6.2f}  R={risk:4.2f}  "
                                  f"→ {score_100:6.2f}")

        self.stdout.write(self.style.SUCCESS(
            f"Coverage scores updated for {len(areas)} areas."
        ))
