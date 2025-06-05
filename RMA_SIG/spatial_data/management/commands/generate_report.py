# spatial_data/management/commands/export_competitors_by_region_csv.py
from pathlib import Path
from datetime import datetime

import pandas as pd
from django.core.management.base import BaseCommand
from django.db.models import Count, Q

from spatial_data.models import (
    RegionAdministrative,
    Competitor,
    RMAOffice,
)

# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------
def build_dataframes() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        summary_df  – one row per administrative region
        detailed_df – one row per (region, mandante) pair,
                      *including* a row where Mandante == "RMA"
    """
    summary_rows, detail_rows = [], []

    regions = (
        RegionAdministrative.objects
        .prefetch_related('provinces')
        .all()
    )

    for region in regions:
        provinces = region.provinces.all()
        if not provinces.exists():
            continue

        # Build a single spatial Q object covering all provinces
        q = Q()
        for prov in provinces:
            q |= Q(location__within=prov.boundary)

        competitors_qs = Competitor.objects.filter(q)     # competitors
        rma_qs         = RMAOffice.objects.filter(q)      # RMA offices

        comp_total = competitors_qs.count()
        rma_total  = rma_qs.count()
        total_agencies = comp_total + rma_total

        # -------------------------------- summary --------------------------------
        mandante_breakdown = (
            competitors_qs.values('mandante')
                          .annotate(cnt=Count('id'))
                          .order_by('-cnt')
        )
        unique_mandantes_incl_rma = mandante_breakdown.count() + 1  # “+1” for RMA

        summary_rows.append({
            'Administrative_Region'        : region.name,
            'Total_Agencies'               : total_agencies,
            'Total_Competitors'            : comp_total,
            'RMA_Offices_Count'            : rma_total,
            'Unique_Mandantes_Including_RMA': unique_mandantes_incl_rma,
            'Provinces_Count'              : provinces.count(),
        })

        # ------------------------------- detailed --------------------------------
        # a) competitors (each mandante)
        for item in mandante_breakdown:
            m_name, m_cnt = item['mandante'], item['cnt']
            type_break = (
                competitors_qs.filter(mandante=m_name)
                              .values('competitor_type')
                              .annotate(num=Count('id'))
            )
            type_str = ', '.join(f"{tb['competitor_type']}: {tb['num']}"
                                 for tb in type_break)

            detail_rows.append({
                'Administrative_Region': region.name,
                'Mandante'             : m_name,
                'Agency_Count'         : m_cnt,
                'Types_Breakdown'      : type_str,
                'Provinces_in_Region'  : ', '.join(p.name for p in provinces),
            })

        # b) RMA *as* a mandante
        if rma_total:
            detail_rows.append({
                'Administrative_Region': region.name,
                'Mandante'             : 'RMA',
                'Agency_Count'         : rma_total,
                'Types_Breakdown'      : f"RMAOffice: {rma_total}",
                'Provinces_in_Region'  : ', '.join(p.name for p in provinces),
            })

    return pd.DataFrame(summary_rows), pd.DataFrame(detail_rows)


# ------------------------------------------------------------------
# command
# ------------------------------------------------------------------
class Command(BaseCommand):
    """
    Creates three CSV reports:

      • summary_<timestamp>.csv
      • detailed_by_mandante_<timestamp>.csv   (now includes an “RMA” row)
      • pivot_region_vs_mandante_<timestamp>.csv
    """

    help = "Export competitor *and RMA office* statistics by administrative region (CSV)."

    def add_arguments(self, parser):
        parser.add_argument(
            '--output-dir',
            default='.',
            help='Directory where CSV files will be written (default: current dir).'
        )

    # --------------------------------------------------------------
    def handle(self, *args, **opts):
        out_dir = Path(opts['output_dir']).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        self.stdout.write("📊 Building data …")
        summary_df, detailed_df = build_dataframes()

        if summary_df.empty:
            self.stdout.write(self.style.WARNING("⚠️  No data found – nothing to export."))
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1) summary -------------------------------------------------------
        summary_path = out_dir / f"summary_{ts}.csv"
        summary_df.to_csv(summary_path, index=False)
        self.stdout.write(self.style.SUCCESS(f"  • summary  →  {summary_path}"))

        # 2) detailed ------------------------------------------------------
        detailed_path = out_dir / f"detailed_by_mandante_{ts}.csv"
        detailed_df.to_csv(detailed_path, index=False)
        self.stdout.write(self.style.SUCCESS(f"  • detailed →  {detailed_path}"))

        # 3) pivot (optional) ---------------------------------------------
        if not detailed_df.empty:
            pivot_df = detailed_df.pivot_table(
                values='Agency_Count',
                index='Administrative_Region',
                columns='Mandante',
                aggfunc='sum',
                fill_value=0
            )
            pivot_path = out_dir / f"pivot_region_vs_mandante_{ts}.csv"
            pivot_df.to_csv(pivot_path)
            self.stdout.write(self.style.SUCCESS(f"  • pivot    →  {pivot_path}"))

        self.stdout.write(self.style.SUCCESS("✅ CSV export completed."))
