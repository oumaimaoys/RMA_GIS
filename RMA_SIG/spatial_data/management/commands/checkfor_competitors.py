# spatial_data/management/commands/check_new_osm_competitors.py
import logging
import time
from typing import Optional, Dict, Any, List

import overpy
from django.conf import settings
from django.contrib.gis.db.models.functions import Distance
from django.contrib.gis.geos import Point
from django.contrib.gis.measure import D
from django.core.mail import send_mail
from django.core.management.base import BaseCommand
from django.utils import timezone

from spatial_data.models import (
    RMAOffice,
    OSMDiscoveredCompetitor,
    AgencyOSMCompetitorProximity,
)

# ------------------------------------------------------------------ #
#  CONFIG                                                            #
# ------------------------------------------------------------------ #
DEFAULT_RADIUS = getattr(settings, "DEFAULT_SEARCH_RADIUS_KM", 5)
OVERPASS_DELAY = 1.2           # seconds between requests (OSM-friendly)
OVERPASS_TIMEOUT = 60          # per request, seconds

EMAIL_TO = getattr(settings, "ALERT_EMAIL_RECIPIENTS", [])
EMAIL_FROM = getattr(settings, "DEFAULT_FROM_EMAIL", None)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  HELPERS                                                           #
# ------------------------------------------------------------------ #
def overpass_query(lat: float, lon: float, radius_m: int) -> str:
    """Return an Overpass QL query for insurance offices/shops."""
    return f"""
    (
      node["office"="insurance"](around:{radius_m},{lat},{lon});
      way ["office"="insurance"](around:{radius_m},{lat},{lon});
      node["shop"="insurance"](around:{radius_m},{lat},{lon});
    );
    out center tags;
    """


def send_summary_mail(entries: List[Dict[str, Any]]) -> None:
    """E-mail the list of new proximities (HTML + plain)."""
    if not entries or not EMAIL_TO or not EMAIL_FROM:
        return

    subject = "[RMA-SIG] New OSM competitor agencies near RMA offices"

    plain_lines = ["New competitor agencies detected:\n"]
    html_lines = ["<html><body><h2>New OSM competitor agencies</h2><ul>"]

    for e in entries:
        plain_lines.append(
            f"- {e['competitor_name']} ({e['osm_key']}, {e['dist_km']:.2f} km) "
            f"near {e['office_name']} [{e['office_city']}]"
        )
        html_lines.append(
            f"<li><strong>{e['competitor_name']}</strong> "
            f"({e['osm_key']}, {e['dist_km']:.2f}&nbsp;km)<br>"
            f"near <em>{e['office_name']}</em> – {e['office_city']}</li>"
        )

    html_lines.append("</ul></body></html>")

    send_mail(
        subject,
        "\n".join(plain_lines),
        EMAIL_FROM,
        EMAIL_TO,
        html_message="".join(html_lines),
        fail_silently=False,
    )


# ------------------------------------------------------------------ #
#  COMMAND                                                           #
# ------------------------------------------------------------------ #
class Command(BaseCommand):
    """Fast competitor checker (Overpass + bulk operations)."""

    help = "Detect new insurance agencies in OSM near each RMA office."

    def add_arguments(self, parser):
        parser.add_argument(
            "--radius_km",
            type=float,
            default=DEFAULT_RADIUS,
            help="Search radius in kilometres.",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Run without writing to the DB or sending e-mails.",
        )

    # -------------------- MAIN ------------------------------------ #
    def handle(self, *args, **opts):
        radius_km: float = opts["radius_km"]
        dry_run: bool = opts["dry_run"]

        api = overpy.Overpass(max_retry_count=3)

        offices = (
            RMAOffice.objects.exclude(location__isnull=True)
            .exclude(location__isempty=True)
        )
        self.stdout.write(
            f"🔍 Scanning {offices.count()} offices – radius {radius_km} km "
            f"(dry-run={dry_run})"
        )

        # Existing OSM ids to skip duplicates fast
        known_ids = set(
            OSMDiscoveredCompetitor.objects.values_list("osm_id", flat=True)
        )

        new_competitors: List[OSMDiscoveredCompetitor] = []
        summary_rows: List[Dict[str, Any]] = []
        new_proximities: List[AgencyOSMCompetitorProximity] = []

        # ---------------- LOOP OVER OFFICES ----------------------- #
        for idx, office in enumerate(offices, 1):
            lat, lon = office.location.y, office.location.x
            radius_m = int(radius_km * 1000)

            try:
                ov_result = api.query(overpass_query(lat, lon, radius_m))
            except overpy.exception.OverpassTooManyRequests:
                time.sleep(30)
                ov_result = api.query(overpass_query(lat, lon, radius_m))
            except Exception as e:
                logger.error("Overpass error for %s: %s", office.name, e)
                continue

            time.sleep(OVERPASS_DELAY)  # politeness pause

            # Helper to process node/way rows uniformly
            def process_osm(
                osm_id: int,
                osm_type: str,
                lat_raw: Optional[str],
                lon_raw: Optional[str],
                tags: Dict[str, Any],
            ) -> None:
                if lat_raw is None or lon_raw is None:
                    return  # missing centre (rare)
                try:
                    y = float(lat_raw)
                    x = float(lon_raw)
                except (TypeError, ValueError):
                    return

                if osm_id in known_ids:
                    # Already known competitor: ensure proximity exists
                    if not dry_run:
                        try:
                            comp = OSMDiscoveredCompetitor.objects.get(osm_id=osm_id)
                        except OSMDiscoveredCompetitor.DoesNotExist:
                            # It’s a freshly-found competitor that we haven’t flushed
                            # to the database yet; proximity will be created after the
                            # bulk_insert section, so just skip for now.
                            pass
                        else:
                            self._make_proximity(comp, office, radius_km, new_proximities)
                    return

                pt = Point(x, y, srid=4326)
                name = tags.get("name") or tags.get("operator") or "Unnamed"

                # Candidate competitor object
                comp_obj = OSMDiscoveredCompetitor(
                    osm_id=osm_id,
                    osm_type=osm_type,
                    name_from_osm=name,
                    osm_location=pt,
                    tags_from_osm=tags,
                    latitude=y,
                    longitude=x,
                )
                if not dry_run:
                    new_competitors.append(comp_obj)

                # Distance via PostGIS
                dist_km = office.location.distance(pt) * 100
                if dist_km <= radius_km:
                    summary_rows.append(
                        {
                            "office_name": office.name,
                            "office_city": office.city,
                            "competitor_name": name,
                            "osm_key": f"{osm_type}/{osm_id}",
                            "dist_km": dist_km,
                        }
                    )
                    if not dry_run:
                        new_proximities.append(
                            AgencyOSMCompetitorProximity(
                                rma_office=office,
                                osm_competitor=comp_obj,  # temp, id resolved later
                            )
                        )
                known_ids.add(osm_id)

            # Nodes
            for n in ov_result.nodes:
                process_osm(n.id, "node", n.lat, n.lon, n.tags)
            # Ways (use centre)
            for w in ov_result.ways:
                process_osm(
                    w.id,
                    "way",
                    getattr(w, "center_lat", None),
                    getattr(w, "center_lon", None),
                    w.tags,
                )

            self.stdout.write(f"  {idx}/{offices.count()} – {office.name} done.")

        # ---------------- BULK SAVE COMPETITORS -------------------- #
        if not dry_run and new_competitors:
            OSMDiscoveredCompetitor.objects.bulk_create(
                new_competitors, ignore_conflicts=True
            )
            # Fetch assigned IDs
            id_map = dict(
                OSMDiscoveredCompetitor.objects.filter(
                    osm_id__in=[c.osm_id for c in new_competitors]
                ).values_list("osm_id", "id")
            )
            for comp in new_competitors:
                comp.id = id_map[comp.osm_id]

        # ------------ RESOLVE FK & BULK SAVE PROXIMITIES ---------- #
        if not dry_run and new_proximities:
            for prox in new_proximities:
                prox.osm_competitor_id = prox.osm_competitor.id
            AgencyOSMCompetitorProximity.objects.bulk_create(
                new_proximities, ignore_conflicts=True
            )

        # ------------------------ SUMMARY -------------------------- #
        if summary_rows:
            self.stdout.write(
                self.style.WARNING(
                    f"\n🚨 {len(summary_rows)} new competitor agencies!"
                )
            )
            for r in summary_rows:
                self.stdout.write(
                    f"- {r['competitor_name']} near {r['office_name']} "
                    f"({r['dist_km']:.1f} km)"
                )
            if not dry_run:
                send_summary_mail(summary_rows)
            else:
                self.stdout.write(self.style.NOTICE("Dry-run: e-mail suppressed."))
        else:
            self.stdout.write(self.style.SUCCESS("\n✅ No new competitor agencies."))

    # ------------ helper: guarantee proximity row --------------- #
    def _make_proximity(
        self,
        comp: OSMDiscoveredCompetitor,
        office: RMAOffice,
        radius_km: float,
        bulk_list: Optional[List[AgencyOSMCompetitorProximity]] = None,
    ) -> None:
        """
        Ensure a proximity row exists; in bulk mode append to list,
        otherwise create if missing.
        """
        if comp.osm_location is None:
            return
        if office.location.distance(comp.osm_location) * 100 > radius_km:
            return

        if bulk_list is not None:
            bulk_list.append(
                AgencyOSMCompetitorProximity(
                    rma_office=office, osm_competitor=comp
                )
            )
        else:
            AgencyOSMCompetitorProximity.objects.get_or_create(
                rma_office=office, osm_competitor=comp
            )
