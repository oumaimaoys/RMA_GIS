# spatial_data/management/commands/check_new_osm_competitors.py
import logging
import time
from typing import Optional, Dict, Any, List

import overpy
from django.conf import settings
from django.contrib.auth import get_user_model
from django.contrib.gis.geos import Point
from django.core.mail import send_mail
from django.core.management.base import BaseCommand
from django.utils import timezone

from frontend.models import Notification
from spatial_data.models import (
    RMAOffice,
    OSMDiscoveredCompetitor,
    AgencyOSMCompetitorProximity,
)

# ────────────────────────────────────────────────────────────────────
#  CONFIG
# ────────────────────────────────────────────────────────────────────
DEFAULT_RADIUS       = getattr(settings, "DEFAULT_SEARCH_RADIUS_KM", 5)
OVERPASS_DELAY       = 1.2         # polite pause between requests (s)
OVERPASS_TIMEOUT     = 60          # not used (overpy default = 180 s)
EMAIL_TO             = getattr(settings, "ALERT_EMAIL_RECIPIENTS", [])
EMAIL_FROM           = getattr(settings, "DEFAULT_FROM_EMAIL", None)

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────
#  HELPER FUNCTIONS
# ────────────────────────────────────────────────────────────────────
def overpass_query(lat: float, lon: float, radius_m: int) -> str:
    """Return an Overpass-QL query for insurance offices/shops."""
    return f"""
    (
      node["office"="insurance"](around:{radius_m},{lat},{lon});
      way ["office"="insurance"](around:{radius_m},{lat},{lon});
      node["shop"="insurance"](around:{radius_m},{lat},{lon});
    );
    out center tags;
    """


def send_summary_mail(entries: List[Dict[str, Any]]) -> None:
    """Plain + HTML e-mail listing the new proximities."""
    if not entries or not EMAIL_TO or not EMAIL_FROM:
        return

    subject = "[RMA-SIG] New OSM competitor agencies near RMA offices"

    plain = ["New competitor agencies detected:\n"]
    html  = ["<html><body><h2>New OSM competitor agencies</h2><ul>"]

    for e in entries:
        line = (
            f"- {e['competitor_name']} ({e['osm_key']}, {e['dist_km']:.2f} km) "
            f"near {e['office_name']} [{e['office_city']}]"
        )
        plain.append(line)
        html.append(
            f"<li><strong>{e['competitor_name']}</strong> "
            f"({e['osm_key']}, {e['dist_km']:.2f}&nbsp;km)<br>"
            f"near <em>{e['office_name']}</em> – {e['office_city']}</li>"
        )

    html.append("</ul></body></html>")

    send_mail(
        subject,
        "\n".join(plain),
        EMAIL_FROM,
        EMAIL_TO,
        html_message="".join(html),
        fail_silently=False,
    )


def notify_staff_competitor(comp_name: str, rma_office: RMAOffice) -> None:
    """
    Creates ONE Notification per staff user for this competitor–office pair.
    """
    staff_qs = get_user_model().objects.filter(is_staff=True, is_active=True)
    if not staff_qs.exists():
        return

    notif_objs = [
        Notification(
            recipient=u,
            verb="New competitor nearby",
            description=f"{comp_name} detected near {rma_office.name}",
            url="/admin/spatial_data/osmdiscoveredcompetitor/",   # jump to admin list
        )
        for u in staff_qs
    ]
    Notification.objects.bulk_create(notif_objs, ignore_conflicts=True)


# ────────────────────────────────────────────────────────────────────
#  MANAGEMENT COMMAND
# ────────────────────────────────────────────────────────────────────
class Command(BaseCommand):
    """Detect new insurance agencies in OSM near each RMA office."""

    help = "Scrapes Overpass-API and stores competitor + proximity rows."

    def add_arguments(self, parser):
        parser.add_argument(
            "--radius_km",
            type=float,
            default=DEFAULT_RADIUS,
            help="Search radius around each office (km).",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Run without writing to DB or sending e-mails/notifications.",
        )

    # ────────────────────────────────────────────────────────────────
    #  MAIN
    # ────────────────────────────────────────────────────────────────
    def handle(self, *args, **opts):
        radius_km: float = opts["radius_km"]
        dry_run:   bool  = opts["dry_run"]

        api = overpy.Overpass(max_retry_count=3)

        offices = (
            RMAOffice.objects
            .exclude(location__isnull=True)
            .exclude(location__isempty=True)
        )
        self.stdout.write(
            f"🔍 Scanning {offices.count()} offices – "
            f"radius {radius_km} km (dry-run = {dry_run})"
        )

        # cache of OSM ids already known globally
        known_ids = set(
            OSMDiscoveredCompetitor.objects.values_list("osm_id", flat=True)
        )

        new_competitors:  List[OSMDiscoveredCompetitor]           = []
        new_proximities:  List[AgencyOSMCompetitorProximity]      = []
        summary_rows:     List[Dict[str, Any]]                    = []

        # ───────────── iterate over offices ─────────────
        for idx, office in enumerate(offices, 1):
            lat, lon = office.location.y, office.location.x
            radius_m = int(radius_km * 1000)

            try:
                result = api.query(overpass_query(lat, lon, radius_m))
            except overpy.exception.OverpassTooManyRequests:
                time.sleep(30)
                result = api.query(overpass_query(lat, lon, radius_m))
            except Exception as exc:
                logger.error("Overpass error for %s: %s", office.name, exc)
                continue

            time.sleep(OVERPASS_DELAY)      # be nice to OSM

            # inner helper to treat nodes & ways alike
            def handle_osm(
                osm_id: int,
                osm_type: str,
                lat_raw: Optional[str],
                lon_raw: Optional[str],
                tags: Dict[str, Any],
            ) -> None:
                if lat_raw is None or lon_raw is None:
                    return
                try:
                    y = float(lat_raw)
                    x = float(lon_raw)
                except (TypeError, ValueError):
                    return

                # ensure distance is inside radius (cheap PostGIS shortcut)
                pt = Point(x, y, srid=4326)
                if office.location.distance(pt) * 100 > radius_km:
                    return

                # ----------------------------------------------------------------
                #  (A) competitor already known globally
                # ----------------------------------------------------------------
                if osm_id in known_ids:
                    if dry_run:
                        return
                    try:
                        comp = OSMDiscoveredCompetitor.objects.get(osm_id=osm_id)
                    except OSMDiscoveredCompetitor.DoesNotExist:
                        return
                    self._append_proximity(comp, office, new_proximities, dry_run)
                    return  # done

                # ----------------------------------------------------------------
                #  (B) brand-new competitor
                # ----------------------------------------------------------------
                name = (
                    tags.get("name")
                    or tags.get("operator")
                    or "Unnamed"
                )
                comp_obj = OSMDiscoveredCompetitor(
                    osm_id=osm_id,
                    osm_type=osm_type,
                    name_from_osm=name,
                    osm_location=pt,
                    latitude=y,
                    longitude=x,
                    tags_from_osm=tags,
                )
                if not dry_run:
                    new_competitors.append(comp_obj)
                known_ids.add(osm_id)

                # proximity & summary
                self._append_proximity(comp_obj, office, new_proximities, dry_run)
                summary_rows.append(
                    {
                        "office_name": office.name,
                        "office_city": office.city,
                        "competitor_name": name,
                        "osm_key": f"{osm_type}/{osm_id}",
                        "dist_km": office.location.distance(pt) * 100,
                    }
                )

            # Nodes
            for n in result.nodes:
                handle_osm(n.id, "node", n.lat, n.lon, n.tags)
            # Ways (use computed centre)
            for w in result.ways:
                handle_osm(
                    w.id,
                    "way",
                    getattr(w, "center_lat", None),
                    getattr(w, "center_lon", None),
                    w.tags,
                )

            self.stdout.write(f"  {idx}/{offices.count()} – {office.name} done.")

        # ───────────── commit bulk objects ─────────────
        if not dry_run and new_competitors:
            OSMDiscoveredCompetitor.objects.bulk_create(
                new_competitors, ignore_conflicts=True
            )
            # map temporary objects to DB ids
            id_map = dict(
                OSMDiscoveredCompetitor.objects.filter(
                    osm_id__in=[c.osm_id for c in new_competitors]
                ).values_list("osm_id", "id")
            )
            for comp in new_competitors:
                comp.id = id_map[comp.osm_id]

        if not dry_run and new_proximities:
            for prox in new_proximities:
                prox.osm_competitor_id = prox.osm_competitor.id
            AgencyOSMCompetitorProximity.objects.bulk_create(
                new_proximities,
                ignore_conflicts=True,
            )

        # ───────────── summary / notifications ─────────────
        if not summary_rows:
            self.stdout.write(self.style.SUCCESS("\n✅ No new competitor agencies."))
            return

        # Console overview
        self.stdout.write(
            self.style.WARNING(
                f"\n🚨 {len(summary_rows)} new competitor agencies detected!"
            )
        )
        for r in summary_rows:
            self.stdout.write(
                f"- {r['competitor_name']} near {r['office_name']} "
                f"({r['dist_km']:.1f} km)"
            )

        if dry_run:
            self.stdout.write(self.style.NOTICE("Dry-run → e-mail & notifications skipped."))
            return

        # 1) E-mail
        send_summary_mail(summary_rows)

        # 2) In-app notifications (deduplicated per (osm_id, office_id))
        notified_pairs = set()
        for prox in new_proximities:
            pair = (prox.osm_competitor.osm_id, prox.rma_office_id)
            if pair in notified_pairs:
                continue
            notified_pairs.add(pair)
            notify_staff_competitor(
                prox.osm_competitor.name_from_osm or "Unnamed",
                prox.rma_office,
            )

    # ────────────────────── helper ──────────────────────
    @staticmethod
    def _append_proximity(
        competitor: OSMDiscoveredCompetitor,
        office: RMAOffice,
        bulk_list: List[AgencyOSMCompetitorProximity],
        dry_run: bool,
    ) -> None:
        """Add a proximity row (bulk or immediate)."""
        if dry_run:
            return
        bulk_list.append(
            AgencyOSMCompetitorProximity(
                rma_office=office,
                osm_competitor=competitor,
                first_seen_near_office=timezone.now(),
            )
        )
