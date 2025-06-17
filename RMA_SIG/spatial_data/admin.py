from django.contrib import admin
from django.contrib.gis.admin import OSMGeoAdmin
from django.urls import path
from django.shortcuts import render, redirect
from django import forms
from django.contrib import messages
from .models import RMAOffice, RMABGD, RMAAgent, Bank, Competitor, Area, CoverageScore, Commune, Province, LossRatio, CoverageStats, Variables, CA, RegionAdministrative, OSMDiscoveredCompetitor, AgencyOSMCompetitorProximity
import pandas as pd
from django.http import HttpResponseRedirect
from .forms import RMABGDForm
from django.contrib.gis.db import models
from django.contrib import admin
from django.utils.html import format_html
from datetime import datetime
import json
from django.contrib.gis.geos import GEOSGeometry, MultiPolygon, Point
from django.forms import ModelForm
from .forms import LatLonPointMixin
from decimal import Decimal
from django.contrib.gis import admin as geoadmin
from django.utils.timezone import now


class CsvImportForm(forms.Form):
    csv_file = forms.FileField()


class BaseModelAdmin(OSMGeoAdmin):
    pass

class LatLonPointAdminMixin:
    """
    Mixin for ModelAdmin / InlineModelAdmin that replaces the
    default map widget with the Lat/Lon form created on the fly.
    """

    def get_form(self, request, obj=None, **kwargs):
        """
        1. Dynamically create a new class that inherits from
           LatLonPointMixin AND ModelForm.
        2. Fill in Meta.model with self.model so it’s concrete.
        3. Return that class to Django.
        """
        attrs = {
            "Meta": type(
                "Meta",
                (LatLonPointMixin.Meta,),
                {"model": self.model},      # plug in the real model
            )
        }
        DynamicForm = type(
            f"{self.model.__name__}LatLonForm",
            (LatLonPointMixin, ModelForm),
            attrs,
        )
        kwargs["form"] = DynamicForm
        return super().get_form(request, obj, **kwargs)



@admin.register(RMABGD)
class RMABGDAdmin(LatLonPointAdminMixin, BaseModelAdmin):
    list_display = ('name', 'code_RMA', 'code_ACAPS', 'address', 'city','location', 'type_BGD', 'Partenaire', 'formatted_date_creation', 'RMA_BGD_state')
    list_filter = ('type_BGD', 'RMA_BGD_state', 'city')
    search_fields = ('name', 'code_RMA', 'Partenaire')
    form = RMABGDForm  

    add_form_template = 'admin/spatial_data/RMABGD/import_form.html'
    
    
    # Set date format for normal Django admin forms
    formfield_overrides = {
        models.DateField: {'widget': forms.DateInput(format='%d-%m-%Y', attrs={'placeholder': 'dd-mm-yyyy'})},
    }

    def add_view(self, request, form_url='', extra_context=None):
        """
        • Default:     show the Excel-import wizard (self.add_form_template).
        • ?manual=1:   temporarily disable add_form_template so Django falls
                       back to the standard add-object form.
        """
        if request.GET.get("manual"):
            original = self.add_form_template
            self.add_form_template = None            # use the stock template
            try:
                return super().add_view(request, form_url, extra_context)
            finally:
                self.add_form_template = original    # restore for next time
        return super().add_view(request, form_url, extra_context)
    
    # Custom method to format date in d-m-yyyy format for admin list display
    def formatted_date_creation(self, obj):
        if obj.date_creation:
            return obj.date_creation.strftime('%d-%m-%Y')
        return '-'
    formatted_date_creation.short_description = 'Date création'
    formatted_date_creation.admin_order_field = 'date_creation'  # Makes column sortable

    def process_excel_import(self, request):
        f = request.FILES.get('excel_file')
        if not f:
            messages.error(request, "Please choose an Excel file.")
            return False

        try:
            df = pd.read_excel(f)
            df.columns = df.columns.str.strip()
            df.columns = df.columns.str.replace('\xa0', ' ', regex=False)
            df.columns = df.columns.str.replace('\u200b', '', regex=False)
            print(list(df.columns))

            required = ["Code RMA","CODE ACAPS","Dénomination RMA","Ville",
                        "Adresse","Longitude","Latitude","Type BGD",
                        "Partenaire","Date création","Etat BGD RMA"]
            missing = [h for h in required if h not in df.columns]
            if missing:
                messages.error(request, f"Missing columns: {', '.join(missing)}")
                return False

            imported = 0
            errors = 0
            for i, row in df.iterrows():
                try:
                    # Parse the date in d-m-yyyy format
                    date_str = str(row["Date création"]).strip()
                    date_creation = None
                    
                    # Try different date formats (robust parsing)
                    if pd.notna(row["Date création"]):
                        try:
                            # First try Excel's native date format (already parsed by pandas)
                            if isinstance(row["Date création"], datetime):
                                date_creation = row["Date création"]
                            # Then try d-m-yyyy format
                            elif '-' in date_str:
                                date_creation = datetime.strptime(date_str, '%d-%m-%Y')
                            # Try d/m/yyyy format
                            elif '/' in date_str:
                                date_creation = datetime.strptime(date_str, '%d/%m/%Y')
                            else:
                                raise ValueError(f"Date format not recognized: {date_str}")
                        except ValueError as e:
                            messages.warning(request, f"Row {i+1}: Date format error ({date_str}). {e}")
                            errors += 1
                            continue
                    
                    RMABGD.objects.create(
                        code_ACAPS=row["CODE ACAPS"],
                        code_RMA=row["Code RMA"],
                        name=row["Dénomination RMA"],
                        address=row["Adresse"],
                        city=row["Ville"],
                        location=f'POINT({row["Longitude"]} {row["Latitude"]})',
                        type_BGD=row["Type BGD"],
                        Partenaire=row["Partenaire"],
                        date_creation=date_creation,
                        RMA_BGD_state=row["Etat BGD RMA"]
                    )
                    imported += 1
                except Exception as e:
                    messages.error(request, f"Row {i+1}: {e}")
                    errors += 1

            if imported:
                messages.success(request, f"Imported {imported} rows")
                if errors:
                    messages.warning(request, f"{errors} rows had errors and were skipped")
            else:
                messages.warning(request, "No rows were imported")
            return imported > 0

        except Exception as e:
            messages.error(request, f"Error processing file: {e}")
            return False

    def changeform_view(self, request, object_id=None, form_url='', extra_context=None):
        if request.method == 'POST' and '_import_file' in request.POST:
            self.process_excel_import(request)
            return HttpResponseRedirect(request.path)
        return super().changeform_view(request, object_id, form_url, extra_context)


@admin.register(RMAAgent)
class RMAAgentAdmin(LatLonPointAdminMixin, BaseModelAdmin):
    list_display = ('name', 'code_RMA', 'code_ACAPS', 'address', 'city','location')
    list_filter = ('city',)
    search_fields = ('name', 'code_RMA', 'address', 'city')

    add_form_template = 'admin/spatial_data/RMAAgent/import_form.html'

    def add_view(self, request, form_url='', extra_context=None):
        """
        • Default:     show the Excel-import wizard (self.add_form_template).
        • ?manual=1:   temporarily disable add_form_template so Django falls
                       back to the standard add-object form.
        """
        if request.GET.get("manual"):
            original = self.add_form_template
            self.add_form_template = None            # use the stock template
            try:
                return super().add_view(request, form_url, extra_context)
            finally:
                self.add_form_template = original    # restore for next time
        return super().add_view(request, form_url, extra_context)
    

    def process_excel_import(self, request):
        f = request.FILES.get('excel_file')
        if not f:
            messages.error(request, "Please choose an Excel file.")
            return False

        try:
            df = pd.read_excel(f)
            required = ["Code RMA-Agent", "Code ACAPS", "Agence", "ville", "Adresse de l'agence", "Longitude", "Latitude"]

            missing = [h for h in required if h not in df.columns]
            if missing:
                messages.error(request, f"Missing columns: {', '.join(missing)}")
                return False

            imported = 0
            for i, row in df.iterrows():
                try:
                    RMAAgent.objects.create(
                        code_ACAPS = row["Code ACAPS"],
                        code_RMA = row["Code RMA-Agent"],
                        name = row["Agence"],
                        address = row["Adresse de l'agence"],
                        city = row["ville"],
                        location = f'POINT({row["Longitude"]} {row["Latitude"]})',
            )
                    imported += 1
                except Exception as e:
                    messages.error(request, f"Row {i+1}: {e}")

            if imported:
                messages.success(request, f"Imported {imported} rows")
            else:
                messages.warning(request, "No rows were imported")
            return imported > 0

        except Exception as e:
            messages.error(request, f"Error processing file: {e}")
            return False

    def changeform_view(self, request, object_id=None, form_url='', extra_context=None):
        if request.method == 'POST' and '_import_file' in request.POST:
            self.process_excel_import(request)
            return HttpResponseRedirect(request.path)
        return super().changeform_view(request, object_id, form_url, extra_context)

    


@admin.register(Bank)
class BankAdmin(LatLonPointAdminMixin, BaseModelAdmin):
    list_display = ('institution_name',)
    search_fields = ('institution_name',)

    add_form_template = "admin/spatial_data/Bank/import_form.html"
    
    def add_view(self, request, form_url='', extra_context=None):
        """
        • Default:     show the Excel-import wizard (self.add_form_template).
        • ?manual=1:   temporarily disable add_form_template so Django falls
                       back to the standard add-object form.
        """
        if request.GET.get("manual"):
            original = self.add_form_template
            self.add_form_template = None            # use the stock template
            try:
                return super().add_view(request, form_url, extra_context)
            finally:
                self.add_form_template = original    # restore for next time
        return super().add_view(request, form_url, extra_context)

    def process_excel_import(self, request):
        f = request.FILES.get('excel_file')
        if not f:
            messages.error(request, "Please choose an Excel file.")
            return False

        try:
            df = pd.read_excel(f)
            required = ["bank", "longitude", "latitude"]

            missing = [h for h in required if h not in df.columns]
            if missing:
                messages.error(request, f"Missing columns: {', '.join(missing)}")
                return False

            imported = 0
            for i, row in df.iterrows():
                try:
                    Bank.objects.create(
                        institution_name=row['bank'],
                        location=f'POINT({row["longitude"]} {row["latitude"]})',
            )
                    imported += 1
                except Exception as e:
                    messages.error(request, f"Row {i+1}: {e}")

            if imported:
                messages.success(request, f"Imported {imported} rows")
            else:
                messages.warning(request, "No rows were imported")
            return imported > 0

        except Exception as e:
            messages.error(request, f"Error processing file: {e}")
            return False

    def changeform_view(self, request, object_id=None, form_url='', extra_context=None):
        if request.method == 'POST' and '_import_file' in request.POST:
            self.process_excel_import(request)
            return HttpResponseRedirect(request.path)
        return super().changeform_view(request, object_id, form_url, extra_context)



@admin.register(Competitor)
class CompetitorAdmin(LatLonPointAdminMixin, BaseModelAdmin):
    list_display = ('agency_name', 'code_ACAPS', 'competitor_type', 'mandante', 'city')
    list_filter = ('competitor_type', 'mandante')
    search_fields = ('company_name', 'code_ACAPS', 'mandante', 'city',)

    add_form_template = "admin/spatial_data/Competitor/import_form.html"
    
    def add_view(self, request, form_url='', extra_context=None):
        """
        • Default:     show the Excel-import wizard (self.add_form_template).
        • ?manual=1:   temporarily disable add_form_template so Django falls
                       back to the standard add-object form.
        """
        if request.GET.get("manual"):
            original = self.add_form_template
            self.add_form_template = None            # use the stock template
            try:
                return super().add_view(request, form_url, extra_context)
            finally:
                self.add_form_template = original    # restore for next time
        return super().add_view(request, form_url, extra_context)

    def process_excel_import(self, request):
        f = request.FILES.get('excel_file')
        if not f:
            messages.error(request, "Please choose an Excel file.")
            return False

        try:
            df = pd.read_excel(f)
            required = ["code_acaps", "nom", "qualité", "mandante", "adresse", "localité", "longitude", "latitude"]

            missing = [h for h in required if h not in df.columns]
            if missing:
                messages.error(request, f"Missing columns: {', '.join(missing)}")
                return False

            imported = 0
            for i, row in df.iterrows():
                try:
                    Competitor.objects.create(
                        code_ACAPS=row['code_acaps'],
                        company_name=row['nom'],
                        competitor_type=row['qualité'],
                        mandante= row['mandante'],
                        address=row['adresse'],
                        city = row['localité'],
                        location=f'POINT({row["longitude"]} {row["latitude"]})',
            )
                    imported += 1
                except Exception as e:
                    messages.error(request, f"Row {i+1}: {e}")

            if imported:
                messages.success(request, f"Imported {imported} rows")
            else:
                messages.warning(request, "No rows were imported")
            return imported > 0

        except Exception as e:
            messages.error(request, f"Error processing file: {e}")
            return False

    def changeform_view(self, request, object_id=None, form_url='', extra_context=None):
        if request.method == 'POST' and '_import_file' in request.POST:
            self.process_excel_import(request)
            return HttpResponseRedirect(request.path)
        return super().changeform_view(request, object_id, form_url, extra_context)



@admin.register(Commune)
class CommuneAdmin(admin.ModelAdmin):
    list_display = ('id','name', 'population',  'insured_population', 'estimated_vehicles','bank_count', 'bank_intensity', 'competition_intensity')
    search_fields = ('name',)

    add_form_template = "admin/spatial_data/Commune/import_form.html"


    def process_geojson_import(self, request):
        f = request.FILES.get('geojson_file')
        if not f:
            messages.error(request, "Please choose a GeoJSON file.")
            return False

        try:
            data = json.load(f)
            features = data.get('features', [])
            if not features:
                messages.error(request, "No features found in the GeoJSON.")
                return False

            imported = 0
            skipped = 0
            errors = 0
            
            # Collect error messages to report later
            error_messages = []
            
            for idx, feature in enumerate(features, start=1):
                props = feature.get('properties', {})
                geom_dict = feature.get('geometry')
                name = props.get('Nom_Commun')
                population_2004 = props.get('Pop2004')
                population = props.get('Population')

                # Check for required fields
                if not name:
                    error_messages.append(f"Feature {idx}: missing 'Nom_Commun'")
                    errors += 1
                    continue
                
                # Fix the population logic - this was causing the issue
                if population is None:
                    if population_2004 is not None:
                        population = population_2004
                    else:
                        # Default population value if both are missing
                        population = 0
                
                if not geom_dict:
                    error_messages.append(f"Feature {idx}: missing geometry")
                    errors += 1
                    continue

                try:
                    # Build a GEOS geometry from the GeoJSON geometry dict
                    geom = GEOSGeometry(json.dumps(geom_dict))
                    
                    # Validate geometry
                    if not geom.valid:
                        # Try to make it valid
                        try:
                            geom = geom.buffer(0)
                            if not geom.valid:
                                error_messages.append(f"Feature {idx}: invalid geometry and couldn't repair")
                                errors += 1
                                continue
                        except Exception as e:
                            error_messages.append(f"Feature {idx}: couldn't repair invalid geometry: {e}")
                            errors += 1
                            continue
                    
                    # If it's a Polygon, wrap it in a MultiPolygon
                    if geom.geom_type == 'Polygon':
                        geom = MultiPolygon(geom)
                    elif geom.geom_type != 'MultiPolygon':
                        error_messages.append(f"Feature {idx}: unexpected geometry type {geom.geom_type}")
                        errors += 1
                        continue

                    # Check if commune with this name already exists
                    if Commune.objects.filter(name=name).exists():
                        skipped += 1
                        continue

                    # Create the commune
                    Commune.objects.create(
                        name=name,
                        population=population,
                        boundary=geom
                    )
                    imported += 1
                except Exception as e:
                    error_messages.append(f"Feature {idx}: {e}")
                    errors += 1

            # Show summary message
            if imported > 0:
                messages.success(request, f"Imported {imported} communes successfully.")
            if skipped > 0:
                messages.warning(request, f"Skipped {skipped} communes (already exist).")
            if errors > 0:
                messages.error(request, f"Failed to import {errors} communes.")
                
            # Show detailed error messages (limit to first 20 to avoid flooding the admin)
            for msg in error_messages[:20]:
                messages.error(request, msg)
                
            if len(error_messages) > 20:
                messages.error(request, f"... and {len(error_messages) - 20} more errors (not shown)")
                
            return imported > 0

        except Exception as e:
            messages.error(request, f"Error processing GeoJSON: {e}")
            import traceback
            messages.error(request, traceback.format_exc())
            return False

    def changeform_view(self, request, object_id=None, form_url='', extra_context=None):
        if request.method == 'POST' and '_import_file' in request.POST:
            self.process_geojson_import(request)
            return HttpResponseRedirect(request.path)
        return super().changeform_view(request, object_id, form_url, extra_context)

    
@admin.register(Province)
class ProvinceAdmin(admin.ModelAdmin):
    list_display = ('name', 'population', 'insured_population', 'estimated_vehicles','bank_count', 'bank_intensity', 'competition_intensity')
    search_fields = ('name',)

    add_form_template = "admin/spatial_data/Province/import_form.html"

    def process_geojson_import(self, request):
        f = request.FILES.get('geojson_file')
        if not f:
            messages.error(request, "Please choose a GeoJSON file.")
            return False

        try:
            data = json.load(f)
            features = data.get('features', [])
            if not features:
                messages.error(request, "No features found in the GeoJSON.")
                return False

            imported = 0
            for idx, feature in enumerate(features, start=1):
                props = feature.get('properties', {})
                geom_dict = feature.get('geometry')
                name = props.get('shapeName')
                population = props.get('population')

                if not name:
                    messages.error(
                        request,
                        f"Feature {idx}: missing 'Nom_commun'."
                    )
                    continue
                if not population:
                    messages.error(
                        request,
                        f"Feature {idx}: missing 'population'."
                    )
                    continue
                if not geom_dict:
                    messages.error(
                        request,
                        f"Feature {idx}: missing geometry"
                    )
                    continue


                try:
                    # Build a GEOS geometry from the GeoJSON geometry dict
                    geom = GEOSGeometry(json.dumps(geom_dict))
                    # If it's a Polygon, wrap it in a MultiPolygon
                    if geom.geom_type == 'Polygon':
                        geom = MultiPolygon(geom)
                    Province.objects.create(
                        name=name,
                        population=population,
                        boundary=geom,
                    )
                    imported += 1
                except Exception as e:
                    messages.error(request, f"Feature {idx}: {e}")

            if imported:
                messages.success(request, f"Imported {imported} features.")
            else:
                messages.warning(request, "No features were imported.")
            return imported > 0

        except Exception as e:
            messages.error(request, f"Error processing GeoJSON: {e}")
            return False

    def changeform_view(self, request, object_id=None, form_url='', extra_context=None):
        if request.method == 'POST' and '_import_file' in request.POST:
            self.process_geojson_import(request)
            return HttpResponseRedirect(request.path)
        return super().changeform_view(request, object_id, form_url, extra_context)
    
@admin.register(LossRatio)
class LossRatioAdmin(admin.ModelAdmin):
    list_display = ('province', 'commune', 'RMA_office', 'loss_ratio')
    search_fields = ('province', 'commune', 'RMA_office',)
    add_form_template = "admin/spatial_data/LossRatio/import_form.html"
    
    def add_view(self, request, form_url='', extra_context=None):
        """
        • Default:     show the Excel-import wizard (self.add_form_template).
        • ?manual=1:   temporarily disable add_form_template so Django falls
                       back to the standard add-object form.
        """
        if request.GET.get("manual"):
            original = self.add_form_template
            self.add_form_template = None            # use the stock template
            try:
                return super().add_view(request, form_url, extra_context)
            finally:
                self.add_form_template = original    # restore for next time
        return super().add_view(request, form_url, extra_context)

    def process_excel_import(self, request):
        f = request.FILES.get('excel_file')
        if not f:
            messages.error(request, "Please choose an Excel file.")
            return False

        try:
            # Read Excel file
            print("DEBUG: Reading Excel file...")
            df = pd.read_excel(f, header=1)
            df.columns = (
                df.columns
                .str.strip()
                .str.replace('\xa0',' ', regex=False)
                .str.replace('\u200b','', regex=False)
            )
            df.columns = df.columns.str.replace(r'\s*\.\s*', '.', regex=True)
            
            # Verify expected columns exist
            expected_cols = ["Réseau", "Code Inter", "Nom Inter", "Usage"]
            missing = [col for col in expected_cols if col not in df.columns]
            if missing:
                messages.error(request, f"Missing required columns: {', '.join(missing)}")
                return False
                
            # Rename columns for consistency
            df = df.rename(columns={
                "Prime"  : "Prime_2022",
                "S/P"    : "SP_2022",
                "Prime.1": "Prime_2023",
                "S/P.1"  : "SP_2023",
                "Prime.2": "Prime_2024",
                "S/P.2"  : "SP_2024",
                "Prime.3": "Prime_Total",
                "S/P.3"  : "SP_Total",
            })
            
            # Filter to only include totals
            tots = df[df["Usage"].str.strip().eq("Total")]
            out = tots[["Code Inter", "Nom Inter", "Prime_2024", "SP_2024"]].copy()
            print(f"DEBUG: Found {len(out)} totals")

            # Clean up SP_2024 values
            def clean_sp(val):
                if pd.isna(val):
                    return None
                s = str(val).strip()
                if s.endswith('%'):
                    # "80%" -> 0.8
                    return float(s.rstrip('%')) / 100
                else:
                    # already a decimal or integer
                    return float(s)

            out["SP_2024"] = out["SP_2024"].apply(clean_sp)
            print("DEBUG: cleaned SP_2024 dtype:", out["SP_2024"].dtype)
            
            # Import the data to the database
            imported = errors = 0
            for i, row in out.iterrows():
                try:
                    raw = row["Code Inter"]
                    # -- normalize the code:
                    if isinstance(raw, float):
                        code_str = str(int(raw)) if raw.is_integer() else str(raw)
                    else:
                        code_str = str(raw).strip().rstrip('.').replace(' ', '')
                    
                    # -- lookup safely:
                    office = RMAOffice.objects.filter(code_RMA__iexact=code_str).first()
                    if not office:
                        print(f"WARNING: no office for code `{code_str}`, skipping row {i+1}")
                        errors += 1
                        continue

                    pt = office.location
                    prov  = Province.objects.filter(boundary__contains=pt).first()
                    comm  = Commune.objects.filter(boundary__contains=pt).first()
                    loss_ratio = row["SP_2024"]
                    if pd.isna(loss_ratio):
                        print(f"WARNING: empty SP_2024 on row {i+1}, skipping")
                        errors += 1
                        continue

                    obj, created = LossRatio.objects.update_or_create(
                        RMA_office=office,
                        defaults={'province': prov, 'commune': comm, 'loss_ratio': loss_ratio}
                    )
                    imported += 1
                except Exception as e:
                    print(f"ERROR: row {i+1} import failed:", e)
                    messages.error(request, f"Row {i+1}: {e}")
                    errors += 1


            print(f"DEBUG: import complete, imported={imported}, errors={errors}")
            if imported:
                messages.success(request, f"Imported {imported} records.")
                if errors:
                    messages.warning(request, f"{errors} rows were skipped.")
            else:
                messages.warning(request, "No rows were imported.")

            return imported > 0
        except Exception as e:
            print(f"ERROR: Import failed: {e}")
            messages.error(request, f"Import failed: {e}")
            return False

    def changeform_view(self, request, object_id=None, form_url='', extra_context=None):
        if request.method == 'POST' and '_import_file' in request.POST:
            self.process_excel_import(request)
            return HttpResponseRedirect(request.path)
        return super().changeform_view(request, object_id, form_url, extra_context)

@admin.register(CoverageScore)
class CoverageScoreAdmin(admin.ModelAdmin):
    list_display = (
        "area",                   # FK – shows __str__ of Area
        "score", "potential",
        "demand_score", "competition_score",
        "economic_score", "accessibility_score",
        "risk_score",
        "travel_time_to_centroid_minutes",
        "calculation_date",
    )

    list_filter = ('potential',)
    search_fields = ('area__name',)

@admin.register(CoverageStats)
class CoverageStatsAdmin(admin.ModelAdmin):
    list_display = (
        "area_type", "calculation_date",
        "pop_mean",  "pop_std",
        "gap_mean",  "gap_std",
        "veh_mean",  "veh_std",
        "bank_intensity_mean",  "bank_intensity_std",
        "comp_intensity_mean",  "comp_intensity_std",
        "loss_ratio_mean",      "loss_ratio_std",
        "market_potential_mean","market_potential_std",
    )

    # (optional quality-of-life tweaks)
    list_filter   = ("area_type",)
    search_fields = ('calculation_date',)

    def get_readonly_fields(self, request, obj=None):
        return ['calculation_date']  # Make calc_date read-only
    
@admin.register(Variables)
class VariablesAdmin(admin.ModelAdmin):
    list_display = ('name', 'value', 'description')
    search_fields = ('name',)


@admin.register(CA)
class CAAdmin(admin.ModelAdmin):
    list_display      = ('agency', 'year', 'CA_value')
    search_fields     = ('agency__name', 'agency__code_RMA', 'year')
    add_form_template = 'admin/spatial_data/CA/import_form.html'

    # ─────────────────────────────────────────────────────────────
    # 1) Inject <years> list into the custom “add” template
    # ─────────────────────────────────────────────────────────────
    def add_view(self, request, form_url='', extra_context=None):
        """
        – Default: show the custom Excel-import wizard, with
                a <select> populated by `years`.
        – ?manual=1: temporarily disable the wizard template so the
                    stock Django “Add CA” form appears instead.
        """

        # 1️⃣  provide the <years> list to the template
        extra_context = extra_context or {}
        extra_context["years"] = list(range(2020, 2031))   # 2020 … 2030

        # 2️⃣  honour the manual-entry toggle
        if request.GET.get("manual"):
            original = self.add_form_template
            self.add_form_template = None                  # use default template
            try:
                return super().add_view(request, form_url, extra_context)
            finally:
                self.add_form_template = original          # restore for next time

        # 3️⃣  regular wizard
        return super().add_view(request, form_url, extra_context)


    # ─────────────────────────────────────────────────────────────
    # 2) Excel importer
    # ─────────────────────────────────────────────────────────────
    def process_excel_import(self, request):
        year = request.POST.get('year')
        f    = request.FILES.get('excel_file')

        if not year or not f:
            messages.error(request, "Sélectionnez une année et un fichier.")
            return False

        try:
            df = pd.read_excel(f, dtype=str)
            df.columns = (df.columns.str.strip()
                                    .str.replace('\xa0', ' ', regex=False)
                                    .str.replace('\u200b', '', regex=False))

            required = {"CODEAGENT", year}
            if missing := required - set(df.columns):
                messages.error(request, f"Colonnes manquantes : {', '.join(missing)}")
                return False

            imported = skipped = 0
            for idx, row in df.iterrows():
                code = (row["CODEAGENT"] or "").strip()
                raw  = (row[year]        or "").strip().replace(' ', '').replace(',', '')
                if not code or not raw:
                    skipped += 1
                    continue

                try:
                    office = RMAOffice.objects.get(code_RMA__iexact=code)
                except RMAOffice.DoesNotExist:
                    messages.warning(request, f"Ligne {idx+2}: CODEAGENT {code} inconnu.")
                    skipped += 1
                    continue

                CA.objects.update_or_create(
                    agency=office,
                    year=int(year),
                    defaults={"CA_value": Decimal(raw)}
                )
                imported += 1

            if imported:
                messages.success(request, f"{imported} lignes importées.")
            if skipped:
                messages.warning(request, f"{skipped} lignes ignorées.")
            return bool(imported)

        except Exception as exc:
            messages.error(request, f"Erreur d'import : {exc}")
            return False

    # ─────────────────────────────────────────────────────────────
    # 3) Hook “Import” button on the add page
    # ─────────────────────────────────────────────────────────────
    def changeform_view(self, request, object_id=None,
                        form_url='', extra_context=None):
        if request.method == 'POST' and '_import_file' in request.POST:
            self.process_excel_import(request)
            return HttpResponseRedirect(request.path)
        return super().changeform_view(request, object_id, form_url, extra_context)
    
@admin.register(RegionAdministrative)
class RegionAdministrativeAdmin(admin.ModelAdmin):
    list_display  = ('name', 'province_count', 'short_province_list')
    search_fields = ('name', 'provinces__name')
    ordering      = ('name',)

    filter_horizontal  = ('provinces',)        # nice dual-select widget


    @admin.display(description="Nbr. provinces")
    def province_count(self, obj):
        return obj.provinces.count()

    @admin.display(description="Some provinces")
    def short_province_list(self, obj):
        names = [p.name for p in obj.provinces.all()[:5]]
        tail  = " …" if obj.provinces.count() > 5 else ""
        return ", ".join(names) + tail
    

class AgencyOSMCompetitorProximityInline(admin.TabularInline):
    model = AgencyOSMCompetitorProximity
    fk_name = "osm_competitor"
    extra = 0
    verbose_name_plural = "RMA offices nearby"
    raw_id_fields = ("rma_office",)
    readonly_fields = ("first_seen_near_office", "last_seen_near_office")
    can_delete = False


# ------------------------------------------------------------------ #
#  MAIN COMPETITOR ADMIN                                             #
# ------------------------------------------------------------------ #
@admin.register(OSMDiscoveredCompetitor)
class OSMDiscoveredCompetitorAdmin(geoadmin.OSMGeoAdmin):
    """
    • Inherits OSMGeoAdmin → interactive map for osm_location.  
    • Inline shows which RMA offices the competitor is near.
    """
    list_display = (
        "osm_id",
        "osm_type",
        "name_from_osm",
        "osm_location",
        "first_seen_globally",
        "last_seen_globally",
        "age_days",
    )
    list_filter = ("osm_type", "first_seen_globally")
    search_fields = (
        "osm_id",
        "name_from_osm",
        "address_from_osm",
        "tags_from_osm__name",   # JSON → icontains works on most DBs
    )

    readonly_fields = (
        "osm_id",
        "osm_type",
        "first_seen_globally",
        "last_seen_globally",
        "age_days",
    )

    raw_id_fields = ()   # none; Point field already handled by OSMGeoAdmin
    list_per_page = 50
    default_lon =  -6.8400        # centre the OpenLayers widget on Morocco
    default_lat =  33.9600
    default_zoom = 6

    inlines = (AgencyOSMCompetitorProximityInline,)

    # ---------- helper columns ------------------------------------ #
    @admin.display(description="Address", ordering="address_from_osm")
    def pretty_address(self, obj):
        if obj.address_from_osm:
            return obj.address_from_osm[:60] + ("…" if len(obj.address_from_osm) > 60 else "")
        return "—"

    @admin.display(description="Age (days)")
    def age_days(self, obj):
        return (now() - obj.first_seen_globally).days


# ------------------------------------------------------------------ #
#  PROXIMITY ADMIN                                                   #
# ------------------------------------------------------------------ #
@admin.register(AgencyOSMCompetitorProximity)
class AgencyOSMCompetitorProximityAdmin(admin.ModelAdmin):
    """
    One row per (RMA office ↔ OSM competitor) pair.
    """
    list_display = (
        "osm_competitor",
        "rma_office",
        "first_seen_near_office",
        "last_seen_near_office",
    )
    list_filter = ("rma_office__city",)
    search_fields = (
        "osm_competitor__name_from_osm",
        "osm_competitor__address_from_osm",
        "rma_office__name",
        "rma_office__city",
    )
    raw_id_fields = ("osm_competitor", "rma_office")
    readonly_fields = ("first_seen_near_office", "last_seen_near_office")
    list_per_page = 50