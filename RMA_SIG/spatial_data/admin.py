from django.contrib import admin
from django.contrib.gis.admin import OSMGeoAdmin
from django.urls import path
from django.shortcuts import render, redirect
from django import forms
from django.contrib import messages
from .models import RMAOffice, RMABGD, RMAAgent, Bank, Competitor, Area, CoverageScore, Commune, Province, LossRatio
import pandas as pd
from django.http import HttpResponseRedirect
from .forms import RMABGDForm
from django.contrib.gis.db import models
from django.contrib import admin
from django.utils.html import format_html
from datetime import datetime
import json
from django.contrib.gis.geos import GEOSGeometry, MultiPolygon, Point

class CsvImportForm(forms.Form):
    csv_file = forms.FileField()


class BaseModelAdmin(OSMGeoAdmin):
    pass



@admin.register(RMABGD)
class RMABGDAdmin(BaseModelAdmin):
    list_display = ('name', 'code_RMA', 'code_ACAPS', 'address', 'city','location', 'type_BGD', 'Partenaire', 'formatted_date_creation', 'RMA_BGD_state')
    list_filter = ('type_BGD', 'RMA_BGD_state', 'city')
    search_fields = ('name', 'code_RMA', 'Partenaire')
    form = RMABGDForm  

    add_form_template = 'admin/spatial_data/RMABGD/change_form.html'
    change_form_template = add_form_template
    
    # Set date format for normal Django admin forms
    formfield_overrides = {
        models.DateField: {'widget': forms.DateInput(format='%d-%m-%Y', attrs={'placeholder': 'dd-mm-yyyy'})},
    }
    
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
class RMAAgentAdmin(BaseModelAdmin):
    list_display = ('name', 'code_RMA', 'code_ACAPS', 'address', 'city','location')
    list_filter = ('city',)
    search_fields = ('name', 'code_RMA', 'address', 'city')

    add_form_template = 'admin/spatial_data/RMAAgent/change_form.html'
    change_form_template = add_form_template
    

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
class BankAdmin(BaseModelAdmin):
    list_display = ('institution_name',)
    search_fields = ('institution_name',)

    add_form_template = "admin/spatial_data/Bank/change_form.html"
    change_form_template = add_form_template
    

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
class CompetitorAdmin(BaseModelAdmin):
    list_display = ('company_name', 'code_ACAPS', 'competitor_type', 'mandante', 'city')
    list_filter = ('competitor_type', 'mandante')
    search_fields = ('company_name', 'code_ACAPS', 'mandante', 'city',)

    add_form_template = "admin/spatial_data/Competitor/change_form.html"
    change_form_template = add_form_template
    

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
    list_display = ('name', 'population',  'insured_population', 'estimated_vehicles')
    search_fields = ('name',)

    add_form_template = "admin/spatial_data/Commune/change_form.html"
    change_form_template = add_form_template

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
    list_display = ('name', 'population', 'insured_population', 'estimated_vehicles')
    search_fields = ('name',)

    add_form_template = "admin/spatial_data/Province/change_form.html"
    change_form_template = add_form_template

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
    add_form_template = change_form_template = "admin/spatial_data/LossRatio/change_form.html"

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
    list_display = ('area', 'score', 'potential', "calculation_date")
    list_filter = ('area', 'potential')
    search_fields = ('area__name',)
