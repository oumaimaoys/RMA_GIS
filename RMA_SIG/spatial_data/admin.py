from django.contrib import admin
from django.contrib.gis.admin import OSMGeoAdmin
from django.urls import path
from django.shortcuts import render, redirect
from django import forms
from django.contrib import messages
from .models import RMAOffice, RMABGD, RMAAgent, Bank, Competitor, PopulationArea, CoverageScore
import pandas as pd
from django.http import HttpResponseRedirect
from .forms import RMABGDForm
from django.contrib.gis.db import models
from django.contrib import admin
from django.utils.html import format_html
from datetime import datetime

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

            required = ["Code RMA","Code ACAPS","Dénomination RMA","Ville",
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
                        code_ACAPS=row["Code ACAPS"],
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
            required = ["code RMA", "code ACAPS", "Dénomination RMA", "Ville", "Adresse", "Longitude", "Latitude"]

            missing = [h for h in required if h not in df.columns]
            if missing:
                messages.error(request, f"Missing columns: {', '.join(missing)}")
                return False

            imported = 0
            for i, row in df.iterrows():
                try:
                    RMAAgent.objects.create(
                        code_ACAPS = row["code ACAPS"],
                        code_RMA = row["code RMA"],
                        name = row["Dénomination RMA"],
                        address = row["Adresse"],
                        city = row["Ville"],
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
    list_filter = ('competitor_type', 'city')
    search_fields = ('company_name', 'code_ACAPS', 'mandante', 'city')

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



@admin.register(PopulationArea)
class PopulationAreaAdmin(BaseModelAdmin):
    list_display = ('name', 'total_population', 'estimated_vehicles')
    search_fields = ('name',)

    add_form_template = "admin/spatial_data/PopulationArea/change_form.html"
    change_form_template = add_form_template
    

    def process_excel_import(self, request):
        f = request.FILES.get('excel_file')
        if not f:
            messages.error(request, "Please choose an Excel file.")
            return False

        try:
            df = pd.read_excel(f)
            required = ["Collectivités territoriales", "Population"]

            missing = [h for h in required if h not in df.columns]
            if missing:
                messages.error(request, f"Missing columns: {', '.join(missing)}")
                return False

            imported = 0
            for i, row in df.iterrows():
                try:
                    PopulationArea.objects.create(
                        province_name=row['Collectivités territoriales'],
                        population=row['Population'],
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



@admin.register(CoverageScore)
class CoverageScoreAdmin(admin.ModelAdmin):
    list_display = ('area', 'score', 'coverage_percentage', 'population_covered', 'calculation_date')
    list_filter = ('calculation_date',)

    change_list_template = "admin/import_change_list.html"