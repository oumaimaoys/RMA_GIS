from django import forms
from django.forms.widgets import DateInput
from .models import RMABGD
from django.contrib.gis.geos import Point

class CustomDateInput(DateInput):
    input_type = 'text'
    
    def __init__(self, attrs=None, format=None):
        if format is None:
            format = '%d-%m-%Y'
        self.format = format
        super().__init__(attrs={'placeholder': 'dd-mm-yyyy', 'class': 'datepicker', **(attrs or {})}, 
                         format=format)

class RMABGDForm(forms.ModelForm):
    class Meta:
        model = RMABGD
        fields = '__all__'
        widgets = {
            'date_creation': CustomDateInput(),
        }
        
    def clean_date_creation(self):
        """Custom validation for date_creation field"""
        date = self.cleaned_data.get('date_creation')
        # Django's form handling should already convert this to a Python date
        # if the format is correct, but you can add additional validation here
        return date
    


class LatLonPointMixin(forms.ModelForm):
    """
    Adds numeric latitude/longitude inputs and converts them
    to / from a GeoDjango Point (SRID 4326).
    """

    latitude  = forms.FloatField(
        label="Latitude (-90 … 90)",
        widget=forms.NumberInput(attrs={"step": "any"}),
        required=True,
    )
    longitude = forms.FloatField(
        label="Longitude (-180 … 180)",
        widget=forms.NumberInput(attrs={"step": "any"}),
        required=True,
    )

    class Meta:
        abstract = True          # <—  key point

        # location will be injected in `clean`, so hide it
        exclude = ("location",)

    # ---------- initialise lat/lon on the change page ----------
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        loc = getattr(self.instance, "location", None)
        if loc:
            self.fields["latitude"].initial  = loc.y
            self.fields["longitude"].initial = loc.x

    # ---------- build a Point from lat / lon ----------
    def clean(self):
        cleaned = super().clean()
        lat = cleaned.get("latitude")
        lon = cleaned.get("longitude")

        if lat is not None and lon is not None:
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                raise forms.ValidationError("Latitude or longitude out of range.")
            cleaned["location"] = Point(lon, lat, srid=4326)

        return cleaned

    def save(self, commit=True):
        # make sure the instance gets the Point
        self.instance.location = self.cleaned_data["location"]
        return super().save(commit)
