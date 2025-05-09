from django import forms
from django.forms.widgets import DateInput
from .models import RMABGD

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