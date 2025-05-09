from django.contrib.gis.db import models
from django.core.validators import MinValueValidator, MaxValueValidator

# models

class RMAOffice(models.Model):
    """RMA Network Offices"""
    code_ACAPS = models.CharField(max_length=50, unique=True)
    code_RMA = models.CharField(max_length=50, unique=True)
    name = models.CharField(max_length=255)
    address = models.CharField(max_length=255)
    city = models.CharField(max_length=100)
    location = models.PointField()
    
    
    def __str__(self):
        return self.name
    
class RMABGD(RMAOffice):
    type_BGD = models.CharField(max_length=100)
    Partenaire = models.CharField(max_length=255)
    date_creation = models.DateField(null=True, blank=True)
    RMA_BGD_state = models.CharField(max_length=100)
    
    # Add this property to get formatted date
    @property
    def formatted_date(self):
        if self.date_creation:
            return self.date_creation.strftime('%d-%m-%Y')
        return ""
    
    class Meta:
        verbose_name = "RMA BGD"
        verbose_name_plural = "RMA BGDs"
        
    def __str__(self):
        return self.name

class RMAAgent(RMAOffice):
    pass

class manager(models.Model): #responsable_salarié
    CIN = models.CharField(max_length=10)
    first_name = models.CharField(max_length=50)
    last_name = models.CharField(max_length=50)
    date_of_birth = models.DateField()
    

class Bank(models.Model):
    """Bank locations"""
    institution_name = models.CharField(max_length=255)
    location=models.PointField()

    def __str__(self):
        return f"{self.institution_name}"

class Competitor(models.Model):
    """Competitor locations"""
    code_ACAPS = models.CharField(max_length=50, unique=True)
    company_name = models.CharField(max_length=255)
    mandante = models.CharField(max_length=255)
    #market_share = models.FloatField(null=True, blank=True, 
    #   help_text="Estimated market share percentage",
    #   validators=[MinValueValidator(0), MaxValueValidator(100)])
    competitor_type = models.CharField(max_length=50, choices=[
        ('AGENT', 'agent'),
        ('BGD', 'bgd'),
        ('COURTIER', 'courtier'),
    ])
    address = models.CharField(max_length=255)
    city = models.CharField(max_length=100)
    location = models.PointField()
    
    def __str__(self):
        return self.company_name

class PopulationArea(models.Model):
    """Population data by commune"""
    name = models.CharField(max_length=255)
    boundary = models.MultiPolygonField(srid=4326)
    total_population = models.IntegerField()
    total_vihicules = models.IntegerField()
    date_updated = models.DateField(help_text="Date this population data was updated")
    
    
    def __str__(self):
        return f"{self.name}"
    
class LossExperience(models.Model): # sinistralité
    area = models.ForeignKey(PopulationArea, on_delete=models.CASCADE)
    loss_experience_rate = models.FloatField()


class CoverageScore(models.Model):
    """Scoring system for network coverage"""
    area = models.ForeignKey(PopulationArea, on_delete=models.CASCADE, related_name='coverage_scores')
    score = models.FloatField(validators=[MinValueValidator(0), MaxValueValidator(100)],
                             help_text="Coverage score (0-100)")
    population_covered = models.IntegerField(help_text="Estimated population covered by RMA network")
    coverage_percentage = models.FloatField(validators=[MinValueValidator(0), MaxValueValidator(100)],
                                         help_text="Percentage of population covered")
    nearest_office = models.ForeignKey(RMAOffice, on_delete=models.SET_NULL, null=True, 
                                      related_name='primary_coverage_areas')
    calculation_date = models.DateTimeField(auto_now=True)
    
    # Additional scoring factors
    competitor_factor = models.FloatField(default=0, 
        help_text="Impact of competitors on the score (-10 to +10)",
        validators=[MinValueValidator(-10), MaxValueValidator(10)])
    bank_partnership_factor = models.FloatField(default=0, 
        help_text="Impact of bank partnerships on the score (0 to +20)",
        validators=[MinValueValidator(0), MaxValueValidator(20)])
    
    class Meta:
        unique_together = ('area', 'calculation_date')
    
    def __str__(self):
        return self.area