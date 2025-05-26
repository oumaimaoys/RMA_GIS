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

class Area(models.Model):
    """Population data by area."""
    name = models.CharField(max_length=255, serialize=True)
    boundary = models.MultiPolygonField(srid=4326, serialize=True)
    population = models.IntegerField(serialize=True)
    insured_population = models.IntegerField(default=0, editable=False, serialize=True)
    estimated_vehicles = models.IntegerField(default=0, editable=False, serialize=True)
    competition_count = models.IntegerField(default=0, editable=False, serialize=True)
    bank_count = models.IntegerField(default=0, editable=False, serialize=True)

    def save(self, *args, **kwargs):
        # recalc before saving
        self.estimated_vehicles = int(self.population * 0.1147)
        self.insured_population = int(self.population * 0.17)

        if self.boundary:
            self.competition_count = Competitor.objects.filter(location__within=self.boundary).count()
            self.bank_count = Bank.objects.filter(location__within=self.boundary).count()

        super().save(*args, **kwargs)

    def __str__(self):
        return self.name
    
class Commune(Area):
    pass
class Province(Area):
    pass
    
class LossRatio(models.Model): # sinistralité
    province = models.ForeignKey(Province, on_delete=models.CASCADE)
    commune = models.ForeignKey(Commune, on_delete=models.CASCADE)
    RMA_office = models.ForeignKey(RMAOffice, on_delete=models.CASCADE)
    loss_ratio = models.FloatField(default=0)


class CoverageScore(models.Model):
    """Scoring system for network coverage"""
    area = models.ForeignKey(Area, on_delete=models.CASCADE, related_name='coverage_scores')
    score = models.FloatField(validators=[MinValueValidator(0), MaxValueValidator(100)],
                             help_text="Coverage score (0-100)")
    potential = models.CharField(max_length=50, choices=[
        ('HIGH', 'High'),
        ('MEDIUM', 'Medium'),
        ('LOW', 'Low'),
    ], help_text="Potential for coverage improvement", default='MEDIUM')
    
    calculation_date = models.DateTimeField(auto_now=True)
    
    
    class Meta:
        unique_together = ('area', 'calculation_date')
    
    def __str__(self):
        return self.area