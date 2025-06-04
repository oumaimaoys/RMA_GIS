from django.contrib.gis.db import models
from django.core.validators import MinValueValidator, MaxValueValidator
from django.utils import timezone # For default timestamps

# --- Your existing models (RMAOffice, RMABGD, RMAAgent, manager, Bank, Competitor) ---
# These seem fine for their purpose and are mostly used as data sources
# for the counts/intensities in the Area model.
# ... (keep them as they are) ...

class RMAOffice(models.Model):
    """RMA Network Offices"""
    code_ACAPS = models.CharField(max_length=50, unique=True)
    code_RMA = models.CharField(max_length=50, unique=True)
    name = models.CharField(max_length=255)
    address = models.CharField(max_length=255)
    city = models.CharField(max_length=100)
    location = models.PointField() # srid=4326 is default

    def __str__(self):
        return self.name

class RMABGD(RMAOffice):
    type_BGD = models.CharField(max_length=100)
    Partenaire = models.CharField(max_length=255)
    date_creation = models.DateField(null=True, blank=True)
    RMA_BGD_state = models.CharField(max_length=100)

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

    def __str__(self):
        return f"{self.first_name} {self.last_name}"


class Bank(models.Model):
    """Bank locations"""
    institution_name = models.CharField(max_length=255)
    location = models.PointField()

    def __str__(self):
        return f"{self.institution_name}"

class Competitor(models.Model):
    """Competitor locations"""
    code_ACAPS = models.CharField(max_length=50, unique=True) # Assuming this is unique for competitors
    agency_name = models.CharField(max_length=255)
    mandante = models.CharField(max_length=255) # Parent company or brand
    competitor_type = models.CharField(max_length=50, choices=[
        ('AGENT', 'Agent'),
        ('BGD', 'BGD'), # Bureau de Gestion Délégué?
        ('COURTIER', 'Courtier/Broker'),
    ])
    address = models.CharField(max_length=255)
    city = models.CharField(max_length=100)
    location = models.PointField()

    def __str__(self):
        # return self.company_name # Original had company_name, changed to agency_name
        return f"{self.agency_name} ({self.mandante})"


# --- Core Area Model and its derivatives ---
class Area(models.Model):
    """Base model for geographical areas with demographic and competitive data."""
    name = models.CharField(max_length=255, unique=True) # Ensure name is unique if it's a primary identifier
    boundary = models.MultiPolygonField(srid=4326) # Explicit SRID is good

    # Core Data (often externally sourced or input)
    population = models.IntegerField(help_text="Total population in the area")

    # Calculated/Derived Data (can be updated via save() or periodic tasks)
    insured_population = models.IntegerField(default=0, editable=True, help_text="Estimated insured population")
    estimated_vehicles = models.IntegerField(default=0, editable=True, help_text="Estimated number of vehicles")
    
    competition_count = models.IntegerField(default=0, editable=False, help_text="Number of competitor branches in the area")
    bank_count = models.IntegerField(default=0, editable=False, help_text="Number of bank branches in the area")
    
    # Intensity Metrics (CRUCIAL - how these are defined matters greatly)
    bank_intensity = models.FloatField(default=0.0, editable=False, help_text="Banks per 10k population or similar metric")
    competition_intensity = models.FloatField(default=0.0, editable=False, help_text="Competitors per 10k population or similar metric")
    
    population_density = models.FloatField(null=True, blank=True, editable=False, help_text="Population per square kilometer")
    vehicle_density = models.FloatField(null=True, blank=True, editable=False, help_text="Vehicles per square kilometer")

    # You might also store a pre-calculated average loss ratio if it's stable enough
    # average_loss_ratio = models.FloatField(null=True, blank=True, editable=False)

    def update_derived_fields(self):
        """
        Updates fields that are derived from other data.
        Call this method explicitly when underlying data changes, or in save().
        """
        # Basic estimations (these formulas should be validated/improved)
        self.estimated_vehicles = int(self.population * Variables.objects.get(name="vehicles_factor")) # This factor needs to be robust
        self.insured_population = int(self.population * Variables.objects.get(name="insurable_population_ratio") ) # This factor needs to be robust

        if self.boundary:
            # Ensure boundary is in a planar projection for area calculation (e.g., a local UTM zone or Web Mercator)
            # For simplicity, if your boundary SRID allows area calculation directly, use it.
            # Otherwise, transform. Example using Web Mercator (EPSG:3857) for area.
            try:
                # Area calculations are more accurate in an equal-area projection,
                # but for consistency, if your PROJ_SRID is Web Mercator, use that.
                # If boundary is in 4326 (degrees), area calculation is tricky.
                boundary_proj = self.boundary.transform(3857, clone=True) # Using common PROJ_SRID from your script
                surface_area_sq_meters = boundary_proj.area
                surface_area_sq_km = surface_area_sq_meters / 1_000_000
                
                if surface_area_sq_km > 0:
                    self.population_density = self.population / surface_area_sq_km
                    self.vehicle_density = self.estimated_vehicles / surface_area_sq_km
                else:
                    self.population_density = None # Or 0, depending on how you want to handle tiny/invalid areas
                    self.vehicle_density = None
            except Exception as e:
                # Handle potential errors in transformation or area calculation
                print(f"Warning: Could not calculate density for {self.name}: {e}")
                self.population_density = None
                self.vehicle_density = None

            # Counts within the area
            self.competition_count = Competitor.objects.filter(location__within=self.boundary).count()
            self.bank_count = Bank.objects.filter(location__within=self.boundary).count()

            # Intensity Calculations (EXAMPLES - TUNE THESE DEFINITIONS)
            if self.population > 0:
                # Competitors per 10,000 population
                self.competition_intensity = (self.competition_count / self.population) * 10000
                # Banks per 10,000 population
                self.bank_intensity = (self.bank_count / self.population) * 10000
            else:
                self.competition_intensity = 0.0
                self.bank_intensity = 0.0
        else:
            self.population_density = None
            self.vehicle_density = None
            self.competition_count = 0
            self.bank_count = 0
            self.competition_intensity = 0.0
            self.bank_intensity = 0.0

    def save(self, *args, **kwargs):
        self.update_derived_fields() # Calculate before every save
        super().save(*args, **kwargs)

    def __str__(self):
        return self.name
    
    class Meta:
        abstract = False # Ensure Area itself can be queried if needed, though usually Commune/Province are used.


class Commune(Area):
    class Meta:
        verbose_name = "Commune"
        verbose_name_plural = "Communes"

class Province(Area):
    # Provinces might have specific fields or methods later
    class Meta:
        verbose_name = "Province"
        verbose_name_plural = "Provinces"


# --- Loss Ratio Model ---
class LossRatio(models.Model):
    """
    Stores loss ratio data, ideally at the most granular level available.
    The scoring script then aggregates this to the Area level (Commune/Province).
    """
    RMA_office = models.ForeignKey(RMAOffice, on_delete=models.SET_NULL, null=True, blank=True, related_name='lossratios', help_text="RMA Office this LR is associated with")
    area = models.ForeignKey(Area, on_delete=models.CASCADE, related_name='lossratio_records', null=True, blank=True)
    province = models.ForeignKey(Province, on_delete=models.CASCADE, related_name='province_lossratios', null=True, blank=True)
    commune = models.ForeignKey(Commune, on_delete=models.CASCADE, related_name='commune_lossratios', null=True, blank=True)
    
    loss_ratio = models.FloatField(default=0.0, validators=[MinValueValidator(0.0)], help_text="Loss Ratio (e.g., 0.65 for 65%)")
    
    


    def __str__(self):
        target_area = self.area or self.commune or self.province
        return f"LR {self.loss_ratio} for {target_area.name if target_area else 'N/A'} ({self.period_start_date} to {self.period_end_date})"

    class Meta:
        # Ensure a loss ratio for a given area and period is unique if that's the case
        # unique_together = ('area', 'period_start_date', 'period_end_date', 'product_line')
        verbose_name_plural = "Loss Ratios"


# --- Storing Calculated Scores ---
class CoverageScore(models.Model):
    """Stores the calculated suitability score for an area."""
    area = models.ForeignKey(Area, on_delete=models.CASCADE)
    
    # Overall Score and Potential
    score = models.FloatField(validators=[MinValueValidator(0), MaxValueValidator(100)],
                             help_text="Overall suitability score (0-100)")
    potential = models.CharField(max_length=50, choices=[
        ('EXCELLENT', 'Excellent'),
        ('GOOD', 'Good'),
        ('MEDIUM', 'Medium'),
        ('LOW', 'Low'),
    ], help_text="Categorized potential based on score")
    
    # Store Component Scores for analysis and history
    demand_score = models.FloatField(null=True, blank=True)
    competition_score = models.FloatField(null=True, blank=True)
    economic_score = models.FloatField(null=True, blank=True)
    accessibility_score = models.FloatField(null=True, blank=True)
    risk_score = models.FloatField(null=True, blank=True)
    travel_time_to_centroid_minutes = models.FloatField(null=True, blank=True)

    # Input parameters at the time of calculation (optional, for reproducibility)
    # latitude_input = models.FloatField(null=True, blank=True)
    # longitude_input = models.FloatField(null=True, blank=True)
    
    calculation_date = models.DateTimeField(default=timezone.now, help_text="Timestamp of when this score was calculated")
    
    class Meta:
        # If you want only the LATEST score per area, you might enforce uniqueness differently
        # or have a separate "LatestCoverageScore" model.
        # This setup allows multiple score records per area over time.
        ordering = ['-calculation_date'] # Show newest first
        # unique_together = ('area', 'calculation_timestamp') # Ensures one score per area per exact ms

    def __str__(self):
        return f"Score {self.score:.1f} ({self.potential}) for {self.area.name} on {self.calculation_date.strftime('%Y-%m-%d %H:%M')}"


# --- Storing Aggregate Statistics ---
class CoverageStats(models.Model):
    """Stores aggregate statistics (mean, std_dev) for various metrics across all areas.
       Used for normalization (e.g., Z-scores) in the scoring process.
       Should be updated periodically by a separate management command.
    """
    # Existing fields
    pop_mean    = models.FloatField(default=0.0)
    pop_std     = models.FloatField(default=0.0)
    gap_mean    = models.FloatField(default=0.0)
    gap_std     = models.FloatField(default=0.0)
    veh_mean    = models.FloatField(default=0.0)
    veh_std     = models.FloatField(default=0.0)
    bank_intensity_mean   = models.FloatField(default=0.0, verbose_name="Bank Intensity Mean") # Renamed for clarity
    bank_intensity_std    = models.FloatField(default=0.0, verbose_name="Bank Intensity Std Dev")
    comp_intensity_mean   = models.FloatField(default=0.0, verbose_name="Competition Intensity Mean") # Renamed
    comp_intensity_std    = models.FloatField(default=0.0, verbose_name="Competition Intensity Std Dev")
    loss_ratio_mean = models.FloatField(default=0.0, verbose_name="Loss Ratio Mean") # Added
    loss_ratio_std  = models.FloatField(default=0.0, verbose_name="Loss Ratio Std Dev") # Added

    # --- NEW FIELDS for new metrics ---
    market_potential_mean = models.FloatField(default=0.0, null=True, blank=True) # For 'market_potential_untapped'
    market_potential_std = models.FloatField(default=0.0, null=True, blank=True)

    # Type of area these stats are for (e.g., "Commune" or "Province")
    area_type = models.CharField(max_length=20, choices=[('Commune', 'Commune'), ('Province', 'Province')], default='Commune')
    
    calculation_date = models.DateTimeField(default=timezone.now, help_text="When these stats were computed")


class Variables(models.Model):
    """Stores variables used in the scoring process, such as weights and factors."""
    name = models.CharField(max_length=100, unique=True)
    value = models.FloatField(help_text="Value of the variable")
    description = models.TextField(blank=True, null=True, help_text="Description of what this variable is used for")
    
    def __str__(self):
        return f"{self.name}: {self.value}"