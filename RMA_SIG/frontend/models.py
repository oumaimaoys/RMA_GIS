from django.conf import settings
from django.db import models
from django.utils import timezone
# Create your models here.

class Notification(models.Model):
    """
    A generic in-app notification destined to a single user.
    Extend with a `target_content_type / object_id` pair if you need
    polymorphic “actors” later.
    """
    recipient   = models.ForeignKey(settings.AUTH_USER_MODEL,
                                    on_delete=models.CASCADE,
                                    related_name="notifications")
    verb        = models.CharField(max_length=140)               # “opened”
    description = models.TextField(blank=True)                   # free text
    url         = models.URLField(blank=True)                    # where to go
    created_at  = models.DateTimeField(default=timezone.now)
    read        = models.BooleanField(default=False)

    class Meta:
        ordering = ("-created_at",)

    def mark_read(self):
        if not self.read:
            self.read = True
            self.save(update_fields=["read"])