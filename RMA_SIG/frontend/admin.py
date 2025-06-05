from django.contrib import admin
from frontend.models import Notification

# Register your models here.
@admin.register(Notification)
class NotificationAdmin(admin.ModelAdmin):
    list_display  = ("verb", "description", "recipient", "created_at", "read")
    list_filter   = ("read", "created_at")
    search_fields = ("description", "recipient__username")
    raw_id_fields = ("recipient",)