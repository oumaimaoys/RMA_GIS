# frontend/context_processors.py
from .models import Notification # Make sure this import path is correct for your Notification model

def notifications_processor(request):
    if request.user.is_authenticated:
        unread_notifications = request.user.notifications.filter(read=False).order_by('-created_at')
        # You might want to limit the number of notifications shown in the dropdown, e.g., [:5]
        # unread_notifications_dropdown = unread_notifications[:5]
        return {
            'unread_notifications': unread_notifications, # or unread_notifications_dropdown
            'unread_notifications_count': unread_notifications.count(),
        }
    return {}

def unread_notifications(request):
    if request.user.is_authenticated:
        qs = request.user.notifications
        return {
            "unread_notifications_count": qs.filter(read=False).count(),
            "unread_notifications"      : qs.order_by("-created_at")[:5],  # for dropdown
        }
    return {}