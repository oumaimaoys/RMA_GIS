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
    """
    Always inject two variables into the template context:
      • unread_notifications_count – the # of unread notifications
      • unread_notifications       – a QuerySet of the most recent unread notifications
    """
    if not request.user.is_authenticated:
        return {
            "unread_notifications_count": 0,
            "unread_notifications":       []
        }

    # Assuming your Notification model has a `read` boolean field,
    # a `created_at` DateTime, and you have related_name="notifications"
    # on the user ForeignKey. Adjust as needed.
    qs = request.user.notifications.all().order_by("-created_at")
    unread = qs.filter(read=False)
    return {
        "unread_notifications_count": unread.count(),
        "unread_notifications":       unread[:5],  # limit to latest 5
    }