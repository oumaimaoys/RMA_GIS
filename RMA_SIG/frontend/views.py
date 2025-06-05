from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect, get_object_or_404
from .models import Notification
from django.http import JsonResponse, HttpResponseRedirect
from django.shortcuts import get_object_or_404, resolve_url

def map_view(request):
    return render(request, "frontend/map.html")

# frontend/views.py


@login_required
def notif_list(request):
    qs = request.user.notifications.all()
    return render(request, "notifications/list.html", {"notifications": qs})

@login_required
def notif_read(request, pk):
    n = get_object_or_404(Notification, pk=pk, recipient=request.user)
    n.mark_read()
    return redirect(n.url or "notifications:list")


from django.shortcuts import get_object_or_404, redirect, resolve_url
from django.contrib.auth.decorators import login_required

@login_required
def read_and_redirect_notification(request, pk):
    notification = get_object_or_404(Notification, pk=pk, user=request.user)
    if not notification.read:
        notification.read = True
        notification.save(update_fields=['read'])

    # Assuming your Notification model has a 'target_url' or similar
    # or can derive it from related objects
    if hasattr(notification, 'get_absolute_url_for_target'): # Example method
        redirect_url = notification.get_absolute_url_for_target()
    elif notification.target_url: # If you store a URL directly
         redirect_url = notification.target_url
    else:
        # Fallback to the main notification list or a default page
        redirect_url = resolve_url('frontend:list')

    return redirect(redirect_url)
@login_required
def mark_all_read(request):
    if request.method == "POST":
        request.user.notifications.filter(read=False).update(read=True)
        if request.headers.get("Hx-Request") or request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return JsonResponse({"ok": True})      # AJAX
    return HttpResponseRedirect(request.META.get("HTTP_REFERER", resolve_url("frontend:list")))

@login_required
def toggle_read(request, pk):
    """
    Flip the read flag for one notif and return JSON with the new state
    """
    n = get_object_or_404(Notification, pk=pk, recipient=request.user)
    n.read = not n.read
    n.save(update_fields=["read"])
    return JsonResponse({"read": n.read})