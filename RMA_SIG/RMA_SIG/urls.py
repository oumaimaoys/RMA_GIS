"""
URL configuration for RMA_SIG project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from spatial_data import views
from django.urls import path, include
from django.views.generic import TemplateView
from django.contrib.auth.decorators import login_required

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('frontend.urls')),
    path('__debug__/', include('debug_toolbar.urls')),  # Add this line
    # 1) Auth endpoints: login, logout
    path('accounts/', include('django.contrib.auth.urls')),

    # 2) Home page (requires login)
    path(
        '',
        login_required(TemplateView.as_view(template_name='home.html')),
        name='home'
    ),
    path("/", include("frontend.urls")),      # or point directly at frontend.views.map_view
    path('spatial/', include('spatial_data.urls')),  # ✅ this must be present

]
