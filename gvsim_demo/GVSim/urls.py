
from django.conf.urls import url

from .views import GVHome

urlpatterns = [
    url(r'^$', GVHome.as_view(), name='gvhome'),
]
