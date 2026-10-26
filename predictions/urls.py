from django.urls import path
from .views import health, home, predict_price

urlpatterns = [
    path("health/", health, name="health"),
    path("", home, name="home"),
    path("predict/", predict_price, name="predict_price"),
]