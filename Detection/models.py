from django.db import models

# Create your models here.

class Model_path(models.Model):
    nom = models.CharField(max_length=30, verbose_name='nom chemin', null=True, blank=True)
    chemin = models.CharField(max_length=100, verbose_name='chemin acces', null=True, blank=True)