from django.contrib import admin
from Detection.models import Model_path


# Register your models here.

class model_pathAdmin(admin.ModelAdmin):
    list_display = ['nom', 'chemin']

admin.site.register(Model_path, model_pathAdmin)