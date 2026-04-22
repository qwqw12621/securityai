from django.contrib import admin
from .models import AnalysisSession, Alert, CNNResult

@admin.register(AnalysisSession)
class SessionAdmin(admin.ModelAdmin):
    list_display  = ('label', 'mode', 'packet_count', 'alert_count', 'created_at')
    list_filter   = ('mode',)
    search_fields = ('label',)

@admin.register(Alert)
class AlertAdmin(admin.ModelAdmin):
    list_display  = ('attack_type', 'severity', 'src_ip', 'session', 'timestamp')
    list_filter   = ('severity', 'attack_type')
    search_fields = ('src_ip', 'attack_type')

@admin.register(CNNResult)
class CNNResultAdmin(admin.ModelAdmin):
    list_display = ('session', 'normal_count', 'anomaly_count', 'threshold')
