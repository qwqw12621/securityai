# analyzer/models.py
# 資料庫模型：記錄每次分析的 Session 與告警歷史
from django.db import models


class AnalysisSession(models.Model):
    """每次 PCAP 上傳 / 即時擷取對應一個 Session"""
    MODE_CHOICES = [('pcap', 'PCAP 離線分析'), ('live', '即時擷取')]

    mode       = models.CharField(max_length=10, choices=MODE_CHOICES)
    label      = models.CharField(max_length=100, blank=True)
    pcap_file  = models.FileField(upload_to='uploads/', null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    packet_count = models.IntegerField(default=0)
    alert_count  = models.IntegerField(default=0)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"[{self.mode}] {self.label} ({self.created_at:%Y-%m-%d %H:%M})"


class Alert(models.Model):
    """單筆告警紀錄"""
    SEVERITY_CHOICES = [
        ('LOW',      'Low'),
        ('MEDIUM',   'Medium'),
        ('HIGH',     'High'),
        ('CRITICAL', 'Critical'),
    ]

    session     = models.ForeignKey(
        AnalysisSession, on_delete=models.CASCADE, related_name='alerts'
    )
    attack_type = models.CharField(max_length=50)
    severity    = models.CharField(max_length=10, choices=SEVERITY_CHOICES)
    src_ip      = models.CharField(max_length=50)
    detail      = models.TextField(blank=True)
    suggestion  = models.TextField(blank=True)
    timestamp   = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-timestamp']

    def __str__(self):
        return f"[{self.severity}] {self.attack_type} from {self.src_ip}"


class CNNResult(models.Model):
    """CNN Autoencoder 異常分析結果"""
    session          = models.OneToOneField(
        AnalysisSession, on_delete=models.CASCADE, related_name='cnn_result'
    )
    threshold        = models.FloatField()
    normal_count     = models.IntegerField(default=0)
    anomaly_count    = models.IntegerField(default=0)
    avg_normal_error = models.FloatField(default=0)
    avg_attack_error = models.FloatField(default=0)
    heatmap_image    = models.ImageField(upload_to='reports/', null=True, blank=True)
    created_at       = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"CNN Result for Session {self.session_id}"
