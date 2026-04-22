# network_platform/urls.py
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/',     admin.site.urls),
    path('',           include('analyzer.urls')),   # 前端頁面
    path('api/',       include('api.urls')),         # REST API
]

# 開發環境提供 media 檔案（PCAP / 圖表）
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL,  document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
