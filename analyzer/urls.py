# analyzer/urls.py
from django.urls import path
from . import views

app_name = 'analyzer'

urlpatterns = [
    path('',            views.dashboard,      name='dashboard'),    # 首頁儀表板
    path('upload/',     views.upload_pcap,    name='upload'),       # 上傳 PCAP
    path('sessions/',   views.session_list,   name='sessions'),     # 歷史紀錄
    path('sessions/<int:pk>/', views.session_detail, name='session_detail'),
    path('live/',       views.live_monitor,   name='live'),         # 即時監控
    path('ai-chat/',    views.ai_chat,        name='ai_chat'),      # AI Agent 對話
]
