# api/urls.py - REST API 端點
from django.urls import path
from . import views

app_name = 'api'

urlpatterns = [
    # Session
    path('sessions/',              views.SessionListAPI.as_view(),   name='session_list'),
    path('sessions/<int:pk>/',     views.SessionDetailAPI.as_view(), name='session_detail'),

    # 告警
    path('alerts/',                views.AlertListAPI.as_view(),     name='alert_list'),

    # 即時統計（即時監控頁面輪詢用）
    path('live/stats/',            views.LiveStatsAPI.as_view(),     name='live_stats'),
    path('live/start/',            views.LiveStartAPI.as_view(),     name='live_start'),
    path('live/stop/',             views.LiveStopAPI.as_view(),      name='live_stop'),

    # CNN 分析
    path('cnn/analyze/',           views.CNNAnalyzeAPI.as_view(),    name='cnn_analyze'),
    path('cnn/result/<int:pk>/',   views.CNNResultAPI.as_view(),     name='cnn_result'),

    # AI Agent（n8n Webhook 中繼）
    path('ai/chat/',               views.AIChatAPI.as_view(),        name='ai_chat'),
]
