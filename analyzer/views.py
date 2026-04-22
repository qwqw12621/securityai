# analyzer/views.py
# 前端頁面的 View 函式
import os
import json
from django.shortcuts import render, get_object_or_404, redirect
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from .models import AnalysisSession, Alert, CNNResult

# ── 核心引擎匯入（來自 core/ 目錄）────────────────────────
import sys
sys.path.insert(0, str(settings.BASE_DIR / 'core'))

from pcap_analyzer import PcapAnalyzer
from anomaly_detector import AnomalyDetector
from packet_visualizer import PacketVisualizer


def dashboard(request):
    """首頁儀表板：顯示最近 Session 與統計摘要"""
    recent_sessions = AnalysisSession.objects.all()[:10]
    total_alerts    = Alert.objects.count()
    total_sessions  = AnalysisSession.objects.count()

    # 告警嚴重程度分布（供 Chart.js 圓餅圖使用）
    severity_data = {
        'CRITICAL': Alert.objects.filter(severity='CRITICAL').count(),
        'HIGH':     Alert.objects.filter(severity='HIGH').count(),
        'MEDIUM':   Alert.objects.filter(severity='MEDIUM').count(),
        'LOW':      Alert.objects.filter(severity='LOW').count(),
    }

    return render(request, 'analyzer/dashboard.html', {
        'recent_sessions': recent_sessions,
        'total_alerts':    total_alerts,
        'total_sessions':  total_sessions,
        'severity_data':   json.dumps(severity_data),
    })


def upload_pcap(request):
    """
    PCAP 上傳與分析頁面

    GET ：顯示上傳表單
    POST：接收 PCAP 檔案，執行分析，跳轉到結果頁面
    """
    if request.method == 'POST':
        pcap_file = request.FILES.get('pcap_file')
        label     = request.POST.get('label', '')

        if not pcap_file:
            return render(request, 'analyzer/upload.html',
                          {'error': '請選擇 PCAP 檔案'})

        # 建立 Session 紀錄
        session = AnalysisSession.objects.create(
            mode      = 'pcap',
            label     = label or pcap_file.name,
            pcap_file = pcap_file,
        )

        # 執行封包分析
        pcap_path = str(settings.MEDIA_ROOT / session.pcap_file.name)
        analyzer  = PcapAnalyzer(pcap_path)
        analyzer.load()
        analyzer.summary()

        # 規則式攻擊偵測
        alerts_raw = analyzer.detect_attacks()

        # 儲存告警到資料庫
        for a in (alerts_raw or []):
            Alert.objects.create(
                session     = session,
                attack_type = a.get('attack_type', ''),
                severity    = a.get('severity', 'LOW'),
                src_ip      = a.get('src_ip', ''),
                detail      = a.get('detail', ''),
                suggestion  = a.get('suggestion', ''),
            )

        # 更新統計
        session.alert_count = session.alerts.count()
        session.save()

        return redirect('analyzer:session_detail', pk=session.pk)

    return render(request, 'analyzer/upload.html')


def session_list(request):
    """歷史 Session 列表"""
    sessions = AnalysisSession.objects.all()
    return render(request, 'analyzer/sessions.html', {'sessions': sessions})


def session_detail(request, pk):
    """單一 Session 的詳細分析結果"""
    session = get_object_or_404(AnalysisSession, pk=pk)
    alerts  = session.alerts.all()
    return render(request, 'analyzer/session_detail.html', {
        'session': session,
        'alerts':  alerts,
    })


def live_monitor(request):
    """即時監控頁面（前端透過 WebSocket 或輪詢 API 取得資料）"""
    return render(request, 'analyzer/live.html')


def ai_chat(request):
    """AI Agent 對話頁面（串接 n8n Webhook）"""
    return render(request, 'analyzer/ai_chat.html')
