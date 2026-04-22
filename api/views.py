# api/views.py - REST API View 類別
import os
import json
import threading
import requests
from django.conf import settings
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser

import sys
sys.path.insert(0, str(settings.BASE_DIR / 'core'))

from analyzer.models import AnalysisSession, Alert, CNNResult

# 全域即時擷取狀態（實際部署建議改用 Redis）
_live_capture_thread = None
_live_stats = {
    'running': False,
    'packet_count': 0,
    'pps': 0,
    'alerts': [],
    'protocols': {},
}


class SessionListAPI(APIView):
    """GET /api/sessions/ - 列出所有 Session"""
    def get(self, request):
        sessions = AnalysisSession.objects.all().values(
            'id', 'mode', 'label', 'created_at',
            'packet_count', 'alert_count'
        )
        return Response(list(sessions))


class SessionDetailAPI(APIView):
    """GET /api/sessions/<pk>/ - 單一 Session 詳情"""
    def get(self, request, pk):
        try:
            session = AnalysisSession.objects.get(pk=pk)
        except AnalysisSession.DoesNotExist:
            return Response({'error': 'Not found'}, status=404)

        alerts = list(session.alerts.values(
            'attack_type', 'severity', 'src_ip', 'detail', 'timestamp'
        ))
        return Response({
            'id':           session.id,
            'mode':         session.mode,
            'label':        session.label,
            'created_at':   session.created_at,
            'packet_count': session.packet_count,
            'alert_count':  session.alert_count,
            'alerts':       alerts,
        })


class AlertListAPI(APIView):
    """GET /api/alerts/ - 列出所有告警（可用 ?severity=HIGH 過濾）"""
    def get(self, request):
        qs = Alert.objects.all()
        severity = request.query_params.get('severity')
        if severity:
            qs = qs.filter(severity=severity.upper())
        return Response(list(qs.values(
            'id', 'session_id', 'attack_type', 'severity',
            'src_ip', 'detail', 'timestamp'
        )))


class LiveStatsAPI(APIView):
    """GET /api/live/stats/ - 即時監控統計（前端每秒輪詢）"""
    def get(self, request):
        return Response(_live_stats)


class LiveStartAPI(APIView):
    """POST /api/live/start/ - 啟動即時擷取"""
    def post(self, request):
        global _live_capture_thread, _live_stats

        if _live_stats['running']:
            return Response({'message': '已在擷取中'})

        interface = request.data.get('interface', '')

        # 啟動背景擷取執行緒
        _live_stats['running']      = True
        _live_stats['packet_count'] = 0
        _live_stats['alerts']       = []
        _live_stats['protocols']    = {}

        def capture_worker():
            from capture import LiveCapture
            cap = LiveCapture(
                interface    = interface,
                count        = 0,
                stats_interval = 5,
            )
            cap.start()

        _live_capture_thread = threading.Thread(
            target=capture_worker, daemon=True
        )
        _live_capture_thread.start()

        return Response({'message': f'已啟動擷取：{interface}'})


class LiveStopAPI(APIView):
    """POST /api/live/stop/ - 停止即時擷取"""
    def post(self, request):
        _live_stats['running'] = False
        return Response({'message': '已停止擷取'})


class CNNAnalyzeAPI(APIView):
    """
    POST /api/cnn/analyze/
    上傳 PCAP 或使用現有 Session，執行 CNN Autoencoder 異常分析
    """
    parser_classes = [MultiPartParser]

    def post(self, request):
        session_id = request.data.get('session_id')
        if not session_id:
            return Response({'error': '請提供 session_id'}, status=400)

        try:
            session = AnalysisSession.objects.get(pk=session_id)
        except AnalysisSession.DoesNotExist:
            return Response({'error': 'Session 不存在'}, status=404)

        model_path = str(settings.CNN_MODEL_PATH)
        if not os.path.exists(model_path):
            return Response({'error': 'CNN 模型尚未訓練，請先執行 run_training.py'}, status=400)

        try:
            import torch
            from cnn_autoencoder import CNNAutoencoder
            from packet_visualizer import PacketVisualizer
            from scapy.all import rdpcap

            # 載入模型
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model  = CNNAutoencoder(latent_dim=settings.CNN_LATENT_DIM)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()

            # 讀取 PCAP 並影像化
            pcap_path = str(settings.MEDIA_ROOT / session.pcap_file.name)
            packets   = rdpcap(pcap_path)
            vis       = PacketVisualizer('medium', apply_mask=True, skip_ethernet=True)

            import numpy as np
            images = []
            for pkt in packets:
                try:
                    img = vis.bytes_to_image(bytes(pkt))
                    images.append(img)
                except Exception:
                    continue

            if not images:
                return Response({'error': '無法從 PCAP 提取影像'}, status=400)

            X = np.array(images, dtype=np.float32)
            X_tensor = torch.from_numpy(X[:, np.newaxis, :, :]).to(device)

            with torch.no_grad():
                errors = model.reconstruction_error(X_tensor).cpu().numpy()

            threshold    = float(settings.CNN_THRESHOLD)
            anomaly_mask = errors > threshold
            n_normal  = int((~anomaly_mask).sum())
            n_anomaly = int(anomaly_mask.sum())

            # 儲存結果
            cnn_result, _ = CNNResult.objects.update_or_create(
                session=session,
                defaults={
                    'threshold':        threshold,
                    'normal_count':     n_normal,
                    'anomaly_count':    n_anomaly,
                    'avg_normal_error': float(errors[~anomaly_mask].mean()) if n_normal > 0 else 0,
                    'avg_attack_error': float(errors[anomaly_mask].mean())  if n_anomaly > 0 else 0,
                }
            )

            return Response({
                'session_id':     session.id,
                'threshold':      threshold,
                'total_packets':  len(errors),
                'normal_count':   n_normal,
                'anomaly_count':  n_anomaly,
                'detection_rate': n_anomaly / (len(errors) + 1e-9),
            })

        except Exception as e:
            return Response({'error': str(e)}, status=500)


class CNNResultAPI(APIView):
    """GET /api/cnn/result/<pk>/ - 取得 CNN 分析結果"""
    def get(self, request, pk):
        try:
            result = CNNResult.objects.get(session_id=pk)
        except CNNResult.DoesNotExist:
            return Response({'error': 'CNN 結果不存在'}, status=404)

        return Response({
            'threshold':        result.threshold,
            'normal_count':     result.normal_count,
            'anomaly_count':    result.anomaly_count,
            'avg_normal_error': result.avg_normal_error,
            'avg_attack_error': result.avg_attack_error,
        })


class AIChatAPI(APIView):
    """
    POST /api/ai/chat/
    轉發使用者訊息到 n8n Webhook，n8n 呼叫 Gemini API 後回傳答案

    n8n 設定方式：
      1. 在 n8n 建立 Webhook 節點，取得 Webhook URL
      2. 將 URL 填入 settings.py 的 N8N_WEBHOOK_URL
      3. n8n 工作流程：Webhook → HTTP Request（Gemini API）→ Respond to Webhook
    """
    N8N_WEBHOOK_URL = 'http://localhost:5678/webhook/ai-chat'  # 替換為實際 URL

    def post(self, request):
        message    = request.data.get('message', '').strip()
        session_id = request.data.get('session_id')
        context    = ''

        if not message:
            return Response({'error': '請輸入訊息'}, status=400)

        # 若有 Session，附帶分析結果作為 context
        if session_id:
            try:
                session = AnalysisSession.objects.get(pk=session_id)
                alerts  = session.alerts.all()[:5]
                context = (
                    f"當前分析 Session：{session.label}，"
                    f"共 {session.packet_count} 個封包，"
                    f"發現 {session.alert_count} 個告警。"
                    f"告警類型：{', '.join(a.attack_type for a in alerts)}"
                )
            except AnalysisSession.DoesNotExist:
                pass

        # 轉發到 n8n
        try:
            resp = requests.post(
                self.N8N_WEBHOOK_URL,
                json={'message': message, 'context': context},
                timeout=30
            )
            data = resp.json()
            return Response({'reply': data.get('reply', data.get('text', ''))})
        except requests.exceptions.ConnectionError:
            return Response({
                'reply': 'AI 服務暫時無法連線，請確認 n8n 是否已啟動。'
            })
        except Exception as e:
            return Response({'error': str(e)}, status=500)
