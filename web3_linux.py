# -*- coding: utf-8 -*-
import cv2
import numpy as np
import threading
import time
import os
import multiprocessing
from queue import Queue
import json
import subprocess
from flask import Flask, Response, send_file, request

app = Flask(__name__)

NUM_CORES = max(1, multiprocessing.cpu_count())
RECOGNIZE_INTERVAL = 0.3
NUM_WORKER = min(2, NUM_CORES)

print(f"System cores: {NUM_CORES}, using {NUM_WORKER} worker threads")

CAMERA_IDS = [0, 1, 2, 3, 4, 5, 20, 21]
TEA_MODEL_PATH = "best.onnx"
TONGUE_MODEL_PATH = "tongue.q.onnx"

TEA_NAMES = {
    0: "Biluochun",
    1: "Jinjunmei",
    2: "Tieguanyin",
    3: "Gongmei",
    4: "Jasmine",
    5: "Gaoshan",
    6: "Zhengshan",
    7: "Shuixian",
    8: "Shoumei"
}

TEA_CN = {
    "Biluochun": "\u78a7\u87ba\u6625",
    "Jinjunmei": "\u91d1\u9a6c\u7709",
    "Tieguanyin": "\u94c1\u89c2\u97f3",
    "Gongmei": "\u8d21\u7709",
    "Jasmine": "\u8309\u8389\u82b1\u8336",
    "Gaoshan": "\u9ad8\u5c71\u8336",
    "Zhengshan": "\u6b63\u5c71\u5c0f\u79cd",
    "Shuixian": "\u6c34\u4ed9",
    "Shoumei": "\u5bff\u7709",
    "Unknown": "\u672a\u77e5"
}

TEA_INFO = {
    0: "\u78a7\u87ba\u6625\u662f\u6c5f\u82cf\u82cf\u5dde\u7684\u9876\u7ea7\u540d\u8336\uff0c\u8336\u6c64\u78a7\u7eff\uff0c\u9999\u6c14\u6e05\u65b0\u3002",
    1: "\u91d1\u9a6c\u7709\u662f\u798f\u5efa\u6b66\u5c71\u7684\u9876\u7ea7\u7ea2\u8336\uff0c\u6c64\u8272\u91d1\u9ec4\uff0c\u53e3\u611f\u751c\u6da6\u3002",
    2: "\u94c1\u89c2\u97f3\u662f\u798f\u5efa\u5b89\u6eaa\u8457\u540d\u7684\u4e4c\u9f99\u8336\uff0c\u5177\u6709\u72ec\u7279\u7684\u5170\u82b1\u9999\u6c14\u3002",
    3: "\u8d21\u7709\u662f\u798f\u5efa\u798f\u9876\u7684\u767d\u8336\uff0c\u53e3\u611f\u6e05\u751c\uff0c\u8336\u6c64\u767d\u7eff\u3002",
    4: "\u8309\u8389\u82b1\u8336\u662f\u4ee5\u7eff\u8336\u4e3a\u5e95\u6761\uff0c\u7b3c\u5165\u8309\u8389\u82b1\u9999\u5236\u6210\u7684\u82b1\u8336\u3002",
    5: "\u9ad8\u5c71\u8336\u751f\u957f\u5728\u6d77\u62d41000\u7c73\u4ee5\u4e0a\uff0c\u53e3\u611f\u539a\u91cd\uff0c\u8336\u6c64\u6e05\u7eef\u3002",
    6: "\u6b63\u5c71\u5c0f\u79cd\u662f\u4e16\u754c\u4e0a\u6700\u65e9\u7684\u7ea2\u8336\uff0c\u4ea7\u4e8e\u798f\u5efa\u6b66\u5c71\u3002",
    7: "\u6c34\u4ed9\u662f\u4e4c\u9f99\u8336\uff0c\u53e3\u611f\u6e05\u723d\uff0c\u9999\u6c14\u6e05\u65b0\u3002",
    8: "\u5bff\u7709\u662f\u767d\u8336\uff0c\u6709\u6e05\u70ed\u964d\u706b\u7684\u529f\u6548\uff0c\u53e3\u611f\u7ec6\u817b\u3002"
}

TONGUE_CLASSES = [
    "jiankangshe", "botaishe", "hongshe", "zishe",
    "pangdashe", "shoushe", "hongdianshe", "liewenshe",
    "chihenshe", "baitaishe", "huangtaishe", "heitaishe",
    "huataishe", "shenquao", "shenqutu", "gandanao",
    "gandantu", "piweiao", "xinfeitu", "xinfeiao"
]

TONGUE_CLASSES_CN = {
    "jiankangshe": "\u5065\u5eb7\u820c",
    "botaishe": "\u8584\u82d4\u820c",
    "hongshe": "\u7ea2\u820c",
    "zishe": "\u7d2b\u820c",
    "pangdashe": "\u80d6\u5927\u820c",
    "shoushe": "\u7626\u820c",
    "hongdianshe": "\u7ea2\u70b9\u820c",
    "liewenshe": "\u88c2\u7eb9\u820c",
    "chihenshe": "\u9f7f\u75d5\u820c",
    "baitaishe": "\u767d\u82d4\u820c",
    "huangtaishe": "\u9ec4\u82d4\u820c",
    "heitaishe": "\u9ed1\u82d4\u820c",
    "huataishe": "\u82b1\u82d4\u820c",
    "shenquao": "\u80be\u533a\u51f9",
    "shenqutu": "\u80be\u533a\u51f8",
    "gandanao": "\u809d\u5355\u51f9",
    "gandantu": "\u809d\u5355\u51f8",
    "piweiao": "\u813e\u80c3\u51f9",
    "xinfeitu": "\u5fc3\u80ba\u51f8",
    "xinfeiao": "\u5fc3\u80ba\u51f9"
}

TONGUE_RECOMMENDATION = {
    "jiankangshe": "\u7eff\u8336",
    "botaishe": "\u7eff\u8336",
    "hongshe": "\u767d\u8336",
    "zishe": "\u7ea2\u8336",
    "pangdashe": "\u4e4c\u9f99\u8336",
    "shoushe": "\u7ea2\u8336",
    "hongdianshe": "\u767d\u8336",
    "liewenshe": "\u9ec4\u8336",
    "chihenshe": "\u4e4c\u9f99\u8336",
    "baitaishe": "\u7ea2\u8336",
    "huangtaishe": "\u7eff\u8336",
    "heitaishe": "\u9ed1\u8336",
    "huataishe": "\u4e4c\u9f99\u8336",
    "shenquao": "\u9ed1\u8336",
    "shenqutu": "\u9ed1\u8336",
    "gandanao": "\u7eff\u8336",
    "gandantu": "\u7eff\u8336",
    "piweiao": "\u7ea2\u8336",
    "xinfeitu": "\u767d\u8336",
    "xinfeiao": "\u767d\u8336"
}

TONGUE_COLOR_CN = {"normal": "\u6b63\u5e38", "white": "\u767d\u8272", "yellow": "\u9ec4\u8272", "red": "\u7ea2\u8272", "black": "\u9ed1\u8272", "purple": "\u7d2b\u8272", "mixed": "\u6742\u8272"}
TONGUE_THICKNESS_CN = {"normal": "\u9002\u4e2d", "thin": "\u8584", "thick": "\u539a"}
TONGUE_MOISTURE_CN = {"normal": "\u6e7f\u6da6", "dry": "\u5e72\u67d0", "wet": "\u6e7f"}

TONGUE_CLASS_ATTRS = {
    "jiankangshe": {"color": "normal", "thickness": "normal", "moisture": "normal"},
    "botaishe": {"color": "white", "thickness": "thin", "moisture": "normal"},
    "hongshe": {"color": "red", "thickness": "normal", "moisture": "normal"},
    "zishe": {"color": "purple", "thickness": "normal", "moisture": "normal"},
    "pangdashe": {"color": "normal", "thickness": "normal", "moisture": "normal"},
    "shoushe": {"color": "normal", "thickness": "thin", "moisture": "dry"},
    "hongdianshe": {"color": "red", "thickness": "normal", "moisture": "normal"},
    "liewenshe": {"color": "normal", "thickness": "normal", "moisture": "dry"},
    "chihenshe": {"color": "normal", "thickness": "normal", "moisture": "normal"},
    "baitaishe": {"color": "white", "thickness": "thick", "moisture": "normal"},
    "huangtaishe": {"color": "yellow", "thickness": "thick", "moisture": "normal"},
    "heitaishe": {"color": "black", "thickness": "thick", "moisture": "normal"},
    "huataishe": {"color": "mixed", "thickness": "thick", "moisture": "normal"},
    "shenquao": {"color": "normal", "thickness": "normal", "moisture": "normal"},
    "shenqutu": {"color": "normal", "thickness": "normal", "moisture": "normal"},
    "gandanao": {"color": "normal", "thickness": "normal", "moisture": "normal"},
    "gandantu": {"color": "normal", "thickness": "normal", "moisture": "normal"},
    "piweiao": {"color": "normal", "thickness": "normal", "moisture": "normal"},
    "xinfeitu": {"color": "normal", "thickness": "normal", "moisture": "normal"},
    "xinfeiao": {"color": "normal", "thickness": "normal", "moisture": "normal"}
}


HTML_CONTENT = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=2.0, user-scalable=yes">
    <title>智能识别系统</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: "Microsoft YaHei", "SimHei", sans-serif; background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460); min-height: 100vh; color: #fff; overflow-x: hidden; overflow-y: auto; }
        .container { position: relative; z-index: 1; min-height: 100vh; }
        header { position: sticky; top: 0; z-index: 100; display: flex; align-items: center; justify-content: space-between; padding: 8px 12px; background: rgba(0,0,0,0.8); backdrop-filter: blur(10px); }
        .logo { font-size: 16px; font-weight: bold; color: #76d7c4; display: flex; align-items: center; gap: 6px; }
        .header-right { display: flex; gap: 8px; }
        .mode-btn { padding: 6px 14px; border: 1px solid rgba(255,255,255,0.2); border-radius: 20px; background: rgba(255,255,255,0.05); color: #fff; cursor: pointer; font-size: 12px; transition: all 0.3s; }
        .mode-btn.active.tea-btn { background: rgba(76,175,80,0.3); border-color: #4caf50; color: #81c784; }
        .mode-btn.active.tongue-btn { background: rgba(244,67,54,0.3); border-color: #f44336; color: #ef9a9a; }
        .main-content { display: flex; flex-direction: column; }
        .video-section { position: relative; width: 100%; height: 400px; background: #000; }
        #video-feed { width: 100%; height: 100%; object-fit: cover; }
        .video-overlay { position: absolute; top: 10px; left: 10px; }
        .status-badge { padding: 6px 14px; border-radius: 20px; font-size: 11px; font-weight: bold; background: rgba(0,0,0,0.7); display: flex; align-items: center; gap: 6px; }
        .status-badge.tea { border: 1px solid rgba(76,175,80,0.5); color: #81c784; }
        .status-badge.tongue { border: 1px solid rgba(244,67,54,0.5); color: #ef9a9a; }
        .status-dot { width: 8px; height: 8px; border-radius: 50%; animation: pulse 2s infinite; }
        .status-dot.tea { background: #4caf50; }
        .status-dot.tongue { background: #f44336; }
        @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.4; } }
        .menu-section { position: absolute; top: 0; left: 0; width: 100%; height: 100%; display: flex; flex-direction: column; align-items: center; justify-content: center; background: rgba(0,0,0,0.6); }
        .menu-title { font-size: 28px; font-weight: bold; color: #76d7c4; margin-bottom: 8px; text-align: center; }
        .menu-subtitle { font-size: 14px; color: #a0aec0; margin-bottom: 40px; }
        .menu-buttons { display: flex; flex-direction: column; gap: 20px; width: 80%; max-width: 280px; }
        .menu-btn { padding: 25px 20px; border: 2px solid rgba(255,255,255,0.1); border-radius: 20px; background: rgba(255,255,255,0.05); cursor: pointer; transition: all 0.3s; }
        .menu-btn:hover { transform: scale(1.02); }
        .menu-btn.tea-btn:hover { border-color: rgba(76,175,80,0.6); background: rgba(76,175,80,0.1); }
        .menu-btn.tongue-btn:hover { border-color: rgba(244,67,54,0.6); background: rgba(244,67,54,0.1); }
        .menu-btn-icon { font-size: 40px; margin-bottom: 10px; display: block; text-align: center; }
        .menu-btn-title { font-size: 20px; font-weight: bold; text-align: center; }
        .menu-btn.tea-btn .menu-btn-title { color: #81c784; }
        .menu-btn.tongue-btn .menu-btn-title { color: #ef9a9a; }
        .menu-btn-desc { font-size: 12px; color: #888; text-align: center; margin-top: 5px; }
        .result-panel { width: 100%; background: rgba(0,0,0,0.9); padding: 15px; }
        .result-panel::-webkit-scrollbar { width: 4px; }
        .result-panel::-webkit-scrollbar-track { background: rgba(255,255,255,0.05); }
        .result-panel::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.2); border-radius: 2px; }
        .card-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px; }
        .card-title { font-size: 16px; font-weight: bold; }
        .card-title.tea { color: #81c784; }
        .card-title.tongue { color: #ef9a9a; }
        .main-result { text-align: center; padding: 15px; border-radius: 12px; margin-bottom: 12px; }
        .main-result.tea { background: rgba(76,175,80,0.15); border: 1px solid rgba(76,175,80,0.3); }
        .main-result.tongue { background: rgba(244,67,54,0.15); border: 1px solid rgba(244,67,54,0.3); }
        .result-value { font-size: 28px; font-weight: bold; margin-bottom: 8px; }
        .result-value.tea { color: #4caf50; }
        .result-value.tongue { color: #f44336; }
        .confidence-display { font-size: 13px; margin-bottom: 8px; }
        .confidence-bar-container { width: 100%; height: 8px; background: rgba(255,255,255,0.1); border-radius: 4px; overflow: hidden; }
        .confidence-bar-fill { height: 100%; border-radius: 4px; transition: width 0.5s; }
        .confidence-bar-fill.tea { background: #4caf50; }
        .confidence-bar-fill.tongue { background: #f44336; }
        .info-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 12px; }
        .info-item { background: rgba(255,255,255,0.05); padding: 12px; border-radius: 10px; text-align: center; }
        .info-label { font-size: 11px; color: #a0a0a0; margin-bottom: 4px; }
        .info-value { font-size: 16px; font-weight: bold; color: #ef9a9a; }
        .tea-info-card { background: rgba(76,175,80,0.05); border-radius: 10px; padding: 12px; border-left: 3px solid #4caf50; margin-bottom: 12px; }
        .tea-info-label { font-size: 11px; color: #81c784; margin-bottom: 6px; font-weight: bold; }
        .tea-info-content { font-size: 13px; color: #a5d6a7; line-height: 1.5; }
        .recommendation-box { background: rgba(76,175,80,0.1); border-radius: 12px; padding: 15px; border: 1px solid rgba(76,175,80,0.3); margin-bottom: 12px; }
        .recommendation-header { font-size: 12px; color: #81c784; margin-bottom: 8px; }
        .recommendation-content { font-size: 20px; font-weight: bold; color: #4caf50; text-align: center; }
        .action-buttons { display: flex; gap: 10px; }
        .action-btn { flex: 1; padding: 12px; border: none; border-radius: 10px; font-size: 14px; font-weight: bold; cursor: pointer; transition: all 0.3s; display: flex; align-items: center; justify-content: center; gap: 6px; }
        .speak-btn { background: linear-gradient(135deg, #667eea, #764ba2); color: white; }
        .speak-btn:hover { opacity: 0.9; }
        .back-btn { background: rgba(255,255,255,0.1); color: #fff; }
        .back-btn:hover { background: rgba(255,255,255,0.15); }
        .hidden { display: none !important; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="logo">🍃 智能识别</div>
            <div class="header-right">
                <button class="mode-btn tea-btn" onclick="switchMode('tea')">🍵 茶叶识别</button>
                <button class="mode-btn tongue-btn" onclick="switchMode('tongue')">👅 舌苔检测</button>
            </div>
        </header>
        <div class="main-content">
            <div class="video-section">
                <img id="video-feed" src="/video_feed" alt="Camera Feed">
                <div class="video-overlay">
                    <div class="status-badge" id="status-badge">
                        <span class="status-dot"></span>
                        <span id="status-text">待机中</span>
                    </div>
                </div>
                <div id="menu-panel" class="menu-section">
                    <div class="menu-title">智能识别系统</div>
                    <div class="menu-subtitle">AI 人工智能识别解决方案</div>
                    <div class="menu-buttons">
                        <button class="menu-btn tea-btn" onclick="switchMode('tea')">
                            <span class="menu-btn-icon">🍵</span>
                            <div class="menu-btn-title">茶叶识别</div>
                            <div class="menu-btn-desc">识别 9 种茶叶品种</div>
                        </button>
                        <button class="menu-btn tongue-btn" onclick="switchMode('tongue')">
                            <span class="menu-btn-icon">👅</span>
                            <div class="menu-btn-title">舌苔检测</div>
                            <div class="menu-btn-desc">分析舌苔颜色、厚度等特征</div>
                        </button>
                    </div>
                </div>
                <div id="tea-panel" class="result-panel hidden">
                    <div class="card-header"><div class="card-title tea">🎯 识别结果</div></div>
                    <div class="main-result tea">
                        <div class="result-value tea" id="tea-name">等待识别...</div>
                        <div class="confidence-display" id="tea-confidence">置信度: --%</div>
                        <div class="confidence-bar-container"><div class="confidence-bar-fill tea" id="tea-confidence-fill" style="width:0%"></div></div>
                    </div>
                    <div class="tea-info-card" id="tea-info">
                        <div class="tea-info-label">📖 茶叶介绍</div>
                        <div class="tea-info-content">启动识别后，这里会显示对应茶叶的详细信息...</div>
                    </div>
                    <div class="action-buttons">
                        <button class="action-btn speak-btn" onclick="speakTea()"><span>🔊</span><span>语音播报</span></button>
                        <button class="action-btn back-btn" onclick="switchMode('menu')"><span>←</span><span>返回</span></button>
                    </div>
                </div>
                <div id="tongue-panel" class="result-panel hidden">
                    <div class="card-header"><div class="card-title tongue">🔍 舌苔检测结果</div></div>
                    <div class="main-result tongue">
                        <div class="result-value tongue" id="tongue-class">等待检测...</div>
                        <div class="confidence-display" id="tongue-confidence">置信度: --%</div>
                        <div class="confidence-bar-container"><div class="confidence-bar-fill tongue" id="tongue-confidence-fill" style="width:0%"></div></div>
                    </div>
                    <div class="info-grid">
                        <div class="info-item"><div class="info-label">舌苔颜色</div><div class="info-value" id="tongue-color">--</div></div>
                        <div class="info-item"><div class="info-label">舌苔厚度</div><div class="info-value" id="tongue-thickness">--</div></div>
                        <div class="info-item"><div class="info-label">舌面湿度</div><div class="info-value" id="tongue-moisture">--</div></div>
                    </div>
                    <div class="recommendation-box">
                        <div class="recommendation-header">🍵 推荐茶饮</div>
                        <div class="recommendation-content" id="tongue-recommendation">等待检测...</div>
                    </div>
                    <div class="action-buttons">
                        <button class="action-btn speak-btn" onclick="speakTongue()"><span>🔊</span><span>语音播报</span></button>
                        <button class="action-btn back-btn" onclick="switchMode('menu')"><span>←</span><span>返回</span></button>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <script>
        var currentMode = 'menu';
        var teaData = { class_id: -1, prob: 0 };
        var tongueData = { class_id: -1, class_name: '', confidence: 0, recommendation: '', color: '', thickness: '', moisture: '' };
        var updateInterval = null;
        
        var TEA_CN_ARRAY = ['碧螺春', '金骏眉', '铁观音', '贡眉', '茉莉花茶', '高山茶', '正山小种', '水仙', '寿眉'];
        var TEA_INFO_CN = {
            '碧螺春': '碧螺春是江苏苏州的顶级名茶，茶汤碧绿，香气清新。',
            '金骏眉': '金骏眉是福建武夷山的顶级红茶，汤色金黄，口感甜润。',
            '铁观音': '铁观音是福建安溪著名的乌龙茶，具有独特的兰花香。',
            '贡眉': '贡眉是福建福鼎的白茶，口感清甜，茶汤白绿。',
            '茉莉花茶': '茉莉花茶是以绿茶为底料，窨入茉莉花香制成的花茶。',
            '高山茶': '高山茶生长在海拔1000米以上，口感厚重，茶汤清澈。',
            '正山小种': '正山小种是世界上最早的红茶，产于福建武夷山。',
            '水仙': '水仙是乌龙茶，口感清爽，香气清新。',
            '寿眉': '寿眉是白茶，有清热降火的功效，口感细腻。'
        };
        var TONGUE_CLASSES = ['jiankangshe','botaishe','hongshe','zishe','pangdashe','shoushe','hongdianshe','liewenshe','chihenshe','baitaishe','huangtaishe','heitaishe','huataishe','shenquao','shenqutu','gandanao','gandantu','piweiao','xinfeitu','xinfeiao'];
        var TONGUE_CLASSES_CN = {'jiankangshe':'\u5065\u5eb7\u820c','botaishe':'\u8584\u82d4\u820c','hongshe':'\u7ea2\u820c','zishe':'\u7d2b\u820c','pangdashe':'\u80d6\u5927\u820c','shoushe':'\u7626\u820c','hongdianshe':'\u7ea2\u70b9\u820c','liewenshe':'\u88c2\u7eb9\u820c','chihenshe':'\u9f7f\u75d5\u820c','baitaishe':'\u767d\u82d4\u820c','huangtaishe':'\u9ec4\u82d4\u820c','heitaishe':'\u9ed1\u82d4\u820c','huataishe':'\u82b1\u82d4\u820c','shenquao':'\u80be\u533a\u51f9','shenqutu':'\u80be\u533a\u51f8','gandanao':'\u809d\u5355\u51f9','gandantu':'\u809d\u5355\u51f8','piweiao':'\u813e\u80c3\u51f9','xinfeitu':'\u5fc3\u80ba\u51f8','xinfeiao':'\u5fc3\u80ba\u51f9'};
        var TONGUE_RECOMMENDATION = {'jiankangshe':'\u7eff\u8336','botaishe':'\u7eff\u8336','hongshe':'\u767d\u8336','zishe':'\u7ea2\u8336','pangdashe':'\u4e4c\u9f99\u8336','shoushe':'\u7ea2\u8336','hongdianshe':'\u767d\u8336','liewenshe':'\u9ec4\u8336','chihenshe':'\u4e4c\u9f99\u8336','baitaishe':'\u7ea2\u8336','huangtaishe':'\u7eff\u8336','heitaishe':'\u9ed1\u8336','huataishe':'\u4e4c\u9f99\u8336','shenquao':'\u9ed1\u8336','shenqutu':'\u9ed1\u8336','gandanao':'\u7eff\u8336','gandantu':'\u7eff\u8336','piweiao':'\u7ea2\u8336','xinfeitu':'\u767d\u8336','xinfeiao':'\u767d\u8336'};
        var TONGUE_COLOR_CN = {'normal':'\u6b63\u5e38','white':'\u767d\u8272','yellow':'\u9ec4\u8272','red':'\u7ea2\u8272','black':'\u9ed1\u8272','purple':'\u7d2b\u8272','mixed':'\u6742\u8272'};
        var TONGUE_THICKNESS_CN = {'normal':'\u9002\u4e2d','thin':'\u8584','thick':'\u539a'};
        var TONGUE_MOISTURE_CN = {'normal':'\u6e7f\u6da6','dry':'\u5e72\u67d0','wet':'\u6e7f'};

        function switchMode(mode) {
            currentMode = mode;
            document.getElementById('menu-panel').classList.add('hidden');
            document.getElementById('tea-panel').classList.add('hidden');
            document.getElementById('tongue-panel').classList.add('hidden');
            document.querySelectorAll('.mode-btn').forEach(function(btn) { btn.classList.remove('active'); });
            var badge = document.getElementById('status-badge');
            var dot = badge.querySelector('.status-dot');
            badge.className = 'status-badge';
            dot.className = 'status-dot';
            if (mode === 'menu') {
                document.getElementById('menu-panel').classList.remove('hidden');
                document.getElementById('status-text').textContent = '待机中';
            } else if (mode === 'tea') {
                document.getElementById('tea-panel').classList.remove('hidden');
                document.querySelector('.mode-btn.tea-btn').classList.add('active');
                badge.classList.add('tea');
                dot.classList.add('tea');
                document.getElementById('status-text').textContent = '识别中...';
            } else if (mode === 'tongue') {
                document.getElementById('tongue-panel').classList.remove('hidden');
                document.querySelector('.mode-btn.tongue-btn').classList.add('active');
                badge.classList.add('tongue');
                dot.classList.add('tongue');
                document.getElementById('status-text').textContent = '检测中...';
            }
            fetch('/api/set_mode/' + mode);
            if (updateInterval) clearInterval(updateInterval);
            if (mode === 'tea' || mode === 'tongue') {
                updateInterval = setInterval(updateResults, 500);
            }
        }
        function updateResults() {
            if (currentMode === 'tea') {
                fetch('/api/tea_result').then(function(res) { return res.json(); }).then(function(data) {
                    if (data.status === 'success') {
                        teaData = data;
                        var teaName = (data.class_id >= 0 && data.class_id < TEA_CN_ARRAY.length) ? TEA_CN_ARRAY[data.class_id] : '未知';
                        document.getElementById('tea-name').textContent = teaName;
                        document.getElementById('tea-confidence').textContent = '置信度: ' + (data.prob * 100).toFixed(1) + '%';
                        document.getElementById('tea-confidence-fill').style.width = (data.prob * 100) + '%';
                        var teaInfo = TEA_INFO_CN[teaName] || '';
                        if (teaInfo) {
                            document.getElementById('tea-info').innerHTML = '<div class="tea-info-label">📖 茶叶介绍</div><div class="tea-info-content">' + teaInfo + '</div>';
                        }
                    }
                }).catch(function(e) { console.error(e); });
            } else if (currentMode === 'tongue') {
                fetch('/api/tongue_result').then(function(res) { return res.json(); }).then(function(data) {
                    if (data.status === 'success') {
                        tongueData = data;
                        var cn = TONGUE_CLASSES_CN[data.class_name] || '未知';
                        document.getElementById('tongue-class').textContent = cn;
                        document.getElementById('tongue-confidence').textContent = '置信度: ' + (data.confidence * 100).toFixed(1) + '%';
                        document.getElementById('tongue-confidence-fill').style.width = (data.confidence * 100) + '%';
                        document.getElementById('tongue-color').textContent = TONGUE_COLOR_CN[data.color] || '--';
                        document.getElementById('tongue-thickness').textContent = TONGUE_THICKNESS_CN[data.thickness] || '--';
                        document.getElementById('tongue-moisture').textContent = TONGUE_MOISTURE_CN[data.moisture] || '--';
                        var rec = TONGUE_RECOMMENDATION[data.class_name] || '绿茶';
                        document.getElementById('tongue-recommendation').textContent = rec;
                    }
                }).catch(function(e) { console.error(e); });
            }
        }
        function speakTea() {
            var teaName = (teaData.class_id >= 0 && teaData.class_id < TEA_CN_ARRAY.length) ? TEA_CN_ARRAY[teaData.class_id] : '未识别到茶叶';
            if (!teaName || teaName === '未识别到茶叶') { alert('请等待识别结果'); return; }
            playSpeech(teaName);
        }
        function speakTongue() {
            if (!tongueData.class_name) { alert('\u8bf7\u7b49\u5f85\u68c0\u6d4b\u7ed3\u679c'); return; }
            var cn = TONGUE_CLASSES_CN[tongueData.class_name] || '\u672a\u77e5';
            var colorCn = TONGUE_COLOR_CN[tongueData.color] || '';
            var thickCn = TONGUE_THICKNESS_CN[tongueData.thickness] || '';
            var moistCn = TONGUE_MOISTURE_CN[tongueData.moisture] || '';
            var rec = TONGUE_RECOMMENDATION[tongueData.class_name] || '\u7eff\u8336';
            playSpeech('\u820c\u82d4\u68c0\u6d4b\u7ed3\u679c\u003a' + cn + '\uff0c\u989c\u8272\u003a' + colorCn + '\uff0c\u539a\u5ea6\u003a' + thickCn + '\uff0c\u6e7f\u5ea6\u003a' + moistCn + '\uff0c\u63a8\u8350\u996e\u7528\u003a' + rec);
        }
        function playSpeech(text) {
            fetch('/tts?text=' + encodeURIComponent(text))
            .then(function(response) { return response.blob(); })
            .then(function(blob) {
                var audioUrl = URL.createObjectURL(blob);
                var audio = new Audio(audioUrl);
                audio.play();
                audio.onended = function() { URL.revokeObjectURL(audioUrl); };
            })
            .catch(function(error) {
                console.error('播放失败:', error);
                alert('语音播放失败，请检查后端服务');
            });
        }
    </script>
</body>
</html>"""

class CameraReader:
    def __init__(self):
        self.cap = None
        self.cam_id = -1
    def open(self):
        for cam_id in CAMERA_IDS:
            self.cap = cv2.VideoCapture(cam_id, cv2.CAP_V4L2)
            if self.cap.isOpened():
                self.cam_id = cam_id
                print(f"Camera opened: /dev/video{cam_id}")
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                return True
            if self.cap:
                self.cap.release()
                self.cap = None
        print(f"None of the camera indices {CAMERA_IDS} worked")
        return False
    def read(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and frame is not None:
                frame = cv2.flip(frame, 1)
                return True, frame
            else:
                print(f"Camera read failed for /dev/video{self.cam_id}, ret={ret}, frame={frame is not None}")
        return False, None
    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None

class TeaClassifier:
    def __init__(self):
        self.session = None
        self.input_name = None
        self.output_name = None
        self.h, self.w = 640, 640
        self.use_model = False
        self.load_model()
    def load_model(self):
        if not os.path.exists(TEA_MODEL_PATH):
            print(f"Tea model not found: {TEA_MODEL_PATH}")
            return
        try:
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
            so.intra_op_num_threads = NUM_CORES
            so.inter_op_num_threads = NUM_CORES
            so.add_session_config_entry('session.disable_fusion', '1')
            self.session = ort.InferenceSession(TEA_MODEL_PATH, sess_options=so, providers=['CPUExecutionProvider'])
            input_meta = self.session.get_inputs()[0]
            self.input_name = input_meta.name
            self.h, self.w = input_meta.shape[2], input_meta.shape[3]
            self.output_name = self.session.get_outputs()[0].name
            self.use_model = True
            print(f"Tea model loaded: {TEA_MODEL_PATH}")
        except Exception as e:
            print(f"Tea model load failed: {e}")
    def preprocess(self, frame):
        img = cv2.resize(frame, (self.w, self.h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        return np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)
    def infer(self, frame):
        if not self.use_model:
            return {"name": "Unknown", "prob": 0.5}
        try:
            preprocessed = self.preprocess(frame)
            outputs = self.session.run([self.output_name], {self.input_name: preprocessed})[0]
            best_prob, best_class = 0, 0
            for i in range(outputs.shape[2]):
                prob = float(np.max(outputs[0, 4:, i]))
                if prob > best_prob:
                    best_prob = prob
                    best_class = int(np.argmax(outputs[0, 4:, i]))
            return {"name": TEA_NAMES.get(best_class, "Unknown"), "prob": best_prob}
        except:
            return {"name": "Unknown", "prob": 0.5}

class TongueAnalyzer:
    def __init__(self):
        self.session = None
        self.input_name = None
        self.output_names = []
        self.h, self.w = 640, 640
        self.use_model = False
        self.load_model()
    def load_model(self):
        if not os.path.exists(TONGUE_MODEL_PATH):
            print(f"Tongue model not found: {TONGUE_MODEL_PATH}, using traditional CV analysis")
            return
        try:
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
            so.intra_op_num_threads = 1
            so.inter_op_num_threads = 1
            so.add_session_config_entry('session.disable_fusion', '1')
            self.session = ort.InferenceSession(TONGUE_MODEL_PATH, sess_options=so, providers=['CPUExecutionProvider'])
            input_meta = self.session.get_inputs()[0]
            self.input_name = input_meta.name
            self.h, self.w = input_meta.shape[2], input_meta.shape[3]
            outputs = self.session.get_outputs()
            self.output_names = [o.name for o in outputs]
            self.use_model = True
            print(f"Tongue model loaded: {TONGUE_MODEL_PATH}")
            print(f"  Output names: {self.output_names}")
        except Exception as e:
            print(f"Tongue model load failed: {e}, using traditional CV analysis")
    def preprocess(self, frame):
        img = cv2.resize(frame, (self.w, self.h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        return np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)
    def detect_tongue(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 40, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([165, 40, 70])
        upper_red2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)
        lower_pink = np.array([140, 30, 90])
        upper_pink = np.array([170, 180, 255])
        pink_mask = cv2.inRange(hsv, lower_pink, upper_pink)
        mask = cv2.bitwise_or(red_mask, pink_mask)
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            for contour in contours[:5]:
                area = cv2.contourArea(contour)
                if area < 500:
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                if w < 30 or h < 30:
                    continue
                if w > frame.shape[1] * 0.95 or h > frame.shape[0] * 0.95:
                    continue
                aspect_ratio = w / h
                if 0.3 < aspect_ratio < 3.0:
                    if y > frame.shape[0] * 0.02:
                        return frame[y:y+h, x:x+w], (x, y, w, h)
        return None, None
    def analyze(self, frame):
        if self.use_model:
            try:
                preprocessed = self.preprocess(frame)
                outputs = self.session.run(self.output_names, {self.input_name: preprocessed})
                out = outputs[0].astype(np.float32)
                if out.ndim == 2:
                    scores = out[0]
                else:
                    scores = out
                exp_s = np.exp(scores - np.max(scores))
                softmax_s = exp_s / np.sum(exp_s)
                class_id = int(np.argmax(scores))
                confidence = float(softmax_s[class_id])
                class_name = TONGUE_CLASSES[class_id] if class_id < len(TONGUE_CLASSES) else TONGUE_CLASSES[0]
                attrs = TONGUE_CLASS_ATTRS.get(class_name, {"color": "normal", "thickness": "normal", "moisture": "normal"})
                return {
                    "class_name": class_name,
                    "confidence": confidence,
                    "color": attrs["color"],
                    "thickness": attrs["thickness"],
                    "moisture": attrs["moisture"],
                    "bbox": None
                }
            except Exception as e:
                print(f"Tongue model inference error: {e}")
        tongue, bbox = self.detect_tongue(frame)
        if tongue is None or tongue.size == 0:
            return {"class_name": "jiankangshe", "confidence": 0.5, "color": "normal", "thickness": "normal", "moisture": "normal", "bbox": None}
        return {"class_name": "jiankangshe", "confidence": 0.5, "color": "normal", "thickness": "normal", "moisture": "normal", "bbox": bbox}

tea_classifier = TeaClassifier()
tongue_analyzer = TongueAnalyzer()

camera = None
current_frame = None
current_mode = "menu"
tea_result = {"name": "Unknown", "prob": 0}
tongue_result = {"class_name": "jiankangshe", "confidence": 0, "color": "normal", "thickness": "normal", "moisture": "normal", "bbox": None}
running = False
task_queue = Queue(maxsize=1)
frame_lock = threading.Lock()
result_lock = threading.Lock()

def open_camera():
    global camera
    camera = CameraReader()
    if camera.open():
        return True
    print("Camera not found!")
    return False

def recognize_worker():
    global tea_result, tongue_result, running
    while running:
        try:
            task = task_queue.get(timeout=0.1)
            mode, frame = task
            if mode == "tea":
                result = tea_classifier.infer(frame)
                with result_lock:
                    tea_result = result
            elif mode == "tongue":
                result = tongue_analyzer.analyze(frame)
                with result_lock:
                    tongue_result = result
        except:
            pass
        time.sleep(0.01)

def camera_thread():
    global current_frame, running
    last_recognize_time = 0
    while running:
        if camera:
            ret, frame = camera.read()
            if ret:
                with frame_lock:
                    current_frame = frame.copy()
                current_time = time.time()
                if current_time - last_recognize_time >= RECOGNIZE_INTERVAL:
                    with result_lock:
                        mode = current_mode
                    if mode in ["tea", "tongue"]:
                        try:
                            task_queue.get_nowait()
                        except:
                            pass
                        task_queue.put((mode, frame.copy()))
                    last_recognize_time = current_time
            else:
                time.sleep(0.1)
        time.sleep(0.01)

def generate_frames():
    global current_frame, current_mode, running
    print("Video feed started")
    while running:
        try:
            with frame_lock:
                if current_frame is not None:
                    frame = current_frame.copy()
                else:
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(frame, "Camera loading...", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            if current_mode == "menu":
                h, w = frame.shape[:2]
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, h), (20, 30, 50), -1)
                cv2.addWeighted(overlay, 0.95, frame, 0.05, 0, frame)
                cv2.putText(frame, "Smart Recognition", (w//2 - 120, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            elif current_mode == "tea":
                h, w = frame.shape[:2]
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, 60), (20, 40, 80), -1)
                cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)
                cv2.putText(frame, "Tea Recognition", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (76, 175, 80), 2)
            elif current_mode == "tongue":
                h, w = frame.shape[:2]
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, 60), (40, 20, 20), -1)
                cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)
                cv2.putText(frame, "Tongue Detection", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (244, 67, 54), 2)
                with result_lock:
                    result = tongue_result
                if result and result["bbox"]:
                    x, y_box, w_box, h_box = result["bbox"]
                    cv2.rectangle(frame, (x, y_box), (x + w_box, y_box + h_box), (244, 67, 54), 2)
            ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if ret:
                yield b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n' % len(buffer) + buffer.tobytes() + b'\r\n'
        except Exception as e:
            print(f"Generate frames error: {e}")
            time.sleep(0.1)

@app.route('/')
def index():
    global current_mode
    current_mode = "menu"
    return HTML_CONTENT, 200, {'Content-Type': 'text/html; charset=utf-8'}

@app.route('/tea')
def tea():
    global current_mode
    current_mode = "tea"
    return HTML_CONTENT, 200, {'Content-Type': 'text/html; charset=utf-8'}

@app.route('/tongue')
def tongue():
    global current_mode
    current_mode = "tongue"
    return HTML_CONTENT, 200, {'Content-Type': 'text/html; charset=utf-8'}

@app.route('/video_feed')
def video_feed():
    print("Video feed request received")
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/tea_result')
def get_tea_result():
    with result_lock:
        tea = tea_result.copy()
    tea_name = tea["name"]
    tea_class_id = -1
    for idx, name in TEA_NAMES.items():
        if name == tea_name:
            tea_class_id = idx
            break
    data = {
        "status": "success",
        "class_id": tea_class_id,
        "prob": tea["prob"]
    }
    return Response(json.dumps(data), content_type='application/json; charset=utf-8')

@app.route('/api/tongue_result')
def get_tongue_result():
    with result_lock:
        tongue = tongue_result.copy()
    data = {
        "status": "success",
        "class_name": tongue["class_name"],
        "confidence": tongue["confidence"],
        "color": tongue["color"],
        "thickness": tongue["thickness"],
        "moisture": tongue["moisture"]
    }
    return Response(json.dumps(data), content_type='application/json; charset=utf-8')

@app.route('/api/set_mode/<mode>')
def set_mode(mode):
    global current_mode
    if mode in ["menu", "tea", "tongue"]:
        current_mode = mode
        data = {"status": "success", "mode": mode}
        return Response(json.dumps(data, ensure_ascii=False), content_type='application/json; charset=utf-8')
    data = {"status": "error", "message": "Invalid mode"}
    return Response(json.dumps(data, ensure_ascii=False), content_type='application/json; charset=utf-8'), 400

@app.route('/tts')
def tts():
    text = request.args.get('text', '')
    if not text:
        return Response('{"error":"no text"}', status=400, content_type='application/json')
    audio_path = '/tmp/tts_output.wav'
    backends = [
        (['espeak-ng', '-v', 'zh-CN', '-w', audio_path, text], 'espeak-ng zh-CN'),
        (['espeak-ng', '-v', 'zh', '-w', audio_path, text], 'espeak-ng zh'),
        (['espeak', '-v', 'zh', '-w', audio_path, text], 'espeak zh'),
        (['pico2wave', '-l', 'zh-CN', '-w', audio_path, text], 'pico2wave'),
        (['espeak-ng', '-v', 'mb-cn1', '-w', audio_path, text], 'espeak-ng mb-cn1'),
    ]
    for cmd, name in backends:
        try:
            subprocess.run(cmd, check=True, timeout=30, capture_output=True)
            return send_file(audio_path, mimetype='audio/wav')
        except FileNotFoundError:
            continue
        except Exception as e:
            continue
    return Response('{"error":"no tts backend, install: sudo apt install espeak-ng"}',
                    status=500, content_type='application/json')

if __name__ == '__main__':
    if open_camera():
        running = True
        for i in range(NUM_WORKER):
            t = threading.Thread(target=recognize_worker, daemon=True)
            t.start()
        camera_t = threading.Thread(target=camera_thread, daemon=True)
        camera_t.start()
        print("Web server starting...")
        print("Access http://localhost:5000")
        app.run(host='0.0.0.0', port=5000, debug=False)
        running = False
        if camera:
            camera.release()
    else:
        print("Cannot open camera, exiting")