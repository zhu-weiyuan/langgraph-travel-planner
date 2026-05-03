# -*- coding: utf-8 -*-
"""
Web server for LangGraph Travel Planner Agent.

Run: python app.py
Visit: http://localhost:7861
"""

import sys
import io
import json
from uuid import uuid4
from datetime import datetime

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from langchain_core.messages import HumanMessage, AIMessage
from agent.graph import build_graph
from http.server import HTTPServer, BaseHTTPRequestHandler
import os

PORT = 7861
_graph = None


def init():
    global _graph
    use_sqlite = os.environ.get('USE_SQLITE', '0') == '1'
    _graph = build_graph(use_sqlite=use_sqlite)
    print(f"[Server] Travel Planner Agent initialized (sqlite={use_sqlite})")


def run_planner(session_id, user_message):
    """Run the travel planner for a user message."""
    config = {"configurable": {"thread_id": session_id}}
    human_msg = HumanMessage(content=user_message)

    # Check if this is feedback (refinement) or new request
    current_state = _graph.get_state(config)
    refinement_round = 0
    user_feedback = None

    if current_state and current_state.values:
        existing_msgs = current_state.values.get('messages', [])
        # If there's already a final_output, treat new message as feedback
        if current_state.values.get('final_output'):
            user_feedback = user_message
            refinement_round = current_state.values.get('refinement_round', 0)

    input_data = {
        "messages": [human_msg],
        "session_id": session_id,
        "refinement_round": refinement_round,
    }
    if user_feedback:
        input_data["user_feedback"] = user_feedback

    try:
        for event in _graph.stream(input_data, config=config, stream_mode="values"):
            pass  # Run through all nodes
    except Exception as e:
        print(f"[Agent Error] {e}")
        import traceback
        traceback.print_exc()

    state = _graph.get_state(config)
    if state and state.values:
        return {
            'output': state.values.get('final_output', '规划生成中，请稍候...'),
            'destination': state.values.get('destination', ''),
            'days': state.values.get('days', 0),
            'budget': state.values.get('budget'),
            'travel_style': state.values.get('travel_style', ''),
            'refinement_round': state.values.get('refinement_round', 0),
        }
    return {'output': '抱歉，规划生成失败，请重试。'}


# ============================================================
# HTML UI
# ============================================================

CHAT_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>🌍 智能旅游规划助手</title>
<style>
  :root {
    --primary: #0ea5e9;
    --primary-dark: #0284c7;
    --accent: #f59e0b;
    --bg: #f0f9ff;
    --surface: #ffffff;
    --text: #0f172a;
    --text-secondary: #475569;
    --border: #e2e8f0;
    --shadow: 0 4px 24px rgba(14,165,233,0.08);
    --radius: 16px;
  }

  * { margin: 0; padding: 0; box-sizing: border-box; }

  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif;
    background: var(--bg);
    color: var(--text);
    height: 100vh;
    display: flex;
    flex-direction: column;
  }

  /* Header */
  .header {
    background: linear-gradient(135deg, #0ea5e9 0%, #2563eb 50%, #7c3aed 100%);
    color: white;
    padding: 20px 32px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 4px 20px rgba(14,165,233,0.2);
  }
  .header h1 { font-size: 22px; font-weight: 700; letter-spacing: -0.5px; }
  .header-subtitle { font-size: 13px; opacity: 0.85; margin-top: 2px; }
  .header-actions { display: flex; gap: 8px; align-items: center; }
  .header-btn {
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.25);
    color: white;
    padding: 8px 16px;
    border-radius: 10px;
    cursor: pointer;
    font-size: 13px;
    transition: all 0.2s;
    backdrop-filter: blur(4px);
  }
  .header-btn:hover { background: rgba(255,255,255,0.25); }

  /* Main layout */
  .main { flex: 1; display: flex; overflow: hidden; }

  /* Sidebar */
  .sidebar {
    width: 300px;
    background: var(--surface);
    border-right: 1px solid var(--border);
    padding: 24px;
    overflow-y: auto;
    flex-shrink: 0;
  }
  .sidebar h3 { font-size: 14px; color: var(--text-secondary); margin-bottom: 16px; text-transform: uppercase; letter-spacing: 1px; }

  .quick-cards { display: flex; flex-direction: column; gap: 8px; }
  .quick-card {
    padding: 12px 16px;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 12px;
    cursor: pointer;
    transition: all 0.2s;
    font-size: 13px;
    line-height: 1.5;
  }
  .quick-card:hover { border-color: var(--primary); background: #e0f2fe; transform: translateY(-1px); box-shadow: var(--shadow); }
  .quick-card .emoji { font-size: 18px; margin-right: 8px; }

  /* Chat area */
  .chat-area { flex: 1; display: flex; flex-direction: column; }

  .chat-container {
    flex: 1;
    overflow-y: auto;
    padding: 24px 32px;
    display: flex;
    flex-direction: column;
    gap: 20px;
  }

  .message { display: flex; gap: 12px; max-width: 85%; animation: slideIn 0.3s ease; }
  @keyframes slideIn { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: translateY(0); } }

  .message.user { align-self: flex-end; flex-direction: row-reverse; }
  .message.bot { align-self: flex-start; }

  .avatar {
    width: 40px; height: 40px; border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 20px; flex-shrink: 0;
  }
  .message.user .avatar { background: linear-gradient(135deg, #0ea5e9, #2563eb); }
  .message.bot .avatar { background: linear-gradient(135deg, #f59e0b, #ef4444); }

  .bubble {
    padding: 16px 20px;
    border-radius: var(--radius);
    font-size: 14px;
    line-height: 1.7;
    white-space: pre-wrap;
    word-break: break-word;
  }
  .message.user .bubble {
    background: linear-gradient(135deg, #0ea5e9, #2563eb);
    color: white;
    border-bottom-right-radius: 4px;
  }
  .message.bot .bubble {
    background: var(--surface);
    box-shadow: var(--shadow);
    border-bottom-left-radius: 4px;
  }

  .system-msg {
    align-self: center;
    padding: 8px 20px;
    background: #f1f5f9;
    border-radius: 12px;
    font-size: 12px;
    color: #64748b;
  }

  /* Typing indicator */
  .typing-indicator { display: flex; gap: 5px; padding: 12px 16px; align-items: center; }
  .typing-indicator .dot { width: 8px; height: 8px; background: #94a3b8; border-radius: 50%; animation: bounce 1.4s infinite; }
  .typing-indicator .dot:nth-child(2) { animation-delay: 0.2s; }
  .typing-indicator .dot:nth-child(3) { animation-delay: 0.4s; }
  @keyframes bounce { 0%, 60%, 100% { transform: translateY(0); } 30% { transform: translateY(-8px); } }

  /* Input area */
  .input-area {
    padding: 20px 32px;
    background: var(--surface);
    border-top: 1px solid var(--border);
    display: flex;
    gap: 12px;
    align-items: center;
  }
  .input-area textarea {
    flex: 1;
    padding: 14px 18px;
    border: 2px solid var(--border);
    border-radius: 14px;
    font-size: 14px;
    outline: none;
    resize: none;
    font-family: inherit;
    transition: border-color 0.2s;
    min-height: 48px;
    max-height: 120px;
  }
  .input-area textarea:focus { border-color: var(--primary); }
  .send-btn {
    padding: 14px 28px;
    background: linear-gradient(135deg, #0ea5e9, #2563eb);
    color: white;
    border: none;
    border-radius: 14px;
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s;
    white-space: nowrap;
  }
  .send-btn:hover { transform: translateY(-1px); box-shadow: 0 4px 16px rgba(14,165,233,0.3); }
  .send-btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none; box-shadow: none; }

  /* Info bar */
  .info-bar {
    padding: 8px 32px;
    background: #f8fafc;
    border-top: 1px solid var(--border);
    font-size: 12px;
    color: #64748b;
    display: flex;
    gap: 24px;
  }
  .info-bar span { display: flex; align-items: center; gap: 4px; }

  /* Markdown-like rendering in bot bubbles */
  .bubble h2 { font-size: 16px; margin: 12px 0 8px; color: #0f172a; }
  .bubble h3 { font-size: 14px; margin: 10px 0 6px; color: #1e40af; }

  /* Scrollbar */
  .chat-container::-webkit-scrollbar { width: 6px; }
  .chat-container::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }

  /* Mobile */
  @media (max-width: 768px) {
    .sidebar { display: none; }
    .header { padding: 16px; }
    .chat-container { padding: 16px; }
    .input-area { padding: 12px 16px; }
    .message { max-width: 95%; }
  }

  /* Welcome screen */
  .welcome { text-align: center; padding: 60px 32px; color: var(--text-secondary); }
  .welcome h2 { font-size: 28px; color: var(--text); margin-bottom: 12px; }
  .welcome p { font-size: 15px; line-height: 1.6; max-width: 500px; margin: 0 auto 24px; }
  .welcome-features { display: flex; gap: 16px; justify-content: center; flex-wrap: wrap; }
  .welcome-feature {
    padding: 16px 24px;
    background: var(--surface);
    border-radius: 12px;
    box-shadow: var(--shadow);
    font-size: 13px;
    text-align: left;
    min-width: 160px;
  }
  .welcome-feature .icon { font-size: 24px; margin-bottom: 8px; }
</style>
</head>
<body>

<div class="header">
  <div>
    <h1>🌍 智能旅游规划助手</h1>
    <div class="header-subtitle">基于 LangGraph + Qwen3.6-27B · 本地运行</div>
  </div>
  <div class="header-actions">
    <button class="header-btn" onclick="newSession()">✨ 新规划</button>
    <button class="header-btn" onclick="clearChat()">🗑️ 清空</button>
  </div>
</div>

<div class="main">
  <div class="sidebar">
    <h3>🚀 快速开始</h3>
    <div class="quick-cards">
      <div class="quick-card" onclick="quickTest('帮我规划一个去云南的5天行程，预算5000元')">
        <span class="emoji">🏔️</span> 云南5日游 · ¥5000
      </div>
      <div class="quick-card" onclick="quickTest('我想去日本东京玩7天，两个人，舒适型')">
        <span class="emoji">🗼</span> 东京7日游 · 双人
      </div>
      <div class="quick-card" onclick="quickTest('推荐一个适合夏天的海边度假目的地，3天预算3000')">
        <span class="emoji">🏖️</span> 夏日海滩 · ¥3000
      </div>
      <div class="quick-card" onclick="quickTest('带父母去西安玩4天，文化深度游')">
        <span class="emoji">🏛️</span> 西安文化 · 亲子游
      </div>
      <div class="quick-card" onclick="quickTest('预算1万去欧洲玩10天，推荐路线')">
        <span class="emoji">🌍</span> 欧洲10日 · ¥10000
      </div>
      <div class="quick-card" onclick="quickTest('成都周边2天自驾游推荐')">
        <span class="emoji">🚗</span> 成都周边 · 自驾
      </div>
    </div>

    <h3 style="margin-top:24px">💡 提示</h3>
    <div style="font-size:12px;color:#64748b;line-height:1.6">
      告诉我你的目的地、天数、预算和偏好，我会为你生成详细的行程规划、费用估算和天气信息。<br><br>
      生成后你可以反馈调整，比如"把Day 2的景点换一下"或"预算太高了"。
    </div>
  </div>

  <div class="chat-area">
    <div class="chat-container" id="chatContainer">
      <div class="welcome" id="welcomeScreen">
        <h2>🌏 去哪玩？我来帮你规划</h2>
        <p>告诉我你的旅行想法，我会为你生成详细的行程方案，包括每日安排、费用估算和天气信息。</p>
        <div class="welcome-features">
          <div class="welcome-feature">
            <div class="icon">🗺️</div>
            <strong>智能规划</strong><br>根据你的偏好定制行程
          </div>
          <div class="welcome-feature">
            <div class="icon">💰</div>
            <strong>费用估算</strong><br>详细的预算明细
          </div>
          <div class="welcome-feature">
            <div class="icon">🌤️</div>
            <strong>天气查询</strong><br>实时天气预报
          </div>
          <div class="welcome-feature">
            <div class="icon">🔄</div>
            <strong>迭代优化</strong><br>根据你的反馈调整
          </div>
        </div>
      </div>
    </div>

    <div class="info-bar">
      <span>📍 目的地: <strong id="infoDest">-</strong></span>
      <span>📅 天数: <strong id="infoDays">-</strong></span>
      <span>💰 预算: <strong id="infoBudget">-</strong></span>
      <span>🎯 风格: <strong id="infoStyle">-</strong></span>
    </div>

    <div class="input-area">
      <textarea id="messageInput" placeholder="描述你的旅行计划...（如：帮我规划去成都的4天3夜，预算4000元）" rows="1" oninput="autoResize(this)"></textarea>
      <button class="send-btn" id="sendBtn" onclick="sendMessage()">开始规划 ✈️</button>
    </div>
  </div>
</div>

<script>
let currentSession = null;
let isProcessing = false;

const chatContainer = document.getElementById('chatContainer');
const messageInput = document.getElementById('messageInput');
const sendBtn = document.getElementById('sendBtn');

// Auto-resize textarea
function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 120) + 'px';
}

messageInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

function addMessage(role, content) {
  // Hide welcome screen
  const welcome = document.getElementById('welcomeScreen');
  if (welcome) welcome.remove();

  const div = document.createElement('div');
  div.className = `message ${role}`;

  const avatar = document.createElement('div');
  avatar.className = 'avatar';
  avatar.textContent = role === 'user' ? '🧑' : '🌍';

  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = content;

  div.appendChild(avatar);
  div.appendChild(bubble);
  chatContainer.appendChild(div);
  chatContainer.scrollTop = chatContainer.scrollHeight;
}

function addTyping() {
  const div = document.createElement('div');
  div.className = 'message bot';
  div.id = 'typingIndicator';
  const avatar = document.createElement('div');
  avatar.className = 'avatar';
  avatar.textContent = '🌍';
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.innerHTML = '<div class="typing-indicator"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>';
  div.appendChild(avatar);
  div.appendChild(bubble);
  chatContainer.appendChild(div);
  chatContainer.scrollTop = chatContainer.scrollHeight;
}

function removeTyping() {
  const el = document.getElementById('typingIndicator');
  if (el) el.remove();
}

async function sendMessage() {
  if (isProcessing) return;
  const message = messageInput.value.trim();
  if (!message) return;

  messageInput.value = '';
  messageInput.style.height = 'auto';
  addMessage('user', message);

  isProcessing = true;
  sendBtn.disabled = true;
  addTyping();

  try {
    const session = currentSession || crypto.randomUUID();
    if (!currentSession) currentSession = session;

    const response = await fetch('/api/plan', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, session_id: session })
    });

    removeTyping();
    const data = await response.json();

    if (data.error) {
      addMessage('bot', '❌ 错误: ' + data.error);
    } else {
      addMessage('bot', data.output);

      // Update info bar
      if (data.destination) document.getElementById('infoDest').textContent = data.destination;
      if (data.days) document.getElementById('infoDays').textContent = data.days + '天';
      if (data.budget) document.getElementById('infoBudget').textContent = '¥' + data.budget;
      if (data.travel_style) {
        const styleMap = { budget:'经济', comfortable:'舒适', luxury:'奢华', adventure:'探险', cultural:'文化', beach:'海滨' };
        document.getElementById('infoStyle').textContent = styleMap[data.travel_style] || data.travel_style;
      }
    }
  } catch (err) {
    removeTyping();
    addMessage('bot', '❌ 连接错误: ' + err.message);
  }

  isProcessing = false;
  sendBtn.disabled = false;
  messageInput.focus();
}

function newSession() {
  currentSession = null;
  chatContainer.innerHTML = '';
  document.getElementById('infoDest').textContent = '-';
  document.getElementById('infoDays').textContent = '-';
  document.getElementById('infoBudget').textContent = '-';
  document.getElementById('infoStyle').textContent = '-';
}

function clearChat() { chatContainer.innerHTML = ''; }
function quickTest(text) { messageInput.value = text; sendMessage(); }
</script>
</body>
</html>"""


class ChatHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(CHAT_HTML.encode('utf-8'))
        elif self.path == '/api/health':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps({'ok': True, 'agent': 'Travel Planner'}).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == '/api/plan':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))

            user_message = data.get('message', '')
            session_id = data.get('session_id', str(uuid4()))

            try:
                result = run_planner(session_id, user_message)
                response = json.dumps(result, ensure_ascii=False)
            except Exception as e:
                import traceback
                traceback.print_exc()
                response = json.dumps({'error': str(e)}, ensure_ascii=False)

            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(response.encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        print(f"[HTTP] {args[0]}")


def main():
    init()
    server = HTTPServer(('0.0.0.0', PORT), ChatHandler)
    print(f"[Server] Travel Planner running at http://localhost:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[Server] Shutting down...")
        server.shutdown()


if __name__ == '__main__':
    main()
