# app1.py

from flask import Flask, render_template, request, jsonify, Response
import json
import time
import threading
import conf # 설정 파일 import

try:
    # [OPT] ultra-fast json (있으면 사용, 없으면 표준 json)
    import orjson as _json
    def _dumps(obj): return _json.dumps(obj)
    def _loads(s): return _json.loads(s)
except Exception:
    def _dumps(obj): return json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    def _loads(s): return json.loads(s)

app = Flask(__name__, static_folder="static", template_folder="templates")

# [OPT] Flask JSON 오버헤드 최소화
app.config.update(
    JSONIFY_PRETTYPRINT_REGULAR=False,  # 불필요한 공백 제거
    JSON_SORT_KEYS=False               # 키 정렬 비활성화
)

# [RT] 쓰레드 세이프 공유 상태 (in-place 업데이트로 GC/할당 최소화)
_gaze = {"x": 0, "y": 0}
_lock = threading.Lock()

# [RT] SSE 구독자 제어용 이벤트
_update_event = threading.Event()

@app.route('/')
def index():
    # 템플릿 렌더링 시 conf.py의 논리 해상도 값을 전달
    return render_template("index2.html", 
                           logical_w=conf.LOGICAL_WIDTH, 
                           logical_h=conf.LOGICAL_HEIGHT)

@app.route('/button<int:num>')
def button_page(num):
    return f"<h1>Button {num} 페이지 입니다.</h1>"

@app.route('/api/gaze', methods=['POST'])
def update_gaze():
    """
    [OPT]
    - in-place 갱신(+락)으로 GIL 경합/JIT 할당 줄임
    - JSON 파싱 최소화
    - SSE 구독자에게 이벤트 통지
    """
    try:
        data = request.get_json(force=True, silent=True)  # werkzeug 내부 fast path
        if not data:
            return jsonify(success=False, error="no_json"), 400

        x = int(data.get("x", _gaze["x"]))
        y = int(data.get("y", _gaze["y"]))

        with _lock:
            _gaze["x"] = x
            _gaze["y"] = y

        # 새 좌표가 올 때마다 스트림에 알림
        _update_event.set()

        return jsonify(success=True)
    except Exception as e:
        return jsonify(success=False, error=str(e)), 400

@app.route('/api/gaze', methods=['GET'])
def get_gaze():
    """
    [OPT]
    - jsonify의 pretty/sort 비활성화로 경량 응답
    """
    with _lock:
        payload = {"x": _gaze["x"], "y": _gaze["y"]}
    # orjson 사용 시 bytes 반환 → Response로 직접 반환
    return Response(_dumps(payload), mimetype="application/json")

@app.route('/api/gaze/stream')
def stream_gaze():
    """
    [NEW][RT] Server-Sent Events 스트림 (브라우저 EventSource로 실시간 수신)
    - 폴링 없이 좌표 갱신 즉시 push → 지연/트래픽 감소
    - idle 타임아웃 방지를 위해 주기적 keepalive 전송
    """
    def gen():
        # 첫 전송: 현재 상태
        last = (-1, -1)
        keepalive_ts = time.time()

        while True:
            # 좌표 업데이트 이벤트를 대기 (최대 1초)
            _update_event.wait(timeout=1.0)

            # 갱신 또는 keepalive 전송
            with _lock:
                x, y = _gaze["x"], _gaze["y"]

            if (x, y) != last:
                payload = _dumps({"x": x, "y": y}).decode("utf-8")
                yield f"data: {payload}\n\n"
                last = (x, y)
                # 이벤트 소비 후 플래그 리셋 (다른 구독자와의 경쟁에 영향 없음)
                _update_event.clear()
            else:
                # 10초마다 keepalive
                now = time.time()
                if now - keepalive_ts > 10:
                    yield ":\n\n"  # SSE 주석 프레임
                    keepalive_ts = now

    headers = {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        # Nginx 등 프록시가 버퍼링하지 않도록 권장 (환경에 따라 무시될 수 있음)
        "X-Accel-Buffering": "no"
    }
    return Response(gen(), headers=headers)

@app.route('/healthz')
def healthz():
    return "ok", 200

if __name__ == '__main__':
    # conf.py의 서버 설정값을 사용
    app.run(host=conf.SERVER_HOST, port=conf.SERVER_PORT, debug=False, threaded=True)

