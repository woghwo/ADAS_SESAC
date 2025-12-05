import cv2
import uvicorn
import time
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
import numpy as np

# 사용자 모듈
from jh_detector import VehicleDetector
from jh_visualizer import TrafficVisualizer
import config as cfg

app = FastAPI(title="Autonomous Driving Safety System")

# ---------------------------------------------------------
# [전역 상태] 재생 제어용 변수
# ---------------------------------------------------------
class SystemState:
    def __init__(self):
        self.paused = False

state = SystemState()

# ---------------------------------------------------------
# [시스템 초기화]
# ---------------------------------------------------------
print("Initializing AI System...")
detector = VehicleDetector()
visualizer = TrafficVisualizer()
print("System Ready.")

def video_stream_generator():
    """
    비디오 스트리밍 제너레이터
    - 일시정지 상태일 때는 프레임 처리를 건너뛰고 대기합니다.
    """
    cap = cv2.VideoCapture(cfg.VIDEO_PATH)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {cfg.VIDEO_PATH}")
        return

    # 마지막으로 전송한 프레임을 저장해두기 위한 변수
    encoded_frame_cache = None

    while True:
        # 1. [일시정지 체크]
        if state.paused:
            # CPU 과부하 방지를 위해 살짝 대기
            time.sleep(0.1)
            
            # (선택사항) 브라우저 연결 유지를 위해 마지막 프레임을 계속 보낼 수도 있지만,
            # 최신 브라우저는 데이터 전송이 멈춰도 마지막 이미지를 유지하므로 
            # 대역폭 절약을 위해 아무것도 yield 하지 않고 continue 합니다.
            # 만약 연결이 끊긴다면 아래 코드를 주석 해제하세요.
            # if encoded_frame_cache:
            #     yield (b'--frame\r\n'
            #            b'Content-Type: image/jpeg\r\n\r\n' + encoded_frame_cache + b'\r\n')
            continue

        # 2. 영상 읽기
        ret, frame = cap.read()
        if not ret:
            # 영상 끝나면 처음으로 (무한 루프)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # 3. 리사이즈
        frame_resized = cv2.resize(frame, cfg.TARGET_SIZE)

        # 4. AI 추론 (Detector)
        results, road_mask = detector.run(frame_resized)

        # 5. 시각화 (Visualizer)
        final_frame = visualizer.draw_results(frame_resized, results)

        # 6. 인코딩 및 전송
        _, buffer = cv2.imencode('.jpg', final_frame)
        encoded_frame_cache = buffer.tobytes() # 캐시 저장

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + encoded_frame_cache + b'\r\n')

    cap.release()

@app.get("/")
async def index():
    """메인 대시보드 페이지 (UI + JS 제어 추가)"""
    html_content = """
    <!DOCTYPE html>
    <html>
        <head>
            <title>Autonomous Safety Dashboard</title>
            <style>
                body { background-color: #1a1a1a; color: white; font-family: 'Segoe UI', Arial, sans-serif; text-align: center; margin: 0; padding: 20px; }
                .container { display: inline-block; position: relative; border: 3px solid #333; box-shadow: 0 0 30px rgba(0,0,0,0.7); }
                h1 { margin-bottom: 5px; color: #4CAF50; letter-spacing: 2px; }
                .status-bar { margin-top: 15px; font-size: 14px; color: #aaa; background: #222; padding: 10px; border-radius: 5px; display: inline-block; }
                img { width: 100%; max-width: 1280px; height: auto; display: block; }
                
                /* 일시정지 오버레이 아이콘 */
                .pause-overlay {
                    position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
                    font-size: 80px; color: rgba(255, 255, 255, 0.8);
                    display: none; pointer-events: none; text-shadow: 0 0 10px black;
                }
                
                /* 버튼 스타일 */
                .btn {
                    background-color: #444; color: white; border: none; padding: 10px 20px;
                    text-align: center; text-decoration: none; display: inline-block;
                    font-size: 16px; margin: 4px 2px; cursor: pointer; border-radius: 4px;
                    transition: background 0.3s;
                }
                .btn:hover { background-color: #666; }
                .controls { margin-top: 15px; }
            </style>
        </head>
        <body>
            <h1>👀Open Eye👀</h1>
            
            <div class="container" onclick="togglePause()">
                <img src="/video_feed" id="videoStream" alt="AI Video Stream">
                <div class="pause-overlay" id="pauseIcon">⏸</div>
            </div>
            
            <div class="controls">
                <button class="btn" onclick="togglePause()">⏯ Play / Pause (Space)</button>
            </div>

            <div class="status-bar">
                System Status: <span id="sysStatus" style="color: #00ff00;">LIVE</span> | 
                Mode: <span style="color: cyan;">HYBRID FUSION (AI + GEO)</span> | 
                Device: CUDA (FP16)
            </div>

            <script>
                // 일시정지 제어 함수
                async function togglePause() {
                    try {
                        const response = await fetch('/toggle_pause');
                        const data = await response.json();
                        updateUI(data.paused);
                    } catch (error) {
                        console.error('Error:', error);
                    }
                }

                // UI 업데이트
                function updateUI(isPaused) {
                    const icon = document.getElementById('pauseIcon');
                    const status = document.getElementById('sysStatus');
                    
                    if (isPaused) {
                        icon.style.display = 'block';
                        status.innerText = "PAUSED";
                        status.style.color = "yellow";
                    } else {
                        icon.style.display = 'none';
                        status.innerText = "LIVE";
                        status.style.color = "#00ff00";
                    }
                }

                // 스페이스바 이벤트 리스너
                document.addEventListener('keydown', function(event) {
                    if (event.code === 'Space') {
                        event.preventDefault(); // 스크롤 방지
                        togglePause();
                    }
                });
            </script>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/video_feed")
async def video_feed():
    """비디오 스트리밍 엔드포인트"""
    return StreamingResponse(video_stream_generator(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/toggle_pause")
async def toggle_pause():
    """일시정지 상태 토글 API"""
    state.paused = not state.paused
    return JSONResponse(content={"paused": state.paused})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)