# config.py

# =========================================================
# [시스템 경로 및 기본 설정]
# =========================================================
# ★ 경로를 실제 모델 경로로 수정해주세요 (예: model_best_dist.pth)
WEIGHT_PATH = "/home/elicer/dev/01_script/3_eval_script/clear/model_best_dist.pth"
CONFIG_FILE = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"

# 추론 시 리사이즈 크기 (가로, 세로)
TARGET_SIZE = (1280, 720) 

# =========================================================
# [거리 계산 튜닝 (핵심)]
# =========================================================
HORIZON_Y = 320        # 소실점 Y좌표
GEO_SCALE = 1000.0     # 위치 기반 거리 스케일
FOCAL_LENGTH = 1100.0  # 차폭 기반 거리 스케일

# 차종별 실제 너비 (m)
REAL_WIDTH_MAP = {
    "vehicle": 1.85,    "othercar": 1.85,
    "bus": 2.20,        "truck": 2.20,
    "motorcycle": 0.80, "bicycle": 0.60,
    "pedestrian": 0.50, "rider": 0.60
}

# =========================================================
# [알고리즘 임계값]
# =========================================================
SIDE_MARGIN = 20          # 화면 가장자리(px) - 사이드 차량 판단 기준
SIDE_BOTTOM_RATIO = 0.80  # 화면 높이의 80% 아래에 바퀴가 있어야 '진입'으로 인정
ENTERING_THRESH = 2.5     # 이 거리(m) 이하일 때 'Entering' 상태로 판단
ROAD_MASK_CHECK = True    # 도로 위 객체 필터링 (빌딩 제거) 기능 켜기/끄기

# [추가] 추적(Tracking) 및 스무딩 설정 (detector.py에서 사용)
MATCH_DISTANCE_THRESH = 100.0 # 프레임 간 객체 매칭 최대 거리 (픽셀)
SMOOTH_ALPHA = 0.3            # 거리값 지수 이동 평균(EMA) 계수 (0~1)
MAX_LOST_FRAMES = 5           # 객체가 사라져도 유지하는 프레임 수 (Ghost 방지)

# =========================================================
# [클래스 정의]
# =========================================================
THING_CLASSES = ["vehicle", "bus", "truck", "othercar", "motorcycle", "bicycle", "pedestrian", "rider", "trafficsign", "trafficlight", "constructionguide", "trafficdrum"]
STUFF_CLASSES = ["freespace", "curb", "sidewalk", "crosswalk", "roadmark", "whitelane", "yellowlane"]
ALL_CLASSES = THING_CLASSES + STUFF_CLASSES

# [추가] Detector가 참조하는 타겟 클래스 (에러 해결용)
TARGET_CLASSES = ALL_CLASSES