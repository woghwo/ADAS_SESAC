import numpy as np
import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from detectron2.modeling import ROI_HEADS_REGISTRY, StandardROIHeads
from torch import nn
import config as cfg
import config

from detectron2.data import MetadataCatalog

# [속도 최적화] FP16 가속
from torch.cuda.amp import autocast

# ---------------------------------------------------------
# [모델 등록] (가중치 로드용)
# ---------------------------------------------------------
@ROI_HEADS_REGISTRY.register()
class DistanceROIHeads(StandardROIHeads):
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        # 1. 입력 채널 수 확인 및 FC 레이어 정의
        input_dim = self.box_head.output_shape.channels if hasattr(self.box_head, 'output_shape') else 1024
        self.distance_fc = nn.Sequential(nn.Linear(input_dim, 1), nn.ReLU())
        self.max_distance = 100.0

    def _forward_box(self, features, proposals):
        # 2. Feature Map에서 필요한 부분 가져오기
        features_list = [features[f] for f in self.box_in_features]
        
        # 3. [수정됨] 학습(Training) 로직 제거 -> 바로 추론 시작
        # 박스 예측을 위한 Feature Pooling
        box_features = self.box_pooler(features_list, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        pred_class_logits, pred_proposal_deltas = self.box_predictor(box_features)

        # 4. 박스 및 클래스 예측 결과 도출 (Inference)
        pred_instances, _ = self.box_predictor.inference(
            (pred_class_logits, pred_proposal_deltas), 
            proposals
        )
        
        # 탐지된 객체가 없으면 빈 결과 반환
        if len(pred_instances) == 0:
            return pred_instances
        
        # 5. [핵심] 거리 예측 (Distance Prediction)
        # 최종 예측된 박스 위치를 기준으로 다시 Feature를 뽑음 (더 정확함)
        pred_boxes = [x.pred_boxes for x in pred_instances]
        final_box_features = self.box_pooler(features_list, pred_boxes)
        final_box_features = self.box_head(final_box_features)
        
        # 거리 예측 실행 (0~1 값)
        pred_normalized = self.distance_fc(final_box_features)
        
        # 미터(m) 단위로 복원
        final_distances = pred_normalized * self.max_distance 
        
        # 6. 결과 인스턴스에 'pred_distances' 필드 추가
        start_idx = 0
        for instances in pred_instances:
            num_boxes = len(instances)
            instances.pred_distances = final_distances[start_idx : start_idx + num_boxes]
            start_idx += num_boxes
            
        return pred_instances

# ---------------------------------------------------------
# [디텍터 클래스]
# ---------------------------------------------------------
class VehicleDetector:
    def __init__(self):
        print("Initializing VehicleDetector (Distance Estimation Mode)...")
        
        # 1. Config 설정
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        
        # 2. [필수] 커스텀 헤드 설정
        self.cfg.MODEL.ROI_HEADS.NAME = "DistanceROIHeads" 
        self.cfg.merge_from_file(model_zoo.get_config_file(config.CONFIG_FILE))
        self.cfg.MODEL.WEIGHTS = config.WEIGHT_PATH
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(config.ALL_CLASSES)
        
        # 3. [옵션] 속도 최적화 (RPN)
        self.cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 250 # 후보 박스 개수 제한 (속도 향상)
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 # 확신하는 것만 탐지
        self.cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        # 4. [중요] 메타데이터 등록 (학습 때 사용한 클래스 순서 그대로!)
        # 이걸 등록해야 시각화할 때 "Vehicle", "Pedestrian" 이름이 뜹니다.
        self.class_names = config.ALL_CLASSES
        MetadataCatalog.get("my_inference").set(thing_classes=self.class_names)
        
        # 5. Predictor 생성
        # self.model = ... 대신 self.predictor를 사용하세요. (이미지 전처리 자동화)
        self.predictor = DefaultPredictor(self.cfg)
        
        # 추적용 변수
        self.tracks = {}
        self.next_track_id = 0
        
        print(f"Model loaded on {self.cfg.MODEL.DEVICE} with Custom Distance Head")

    def run(self, image):
        h_img, w_img = image.shape[:2]
        
        # 1. [FP32 추론] DefaultPredictor가 전처리부터 추론까지 다 해줌
        predictions = self.predictor(image)
        instances = predictions["instances"].to("cpu")
        
        results = {
            "class": [], "distance": [], "is_entering": [],
            "box": [], "score": [], "mask": [], "id": []
        }
        
        combined_road_mask = None 
        
        # 2. 감지된 객체가 없을 때 (추적 로직은 실행하여 Ghost 처리)
        if not instances.has("pred_boxes"):
            return self._process_tracking([], results, combined_road_mask)

        # 데이터 추출 (CPU로 이동된 텐서를 numpy로 변환)
        boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()
        classes = instances.pred_classes.numpy()
        masks = instances.pred_masks.numpy() if instances.has("pred_masks") else None
        
        # [거리 정보 추출] 모델이 예측한 거리값 가져오기
        if instances.has("pred_distances"):
            model_dists = instances.pred_distances.numpy().flatten()
        else:
            # 만약 모델이 거리를 예측하지 않았다면(구조 불일치 등) 0으로 채움
            model_dists = np.zeros(len(boxes))
        
        # 3. 도로 마스크 통합 (Freespace 클래스 필터링)
        if masks is not None:
            combined_road_mask = np.zeros((h_img, w_img), dtype=bool)
            has_road = False
            for i, c_id in enumerate(classes):
                c_name = self._get_class_name(c_id)
                
                # THING_CLASSES에 없는 클래스는 무시
                # (config.py에 정의된 모든 클래스 리스트)
                if c_name not in cfg.ALL_CLASSES: continue 
                
                if c_name == "freespace":
                    combined_road_mask = np.logical_or(combined_road_mask, masks[i])
                    has_road = True
            
            if not has_road: combined_road_mask = None

        # 4. 객체별 로직 처리
        raw_detections = []
        for i in range(len(boxes)):
            c_name = self._get_class_name(classes[i])
            score = scores[i]
            box = boxes[i].astype(int)
            
            # 필터링
            # TARGET_CLASSES 대신 ALL_CLASSES 또는 THING_CLASSES 사용
            if c_name not in cfg.ALL_CLASSES: continue 
            if c_name == "freespace": continue # 도로는 위에서 처리했음
            if not self._check_score(c_name, score): continue
            if c_name != "freespace" and not self._check_on_road(box, combined_road_mask, w_img, h_img):
                continue
            
            # 거리 계산 (Hybrid: Math + AI)
            raw_dist = 0.0
            is_entering = False
            if c_name != "freespace":
                # _calculate_logic에 모델이 예측한 거리(model_dists[i])를 전달
                raw_dist, is_entering = self._calculate_logic(
                    box, c_name, w_img, h_img, model_dist=model_dists[i]
                )
            
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2

            raw_detections.append({
                "class": c_name,
                "box": box,
                "score": score,
                "mask": masks[i] if masks is not None else None,
                "distance": raw_dist,
                "is_entering": is_entering,
                "center": (cx, cy)
            })
            
        # 5. 추적 및 스무딩 처리
        results, _ = self._process_tracking(raw_detections, results, combined_road_mask)

        # 6. 통합된 도로 마스크 추가
        if combined_road_mask is not None:
            dummy_box = np.array([0, 0, 0, 0])
            self._append_result(results, {
                "class": "freespace",
                "distance": 0.0,
                "is_entering": False,
                "box": dummy_box,
                "score": 1.0,
                "mask": combined_road_mask
            }, -1)
            
        return results, combined_road_mask

    def _process_tracking(self, raw_detections, results, road_mask):
        matched_track_ids = set()
        
        for det in raw_detections:
            # 도로는 추적 대상에서 제외 (매 프레임 새로 갱신)
            if det["class"] == "freespace":
                self._append_result(results, det, -1)
                continue

            best_id = -1
            # cfg.MATCH_DISTANCE_THRESH는 픽셀 단위여야 합니다 (예: 50~100픽셀)
            min_dist = cfg.MATCH_DISTANCE_THRESH 
            
            # 현재 감지된 객체(det)와 기존 트랙(track) 간의 거리 비교
            det_center = np.array(det["center"])
            
            for t_id, track in self.tracks.items():
                if t_id in matched_track_ids: continue
                
                # 중심점 간의 유클리드 거리 (Pixel Distance)
                track_center = np.array(track["center"])
                dist = np.linalg.norm(det_center - track_center)
                
                if dist < min_dist:
                    min_dist = dist
                    best_id = t_id
            
            # 매칭 성공: 기존 트랙 업데이트
            if best_id != -1:
                track = self.tracks[best_id]
                
                # [거리 스무딩] 튀는 값을 방지 (alpha가 작을수록 기존 값 유지 성향이 강함)
                alpha = cfg.SMOOTH_ALPHA
                smoothed_dist = (det["distance"] * alpha) + (track["distance"] * (1 - alpha))
                
                # 정보 갱신
                track["distance"] = smoothed_dist
                track["center"] = det["center"]
                track["box"] = det["box"]
                track["lost_count"] = 0 # 찾았으니 카운트 초기화
                track["data"] = det # 최신 데이터로 교체
                
                # 결과에 추가할 때는 스무딩된 거리 사용
                det_to_export = det.copy()
                det_to_export["distance"] = smoothed_dist 
                
                self._append_result(results, det_to_export, best_id)
                matched_track_ids.add(best_id)
            
            # 매칭 실패: 새로운 객체로 등록
            else:
                new_id = self.next_track_id
                self.next_track_id += 1
                self.tracks[new_id] = {
                    "center": det["center"], 
                    "box": det["box"],
                    "distance": det["distance"], 
                    "lost_count": 0, 
                    "data": det
                }
                self._append_result(results, det, new_id)

        # [고스트 처리] 매칭되지 않은 기존 트랙들 관리
        dead_track_ids = []
        for t_id, track in self.tracks.items():
            if t_id not in matched_track_ids:
                track["lost_count"] += 1
                
                # 너무 오래 사라져 있으면 삭제 명단에 추가
                if track["lost_count"] > cfg.MAX_LOST_FRAMES:
                    dead_track_ids.append(t_id)
                else:
                    # 아직 "Ghost"로 유지 (잠깐 가려진 경우 등)
                    # 데이터는 마지막으로 관측된 데이터를 사용하되, 거리는 유지
                    ghost_data = track["data"].copy() # [안전] 원본 보호를 위해 copy 사용
                    ghost_data["distance"] = track["distance"]
                    
                    # 시각화 시 Ghost임을 표시하고 싶다면 여기에 플래그 추가 가능
                    # ghost_data["is_ghost"] = True 
                    
                    self._append_result(results, ghost_data, t_id)
        
        # 죽은 트랙 삭제
        for t_id in dead_track_ids:
            del self.tracks[t_id]
            
        return results, road_mask

    def _append_result(self, results, data, obj_id):
        results["class"].append(data["class"])
        results["distance"].append(float(data["distance"]))
        results["is_entering"].append(bool(data["is_entering"]))
        results["box"].append(data["box"].tolist())
        results["score"].append(float(data["score"]))
        results["mask"].append(data["mask"])
        results["id"].append(obj_id)

    def _get_class_name(self, c_id):
        if c_id < len(self.class_names):
            return self.class_names[c_id]
        return "unknown"

    def _check_score(self, c_name, score):
        if c_name in ["pedestrian", "bicycle", "motorcycle", "rider"]: thresh = 0.15
        elif c_name in ["vehicle", "bus", "truck"]: thresh = 0.3
        else: thresh = 0.25
        return score >= thresh

    def _check_on_road(self, box, road_mask, w, h):
        if not cfg.ROAD_MASK_CHECK or road_mask is None: return True
        x1, y1, x2, y2 = box
        check_y_start = max(0, y2 - 5)
        check_y_end = min(h, y2)
        check_x_start = max(0, x1)
        check_x_end = min(w, x2)
        region = road_mask[check_y_start:check_y_end, check_x_start:check_x_end]
        overlap = np.count_nonzero(region)
        width = check_x_end - check_x_start
        if width > 0 and overlap < (width * 5) * 0.10: return False
        return True

    def _calculate_logic(self, box, c_name, w_img, h_img, model_dist=0.0):
        # 1. 공통 변수 추출
        x1, y1, x2, y2 = box
        box_w = x2 - x1
        box_h = y2 - y1
        
        # 2. 결과 변수 초기화
        final_dist = 0.0
        dist_geo = 0.0 
        
        # [A] 수학적(Geometric) 거리 계산 (Baseline)
        dist_geo = self._calc_dist_geo(y2)
        dist_width = self._calc_dist_width(box_w, c_name)
        aspect_ratio = box_w / box_h
        
        # (1) Math Distance (기본값)
        math_dist = 0.0
        if c_name in ["bus", "truck"]:
            math_dist = (dist_geo * 0.4) + (dist_width * 0.6)
        elif aspect_ratio < 0.8:
            math_dist = dist_geo
        else:
            math_dist = (dist_geo * 0.4) + (dist_width * 0.6)
        
        # (2) AI 결합 (Hybrid Fusion)
        # 기본 비율: Math 5 : AI 5
        MATH_RATIO = 0.5
        AI_RATIO = 0.5
        
        # [안전장치] 모델 예측값이 유효할 때만 섞음
        if model_dist > 0.1: # 0.1m 이하는 노이즈로 간주
            # [추가] 신뢰도 검사: AI 예측이 Math와 너무 차이(3배 이상)나면 AI 신뢰도 낮춤
            # (AI가 가끔 하늘에 뜬 객체에 대해 100m라고 하거나, 바로 앞을 0m라고 할 때 방어)
            diff = abs(model_dist - math_dist)
            
            if diff > math_dist * 2.0: # 차이가 2배 이상 나면
                # AI 비율을 확 줄임 (Math 9 : AI 1)
                final_dist = (math_dist * 0.9) + (model_dist * 0.1)
            else:
                # 정상 범위면 5:5 융합
                final_dist = (math_dist * MATH_RATIO) + (model_dist * AI_RATIO)
        else:
            # 모델 값이 없으면 수학적 계산만 사용
            final_dist = math_dist
            
        # -------------------------------------------------------
        # 3. 사이드 컷인 로직 (Entering) - 공통 후처리
        # -------------------------------------------------------
        is_entering = False
        is_touching_side = (x1 < cfg.SIDE_MARGIN) or (x2 > w_img - cfg.SIDE_MARGIN)
        
        if is_touching_side:
            is_really_close = y2 > (h_img * cfg.SIDE_BOTTOM_RATIO)
            
            if is_really_close:
                if dist_geo == 0.0: dist_geo = self._calc_dist_geo(y2)
                
                # 사이드에서 접근 시 더 보수적으로(가깝게) 판단
                final_dist = dist_geo * 0.5
                
                if y2 > h_img * 0.90: final_dist = min(final_dist, 2.0)
                elif y2 > h_img * 0.85: final_dist = min(final_dist, 4.0)
                
                if final_dist <= cfg.ENTERING_THRESH: is_entering = True
                
        return final_dist, is_entering

    def _calc_dist_geo(self, y_bottom):
        if y_bottom <= cfg.HORIZON_Y: return 100.0
        dist = cfg.GEO_SCALE / (y_bottom - cfg.HORIZON_Y)
        return max(1.0, dist)

    def _calc_dist_width(self, w_px, c_name):
        real_w = cfg.REAL_WIDTH_MAP.get(c_name, 1.85)
        if w_px <= 0: return 100.0
        dist = (real_w * cfg.FOCAL_LENGTH) / w_px
        return max(1.0, dist)