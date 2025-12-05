import numpy as np

import torch

from detectron2.config import get_cfg

from detectron2.engine import DefaultPredictor

from detectron2 import model_zoo

from detectron2.modeling import ROI_HEADS_REGISTRY, StandardROIHeads

from torch import nn

import config as cfg

import cv2



# [속도 최적화] FP16 가속

#from torch.cuda.amp import autocast

import torch



# ---------------------------------------------------------

# [모델 등록] (가중치 로드용)

# ---------------------------------------------------------

@ROI_HEADS_REGISTRY.register()

class DistanceROIHeads(StandardROIHeads):

    def __init__(self, cfg, input_shape):

        super().__init__(cfg, input_shape)

        input_dim = self.box_head.output_shape.channels if hasattr(self.box_head, 'output_shape') else 1024

        self.distance_fc = nn.Sequential(nn.Linear(input_dim, 1), nn.ReLU())

        self.max_distance = 100.0



    def _forward_box(self, features, proposals):

        features_list = [features[f] for f in self.box_in_features]

        box_features = self.box_pooler(features_list, [x.proposal_boxes for x in proposals])

        box_features = self.box_head(box_features)

        pred_class_logits, pred_proposal_deltas = self.box_predictor(box_features)

       

        # 추론 모드에서 거리 예측값도 함께 반환하도록 수정

        if self.training: return {}

       

        pred_instances, _ = self.box_predictor.inference((pred_class_logits, pred_proposal_deltas), proposals)

        if len(pred_instances) == 0: return pred_instances

       

        # 거리 예측 수행 (모델의 지능 활용)

        pred_boxes = [x.pred_boxes for x in pred_instances]

        final_box_features = self.box_pooler(features_list, pred_boxes)

        final_box_features = self.box_head(final_box_features)

       

        pred_normalized = self.distance_fc(final_box_features)

        final_distances = pred_normalized * self.max_distance

       

        # 예측된 거리를 instances에 저장

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

        print("Initializing VehicleDetector (FP16 + Hybrid Mode)...")

        self.cfg = get_cfg()

        #self.cfg.merge_from_file(model_zoo.get_config_file(cfg.CONFIG_FILE))

        # model_zoo 우회하고, config.py에 적어둔 실제 경로를 그대로 사용

        self.cfg.merge_from_file(cfg.CONFIG_FILE)

        self.cfg.MODEL.WEIGHTS = cfg.WEIGHT_PATH

        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 19

        self.cfg.MODEL.ROI_HEADS.NAME = "DistanceROIHeads"

        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.15

        self.cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

       

        # [속도 최적화] RPN 후보 수 줄이기

        self.cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 250

       

        # [속도 최적화] FP16 적용을 위해 모델 직접 로드

        self.model = DefaultPredictor(self.cfg).model

        self.model.eval()

       

        # 추적용 메모리

        self.tracks = {}

        self.next_track_id = 0
        print(f"Model loaded on {self.cfg.MODEL.DEVICE} with RPN_TOPK=500")



    def run(self, image):

        h_img, w_img = image.shape[:2]

       

        # 1. FP16 추론 실행

        with torch.no_grad():

            input_image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            inputs = [{"image": input_image.to(self.cfg.MODEL.DEVICE)}]



            if self.cfg.MODEL.DEVICE == "cuda":

                # 새 방식

                with torch.amp.autocast("cuda"):

                    outputs = self.model(inputs)[0]

            else:

                outputs = self.model(inputs)[0]



        instances = outputs["instances"].to("cpu")

       

        results = {

            "class": [], "distance": [], "is_entering": [],

            "box": [], "score": [], "mask": [], "id": []

        }

       

        combined_road_mask = None

       

        # 2. 감지된 객체가 없을 때 (Ghost 처리를 위해 추적 로직은 실행)

        if not instances.has("pred_boxes"):

            return self._process_tracking([], results, combined_road_mask)



        # 데이터 추출

        boxes = instances.pred_boxes.tensor.numpy()

        scores = instances.scores.numpy()

        classes = instances.pred_classes.numpy()

        masks = instances.pred_masks.numpy() if instances.has("pred_masks") else None

       

        # 모델이 예측한 거리값 가져오기

        if instances.has("pred_distances"):

            model_dists = instances.pred_distances.numpy().flatten()

        else:

            model_dists = np.zeros(len(boxes))

       

        # 3. 도로 마스크 통합 (Target Class 필터링 포함)

        if masks is not None:

            combined_road_mask = np.zeros((h_img, w_img), dtype=bool)

            has_road = False

            for i, c_id in enumerate(classes):

                c_name = self._get_class_name(c_id)

                if c_name not in cfg.TARGET_CLASSES: continue

               

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

            if c_name not in cfg.TARGET_CLASSES: continue

            if c_name == "freespace": continue # 도로는 위에서 처리했음

            if not self._check_score(c_name, score): continue

            if c_name != "freespace" and not self._check_on_road(box, combined_road_mask, w_img, h_img):

                continue

           

            current_mask = masks[i] if masks is not None else None



            # 거리 계산 (Hybrid: Math + AI)

            raw_dist = 0.0

            is_entering = False

            if c_name != "freespace":

                raw_dist, is_entering = self._calculate_logic(

                    box, c_name, w_img, h_img,

                    model_dist=model_dists[i],

                    mask=current_mask # <--- 여기 추가!

                )

           

            cx = (box[0] + box[2]) / 2

            cy = (box[1] + box[3]) / 2



            raw_detections.append({

                "class": c_name,

                "box": box,

                "score": score,

                "mask": current_mask,

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

            # 도로는 추적 안 함

            if det["class"] == "freespace":

                self._append_result(results, det, -1)

                continue



            best_id = -1

            min_dist = cfg.MATCH_DISTANCE_THRESH

           

            for t_id, track in self.tracks.items():

                if t_id in matched_track_ids: continue

                dist = np.linalg.norm(np.array(det["center"]) - np.array(track["center"]))

                if dist < min_dist:

                    min_dist = dist

                    best_id = t_id

           

            if best_id != -1:

                track = self.tracks[best_id]

                alpha = cfg.SMOOTH_ALPHA

                smoothed_dist = (det["distance"] * alpha) + (track["distance"] * (1 - alpha))

               

                track["distance"] = smoothed_dist

                track["center"] = det["center"]

                track["box"] = det["box"]

                track["lost_count"] = 0

                track["data"] = det

               

                det_to_export = det.copy()

                det_to_export["distance"] = smoothed_dist

                self._append_result(results, det_to_export, best_id)

                matched_track_ids.add(best_id)

            else:

                new_id = self.next_track_id

                self.next_track_id += 1

                self.tracks[new_id] = {

                    "center": det["center"], "box": det["box"],

                    "distance": det["distance"], "lost_count": 0, "data": det

                }

                self._append_result(results, det, new_id)



        dead_track_ids = []

        for t_id, track in self.tracks.items():

            if t_id not in matched_track_ids:

                track["lost_count"] += 1

                if track["lost_count"] > cfg.MAX_LOST_FRAMES:

                    dead_track_ids.append(t_id)

                else:

                    ghost_data = track["data"]

                    ghost_data["distance"] = track["distance"]

                    self._append_result(results, ghost_data, t_id)

       

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

        return cfg.ALL_CLASSES[c_id] if c_id < len(cfg.ALL_CLASSES) else "unknown"



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



    def _calculate_logic(self, box, c_name, w_img, h_img, model_dist=0.0, mask=None):
        # 1. 공통 변수 추출
        x1, y1, x2, y2 = box
        box_w = x2 - x1
        box_h = y2 - y1
        
        # 중심점 계산
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2  # Y축 중심
        
        # ---------------------------------------------------
        # [기존 거리 계산 로직 유지]
        # ---------------------------------------------------
        dist_geo = self._calc_dist_geo(y2)
        dist_width = self._calc_dist_width(box_w, c_name)
        aspect_ratio = box_w / box_h
        
        math_dist = 0.0
        if c_name in ["bus", "truck"]:
            math_dist = (dist_geo * 0.4) + (dist_width * 0.6)
        elif aspect_ratio < 0.8:
            math_dist = dist_geo
        else:
            math_dist = (dist_geo * 0.4) + (dist_width * 0.6)
        
        MATH_RATIO = 0.5
        AI_RATIO = 0.5
        
        final_dist = math_dist
        if model_dist > 0:
            final_dist = (math_dist * MATH_RATIO) + (model_dist * AI_RATIO)

        # ---------------------------------------------------
        # [Cut-in 판별 로직]
        # ---------------------------------------------------
        is_entering = False
        
        # 사다리꼴 지오메트리 정의
        center_x = int(w_img // 2)
        bottom_w = 450
        top_w = 50
        bottom_y = int(h_img - 20)
        top_y = int(h_img // 3)
        
        # 거리가 너무 먼 객체는 Cut-in 판단에서 제외
        is_close_enough = y2 > top_y 

        if is_close_enough and (mask is not None):
            # [A] 사다리꼴 라인 터치 여부 (기존 로직)
            trap_h = bottom_y - top_y
            current_y = np.clip(y2, top_y, bottom_y)
            ratio = (current_y - top_y) / trap_h
            current_half_w = ((top_w * (1 - ratio)) + (bottom_w * ratio)) / 2
            
            lane_left_x = center_x - current_half_w
            lane_right_x = center_x + current_half_w
            
            touching_left = (x1 < lane_left_x) and (x2 > lane_left_x)
            touching_right = (x1 < lane_right_x) and (x2 > lane_right_x)
            is_touching_corridor = touching_left or touching_right

            # [B] 마스크 면적 비율 계산 (기존 로직)
            if is_touching_corridor:
                corridor_mask = np.zeros((h_img, w_img), dtype=np.uint8)
                pts = np.array([
                    [center_x - bottom_w // 2, bottom_y],
                    [center_x + bottom_w // 2, bottom_y],
                    [center_x + top_w // 2, top_y],
                    [center_x - top_w // 2, top_y]
                ], np.int32)
                cv2.fillPoly(corridor_mask, [pts], 1)
                
                roi_y1, roi_y2 = y1, y2
                roi_x1, roi_x2 = x1, x2
                
                vehicle_roi = mask[roi_y1:roi_y2, roi_x1:roi_x2]
                corridor_roi = corridor_mask[roi_y1:roi_y2, roi_x1:roi_x2]
                
                intersection = np.logical_and(vehicle_roi, corridor_roi)
                overlap_area = np.sum(intersection)
                total_corridor_area = np.sum(corridor_mask)
                
                if total_corridor_area > 0:
                    blockage_ratio = overlap_area / total_corridor_area
                    # 기존 조건: 20% ~ 50% 사이일 때만 진입 중으로 판단
                    if 0.15 <= blockage_ratio < 0.50:
                        is_entering = True

        # ---------------------------------------------------
        # [NEW] 중심점(Center) 체크: 이미 들어왔으면 Cut-in 해제
        # Visualizer의 'too close' 체크 방식과 동일한 로직 적용
        # ---------------------------------------------------
        if is_entering:
            # 객체의 중심 Y(cy)를 기준으로 사다리꼴 폭을 다시 계산
            # (이유: 박스 바닥(y2) 기준이 아니라 몸통 중심이 들어왔는지 봐야 하므로)
            trap_h = bottom_y - top_y
            
            # cy가 사다리꼴 높이 범위 내에 있도록 clip
            safe_cy = np.clip(cy, top_y, bottom_y)
            
            # 중심 Y 위치에서의 사다리꼴 비율 (0.0 ~ 1.0)
            center_ratio = (safe_cy - top_y) / trap_h
            
            # 중심 Y 위치에서의 사다리꼴 절반 폭
            center_half_w = ((top_w * (1 - center_ratio)) + (bottom_w * center_ratio)) / 2
            
            # 중심점 기준 왼쪽/오른쪽 경계선
            center_left_bound = center_x - center_half_w
            center_right_bound = center_x + center_half_w
            
            # 중심점(cx)이 경계선 사이에 존재하는가?
            is_center_inside = (center_left_bound < cx < center_right_bound)
            
            # [결론] 중심이 안에 있으면 "이미 주행로에 있는 차" -> Cut-in 경고 해제
            if is_center_inside:
                is_entering = False
                
        return final_dist, is_entering

   

    # 후방 라이다 센서 이용 > 미리 알림

    def _calc_dist_geo(self, y_bottom):

        if y_bottom <= cfg.HORIZON_Y: return 100.0

        dist = cfg.GEO_SCALE / (y_bottom - cfg.HORIZON_Y)

        return max(1.0, dist)



    def _calc_dist_width(self, w_px, c_name):

        real_w = cfg.REAL_WIDTH_MAP.get(c_name, 1.85)

        if w_px <= 0: return 100.0

        dist = (real_w * cfg.FOCAL_LENGTH) / w_px

        return max(1.0, dist)