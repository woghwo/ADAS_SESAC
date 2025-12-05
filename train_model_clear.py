import os

# =========================================================
# [1. 시스템 설정] 멈춤 방지 및 성능 최적화 (최상단 필수)
# =========================================================
os.environ["TORCH_Dynamo"] = "disable"
os.environ["TORCH_INDUCTOR"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import cv2
cv2.setNumThreads(0) # OpenCV 멀티스레딩 충돌 방지

import random
import numpy as np
import torch
import datetime
import json
import logging
import copy
from collections import defaultdict, OrderedDict
from tqdm import tqdm

# Detectron2 Imports
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultTrainer, HookBase
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_train_loader, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, DatasetEvaluator, DatasetEvaluators
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine.hooks import BestCheckpointer
from detectron2.modeling import ROI_HEADS_REGISTRY, StandardROIHeads
from torch import nn
import torch.nn.functional as F
from detectron2.structures import BoxMode, pairwise_iou
import detectron2.utils.comm as comm

# =========================================================
# [설정 영역] 사용자 환경
# =========================================================

# [디버깅 옵션]
# 0 또는 None: 전체 데이터 사용 (실전 학습)
# 숫자 (예: 1000, 200): 해당 개수만큼만 스마트하게 뽑아서 사용 (빠른 테스트)
DEBUG_TRAIN_LIMIT = 0  # 학습 루프 확인용
DEBUG_VAL_LIMIT = 0     # 검증 및 지표 출력 확인용

# 1. 학습 데이터 (A: 고속도로)
HWAY_TRAIN_JSON = '/home/elicer/data/090_AD_Hway_Day/Training/02_label_data/train_coco_with_distance.json'
HWAY_TRAIN_ROOT = '/home/elicer/data/090_AD_Hway_Day/Training/01_raw_data/image_data'

# 2. 학습 데이터 (B: 도심)
CITY_TRAIN_JSON = '/home/elicer/data/092_AD_City_Day/Training/02_label_data/label_day_clear/train_city_coco_with_dist.json'
CITY_TRAIN_ROOT = '/home/elicer/data/092_AD_City_Day/Training/01_raw_data/image_day_clear'

# 3. 검증 데이터
VAL_JSON_PATH = '/home/elicer/data/090_AD_Hway_Day/Validation/integrated_val.json'
VAL_IMAGE_ROOT = None 

# 4. 저장 경로
OUTPUT_DIR = '/home/elicer/dev/gt/data/model/' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# =========================================================
# [3. 모델 정의] Custom Head (Distance)
# =========================================================
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

        if self.training:
            pred_normalized = self.distance_fc(box_features)
            losses = self.box_predictor.losses((pred_class_logits, pred_proposal_deltas), proposals)
            losses["loss_distance"] = self._get_distance_loss(pred_normalized, proposals)
            return losses 
        else:
            pred_instances, _ = self.box_predictor.inference((pred_class_logits, pred_proposal_deltas), proposals)
            if len(pred_instances) == 0: return pred_instances
            
            pred_boxes = [x.pred_boxes for x in pred_instances]
            final_box_features = self.box_pooler(features_list, pred_boxes)
            final_box_features = self.box_head(final_box_features)
            
            pred_normalized = self.distance_fc(final_box_features)
            final_distances = pred_normalized * self.max_distance 
            
            start_idx = 0
            for instances in pred_instances:
                num_boxes = len(instances)
                instances.pred_distances = final_distances[start_idx : start_idx + num_boxes]
                start_idx += num_boxes
            return pred_instances

    def _get_distance_loss(self, pred_distances, proposals):
        gt_distances = []
        for p in proposals:
            if p.has("gt_distances"):
                gt_distances.append(p.gt_distances)
            else:
                gt_distances.append(torch.full((len(p),), -1.0, device=pred_distances.device))
        
        if not gt_distances: return pred_distances.sum() * 0
        gt_distances = torch.cat(gt_distances).flatten()
        pred_distances = pred_distances.flatten()
        valid_mask = (gt_distances > -0.5) 
        if valid_mask.sum() == 0: return pred_distances.sum() * 0
        
        loss = F.smooth_l1_loss(pred_distances[valid_mask], gt_distances[valid_mask], reduction='mean')
        return loss * 0.5

# =========================================================
# [4. 평가 모듈] Rich Distance Evaluator
# =========================================================
class RichDistanceEvaluator(DatasetEvaluator):
    """거리 구간별 오차(MAE) 및 전체 성능을 평가합니다."""
    def __init__(self, dataset_name, max_dist=100.0):
        self.dataset_name = dataset_name
        self.max_dist = max_dist
        self.reset()

    def reset(self):
        self.buckets = {"0-10m": [], "10-30m": [], "30-50m": [], "50m+": []}
        self.all_errors = []

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            pred = output["instances"].to("cpu")
            if not input.get("instances"): continue
            gt = input["instances"].to("cpu")
            
            if len(pred) == 0 or len(gt) == 0: continue
            
            ious = pairwise_iou(pred.pred_boxes, gt.gt_boxes)
            if ious.numel() == 0: continue
            matched_vals, matched_idxs = ious.max(dim=1)
            
            valid_mask = matched_vals > 0.5
            if valid_mask.sum() == 0: continue
            
            if not hasattr(pred, "pred_distances") or not hasattr(gt, "gt_distances"): continue
            
            pred_dists = pred.pred_distances[valid_mask]
            gt_idxs = matched_idxs[valid_mask]
            gt_dists = gt.gt_distances[gt_idxs]
            
            valid_gt_mask = gt_dists > -0.001 
            if valid_gt_mask.sum() == 0: continue
            
            p_d_m = pred_dists[valid_gt_mask].flatten()
            # GT는 매퍼에서 Normalize 되었으므로 다시 복원해서 비교 (m 단위)
            g_d_m = gt_dists[valid_gt_mask].flatten() * self.max_dist
            
            abs_errs = torch.abs(p_d_m - g_d_m).tolist()
            gt_vals = g_d_m.tolist()
            
            for err, dist in zip(abs_errs, gt_vals):
                self.all_errors.append(err)
                if dist < 10: self.buckets["0-10m"].append(err)
                elif 10 <= dist < 30: self.buckets["10-30m"].append(err)
                elif 30 <= dist < 50: self.buckets["30-50m"].append(err)
                else: self.buckets["50m+"].append(err)

    def evaluate(self):
        results = OrderedDict()
        if self.all_errors:
            results["Total_MAE(m)"] = np.mean(self.all_errors)
        else:
            results["Total_MAE(m)"] = 0.0
            
        print("\n" + "="*40)
        print(" 📏 Distance Prediction Analysis (MAE)")
        print("="*40)
        for range_name, errors in self.buckets.items():
            if len(errors) > 0:
                mae = np.mean(errors)
                count = len(errors)
                results[f"MAE_{range_name}"] = mae
                print(f"   Target {range_name:<7}: {mae:.2f}m  (Count: {count})")
            else:
                print(f"   Target {range_name:<7}: N/A    (Count: 0)")
        
        print(f"   [Total] Average: {results['Total_MAE(m)']:.2f}m")
        print("="*40 + "\n")
        return {"dist_metrics": results}

# =========================================================
# [5. 데이터 로딩] 스마트 샘플링 & 고속 파싱
# =========================================================

def get_custom_dicts_smart(json_path, image_root=None, dataset_name="", limit=0):
    print(f"📂 Loading {dataset_name}: {json_path}")
    
    # 1. JSON 데이터 로드
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    categories = sorted(data['categories'], key=lambda x: x['id'])
    thing_classes = [c['name'] for c in categories]
    all_images = data['images']
    
    # 2. 어노테이션 고속 인덱싱 (리스트 순회 X, 딕셔너리 조회 O)
    print(f"   -> Indexing annotations...")
    img_to_anns = defaultdict(list)
    cat_to_img_ids = defaultdict(set)
    
    for ann in data['annotations']:
        img_to_anns[ann['image_id']].append(ann)
        cat_to_img_ids[ann['category_id']].add(ann['image_id'])

    # 3. 스마트 샘플링 (디버깅용 데이터 제한 시)
    final_images = []
    
    if limit > 0 and limit < len(all_images):
        print(f"⚠️ [DEBUG] {dataset_name}: 클래스 균형을 맞춰 {limit}장을 추출합니다.")
        selected_img_ids = set()
        MIN_PER_CLASS = 10 # 최소 10장씩은 보장
        
        # 희귀 클래스부터 우선 확보
        sorted_cats = sorted(categories, key=lambda c: len(cat_to_img_ids[c['id']]))
        
        for cat in sorted_cats:
            cat_id = cat['id']
            candidates = list(cat_to_img_ids[cat_id])
            random.shuffle(candidates)
            
            count = 0
            for img_id in candidates:
                if count >= MIN_PER_CLASS: break
                if img_id not in selected_img_ids:
                    selected_img_ids.add(img_id)
                    count += 1
        
        # 남은 공간은 랜덤으로 채우기
        remaining_slots = limit - len(selected_img_ids)
        if remaining_slots > 0:
            all_ids = [img['id'] for img in all_images]
            random.shuffle(all_ids)
            for img_id in all_ids:
                if remaining_slots <= 0: break
                if img_id not in selected_img_ids:
                    selected_img_ids.add(img_id)
                    remaining_slots -= 1
        
        # 선택된 ID로 이미지 리스트 생성
        img_lookup = {img['id']: img for img in all_images}
        final_images = [img_lookup[iid] for iid in selected_img_ids]
        print(f"   -> 균형 추출 완료: {len(final_images)}장")
        
    else:
        # 제한 없으면 전체 사용
        final_images = all_images 

    # 4. 데이터셋 딕셔너리 생성 (파싱)
    dataset_dicts = []
    missing_count = 0
    
    for img in tqdm(final_images, desc=f"Parsing {dataset_name}"):
        record = {}
        file_name = img['file_name']
        
        # 경로 처리
        if os.path.isabs(file_name):
            full_path = file_name
        else:
            if image_root is None: continue
            full_path = os.path.join(image_root, file_name)
            
        # [안전장치] 파일 존재 여부 확인 (삭제된 데이터 건너뛰기)
        if not os.path.exists(full_path):
            missing_count += 1
            continue
            
        record["file_name"] = full_path
        record["image_id"] = img['id']
        record["height"] = img['height']
        record["width"] = img['width']
        
        objs = []
        # 인덱싱된 딕셔너리에서 즉시 조회 (O(1))
        current_anns = img_to_anns.get(img['id'], [])
        
        for ann in current_anns:
            dist_val = ann.get("distance", -1.0)
            if dist_val is None: dist_val = -1.0
            
            obj = {
                "bbox": ann["bbox"],
                "bbox_mode": BoxMode.XYWH_ABS, 
                "category_id": ann["category_id"] - 1, 
                "segmentation": ann.get("segmentation", []),
                "iscrowd": ann.get("iscrowd", 0),
                "distance": float(dist_val)
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    
    if missing_count > 0:
        print(f"⚠️ [Warning] {missing_count}장의 이미지를 찾을 수 없어 건너뛰었습니다.")
    
    return dataset_dicts, thing_classes

# =========================================================
# [6. Mappers & Trainer]
# =========================================================
def train_mapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)
    try:
        # [핵심] 이미지 읽기 시도
        image = utils.read_image(dataset_dict["file_name"], format="BGR")
    except Exception as e:
        # [방어] 읽기 실패 시 로그 남기고 None 반환 (알아서 건너뜀)
        print(f"\n⚠️ [Train Skip] 이미지 손상/로딩 실패: {dataset_dict['file_name']} ({e})")
        return None

    # 이미지 크기 확인 (0바이트 파일 등 방어)
    if image is None or image.size == 0:
        print(f"\n⚠️ [Train Skip] 빈 이미지 파일: {dataset_dict['file_name']}")
        return None

    transform_list = [T.Resize((800, 800)), T.RandomFlip(prob=0.5)]
    image, transforms = T.apply_transform_gens(transform_list, image)
    dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

    if "annotations" in dataset_dict:
        annos = [
            utils.transform_instance_annotations(obj, transforms, image.shape[:2])
            for obj in dataset_dict.pop("annotations")
            if obj.get("iscrowd", 0) == 0
        ]
        instances = utils.annotations_to_instances(annos, image.shape[:2])
        
        MAX_DIST = 100.0
        normalized_dists = []
        for obj in annos:
            d = obj.get("distance", -1.0)
            if d != -1.0: d = min(d, MAX_DIST) / MAX_DIST
            normalized_dists.append(d)
            
        instances.gt_distances = torch.tensor(normalized_dists, dtype=torch.float32)
        dataset_dict["instances"] = utils.filter_empty_instances(instances)
    
    return dataset_dict

def test_mapper(dataset_dict):
    """검증용 매퍼 (Augmentation 없음, GT 포함, 에러 방어 포함)"""
    dataset_dict = copy.deepcopy(dataset_dict)
    try:
        # [핵심] 이미지 읽기 시도
        image = utils.read_image(dataset_dict["file_name"], format="BGR")
    except Exception as e:
        # [방어] 검증 도중 멈추지 않도록 Skip
        print(f"\n⚠️ [Val Skip] 이미지 손상/로딩 실패: {dataset_dict['file_name']} ({e})")
        return None
    
    if image is None or image.size == 0:
        print(f"\n⚠️ [Val Skip] 빈 이미지 파일: {dataset_dict['file_name']}")
        return None
    
    # Resize만 (학습과 동일 조건)
    transform_list = [T.Resize((800, 800))]
    image, transforms = T.apply_transform_gens(transform_list, image)
    dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

    if "annotations" in dataset_dict:
        annos = [
            utils.transform_instance_annotations(obj, transforms, image.shape[:2])
            for obj in dataset_dict.pop("annotations")
            if obj.get("iscrowd", 0) == 0
        ]
        instances = utils.annotations_to_instances(annos, image.shape[:2])
        
        MAX_DIST = 100.0
        normalized_dists = []
        for obj in annos:
            d = obj.get("distance", -1.0)
            if d != -1.0:
                d = min(d, MAX_DIST) / MAX_DIST
            normalized_dists.append(d)
            
        instances.gt_distances = torch.tensor(normalized_dists, dtype=torch.float32)
        dataset_dict["instances"] = instances
        
    return dataset_dict

class MyTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=train_mapper, num_workers=4)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name, mapper=test_mapper, num_workers=2)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None: output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return DatasetEvaluators([
            COCOEvaluator(dataset_name, output_dir=output_folder),
            RichDistanceEvaluator(dataset_name)
        ])
    
    def build_hooks(self):
        hooks = super().build_hooks()
        
        # 1. 검증 실행 Hook (이게 먼저 있어야 점수가 나옴)
        hooks.insert(-1, ValidationLossHook(self.cfg))
        
        # 2. Best Model 저장 Hook
        hooks.append(BestCheckpointer(
            self.cfg.TEST.EVAL_PERIOD, 
            self.checkpointer, 
            "val_loss_distance", # 감시할 지표
            "min",
            file_prefix="model_best_dist" 
        ))
        
        # 3. [추가] Early Stopping Hook
        hooks.append(EarlyStoppingHook(
            patience=5,                 # 5번 연속으로 안 좋아지면 멈춤 (5 * 5000 iter = 25,000 iter 동안)
            metric_name="val_loss_distance", # BestCheckpointer와 같은 지표 추천
            mode="min",
            threshold=0.0001            # 0.0001이라도 줄어야 인정
        ))
        
        return hooks

class ValidationLossHook(HookBase):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.clone()
        # Test Dataset으로 Validation Loss 계산
        self.cfg.DATASETS.TRAIN = self.cfg.DATASETS.TEST 
        self._loader = iter(build_detection_train_loader(self.cfg, mapper=test_mapper, num_workers=2))
        
    def after_step(self):
        if (self.trainer.iter + 1) % self.cfg.TEST.EVAL_PERIOD != 0: return
        try: 
            data = next(self._loader)
            if data is None: return
        except StopIteration:
            self._loader = iter(build_detection_train_loader(self.cfg, mapper=test_mapper, num_workers=2))
            data = next(self._loader)
        
        with torch.no_grad():
            self.trainer.model.train()
            loss_dict = self.trainer.model(data)
            loss_dict_reduced = {"val_" + k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            if comm.is_main_process():
                self.trainer.storage.put_scalars(**loss_dict_reduced)

class EarlyStoppingHook(HookBase):
    def __init__(self, patience=5, metric_name="val_loss_distance", mode="min", threshold=0.0001):
        """
        Args:
            patience (int): 성능이 개선되지 않아도 기다려줄 횟수 (검증 주기 기준)
            metric_name (str): 감시할 지표 (예: val_loss_distance, val_bbox_AP)
            mode (str): "min"이면 낮을수록 좋음(Loss), "max"면 높을수록 좋음(AP)
            threshold (float): 개선으로 간주할 최소 변화량
        """
        self.patience = patience
        self.metric_name = metric_name
        self.mode = mode
        self.threshold = threshold
        self.best_score = float("inf") if mode == "min" else float("-inf")
        self.wait_count = 0
        
    def after_step(self):
        # 검증 주기(EVAL_PERIOD)마다 체크
        if (self.trainer.iter + 1) % self.trainer.cfg.TEST.EVAL_PERIOD != 0:
            return

        # Tensorboard/Storage에 기록된 최신 지표 가져오기
        storage = self.trainer.storage
        if self.metric_name not in storage.latest():
            return # 아직 지표가 없으면 패스

        # 현재 점수 (이동 평균이 아닌 최신 값 사용)
        current_score = storage.latest()[self.metric_name][0]

        # 성능 개선 여부 판단
        improved = False
        if self.mode == "min":
            if current_score < self.best_score - self.threshold:
                improved = True
        else: # mode == "max"
            if current_score > self.best_score + self.threshold:
                improved = True

        if improved:
            self.best_score = current_score
            self.wait_count = 0 # 카운트 초기화
            # (옵션) 여기서 Best Model은 BestCheckpointer가 알아서 저장해줌
        else:
            self.wait_count += 1
            print(f"\n⚠️ [EarlyStopping] {self.wait_count}/{self.patience} patience used. (Best: {self.best_score:.4f}, Curr: {current_score:.4f})")

        # 인내심 바닥남 -> 학습 강제 종료
        if self.wait_count >= self.patience:
            print(f"\n🛑 [EarlyStopping] Stopping training early! No improvement for {self.patience} evals.")
            self.trainer.storage.put_scalar("early_stop", 1) # 기록용
            raise StopIteration # 학습 루프 탈출 예외 발생

# =========================================================
# [7. Main Execution]
# =========================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    setup_logger(output=OUTPUT_DIR) 
    
    print(">>> [1/4] Loading Datasets...")
    
    # 1. 학습 데이터 로드 (스마트 샘플링)
    hway_dicts, classes = get_custom_dicts_smart(HWAY_TRAIN_JSON, HWAY_TRAIN_ROOT, "Highway", limit=DEBUG_TRAIN_LIMIT)
    city_dicts, _ = get_custom_dicts_smart(CITY_TRAIN_JSON, CITY_TRAIN_ROOT, "City", limit=DEBUG_TRAIN_LIMIT)
    total_train_dicts = hway_dicts + city_dicts
    
    DatasetCatalog.register("my_train", lambda: total_train_dicts)
    MetadataCatalog.get("my_train").set(thing_classes=classes)
    
    # 2. 검증 데이터 로드 (스마트 샘플링)
    val_dicts, _ = get_custom_dicts_smart(VAL_JSON_PATH, image_root=None, dataset_name="Validation", limit=DEBUG_VAL_LIMIT)
    DatasetCatalog.register("my_val", lambda: val_dicts)
    MetadataCatalog.get("my_val").set(thing_classes=classes)
    
    print(f"✅ Train: {len(total_train_dicts)} / Val: {len(val_dicts)}")

    print(">>> [2/4] Configuring Trainer...")
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    
    cfg.DATASETS.TRAIN = ("my_train",)
    cfg.DATASETS.TEST = ("my_val",)
    cfg.DATALOADER.NUM_WORKERS = 6
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    
    cfg.SOLVER.IMS_PER_BATCH = 8
    cfg.SOLVER.BASE_LR = 0.002
    
    # 반복 횟수 설정
    if DEBUG_TRAIN_LIMIT > 0:
        cfg.SOLVER.MAX_ITER = DEBUG_TRAIN_LIMIT
        cfg.SOLVER.CHECKPOINT_PERIOD = 1000
        cfg.TEST.EVAL_PERIOD = 1000
    else:
        cfg.SOLVER.MAX_ITER = 40000
        cfg.SOLVER.STEPS = (28000, 36000)
        cfg.SOLVER.CHECKPOINT_PERIOD = 8000
        cfg.TEST.EVAL_PERIOD = 8000 
    
    cfg.MODEL.ROI_HEADS.NAME = "DistanceROIHeads"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(classes)
    cfg.OUTPUT_DIR = OUTPUT_DIR
    
    print(">>> [3/4] Starting Training...")
    trainer = MyTrainer(cfg) 
    trainer.register_hooks([ValidationLossHook(cfg)]) # Hook 추가 방식 변경 (명시적 등록)
    trainer.resume_or_load(resume=False)
    trainer.train()

if __name__ == "__main__":
    main()