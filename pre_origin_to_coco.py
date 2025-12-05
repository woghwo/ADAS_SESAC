import os
import json
import numpy as np
import random
from pycocotools import mask as maskUtils
from collections import Counter, defaultdict
from typing import List, Dict, Any

# [설정 영역] 경로와 클래스 정의

# 1. 경로 설정
IMAGE_DIR = '/home/elicer/data/091_AD_Severe_Day/Validation/01_raw_data/image_data'  # 원본 이미지 폴더
LABEL_DIR = '/home/elicer/data/091_AD_Severe_Day/Validation/02_label_data/2D'  # 원본 라벨 폴더

# 중간 생성 파일 (전체 데이터 JSON)
FULL_JSON_PATH = '/home/elicer/data/091_AD_Severe_Day/Validation/02_label_data/train_coco_full.json' 
# 최종 생성 파일 (샘플링된 데이터 JSON)
SAMPLED_JSON_PATH = '/home/elicer/data/091_AD_Severe_Day/Validation/02_label_data/train_coco_stratified.json'

# 2. 샘플링 비율 (전체 데이터 중 몇 %를 사용할지)
SAMPLE_RATIO = 0.3  # 30%만 사용

# 3. 클래스 매핑 (소문자 통일)
CONSOLIDATION_MAP = {
    "freespace": "freespace",
    "trafficsign": "trafficsign",
    "trafficlight": "trafficlight",
    "fense": "background", "fence": "background",
    "policecar": "othercar", "ambulance": "othercar", "schoolbus": "othercar",
    "twowheeler": "motorcycle",
    "safetyzone": "roadmark", "speedbump": "roadmark",
    "bluelane": "roadmark", "redlane": "roadmark", "stoplane": "roadmark",
    "crosswalk": "crosswalk", "sidewalk": "sidewalk"
}

IGNORE_CLASSES = ["nibbercore", "lense", "egovehicle", "warmingtriangle", "background"]

THING_CLASSES = [
    "vehicle", "bus", "truck", "othercar", 
    "motorcycle", "bicycle", "pedestrian", "rider",
    "trafficsign", "trafficlight", "constructionguide", "trafficdrum", 
]

STUFF_CLASSES = [
    "freespace", "curb", "sidewalk", "crosswalk", 
    "roadmark", "whitelane", "yellowlane", 
]

# =========================================================
# [Part 1] Raw Data -> Full COCO JSON 변환
# =========================================================

def get_categories():
    COCO_CLASSES = THING_CLASSES + STUFF_CLASSES
    CATEGORIES = []
    for i, name in enumerate(COCO_CLASSES):
        is_thing = 1 if name in THING_CLASSES else 0
        CATEGORIES.append({"id": i + 1, "name": name, "supercategory": "none", "is_thing": is_thing})
    return CATEGORIES, {cat['name']: cat['id'] for cat in CATEGORIES}

def polygon_to_bbox_and_area(polygon_coords, h, w):
    segmentation = [[float(c) for c in polygon_coords]]
    points = np.array(polygon_coords).reshape(-1, 2)
    if len(points) < 3: return None, 0, None # 유효하지 않은 폴리곤
    
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    width, height = x_max - x_min, y_max - y_min
    bbox = [float(x_min), float(y_min), float(width), float(height)]
    
    try:
        rles = maskUtils.frPyObjects(segmentation, h, w)
        area = float(maskUtils.area(rles).sum())
    except:
        area = float(width * height)
    return bbox, area, segmentation

def convert_raw_to_coco_full():
    print(f"[1단계] Raw 데이터를 읽어 전체 COCO JSON을 생성")
    CATEGORIES, CATEGORY_MAP = get_categories()
    
    coco_output = {
        "info": {}, "licenses": [], "categories": CATEGORIES, 
        "images": [], "annotations": []
    }
    
    image_id = 0
    annotation_id = 0
    
    files = [f for f in os.listdir(LABEL_DIR) if f.endswith('.json')]
    total_files = len(files)
    
    for idx, filename in enumerate(files):
        if idx % 5000 == 0: print(f" - 처리 중: {idx}/{total_files}")
            
        label_path = os.path.join(LABEL_DIR, filename)
        with open(label_path, 'r', encoding='utf-8') as f:
            try: data = json.load(f)
            except: continue

        try:
            img_name = data['information']['filename']
            w = int(data['information']['resolution'][0])
            h = int(data['information']['resolution'][1])
        except: continue

        coco_output["images"].append({
            "id": image_id, "file_name": img_name, "height": h, "width": w
        })
        
        for ann in data.get("annotations", []):
            raw_cls = ann['class'].strip().lower() # 소문자 강제 변환
            
            cls_name = CONSOLIDATION_MAP.get(raw_cls, raw_cls)
            if cls_name in IGNORE_CLASSES: continue
            if cls_name not in CATEGORY_MAP: continue
            
            bbox, area, seg = polygon_to_bbox_and_area(ann['polygon'], h, w)
            if bbox is None or area <= 1.0: continue
            
            coco_output["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": CATEGORY_MAP[cls_name],
                "bbox": bbox,
                "area": area,
                "segmentation": seg,
                "iscrowd": 0
            })
            annotation_id += 1
        
        image_id += 1

    with open(FULL_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(coco_output, f, indent=4, ensure_ascii=False)
    print(f" -> 전체 데이터 JSON 저장 완료: {FULL_JSON_PATH} (이미지 {image_id}장)")
    return FULL_JSON_PATH

# =========================================================
# [Part 2] Stratified Sampling (클래스 비율 유지하며 추출)
# =========================================================

def stratified_sampling_filter(full_json_path, output_json_path, ratio):
    print(f"\n[2단계] 클래스 불균형을 고려한 층화 추출(Stratified Sampling) 시작")
    
    with open(full_json_path, 'r') as f:
        coco = json.load(f)
        
    images = coco["images"]
    annotations = coco["annotations"]
    categories = coco["categories"]
    
    total_images = len(images)
    target_n_samples = int(total_images * ratio)
    print(f"전체 {total_images}장 중 {ratio*100}%인 약 {target_n_samples}장 추출")

    # 1. 클래스별 빈도 및 이미지 매핑
    class_counts = Counter([ann["category_id"] for ann in annotations])
    total_objects = sum(class_counts.values())
    
    # category_id -> [image_id, image_id, ...]
    cat_to_imgs = defaultdict(list)
    for ann in annotations:
        cat_to_imgs[ann["category_id"]].append(ann["image_id"])
    
    # 중복 제거 (한 이미지가 같은 클래스 객체를 여러 개 가질 경우)
    for cid in cat_to_imgs:
        cat_to_imgs[cid] = list(set(cat_to_imgs[cid]))

    # 2. 클래스별 할당량 계산 (비율 기반)
    selected_image_ids = set()
    
    print("\n[클래스별 목표 할당량]")
    for cid, count in class_counts.items():
        # 전체 데이터에서의 비율
        cls_ratio = count / total_objects
        # 목표 샘플 수 (적어도 1장은 포함되도록 max 사용)
        # 주의: 이는 '어노테이션 수' 기준 비율을 '이미지 수'에 단순 대입하는 근사치입니다.
        target_count = max(1, int(cls_ratio * target_n_samples))
        
        candidates = cat_to_imgs[cid]
        if not candidates: continue
        
        # 목표보다 후보가 적으면 전수 조사, 많으면 랜덤 추출
        if len(candidates) <= target_count:
            sampled = candidates
        else:
            sampled = np.random.choice(candidates, target_count, replace=False)
        
        selected_image_ids.update(sampled)

    # 3. 집합 크기 조정 (중복으로 인해 target_n_samples보다 작거나 클 수 있음)
    final_ids = list(selected_image_ids)
    
    # 만약 목표보다 많이 뽑혔으면 랜덤하게 줄임
    if len(final_ids) > target_n_samples:
        final_ids = np.random.choice(final_ids, target_n_samples, replace=False).tolist()
    
    # 만약 목표보다 적게 뽑혔으면 (중복 포함 때문에), 남은 이미지 중에서 랜덤으로 채움
    elif len(final_ids) < target_n_samples:
        all_ids = set(img['id'] for img in images)
        remaining_ids = list(all_ids - set(final_ids))
        needed = target_n_samples - len(final_ids)
        if len(remaining_ids) >= needed:
            additional = np.random.choice(remaining_ids, needed, replace=False).tolist()
            final_ids.extend(additional)
            
    print(f" -> 최종 추출된 이미지 수: {len(final_ids)}")
    
    # 4. 새로운 JSON 생성
    final_ids_set = set(final_ids)
    new_images = [img for img in images if img['id'] in final_ids_set]
    new_annotations = [ann for ann in annotations if ann['image_id'] in final_ids_set]
    
    coco_stratified = {
        "info": coco["info"],
        "licenses": coco["licenses"],
        "categories": coco["categories"],
        "images": new_images,
        "annotations": new_annotations
    }
    
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(coco_stratified, f, indent=4, ensure_ascii=False)
        
    print(f"\n[완료] 샘플링된 JSON이 저장되었습니다: {output_json_path}")
    print(" -> 이미지를 복사할 필요 없이, 학습 코드에서 이 JSON 경로만 지정하면 됩니다.")

# =========================================================
# [Main Execution]
# =========================================================
if __name__ == "__main__":
    # 1. 전체 데이터 변환 (한 번만 제대로 해두면 됩니다)
    if not os.path.exists(FULL_JSON_PATH):
        convert_raw_to_coco_full()
    else:
        print(f"[Skip] 전체 JSON 파일이 이미 존재합니다: {FULL_JSON_PATH}")
        print("다시 만들려면 파일을 삭제하고 재실행하세요.")

    # 2. 층화 추출 수행
    stratified_sampling_filter(FULL_JSON_PATH, SAMPLED_JSON_PATH, SAMPLE_RATIO)