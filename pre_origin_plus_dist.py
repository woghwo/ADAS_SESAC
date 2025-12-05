import json
import os
import numpy as np
from tqdm import tqdm  # 진행률 표시를 위한 라이브러리 (pip install tqdm)

# =========================================================
# [설정 영역] 경로 설정
# =========================================================

# 1. Target: 앞서 생성한 COCO 포맷 파일 (distance가 없는 상태)
COCO_JSON_PATH = '/home/elicer/data/091_AD_Severe_Day/Validation/02_label_data/train_coco_full.json'

# 2. Source: distance 정보가 들어있는 3D 원본 라벨 폴더 (예: 08_174514_221206_01.json 파일들이 있는 곳)
# 주의: 파일명(filename)이 COCO 데이터의 file_name과 일치해야 매칭 가능
SOURCE_LABEL_DIR = '/home/elicer/data/091_AD_Severe_Day/Validation/02_label_data/3D' 

# 3. 결과 저장 경로
OUTPUT_JSON_PATH = '/home/elicer/data/091_AD_Severe_Day/Validation/02_label_data/val_coco_with_distance.json'

# 4. IoU 임계값 (0.0 ~ 1.0)
# 두 박스가 50% 이상 겹쳐야 같은 객체로 간주
IOU_THRESHOLD = 0.5

# =========================================================
# [함수] IoU 계산 및 매칭 로직
# =========================================================

def calculate_iou(boxA, boxB):
    """
    boxA: [x1, y1, x2, y2] (Source: 3D data)
    boxB: [x, y, w, h] (Target: COCO data) -> 변환 필요
    """
    # COCO format [x, y, w, h] -> [x1, y1, x2, y2] 변환
    boxB_converted = [boxB[0], boxB[1], boxB[0] + boxB[2], boxB[1] + boxB[3]]

    # 교집합(Intersection) 좌표 계산
    xA = max(boxA[0], boxB_converted[0])
    yA = max(boxA[1], boxB_converted[1])
    xB = min(boxA[2], boxB_converted[2])
    yB = min(boxA[3], boxB_converted[3])

    # 교집합 넓이 계산
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # 각 박스의 넓이 계산
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB_converted[2] - boxB_converted[0]) * (boxB_converted[3] - boxB_converted[1])

    # 합집합(Union) 넓이 계산
    unionArea = boxAArea + boxBArea - interArea
    
    if unionArea == 0:
        return 0

    # IoU 계산
    iou = interArea / float(unionArea)
    return iou

def merge_distance_data():
    print(">>> 데이터 병합을 시작합니다...")

    # 1. COCO 데이터 로드
    with open(COCO_JSON_PATH, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    
    print(f"Loaded COCO JSON: {len(coco_data['images'])} images, {len(coco_data['annotations'])} annotations")

    # 2. 검색 속도를 위해 COCO 데이터를 인덱싱 (Image ID 기준)
    # image_filename -> image_id 매핑
    filename_to_img_id = {img['file_name']: img['id'] for img in coco_data['images']}
    
    # image_id -> [annotation_objects...] 매핑
    img_id_to_anns = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_id_to_anns:
            img_id_to_anns[img_id] = []
        
        # 기본적으로 distance 필드를 -1로 초기화 (또는 None)
        ann['distance'] = -1 
        img_id_to_anns[img_id].append(ann)

    # 3. Source 폴더의 파일들을 순회하며 매칭
    source_files = [f for f in os.listdir(SOURCE_LABEL_DIR) if f.endswith('.json')]
    matched_count = 0
    total_source_objects = 0
    
    print(f"Matching with {len(source_files)} source files...")

    for filename in tqdm(source_files):
        source_path = os.path.join(SOURCE_LABEL_DIR, filename)
        
        try:
            with open(source_path, 'r', encoding='utf-8') as f:
                source_data = json.load(f)
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue

        # 파일명으로 해당 이미지의 COCO ID 찾기
        # (Source JSON 안의 filename과 COCO의 file_name이 정확히 일치한다고 가정)
        target_filename = source_data['information']['filename']
        if target_filename not in filename_to_img_id:
            continue
            
        target_img_id = filename_to_img_id[target_filename]
        
        # 해당 이미지에 속한 COCO 라벨들 가져오기
        target_anns = img_id_to_anns.get(target_img_id, [])
        if not target_anns:
            continue

        # Source의 각 객체(annotation)를 순회
        for source_ann in source_data.get('annotations', []):
            dist = source_ann.get('distance')
            
            # distance가 없거나 null이면 건너뜀
            if dist is None:
                continue
            
            total_source_objects += 1
            source_bbox = source_ann.get('bbox') # [x1, y1, x2, y2]
            
            if not source_bbox: 
                continue

            # 가장 IoU가 높은 COCO 객체 찾기
            best_iou = 0
            best_match_ann = None

            for target_ann in target_anns:
                # 이미 distance가 할당된 객체는 덮어쓸지 말지 결정 (여기서는 더 높은 IoU가 나오면 덮어쓰거나 pass)
                # 일단은 모든 후보와 비교
                iou = calculate_iou(source_bbox, target_ann['bbox'])
                
                if iou > best_iou:
                    best_iou = iou
                    best_match_ann = target_ann
            
            # 임계값 넘는 매칭이 있으면 distance 업데이트
            if best_iou >= IOU_THRESHOLD and best_match_ann is not None:
                best_match_ann['distance'] = dist
                matched_count += 1

    # 4. 결과 저장
    print("\n>>> 병합 완료")
    print(f"Total Source Objects with Distance: {total_source_objects}")
    print(f"Successfully Matched Objects: {matched_count}")
    print(f"Match Rate: {matched_count / total_source_objects * 100:.2f}%" if total_source_objects > 0 else "0%")
    
    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, indent=4, ensure_ascii=False)
    
    print(f"Saved merged JSON to: {OUTPUT_JSON_PATH}")

if __name__ == "__main__":
    merge_distance_data()