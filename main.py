import cv2
import config as cfg
from detector import VehicleDetector

IMG_PATH = r"/home/elicer/dev/gt/data/test_data/371_ND_000_FC_176.jpg"

def main():
    detector = VehicleDetector()
    
    # 이미지 한 장 로드 (테스트용)
    img_path = IMG_PATH
    cap = cv2.VideoCapture(img_path)
    ret, img = cap.read()
    if not ret: return
    
    img = cv2.resize(img, cfg.TARGET_SIZE)

    # ----------------------------------------------------------
    # [사용 예시] 시각화 담당자가 쓰게 될 코드 패턴
    # ----------------------------------------------------------
    results, road_mask = detector.run(img)
    
    # 1. 원하는 리스트만 쏙쏙 뽑아 쓰기 가능
    print("Classes:", results['class']) # 클래스 이름
    print("Distances:", results['distance']) # 거리(m)
    print("Entering:", results['is_entering']) # 진입 여부
    print("Boxes:", results['box']) # 바운딩 박스 좌표
    print("Scores:", results['score']) # 신뢰도
    print("Masks:", results['mask']) # 마스크(주해 도로)
    
    # 2. for문으로 돌리기 매우 편함 (zip 사용)
    # 시각화 하시는 분은 이렇게 짜면 됩니다:
    for c_name, dist, is_ent, box, score in zip(results['class'], results['distance'], results['is_entering'], results['box'], results['score']):
        if c_name == "freespace": continue

if __name__ == "__main__":
    main()