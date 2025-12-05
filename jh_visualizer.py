import cv2
import numpy as np
import time
import config as cfg

class TrafficVisualizer:
    def __init__(self):
        # -----------------------------------------------------------
        # [설정] 시각화 튜닝
        # -----------------------------------------------------------
        self.map_size = 200     
        self.map_scale = 3.0    
        self.max_dist_vis = 60.0 
        
        self.map_coord_cache = {} 
        self.DEADBAND_METER = 0.5 

        self.VEHICLE_LENGTHS = {
            "vehicle": 4.5, "taxi": 4.5, "othercar": 4.5,
            "bus": 11.0, "truck": 10.0,
            "motorcycle": 2.0, "bicycle": 1.5,
            "pedestrian": 0.5, "rider": 0.5
        }

        self.prev_time = time.time()
        self.fps = 0.0

    def draw_results(self, frame, results):
        h, w = frame.shape[:2]
        center_x = w // 2
        
        # --- [Visualizer 내부에 사다리꼴 스펙 정의] ---
        corridor_bottom_w = 450.0
        corridor_top_w = 50.0
        corridor_bottom_y = float(h) - 20.0
        
        # 1. Drivable Mask 추출
        drivable_mask = None
        for c_name, mask in zip(results["class"], results["mask"]):
            if c_name == 'freespace' and mask is not None:
                drivable_mask = mask
                break
        
        # 2. [Road Painting] 도로 영역을 제일 먼저 그림 (배경)
        # 박스나 사다리꼴이 가려지지 않도록 맨 처음에 처리
        if drivable_mask is not None:
            # 도로 마스크를 반투명하게 칠하기
            color_mask = np.zeros_like(frame)
            color_mask[drivable_mask] = (255, 100, 0) # Blue-ish color
            frame = cv2.addWeighted(frame, 1.0, color_mask, 0.3, 0)

        # -----------------------------------------------------------
        # [Geometry Sync] 사다리꼴 높이(소실점) 미리 계산
        # -----------------------------------------------------------
        limit_y = float(h) // 3.0
        active_top_y = limit_y  # 기본값

        if drivable_mask is not None:
            scan_w = 5
            x1 = max(0, center_x - scan_w)
            x2 = min(w, center_x + scan_w)
            
            start_y = int(corridor_bottom_y)
            end_y = int(limit_y)
            
            for y in range(start_y, end_y, -5):
                if np.mean(drivable_mask[y, x1:x2]) < 0.5:
                    active_top_y = float(y + 10)
                    break
            
            if corridor_bottom_y - active_top_y < 40: 
                active_top_y = corridor_bottom_y - 40
        # -----------------------------------------------------------

        # 3. 위험 거리 계산
        valid_danger_dists = []
        
        for i, c_name in enumerate(results["class"]):
            if c_name == 'freespace': continue
            
            dist = results["distance"][i]
            box = results["box"][i]
            
            # (A) 도로 위 체크
            is_on_road = True
            bx1, by1, bx2, by2 = map(int, box)
            cx = (bx1 + bx2) // 2
            cy = by2 - 5
            
            check_cx = np.clip(cx, 0, w - 1)
            check_cy = np.clip(cy, 0, h - 1)
            
            if drivable_mask is not None:
                if not drivable_mask[check_cy, check_cx]:
                    is_on_road = False
            
            # (B) 사다리꼴 안쪽 체크 (동기화된 active_top_y 사용)
            is_in_corridor = False
            
            if by2 < active_top_y: 
                current_w = corridor_top_w
            elif by2 > corridor_bottom_y: 
                current_w = corridor_bottom_w
            else:
                ratio = (by2 - active_top_y) / (corridor_bottom_y - active_top_y)
                current_w = corridor_top_w + (corridor_bottom_w - corridor_top_w) * ratio
            
            bound_left = center_x - (current_w / 2.0)
            bound_right = center_x + (current_w / 2.0)
            
            # 마진 0.0 (시각적으로 보이는 선 안쪽만 인정)
            margin = 0.0
            if (bound_left - margin) < cx < (bound_right + margin):
                is_in_corridor = True

            if is_on_road and is_in_corridor:
                valid_danger_dists.append(dist)

        min_dist = min(valid_danger_dists) if valid_danger_dists else 999.0
        warnings = sum(results["is_entering"])

        # 4. 그리기 실행 순서 정리
        # (1) 사다리꼴 (바닥)
        frame = self._draw_safety_corridor(frame, w, h, min_dist, warnings, active_top_y)
        # (2) 객체 박스 (사다리꼴 위에)
        frame = self._draw_objects(frame, results)
        # (3) UI 요소들 (맨 위에)
        frame = self._draw_minimap(frame, results, w)
        frame = self._draw_fps(frame)
        frame = self._draw_bottom_status(frame, w, h, min_dist, warnings)
        
        return frame
    
    def _draw_safety_corridor(self, frame, w, h, min_dist, warnings, top_y_in):
        # 1. 색상 결정
        if warnings > 0 or min_dist < 10.0:
            color = (0, 0, 255) # Red
            intensity = 0.35
        elif min_dist < 30.0:
            color = (0, 255, 255) # Yellow
            intensity = 0.25
        else:
            color = (0, 255, 0) # Green
            intensity = 0.2

        center_x = int(w // 2)
        bottom_w = 450
        top_w = 50
        bottom_y = int(h - 20)
        
        # [동기화] 계산된 top_y 사용
        top_y = int(top_y_in)

        pts = np.array([
            [center_x - bottom_w // 2, bottom_y], 
            [center_x + bottom_w // 2, bottom_y], 
            [center_x + top_w // 2, top_y],       
            [center_x - top_w // 2, top_y]        
        ], np.int32)

        # Overlay 그리기
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], color)
        
        # 라인 그리기
        cv2.line(overlay, (center_x, bottom_y), (center_x, top_y), (255, 255, 255), 1)
        
        lane_h = bottom_y - top_y
        line_y = int(bottom_y - lane_h * 0.45)
        
        ratio = 0.45
        current_w = int(bottom_w * (1 - ratio) + top_w * ratio)
        
        cv2.line(overlay, (center_x - current_w // 2, line_y), 
                          (center_x + current_w // 2, line_y), (255, 255, 255), 1)

        cv2.addWeighted(overlay, intensity, frame, 1 - intensity, 0, frame)
        return frame
    
    def _draw_minimap(self, frame, results, w):
        map_bg = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        map_bg[:] = (30, 30, 30) 
        pixels_per_10m = 10.0 * self.map_scale
        for i in range(1, 6):
            y_grid = int(self.map_size - (i * pixels_per_10m))
            if y_grid > 0:
                cv2.line(map_bg, (0, y_grid), (self.map_size, y_grid), (50, 50, 50), 1)

        cx = self.map_size // 2
        bottom = self.map_size - 10
        pts = np.array([[cx, bottom-15], [cx-8, bottom], [cx+8, bottom]], np.int32)
        cv2.fillPoly(map_bg, [pts], (255, 255, 255))
        
        current_frame_ids = set()

        for i, c_name in enumerate(results["class"]):
            if c_name == 'freespace': continue
            
            dist = results["distance"][i]
            if dist > self.max_dist_vis: continue
            
            box = results["box"][i]
            obj_id = results["id"][i]
            is_ent = results["is_entering"][i]
            current_frame_ids.add(obj_id)
            
            box_cx = (box[0] + box[2]) / 2
            img_center = w / 2
            lateral_offset = (box_cx - img_center) / (w / 2) 
            target_x = cx + (lateral_offset * (self.map_size / 2.0))
            target_y = bottom - (dist * self.map_scale)
            
            if obj_id in self.map_coord_cache:
                prev_x, prev_y = self.map_coord_cache[obj_id]
                if dist > 40.0: alpha = 0.05
                elif dist > 20.0: alpha = 0.1
                else: alpha = 0.3
                
                move_dist = np.sqrt((target_x - prev_x)**2 + (target_y - prev_y)**2)
                if move_dist < 2.0: 
                    curr_x, curr_y = prev_x, prev_y
                else:
                    curr_x = prev_x * (1 - alpha) + target_x * alpha
                    curr_y = prev_y * (1 - alpha) + target_y * alpha
            else:
                curr_x, curr_y = target_x, target_y
            
            self.map_coord_cache[obj_id] = (curr_x, curr_y)
            draw_x = int(curr_x)
            draw_y = int(curr_y)
            
            real_len = self.VEHICLE_LENGTHS.get(c_name, 4.5)
            real_width = cfg.REAL_WIDTH_MAP.get(c_name, 1.85)
            rect_h = int(real_len * self.map_scale) 
            rect_w = int(real_width * self.map_scale * 1.5)
            top_left = (draw_x - rect_w//2, draw_y - rect_h//2)
            bottom_right = (draw_x + rect_w//2, draw_y + rect_h//2)
            
            color = (0, 0, 255) if is_ent else cfg.COLOR_MAP.get(c_name, (200, 200, 200))
            cv2.rectangle(map_bg, top_left, bottom_right, color, -1)
            cv2.rectangle(map_bg, top_left, bottom_right, (0, 0, 0), 1)

        self.map_coord_cache = {k: v for k, v in self.map_coord_cache.items() if k in current_frame_ids}

        margin = 20
        y_offset = 70
        x_offset = w - self.map_size - margin
        roi = frame[y_offset:y_offset+self.map_size, x_offset:x_offset+self.map_size]
        combined = cv2.addWeighted(roi, 0.1, map_bg, 0.9, 0)
        frame[y_offset:y_offset+self.map_size, x_offset:x_offset+self.map_size] = combined
        cv2.rectangle(frame, (x_offset, y_offset), (x_offset+self.map_size, y_offset+self.map_size), (150, 150, 150), 2)
        cv2.putText(frame, "BEV Map", (x_offset + 5, y_offset + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        return frame

    def _draw_bottom_status(self, frame, w, h, min_dist, warnings):
        panel_w = 320
        panel_h = 90
        x1 = w - panel_w - 20
        y1 = h - panel_h - 20
        x2 = w - 20
        y2 = h - 20

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        if warnings > 0:
            status_text = "WARNING: CUT-IN"
            status_color = (0, 0, 255)
            bar_len = panel_w 
        elif min_dist < 10.0:
            status_text = "DANGER: TOO CLOSE" 
            status_color = (0, 0, 255)
            bar_len = panel_w 
        elif min_dist < 30.0:
            status_text = "CAUTION: BRAKE"
            status_color = (0, 255, 255) 
            bar_len = int(panel_w * 0.6)
        else:
            status_text = "SYSTEM: STABLE"
            status_color = (0, 255, 0) 
            bar_len = int(panel_w * 0.3)

        cv2.putText(frame, status_text, (x1 + 15, y1 + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        tech_text = "Logic: Hybrid Fusion (AI+Geo)"
        cv2.putText(frame, tech_text, (x1 + 15, y1 + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.rectangle(frame, (x1, y2 - 10), (x1 + bar_len, y2), status_color, -1)
        return frame

    def _draw_fps(self, frame):
        curr_time = time.time()
        delta = curr_time - self.prev_time
        self.prev_time = curr_time
        if delta > 0:
            current_fps = 1.0 / delta
            self.fps = (self.fps * 0.9) + (current_fps * 0.1)
        
        fps_text = f"FPS: {int(self.fps)}"
        cv2.rectangle(frame, (5, 5), (110, 35), (0, 0, 0), -1)
        cv2.putText(frame, fps_text, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        return frame

    def _draw_objects(self, frame, results):
        overlay = frame.copy()
        
        # 1차 루프: 반투명 요소
        for c_name, dist, is_ent, box, score, mask in zip(
            results["class"], results["distance"], results["is_entering"], 
            results["box"], results["score"], results["mask"]
        ):
            if dist >= 50.0: continue
            if c_name == 'freespace': continue # 도로는 위에서 이미 그렸음

            x1, y1, x2, y2 = map(int, box)
            color = cfg.COLOR_MAP.get(c_name, (200, 200, 200))

            if is_ent:
                if mask is not None:
                    mask_uint8 = (mask * 255).astype(np.uint8)
                    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(overlay, contours, -1, (0, 0, 255), -1)
            else:
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                label = f"{c_name} {dist:.1f}m"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                text_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
                cv2.rectangle(overlay, (x1, text_y - th - 5), (x1 + tw, text_y + 5), color, -1)
                cv2.putText(overlay, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        # 2차 루프: 강조 요소
        for c_name, dist, is_ent, box, score, mask in zip(
            results["class"], results["distance"], results["is_entering"], 
            results["box"], results["score"], results["mask"]
        ):
            if dist >= 50.0: continue
            if c_name == 'freespace' or not is_ent: continue 

            x1, y1, x2, y2 = map(int, box)
            
            if mask is not None:
                mask_uint8 = (mask * 255).astype(np.uint8)
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(frame, contours, -1, (0, 0, 255), 3) 
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

            label = "WARNING!"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            text_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
            cv2.rectangle(frame, (x1, text_y - th - 5), (x1 + tw, text_y + 5), (0, 0, 255), -1)
            cv2.putText(frame, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
        return frame