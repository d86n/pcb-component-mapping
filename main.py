import cv2
import numpy as np

def process_arduino_v9_hybrid(image_path):
    print(f"Đang xử lý V9 (Hybrid V5 + V8): {image_path}")
    img = cv2.imread(image_path)
    if img is None: return

    original = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # --- BƯỚC 1: TÁCH MẠCH (GIỮ NGUYÊN V5) ---
    lower_teal = np.array([35, 50, 50])   
    upper_teal = np.array([100, 255, 255]) 
    mask_board = cv2.inRange(hsv, lower_teal, upper_teal)
    
    kernel = np.ones((3,3), np.uint8) 
    mask_board = cv2.morphologyEx(mask_board, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_board = cv2.morphologyEx(mask_board, cv2.MORPH_OPEN, kernel, iterations=1)

    contours_board, _ = cv2.findContours(mask_board, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours_board: return
    c_board = max(contours_board, key=cv2.contourArea)
    mask_roi = np.zeros_like(mask_board)
    cv2.drawContours(mask_roi, [c_board], -1, 255, -1)

    # --- BƯỚC 2: TÌM ỨNG VIÊN (GIỮ NGUYÊN V5) ---
    mask_components = cv2.bitwise_not(mask_board)
    mask_components = cv2.bitwise_and(mask_components, mask_components, mask=mask_roi)
    mask_components = cv2.morphologyEx(mask_components, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask_components, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    count = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    print(f"--- BẮT ĐẦU SÀNG LỌC ---")
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        perimeter = cv2.arcLength(cnt, True)
        
        # 1. Lọc rác siêu nhỏ (Logic V5)
        if area < 30: continue
        if area > (img.shape[0]*img.shape[1] * 0.9): continue

        # Tính toán chỉ số cơ bản
        mask_curr = np.zeros_like(gray)
        cv2.drawContours(mask_curr, [cnt], -1, 255, -1)
        mean_intensity = cv2.mean(gray, mask=mask_curr)[0]
        
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0: continue
        solidity = float(area) / hull_area 
        aspect_ratio = float(w) / h
        
        if perimeter == 0: continue
        circularity = 4 * np.pi * (area / (perimeter * perimeter))

        # ==========================================================
        #                 BỘ LỌC CƠ BẢN CỦA V5 (LOẠI BỎ RÁC)
        # ==========================================================

        # 1. DIỆT SƠN TRẮNG (Logo, Text)
        if mean_intensity > 190: continue

        # 2. DIỆT CHÂN CẮM ĐỰC & CHÂN HÀN
        # (Nhỏ < 350 VÀ Sáng > 100 -> Xóa)
        if area < 350 and mean_intensity > 100:
            continue

        # 3. DIỆT LỖ BẮT ỐC (Tròn & Sáng)
        if circularity > 0.78 and mean_intensity > 130:
            continue
            
        # 4. DIỆT DÒNG KẺ MẢNH / LOGO DẸT
        if w < 10 or h < 10: continue
        if aspect_ratio > 5.0: continue
        if solidity < 0.6: continue

        # ==========================================================
        #           MODULE ĐẶC BIỆT: TÍCH HỢP V8 (X-RAY CHECK)
        # ==========================================================
        # Chỉ áp dụng cho các khối MÀU ĐEN (Candidate for Chip or Header)
        # Intensity < 105 nghĩa là nó có thể là Chip hoặc Header ICSP
        
        is_black_object = (mean_intensity < 105)
        
        if is_black_object:
            # Cắt vùng ảnh con để soi
            roi = gray[y:y+h, x:x+w]
            
            # Tìm các đốm sáng bên trong (Chân kim loại)
            # Chân kim loại sáng > 115 trên nền đen
            _, mask_pins = cv2.threshold(roi, 115, 255, cv2.THRESH_BINARY)
            
            # Đếm số lượng đốm sáng tách biệt
            cnts_pins, _ = cv2.findContours(mask_pins, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            valid_pins = 0
            for p in cnts_pins:
                # Chỉ đếm đốm sáng có kích thước > 5px (để tránh nhiễu/chữ)
                if cv2.contourArea(p) > 5:
                    valid_pins += 1
            
            # LUẬT V8: Nếu có >= 3 chân kim loại bên trong -> Là Header ICSP -> XÓA
            if valid_pins >= 3:
                # print(f"Đã diệt ICSP tại {x},{y} với {valid_pins} chân")
                continue

        # ==========================================================
        #                 QUYẾT ĐỊNH CUỐI CÙNG (WHITELIST V5)
        # ==========================================================
        
        # Nếu đã qua được tất cả các cửa ải trên, ta phân loại lần cuối để chắc ăn
        
        # Nhóm 1: Linh kiện ĐEN (Chip, Trở)
        is_black = (mean_intensity < 105)
        
        # Nhóm 2: Linh kiện TRUNG TÍNH (Tụ nhôm, Thạch anh)
        # Phải đủ LỚN (> 350) 
        is_mid_tone = (105 <= mean_intensity <= 190 and area > 350)
        
        # Nhóm 3: Linh kiện LỚN bất thường (Cổng USB, Jack nguồn)
        is_huge = (area > 800)

        if is_black or is_mid_tone or is_huge:
            count += 1
            
            # Vẽ kết quả
            cv2.rectangle(original, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.circle(original, (x + w//2, y + h//2), 2, (0, 0, 255), -1)

    print(f"Tổng số linh kiện phát hiện: {count}")
    
    cv2.imshow("Result V9 - Hybrid (Best of V5 & V8)", original)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Chạy code
img_path = "img/Arduino Uno Rev3 SMD.jpg"
process_arduino_v9_hybrid(img_path)