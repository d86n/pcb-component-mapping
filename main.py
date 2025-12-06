import cv2
import numpy as np

def process_arduino_v5_final(image_path):
    print(f"Đang xử lý ảnh V5 (Final): {image_path}")
    img = cv2.imread(image_path)
    if img is None: return

    original = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # --- BƯỚC 1: TÁCH MẠCH (GIỮ NGUYÊN) ---
    lower_teal = np.array([35, 50, 50])   
    upper_teal = np.array([100, 255, 255]) 
    mask_board = cv2.inRange(hsv, lower_teal, upper_teal)
    
    # Kernel 3x3 để giữ tách biệt các linh kiện sát nhau
    kernel = np.ones((3,3), np.uint8) 
    mask_board = cv2.morphologyEx(mask_board, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_board = cv2.morphologyEx(mask_board, cv2.MORPH_OPEN, kernel, iterations=1)

    contours_board, _ = cv2.findContours(mask_board, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours_board: return
    c_board = max(contours_board, key=cv2.contourArea)
    mask_roi = np.zeros_like(mask_board)
    cv2.drawContours(mask_roi, [c_board], -1, 255, -1)

    # --- BƯỚC 2: TÌM ỨNG VIÊN ---
    mask_components = cv2.bitwise_not(mask_board)
    mask_components = cv2.bitwise_and(mask_components, mask_components, mask=mask_roi)
    mask_components = cv2.morphologyEx(mask_components, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask_components, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    count = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    print(f"--- BẮT ĐẦU PHÂN LOẠI V5 ---")
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        perimeter = cv2.arcLength(cnt, True)
        
        # 1. Lọc rác siêu nhỏ
        if area < 30: continue
        if area > (img.shape[0]*img.shape[1] * 0.9): continue

        # --- TÍNH TOÁN CHỈ SỐ ---
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
        #                 BỘ LỌC THÉP (NEGATIVE LIST)
        # ==========================================================

        # 1. DIỆT SƠN TRẮNG (Logo, Text) - Đã OK ở V4
        if mean_intensity > 190: continue

        # 2. DIỆT CHÂN CẮM ĐỰC & CHÂN HÀN (FIX MỚI)
        # Logic: Nếu nhỏ (< 350) thì BẮT BUỘC phải ĐEN THUI (< 100).
        # Chân cắm đực là kim loại nên sáng (> 110) -> Xóa.
        # Con trở/chip bé là màu đen (< 80) -> Giữ.
        if area < 350 and mean_intensity > 100:
            continue

        # 3. DIỆT LỖ BẮT ỐC (FIX MỚI)
        # Logic: Tròn (> 0.78) VÀ Khá sáng (> 130) -> Xóa
        # Tụ CS47 (tròn) có chữ in đen và bề mặt nhôm xám nên intensity thường thấp hơn 130 hoặc circularity thấp hơn do chân đế vuông.
        if circularity > 0.78 and mean_intensity > 130:
            continue
            
        # 4. DIỆT DÒNG KẺ MẢNH
        if w < 10 or h < 10: continue
        if aspect_ratio > 5.0: continue

        # ==========================================================
        #                 DANH SÁCH CHẤP NHẬN (WHITELIST)
        # ==========================================================
        
        # Điều kiện cơ bản: Phải là khối đặc
        if solidity < 0.6: continue

        # Nhóm 1: Linh kiện ĐEN (Chip, Trở, Diode, Header base)
        is_black = (mean_intensity < 100)
        
        # Nhóm 2: Linh kiện TRUNG TÍNH (Tụ nâu, Tụ nhôm, Thạch anh)
        # Phải đủ LỚN (> 350) để không bị nhầm với chân hàn
        is_mid_tone = (100 <= mean_intensity <= 190 and area > 350)
        
        # Nhóm 3: Linh kiện LỚN bất thường (Cổng USB, Jack nguồn)
        is_huge = (area > 800)

        if is_black or is_mid_tone or is_huge:
            count += 1
            
            # Vẽ kết quả
            cv2.rectangle(original, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.circle(original, (x + w//2, y + h//2), 2, (0, 0, 255), -1)
            
            # Debug: In thông số để kiểm tra nếu sai
            # text_debug = f"I:{int(mean_intensity)} A:{int(area)}"
            # cv2.putText(original, text_debug, (x, y-5), font, 0.3, (0, 255, 255), 1)

    print(f"Tổng số linh kiện phát hiện: {count}")
    
    cv2.imshow("Final Result V5 - Clean", original)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Chạy code
img_path = "img/Arduino Uno Rev3 SMD.jpg"
process_arduino_v5_final(img_path)