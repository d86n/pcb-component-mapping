import cv2
import numpy as np

def process_arduino_v13_fixed(image_path):
    print(f"Đang xử lý V13 (V9 Base + Rescue Small Chips): {image_path}")
    img = cv2.imread(image_path)
    if img is None: return

    original = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # --- BƯỚC 1: TÁCH MẠCH (GIỮ NGUYÊN V9) ---
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

    # --- BƯỚC 2: TÌM ỨNG VIÊN (GIỮ NGUYÊN V9) ---
    mask_components = cv2.bitwise_not(mask_board)
    mask_components = cv2.bitwise_and(mask_components, mask_components, mask=mask_roi)
    mask_components = cv2.morphologyEx(mask_components, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask_components, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    count = 0
    
    print(f"--- BẮT ĐẦU SÀNG LỌC ---")
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        perimeter = cv2.arcLength(cnt, True)
        
        # 1. Hạ ngưỡng diện tích xuống 20 để bắt chip tí hon
        if area < 20: continue
        if area > (img.shape[0]*img.shape[1] * 0.9): continue

        # Tính toán chỉ số
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
        #           CẤU HÌNH "GIẤY THÔNG HÀNH" (RESCUE)
        # ==========================================================
        # Chip nhỏ có đặc điểm:
        # 1. Nhỏ (Area < 450)
        # 2. RẤT VUÔNG VỨC (Solidity > 0.8) -> Đây là chìa khóa!
        # (Chân hàn/kim loại thường méo mó, solidity thấp ~0.6-0.7)
        
        is_rectangular_chip = (solidity > 0.82) 

        # ==========================================================
        #                 BỘ LỌC RÁC (V9 LOGIC)
        # ==========================================================

        # 1. DIỆT SƠN TRẮNG (Logo, Text)
        if mean_intensity > 195: continue

        # 2. DIỆT CHÂN CẮM ĐỰC & CHÂN HÀN
        # Logic cũ: Nhỏ (< 350) + Sáng (> 100) -> Xóa
        # Logic MỚI: Chỉ xóa nếu nó KHÔNG PHẢI là chip vuông vức
        if not is_rectangular_chip:
            if area < 350 and mean_intensity > 100:
                continue

        # 3. DIỆT LỖ BẮT ỐC
        if circularity > 0.78 and mean_intensity > 130:
            continue
            
        # 4. DIỆT DÒNG KẺ MẢNH
        if w < 5 or h < 5: continue
        if aspect_ratio > 5.0: continue
        
        # Chip thì phải đặc, nếu quá loãng (< 0.5) thì là rác
        if solidity < 0.5: continue

        # ==========================================================
        #           MODULE X-RAY (Soi chân ICSP Header)
        # ==========================================================
        
        is_black_object = (mean_intensity < 105)
        
        # [FIX]: Chỉ soi chân nếu vật thể ĐỦ TO (> 500)
        # Việc này ngăn không cho nó soi 4 con chip nhỏ (vì chip nhỏ cũng đen)
        # Nếu soi chip nhỏ, chữ trên lưng chip sẽ bị tính là chân -> Chip bị xóa oan.
        
        if is_black_object and area > 500:
            roi = gray[y:y+h, x:x+w]
            _, mask_pins = cv2.threshold(roi, 115, 255, cv2.THRESH_BINARY)
            cnts_pins, _ = cv2.findContours(mask_pins, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid_pins = 0
            for p in cnts_pins:
                if cv2.contourArea(p) > 5:
                    valid_pins += 1
            
            # ICSP Header có 6 chân -> Xóa
            if valid_pins >= 3:
                continue

        # ==========================================================
        #                 QUYẾT ĐỊNH CUỐI CÙNG
        # ==========================================================
        
        # Whitelist
        is_black = (mean_intensity < 110)
        is_mid_tone = (110 <= mean_intensity <= 195 and area > 350)
        is_huge = (area > 800)
        
        # [QUAN TRỌNG] Thêm điều kiện nhận chip nhỏ
        is_rescued_chip = (is_rectangular_chip and area > 20 and area < 500)

        if is_black or is_mid_tone or is_huge or is_rescued_chip:
            count += 1
            
            # Vẽ kết quả
            cv2.rectangle(original, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.circle(original, (x + w//2, y + h//2), 2, (0, 0, 255), -1)

    print(f"Tổng số linh kiện phát hiện: {count}")
    
    cv2.imshow("Result V13 - V9 Fixed", original)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

img_path = "img/Arduino Uno Rev3 SMD.jpg"
process_arduino_v13_fixed(img_path)