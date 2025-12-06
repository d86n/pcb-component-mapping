import cv2
import numpy as np

def process_arduino_v4(image_path):
    print(f"Đang xử lý ảnh V4: {image_path}")
    img = cv2.imread(image_path)
    if img is None: return

    original = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # --- BƯỚC 1: TÁCH MẠCH (GIỮ NGUYÊN NHƯ V3) ---
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
    
    print(f"--- BẮT ĐẦU PHÂN LOẠI V4 ---")
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        
        # 1. Lọc rác siêu nhỏ (bụi)
        if area < 30: continue
        # Lọc bóng quá to
        if area > (img.shape[0]*img.shape[1] * 0.9): continue

        # --- PHÂN TÍCH ĐỘ SÁNG (CHÌA KHÓA CỦA V4) ---
        mask_curr = np.zeros_like(gray)
        cv2.drawContours(mask_curr, [cnt], -1, 255, -1)
        mean_intensity = cv2.mean(gray, mask=mask_curr)[0]
        
        # --- BỘ LỌC 1: DIỆT LOGO & TEXT (SƠN TRẮNG) ---
        # Sơn in lụa (Arduino, Digital, PWM, ~) thường rất sáng (> 190)
        # Linh kiện sáng nhất là thạch anh/USB cũng chỉ tầm 150-180
        # -> Ngưỡng cắt: 185. Cái gì sáng hơn 185 là SƠN -> XÓA.
        if mean_intensity > 185: 
            continue 

        # --- BỘ LỌC 2: DIỆT CHÂN CẮM ĐỰC & NHIỄU KIM LOẠI ---
        # Chân cắm đực: Nhỏ (area < 200) + Sáng (màu kim loại > 120)
        # Tại sao phải chặn? Vì chúng sáng hơn chip đen nhưng tối hơn sơn trắng.
        # Chip đen: < 110. Tụ/Thạch anh: To > 500.
        # -> Cái gì BÉ (< 200) mà lại KHÔNG ĐEN (> 120) -> XÓA.
        if area < 200 and mean_intensity > 120:
            continue

        # --- BỘ LỌC 3: HÌNH DÁNG ---
        # Lọc các nét gạch mảnh (ví dụ viền bảng, vạch kẻ)
        if w < 12 or h < 12: continue
        
        # Tính các chỉ số
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0: continue
        solidity = float(area) / hull_area 
        aspect_ratio = float(w) / h
        
        # --- QUYẾT ĐỊNH CUỐI CÙNG (WHITELIST) ---
        
        # Nhóm 1: Linh kiện ĐEN (Chip, Trở, Header nhựa)
        # Đặc điểm: Tối màu (< 115) + Hình khối rõ ràng
        is_black_component = (mean_intensity < 115 and area > 40 and solidity > 0.6)
        
        # Nhóm 2: Linh kiện BẠC/XÁM (Tụ nhôm, Thạch anh, USB)
        # Đặc điểm: Màu trung tính (115-185) + KÍCH THƯỚC PHẢI LỚN (> 300)
        # Lưu ý: Phải đủ to để không bị nhầm với chân hàn/chân cắm đực
        is_silver_component = (115 <= mean_intensity <= 185 and area > 300 and solidity > 0.6)
        
        # Nhóm 3: Linh kiện DÀI/TO bất thường (Header 8 chân, Jack nguồn)
        is_large_structure = (area > 800 and mean_intensity < 180)

        if is_black_component or is_silver_component or is_large_structure:
            # Check phụ: Loại bỏ các thanh quá mảnh (dòng kẻ)
            if aspect_ratio > 5.0: continue
            
            count += 1
            
            # Vẽ kết quả
            cv2.rectangle(original, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.circle(original, (x + w//2, y + h//2), 2, (0, 0, 255), -1)
            
            # Debug: In giá trị độ sáng lên ảnh để bạn kiểm tra nếu sai
            # cv2.putText(original, f"{int(mean_intensity)}", (x, y-5), font, 0.4, (0, 255, 255), 1)

    print(f"Tổng số linh kiện phát hiện: {count}")
    
    cv2.imshow("Final Result V4 - Strict Filter", original)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Chạy với ảnh của bạn
img_path = "img/Arduino Uno Rev3 SMD.jpg" 
process_arduino_v4(img_path)