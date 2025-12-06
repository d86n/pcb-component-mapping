import cv2
import numpy as np

def process_arduino_v3(image_path):
    print(f"Đang xử lý ảnh: {image_path}")
    img = cv2.imread(image_path)
    if img is None: return

    original = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # --- BƯỚC 1: TÁCH MẠCH (GIỮ NGUYÊN) ---
    lower_teal = np.array([35, 50, 50])   
    upper_teal = np.array([100, 255, 255]) 
    mask_board = cv2.inRange(hsv, lower_teal, upper_teal)
    
    # [FIX 1] Giảm kernel xuống 3x3 để tránh dính tụ và diode
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
    
    # Lọc nhiễu muối tiêu trước khi tìm contour
    mask_components = cv2.morphologyEx(mask_components, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask_components, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    count = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    print(f"--- BẮT ĐẦU PHÂN LOẠI CHI TIẾT ---")
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        perimeter = cv2.arcLength(cnt, True)
        
        # 1. Lọc rác siêu nhỏ
        if area < 30: continue
        if area > (img.shape[0]*img.shape[1] * 0.9): continue

        # [FIX 2] Lọc theo Kích thước cạnh (Loại bỏ text ANALOG IN, TM)
        # Linh kiện phải có độ dày nhất định. Text thường nét mảnh hoặc thấp.
        if w < 12 or h < 12: continue 

        # --- TÍNH TOÁN ĐẶC TRƯNG ---
        # Tính độ sáng trung bình
        mask_curr = np.zeros_like(gray)
        cv2.drawContours(mask_curr, [cnt], -1, 255, -1)
        mean_intensity = cv2.mean(gray, mask=mask_curr)[0]
        
        # Tính độ đặc (Solidity)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0: continue
        solidity = float(area) / hull_area 
        
        # Tính tỷ lệ cạnh (Aspect Ratio)
        aspect_ratio = float(w) / h
        
        # [FIX 3] Tính độ tròn (Circularity) để diệt Lỗ Bắt Ốc
        if perimeter == 0: continue
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        
        # --- LOGIC LOẠI BỎ (NEGATIVE LIST) ---
        
        # A. Loại bỏ Logo/Sơn trắng: Diện tích to + Quá Sáng
        if area > 400 and mean_intensity > 195: continue
        
        # B. Loại bỏ Lỗ bắt ốc: Rất Tròn (>0.8) + Rất Sáng (>180) + Diện tích đủ lớn
        if circularity > 0.75 and mean_intensity > 180 and area > 300:
            continue
            
        # C. Loại bỏ dòng chữ dài (nếu sót): Aspect Ratio quá lớn
        if aspect_ratio > 5.0: continue

        # --- LOGIC CHẤP NHẬN (POSITIVE LIST) ---
        # Chỉ nhận nếu thỏa mãn các điều kiện của linh kiện
        
        is_standard = (solidity > 0.6)         # Phải là khối đặc
        is_large = (area > 800)                # Hoặc là cục to (Header, Jack nguồn)
        
        # Điều kiện phụ: Tụ điện tròn (CS 47) có thể bị circularity cao
        # Nhưng tụ điện CS 47 có màu tối hơn lỗ ốc (intensity < 180) nên vẫn được giữ lại
        
        if is_standard or is_large:
            count += 1
            
            # Vẽ đẹp hơn: Khung xanh lá, text nhỏ
            cv2.rectangle(original, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # Vẽ chấm tâm đỏ
            cv2.circle(original, (x + w//2, y + h//2), 2, (0, 0, 255), -1)
            
            # Debug: In ra nếu cần
            # print(f"Comp {count}: Area={area:.0f}, I={mean_intensity:.0f}, Circ={circularity:.2f}")

    print(f"Tổng số linh kiện phát hiện: {count}")
    
    cv2.imshow("Final Result V3", original)
    # cv2.imshow("Mask Debug", mask_components) # Bật cái này nếu muốn soi mask
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Chạy với ảnh của bạn
img_path = "img/Arduino Uno Rev3 SMD.jpg" # Đảm bảo đúng đường dẫn
process_arduino_v3(img_path)