import cv2
import numpy as np

def process_arduino_smd(image_path):
    # 1. Đọc ảnh từ đường dẫn bạn cung cấp
    print(f"Đang xử lý ảnh: {image_path}")
    img = cv2.imread(image_path)
    
    if img is None:
        print("Lỗi: Không tìm thấy file ảnh! Hãy kiểm tra lại đường dẫn.")
        return

    # Resize ảnh nếu quá to để hiển thị vừa màn hình (tùy chọn)
    # img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
    original = img.copy()
    
    # 2. Chuyển sang không gian màu HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # --- BƯỚC 1: TÁCH NỀN MẠCH (BOARD SEGMENTATION) ---
    # Arduino Uno SMD thường có màu Xanh Teal đậm.
    # Dải màu này bao phủ từ xanh lá đậm đến xanh dương nhạt.
    lower_teal = np.array([35, 50, 50])   
    upper_teal = np.array([100, 255, 255]) 
    
    mask_board = cv2.inRange(hsv, lower_teal, upper_teal)
    
    # Xử lý nhiễu: Lấp đầy các lỗ nhỏ trên mạch (do chữ in làm đứt đoạn màu xanh)
    kernel = np.ones((5,5), np.uint8)
    mask_board = cv2.morphologyEx(mask_board, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_board = cv2.morphologyEx(mask_board, cv2.MORPH_OPEN, kernel, iterations=1)

    # --- BƯỚC 2: TẠO ROI (VÙNG QUAN TÂM) ---
    # Mục đích: Chỉ xử lý bên trong cái mạch, bỏ qua nền trắng/bàn gỗ bên ngoài
    contours_board, _ = cv2.findContours(mask_board, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours_board:
        print("Không tìm thấy mạch xanh nào cả! Cần chỉnh lại dải màu HSV.")
        return
        
    # Lấy contour lớn nhất -> Chính là bo mạch Arduino
    c_board = max(contours_board, key=cv2.contourArea)
    mask_roi = np.zeros_like(mask_board)
    cv2.drawContours(mask_roi, [c_board], -1, 255, -1) # Vẽ đặc màu trắng lên nền đen

    # --- BƯỚC 3: TÌM ỨNG VIÊN LINH KIỆN ---
    # Logic: Trong vùng ROI, cái gì KHÔNG PHẢI MÀU XANH thì là Linh kiện (hoặc Chữ/Lỗ kim loại)
    mask_components = cv2.bitwise_not(mask_board)
    # Cắt bỏ phần nhiễu bên ngoài bo mạch
    mask_components = cv2.bitwise_and(mask_components, mask_components, mask=mask_roi)

    # --- BƯỚC 4: PHÂN LOẠI & ĐẾM (BLOB ANALYSIS) ---
    contours, _ = cv2.findContours(mask_components, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    count = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    print(f"--- BẮT ĐẦU QUÉT LINH KIỆN ---")
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        
        # --- BỘ LỌC QUAN TRỌNG (FILTERING) ---
        
        # 1. Lọc rác quá nhỏ (nhiễu ảnh)
        # Chip SMD khá nhỏ nên để min_area thấp (khoảng 30-50 px)
        if area < 50: continue
        
        # 2. Lọc bóng quá to (đôi khi bóng của mạch bị nhận nhầm)
        if area > (img.shape[0]*img.shape[1] * 0.8): continue

        # 3. Tính toán đặc trưng hình học
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0: continue
        
        solidity = float(area) / hull_area  # Độ đặc (Linh kiện thường > 0.7, Chữ < 0.5)
        aspect_ratio = float(w) / h         # Tỷ lệ cạnh (Linh kiện < 4.0, Dòng chữ > 5.0)
        
        # ĐIỀU KIỆN CHẤP NHẬN LÀ LINH KIỆN:
        # TH1: Các linh kiện chuẩn (Chip, Tụ, Header) -> Đặc và gọn
        is_standard_component = (solidity > 0.65 and aspect_ratio < 4.0)
        
        # TH2: Các linh kiện to đặc biệt (Cổng USB, Jack nguồn, Tụ nhôm to)
        # Những thứ này đôi khi có hình dạng phức tạp hoặc bị bóng, nhưng diện tích rất lớn
        is_large_component = (area > 800) 

        if is_standard_component or is_large_component:
            count += 1
            
            # Vẽ Bounding Box (Màu đỏ)
            cv2.rectangle(original, (x, y), (x+w, y+h), (0, 0, 255), 2)
            
            # Vẽ tâm (Chấm xanh lá)
            cx, cy = x + w//2, y + h//2
            cv2.circle(original, (cx, cy), 3, (0, 255, 0), -1)
            
            # Đánh số thứ tự
            # cv2.putText(original, str(count), (x, y-5), font, 0.4, (255, 0, 0), 1)

    print(f"Tổng số linh kiện phát hiện: {count}")
    
    # Hiển thị kết quả
    # Cửa sổ 1: Mask mạch (để kiểm tra xem đã bắt đúng vùng xanh chưa)
    cv2.imshow("Debug: Mask Board", mask_board)
    
    # Cửa sổ 2: Mask linh kiện thô (trước khi lọc)
    cv2.imshow("Debug: Raw Components", mask_components)
    
    # Cửa sổ 3: Kết quả cuối cùng
    cv2.imshow("Final Result: Arduino SMD Analysis", original)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# --- CHẠY CHƯƠNG TRÌNH ---
# Đường dẫn ảnh của bạn
img_path = "img/Arduino Uno Rev3 SMD.jpg"
process_arduino_smd(img_path)