import cv2
import os
import numpy as np

def detect_color(image_name):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    img_path = os.path.join(base_dir, "data", "images", image_name)
    output_dir = os.path.join(base_dir, "outputs", "task3")
    os.makedirs(output_dir, exist_ok=True)

    
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print(f"未找到图片 {image_name}")
        return

    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    
    lower_blue = np.array([100, 100, 50])
    upper_blue = np.array([130, 255, 255])

    
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result_img = img.copy()
    
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            x, y, w, h = cv2.boundingRect(cnt)
            center_x, center_y = x + w // 2, y + h // 2

            cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(result_img, (center_x, center_y), 5, (0, 0, 255), -1)

            text = f"({center_x},{center_y}) Area:{int(area)}"
            cv2.putText(result_img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            print(f"[{image_name}] 找到目标: 中心点({center_x}, {center_y}), 面积: {int(area)}")

    
    name_without_ext = os.path.splitext(image_name)[0]
    mask_path = os.path.join(output_dir, f"{name_without_ext}_mask.png")
    result_path = os.path.join(output_dir, f"{name_without_ext}_result.png")
    
    cv2.imencode('.png', mask)[1].tofile(mask_path)
    cv2.imencode('.png', result_img)[1].tofile(result_path)

if __name__ == "__main__":
    image_list = ["test1.jpg", "test2.jpg", "test3.jpg"]
    for name in image_list:
        detect_color(name)
    print("任务三执行完毕，掩膜和结果图已保存至 outputs/task3/")