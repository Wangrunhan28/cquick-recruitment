import cv2
import os
import numpy as np

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    img_path = os.path.join(base_dir, "data", "images", "test1.jpg") 
    output_dir = os.path.join(base_dir, "outputs", "task2")
    os.makedirs(output_dir, exist_ok=True)

    
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print("未找到图片，请检查路径。")
        return

    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imencode('.jpg', gray)[1].tofile(os.path.join(output_dir, "01_gray.jpg"))

    
    blur_3 = cv2.GaussianBlur(gray, (3, 3), 0)
    blur_9 = cv2.GaussianBlur(gray, (9, 9), 0)
    cv2.imencode('.jpg', blur_3)[1].tofile(os.path.join(output_dir, "02_blur_3x3.jpg"))
    cv2.imencode('.jpg', blur_9)[1].tofile(os.path.join(output_dir, "03_blur_9x9.jpg"))

    
    _, thresh_100 = cv2.threshold(blur_3, 100, 255, cv2.THRESH_BINARY)
    _, thresh_180 = cv2.threshold(blur_3, 180, 255, cv2.THRESH_BINARY)
    cv2.imencode('.jpg', thresh_100)[1].tofile(os.path.join(output_dir, "04_thresh_100.jpg"))
    cv2.imencode('.jpg', thresh_180)[1].tofile(os.path.join(output_dir, "05_thresh_180.jpg"))

    
    edges = cv2.Canny(blur_3, 50, 150)
    cv2.imencode('.jpg', edges)[1].tofile(os.path.join(output_dir, "06_canny.jpg"))

    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    opened = cv2.morphologyEx(thresh_100, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(thresh_100, cv2.MORPH_CLOSE, kernel)
    cv2.imencode('.jpg', opened)[1].tofile(os.path.join(output_dir, "07_morph_open.jpg"))
    cv2.imencode('.jpg', closed)[1].tofile(os.path.join(output_dir, "08_morph_close.jpg"))

    print("任务二执行完毕，中间结果已保存至 outputs/task2/")

if __name__ == "__main__":
    main()