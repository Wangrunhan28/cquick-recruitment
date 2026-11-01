import cv2
import os
import numpy as np

def process_image(image_name):
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    img_path = os.path.join(base_dir, "data", "images", image_name)
    output_dir = os.path.join(base_dir, "outputs", "task1")
    
    os.makedirs(output_dir, exist_ok=True)

    
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    
    if img is None:
        print(f"错误：无法读取图片 {img_path}，请检查路径或文件名！")
        return
    
    height, width, channels = img.shape
    print(f"--- 正在处理: {image_name} ---")
    print(f"读取成功！宽: {width}, 高: {height}, 通道数: {channels}")

    
    resized_img = cv2.resize(img, (width // 2, height // 2))
    cropped_img = img[height // 2 :, width // 2 :]
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    
    name_without_ext = os.path.splitext(image_name)[0]
    resize_path = os.path.join(output_dir, f"{name_without_ext}_resized.jpg")
    crop_path = os.path.join(output_dir, f"{name_without_ext}_cropped.jpg")
    gray_path = os.path.join(output_dir, f"{name_without_ext}_gray.jpg")

    cv2.imencode('.jpg', resized_img)[1].tofile(resize_path)
    cv2.imencode('.jpg', cropped_img)[1].tofile(crop_path)
    cv2.imencode('.jpg', gray_img)[1].tofile(gray_path)

    
    check_img = cv2.imdecode(np.fromfile(gray_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if check_img is not None:
        print("验证通过：保存的灰度图可以成功重新读取。\n")

if __name__ == "__main__":
    image_list = ["test1.jpg", "test2.jpg"] 
    for name in image_list:
        process_image(name)