from PIL import Image
import os

folder_path = 'D:\\CODE\\COMPUTER VISION\\Vision Proj\\DATASET\\not_cheating'  # Đường dẫn tới thư mục chứa các file JPG và PNG

for file_name in os.listdir(folder_path):
    if file_name.endswith('.jpg'):
        # Đọc file và chuyển đổi định dạng thành PNG
        image_path = os.path.join(folder_path, file_name)
        image = Image.open(image_path)
        new_file_name = file_name.rsplit('.', 1)[0] + '.png'  # Đổi tên file thành tên mới với định dạng PNG
        new_image_path = os.path.join(folder_path, new_file_name)
        image.save(new_image_path, 'PNG')
        
        # Xóa file JPG cũ
        os.remove(image_path)