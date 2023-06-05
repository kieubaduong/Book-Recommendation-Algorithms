import pandas as pd
import re

df = pd.read_csv('../../dataset/processed_dataset/users.csv')

def remove_special_characters(text):
    # Định nghĩa biểu thức chính quy để tìm các ký tự đặc biệt
    pattern = r'[^\w\s]'
    
    # Sử dụng hàm sub của module re để thay thế các ký tự đặc biệt bằng ký tự trống
    cleaned_text = re.sub(pattern, '', text)
    
    return cleaned_text

# Xử lý tập dữ liệu người dùng
for index, row in df.iterrows():
    location = row['location']
    cleaned_location = remove_special_characters(location)
    df.at[index, 'location'] = cleaned_location

df.to_csv('../../dataset/processed_dataset/users.csv', index=False)