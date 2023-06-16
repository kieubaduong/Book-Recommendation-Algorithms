import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# Đọc dữ liệu từ file user_features_v2.csv
df_user_features = pd.read_csv("/content/drive/MyDrive/Dataset/featured/user_features_v2.csv")

df_user_features = df_user_features.drop_duplicates(subset='feature', keep='first')
user_features = np.array([np.fromstring(features[1:-1], dtype=float, sep=' ') for features in df_user_features['feature']])

# Xác định số lượng clusters tối ưu bằng phương pháp Elbow
distortions = []
k_values = range(1, 31)  # Giả sử chúng ta xét số lượng clusters từ 1 đến 30
for k in k_values:
    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(user_features)
    distortions.append(kmeans.inertia_)

# Vẽ biểu đồ Elbow
plt.plot(k_values, distortions, 'bx-')
plt.xlabel('Số lượng cụm (k)')
plt.ylabel('Distortion')
plt.title('Phương pháp Elbow để chọn số lượng cụm tối ưu')
plt.show()