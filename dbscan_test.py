import rasterio
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer # <-- Import the imputer
import matplotlib.pyplot as plt
import hdbscan

# 1. Load the exported GeoTIFF data
filepath = '/Users/maarten/Downloads/embeddings.tif' 
with rasterio.open(filepath) as src:
    meta = src.meta
    img_data = src.read()
    pixels = img_data.transpose(1, 2, 0).reshape(-1, meta['count'])

# --- START: NEW FIX ---

# 2. Impute missing values
# Create an imputer that replaces NaN with the mean value of its column (band)
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

# Fit the imputer to the data and transform it (filling the NaNs)
pixels_imputed = imputer.fit_transform(pixels)

# --- END: NEW FIX ---


# 3. Scale the imputed data
# Now we use the clean, imputed data for scaling
scaler = StandardScaler()
pixels_scaled = scaler.fit_transform(pixels_imputed)

# 4. Run DBSCAN
clusterer = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=10, core_dist_n_jobs=-1, algorithm='best')
clusters = clusterer.fit_predict(pixels_scaled)

# 5. Reshape and Visualize
# The rest of the script is the same, but we reshape the final 'clusters' array
cluster_map_2d = clusters.reshape(img_data.shape[1], img_data.shape[2])

plt.figure(figsize=(10, 10))
plt.imshow(cluster_map_2d, cmap='viridis')
plt.title('DBSCAN Clustering Results (with Imputation)')
plt.colorbar(label='Cluster ID (-1 is Noise)')
plt.show()

# 6. Save the final map
meta.update(count=1, dtype='int16')
with rasterio.open('dbscan_clusters_imputed.tif', 'w', **meta) as dst:
    dst.write(cluster_map_2d.astype('int16'), 1)