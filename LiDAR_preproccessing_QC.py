#%%
import laspy
import numpy as np
import open3d as o3d
#%%
# ---- Load LAS ----
las_file = "C:\\Users\\mdlevins\\Desktop\\attempt_2_lidardump-2026.01.20-21.57.13.las"   # <-- change this
las = laspy.read(las_file)
#%%
#Does .LAS have a CRS? If so, print it.
# ---- Basic Diagnostics ----
print("Number of points:", len(las.x))
print("X range:", np.min(las.x), "to", np.max(las.x))
print("Y range:", np.min(las.y), "to", np.max(las.y))
print("Z range:", np.min(las.z), "to", np.max(las.z))
print(las.header.parse_crs())
#%%
# ---- Stack XYZ ----
points = np.vstack((las.x, las.y, las.z)).transpose()
#%%
# ---- Optional: Downsample if large ----
max_points = 2_000_000
if len(points) > max_points:
    print("Downsampling...")
    idx = np.random.choice(len(points), max_points, replace=False)
    points = points[idx]

# ---- Normalize (helps with huge coordinate values like ECEF) ----
points -= np.mean(points, axis=0)
#%%
# ---- Convert to Open3D format ----
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# ---- Color by height ----
z_vals = points[:, 2]
z_norm = (z_vals - z_vals.min()) / (z_vals.max() - z_vals.min())
colors = np.zeros((len(points), 3))
colors[:, 0] = z_norm  # red channel
colors[:, 2] = 1 - z_norm  # blue channel
pcd.colors = o3d.utility.Vector3dVector(colors)

# ---- Visualize ----
o3d.visualization.draw_geometries([pcd])
#####################
#####################
#Convert to COPC format using PDAL (if needed to visualize in QGIS)
# %%
import pdal
import json
#%%
pipeline = {
    "pipeline": [
        "C:\\Users\\mdlevins\\Desktop\\attempt_2_lidardump-2026.01.20-21.57.13.las",
        {
            "type": "writers.copc",
            "filename": "C:\\Users\\mdlevins\\Desktop\\attempt_2_lidardump_copc.laz"
        }
    ]
}

p = pdal.Pipeline(json.dumps(pipeline))
p.execute()

# %%
