
"""
Orchard Canopy Structural Modeling Pipeline
Author: Micah Levinson
Purpose:
    - Normalize LiDAR heights
    - Build canopy height model (CHM)
    - Extract row-aligned structural metrics
    - Compute voxel canopy volume per linear meter

Dependencies:
    pdal
    laspy
    numpy
    scipy
    rasterio
    shapely
    geopandas
"""
# ==============================
# ======= IMPORTS ==============
# ==============================
# %%
import json
import pdal
import laspy
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString
import geopandas as gpd
import open3d as o3d
#%%
#### Load and make sure the data in your .LAS file makes sense before proceeding.
####
las = laspy.read("C:\\Users\\mdlevins\\Desktop\\attempt_2_lidardump-2026.01.20-21.57.13.las")

print("\n--- BASIC INFO ---")
print("Point count:", len(las.points))
print("LAS version:", las.header.version)
print("Point format:", las.header.point_format)

print("\n--- DIMENSIONS PRESENT ---")
for dim in las.point_format.dimension_names:
    print("-", dim)

print("\n--- COORDINATE RANGES ---")
print("X:", np.min(las.x), "to", np.max(las.x))
print("Y:", np.min(las.y), "to", np.max(las.y))
print("Z:", np.min(las.z), "to", np.max(las.z))

# Check classification if present
if "classification" in las.point_format.dimension_names:
    classes = np.unique(las.classification)
    print("\n--- CLASSIFICATIONS FOUND ---")
    print(classes)

# Check return info if present
if "return_number" in las.point_format.dimension_names:
    print("\n--- RETURN INFO ---")
    print("Unique return numbers:", np.unique(las.return_number))

if "number_of_returns" in las.point_format.dimension_names:
    print("Unique number of returns:", np.unique(las.number_of_returns))

# Check intensity
if "intensity" in las.point_format.dimension_names:
    print("\n--- INTENSITY RANGE ---")
    print(np.min(las.intensity), "to", np.max(las.intensity))
# Check what classification values are present
if "classification" in las.point_format.dimension_names:
    print("Unique classifications:", np.unique(las.classification))
#%%
# ==============================
# ======= PARAMETERS ===========
# ==============================

INPUT_LAS = "C:\\Users\\mdlevins\\Desktop\\attempt_2_lidardump-2026.01.20-21.57.13.las"
NORMALIZED_LAS = "normalized.las"
CHM_TIF = "chm.tif"

GROUND_FILTER = {
    "slope": 0.2,
    "threshold": 0.45,
    "window": 16
}

CHM_RESOLUTION = 0.10          # meters
VOXEL_SIZE = 0.10              # meters
ROW_AZIMUTH_DEGREES = 90       # adjust to orchard orientation
ROW_WIDTH = 3.0                # meters
#%%
# ==============================
# ======= 1. NORMALIZE =========
# ==============================

def normalize_heights():
    print("Normalizing heights...")

    pipeline = {
        "pipeline": [
            INPUT_LAS,
            {
                "type": "filters.smrf",
                **GROUND_FILTER
            },
            {
                "type": "filters.hag_nn"
            },
            {
                "type": "writers.las",
                "filename": NORMALIZED_LAS,
                "extra_dims": "all"
            }
        ]
    }

    p = pdal.Pipeline(json.dumps(pipeline))
    p.execute()

    print("Height normalization complete.")


# ==============================
# ======= 2. BUILD CHM =========
# ==============================

def build_chm():
    print("Building CHM...")

    las = laspy.read(NORMALIZED_LAS)
    x = las.x
    y = las.y
    z = las.z  # height above ground

    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    width = int(np.ceil((xmax - xmin) / CHM_RESOLUTION))
    height = int(np.ceil((ymax - ymin) / CHM_RESOLUTION))

    chm = np.zeros((height, width))

    xi = ((x - xmin) / CHM_RESOLUTION).astype(int)
    yi = ((ymax - y) / CHM_RESOLUTION).astype(int)

    xi = np.clip(xi, 0, width - 1)
    yi = np.clip(yi, 0, height - 1)

    for i in range(len(z)):
        if z[i] > chm[yi[i], xi[i]]:
            chm[yi[i], xi[i]] = z[i]

    transform = from_origin(xmin, ymax, CHM_RESOLUTION, CHM_RESOLUTION)

    with rasterio.open(
        CHM_TIF,
        "w",
        driver="GTiff",
        height=chm.shape[0],
        width=chm.shape[1],
        count=1,
        dtype=chm.dtype,
        crs=None,
        transform=transform,
    ) as dst:
        dst.write(chm, 1)

    print("CHM saved.")


# ==============================
# ======= 3. ROW METRICS =======
# ==============================

def extract_row_metrics():
    print("Extracting row-aligned structural metrics...")

    las = laspy.read(NORMALIZED_LAS)
    x = las.x
    y = las.y
    z = las.z

    theta = np.deg2rad(ROW_AZIMUTH_DEGREES)

    # Rotate coordinates
    x_rot = x * np.cos(theta) + y * np.sin(theta)
    y_rot = x * np.sin(theta) + y * np.cos(theta)

    # Bin along row axis
    bin_length = 1.0  # 1 meter linear segments
    bins = np.floor((x_rot - x_rot.min()) / bin_length)

    unique_bins = np.unique(bins)

    metrics = []

    for b in unique_bins:
        mask = bins == b
        heights = z[mask]

        if len(heights) == 0:
            continue

        metrics.append({
            "linear_meter": b,
            "mean_height": np.mean(heights),
            "max_height": np.max(heights),
            "p90_height": np.percentile(heights, 90),
            "point_density": len(heights)
        })

    print("Row metrics complete.")
    return metrics


# ==============================
# ======= 4. VOXEL VOLUME ======
# ==============================

def compute_voxel_volume():
    print("Computing voxel canopy volume...")

    las = laspy.read(NORMALIZED_LAS)
    x = las.x
    y = las.y
    z = las.z

    vox_x = np.floor(x / VOXEL_SIZE)
    vox_y = np.floor(y / VOXEL_SIZE)
    vox_z = np.floor(z / VOXEL_SIZE)

    voxel_ids = set(zip(vox_x, vox_y, vox_z))

    voxel_volume = len(voxel_ids) * (VOXEL_SIZE ** 3)

    print(f"Total canopy voxel volume: {voxel_volume:.2f} m³")
    return voxel_volume
#%%

# ==============================
# ======= MAIN =================
# ==============================

def main():
    normalize_heights()
    build_chm()
    metrics = extract_row_metrics()
    volume = compute_voxel_volume()

    print("\nPipeline complete.")
    print(f"Voxel canopy volume: {volume:.2f} m³")
    print(f"Extracted {len(metrics)} row segments.")


if __name__ == "__main__":
    main()

# %%
####### Visualizing the normalized point cloud.
# Load LAS
las = laspy.read("normalized.las")
## Simple colormap function based on height
def plt_colormap(values):
    cmap = plt.get_cmap("viridis")
    return cmap(values)[:, :3]
# Downsample for visualization (very important)
step = 8  # visualize every Nth point
points = np.vstack((las.x[::step],
                    las.y[::step],
                    las.z[::step])).T

# Normalize height for coloring
z = las.z[::step]
z_norm = (z - z.min()) / (z.max() - z.min())

colors = plt_colormap(z_norm)

# Create Open3D point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)

# Visualize
o3d.visualization.draw_geometries([pcd])
# %%
"""
Orchard 3D Visualization Module
- Clips to row-scale window
- Downsamples for performance
- Builds colored voxel representation
- Optimized for UAV LiDAR orchard data
"""
# ==============================
# PARAMETERS
# ==============================

LAS_FILE = "normalized.las"

# Window size (meters) – adjust as needed
WINDOW_LENGTH = 40     # along row (N-S)
WINDOW_WIDTH = 12       # across row

VOXEL_SIZE = 0.08       # meters
DOWNSAMPLE_STEP = 3    # take every Nth point

# ==============================
# LOAD DATA
# ==============================

las = laspy.read(LAS_FILE)

x = las.x
y = las.y
z = las.z   # use normalized height if available

# ==============================
# CLIP TO WINDOW
# ==============================

# Use center of dataset
x_center = np.mean(x)
y_center = np.mean(y)

mask = (
    (x > x_center - WINDOW_WIDTH/2) &
    (x < x_center + WINDOW_WIDTH/2) &
    (y > y_center - WINDOW_LENGTH/2) &
    (y < y_center + WINDOW_LENGTH/2)
)

points = np.vstack((x[mask],
                    y[mask],
                    z[mask])).T

# Downsample
points = points[::DOWNSAMPLE_STEP]

print(f"Points in window: {len(points)}")

# ==============================
# CENTER FOR STABILITY
# ==============================

points -= points.mean(axis=0)

# ==============================
# BUILD POINT CLOUD
# ==============================

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# ==============================
# CREATE VOXEL GRID
# ==============================

voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
    pcd,
    voxel_size=VOXEL_SIZE
)

# Convert voxels to colored point cloud for better visualization
voxels = voxel_grid.get_voxels()
voxel_centers = []

for voxel in voxels:
    center = voxel.grid_index * VOXEL_SIZE
    voxel_centers.append(center)

voxel_centers = np.array(voxel_centers)

# Color by height
z_vals = voxel_centers[:, 2]
z_norm = (z_vals - z_vals.min()) / (z_vals.max() - z_vals.min())
colors = plt.get_cmap("viridis")(z_norm)[:, :3]

voxel_pcd = o3d.geometry.PointCloud()
voxel_pcd.points = o3d.utility.Vector3dVector(voxel_centers)
voxel_pcd.colors = o3d.utility.Vector3dVector(colors)

# ==============================
# VISUALIZE
# ==============================

o3d.visualization.draw_geometries([voxel_pcd])
# %%
