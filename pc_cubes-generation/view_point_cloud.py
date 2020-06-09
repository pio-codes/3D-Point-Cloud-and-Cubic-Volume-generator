
"""
Point cloud class generated from stereo image pairs.
Classes:
    * ``PointCloud`` - Point cloud with RGB colors
.. image:: classes_point_cloud.svg
"""

from open3d import *
import numpy as np

# Usage: pcd = read_point_cloud("file or file path")

if __name__ == "__main__":
    print("Load a ply point cloud, print it, and render it")
    pcd = read_point_cloud("C:/Users/Promise Okorie/PycharmProjects/final/final_test/test_pair_0/point_cloud_test_0.ply")
    print(pcd)
    print(np.asarray(pcd.points))
    draw_geometries([pcd])
