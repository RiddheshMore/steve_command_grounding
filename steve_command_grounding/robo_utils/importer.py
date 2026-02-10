"""
Unified importer for Open3D components.
Avoids the "type already registered" error by using the top-level API.
"""

try:
    import open3d as o3d
    AxisAlignedBoundingBox = o3d.geometry.AxisAlignedBoundingBox
    PointCloud = o3d.geometry.PointCloud
    TriangleMesh = o3d.geometry.TriangleMesh
    Vector3dVector = o3d.utility.Vector3dVector
except ImportError:
    # Fallback for systems without Open3D
    AxisAlignedBoundingBox = None
    PointCloud = None
    TriangleMesh = None
    Vector3dVector = None
