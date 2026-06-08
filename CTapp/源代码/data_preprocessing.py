"""CT医学影像预处理模块。

支持 DICOM / .mhd / .nii 格式的CT数据加载与预处理，
为游戏关卡1的"数据准备"场景提供底层能力。
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image


def load_ct_scan(path: str) -> np.ndarray:
    """自动识别格式加载CT扫描为3D numpy数组 [D, H, W]。

    支持: DICOM目录 (.dcm), .mhd/.raw, .nii/.nii.gz
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"路径不存在: {path}")

    if path.is_dir():
        return _load_dicom_series(path)
    elif path.suffix.lower() in (".mhd", ".raw"):
        return _load_mhd(path)
    elif path.suffix.lower() in (".nii", ".gz"):
        return _load_nifti(path)
    else:
        raise ValueError(f"不支持的格式: {path.suffix}")


def _load_dicom_series(dicom_dir: Path) -> np.ndarray:
    try:
        import pydicom
    except ImportError:
        raise ImportError("需要安装 pydicom: pip install pydicom")

    slices = []
    for f in sorted(dicom_dir.glob("*")):
        if f.suffix.lower() == ".dcm" or f.suffix == "":
            try:
                ds = pydicom.dcmread(str(f), force=True)
                if hasattr(ds, "pixel_array"):
                    slices.append((float(ds.ImagePositionPatient[2])
                                   if hasattr(ds, "ImagePositionPatient") else 0.0,
                                   ds.pixel_array.astype(np.float32) * ds.RescaleSlope + ds.RescaleIntercept
                                   if hasattr(ds, "RescaleSlope") else ds.pixel_array.astype(np.float32)))
            except Exception:
                continue

    if not slices:
        raise ValueError(f"在 {dicom_dir} 中未找到有效DICOM文件")

    slices.sort(key=lambda x: x[0])
    volume = np.stack([s[1] for s in slices], axis=0)
    return volume


def _load_mhd(mhd_path: Path) -> np.ndarray:
    try:
        import SimpleITK as sitk
    except ImportError:
        raise ImportError("需要安装 SimpleITK: pip install SimpleITK")

    image = sitk.ReadImage(str(mhd_path))
    volume = sitk.GetArrayFromImage(image)
    return volume.astype(np.float32)


def _load_nifti(nifti_path: Path) -> np.ndarray:
    try:
        import SimpleITK as sitk
    except ImportError:
        raise ImportError("需要安装 SimpleITK: pip install SimpleITK")

    image = sitk.ReadImage(str(nifti_path))
    volume = sitk.GetArrayFromImage(image)
    return volume.astype(np.float32)


def apply_lung_window(hu_volume: np.ndarray, wc: int = -600, ww: int = 1500) -> np.ndarray:
    """肺窗映射：HU值 → 0-255 uint8。

    Args:
        hu_volume: HU值数组 (任意shape)
        wc: 窗位 (Window Center)
        ww: 窗宽 (Window Width)
    """
    low = wc - ww / 2
    high = wc + ww / 2
    clipped = np.clip(hu_volume, low, high)
    normalized = (clipped - low) / (high - low)
    return (normalized * 255).astype(np.uint8)


def apply_mediastinal_window(hu_volume: np.ndarray) -> np.ndarray:
    """纵隔窗 (WC=50, WW=350)"""
    return apply_lung_window(hu_volume, wc=50, ww=350)


def apply_bone_window(hu_volume: np.ndarray) -> np.ndarray:
    """骨窗 (WC=400, WW=1800)"""
    return apply_lung_window(hu_volume, wc=400, ww=1800)


def normalize_hu(hu_volume: np.ndarray, min_hu: int = -1000, max_hu: int = 400) -> np.ndarray:
    """HU值裁剪+归一化到[0, 1]"""
    clipped = np.clip(hu_volume, min_hu, max_hu)
    return (clipped - min_hu) / (max_hu - min_hu)


def hu_to_uint8(hu_slice: np.ndarray, wc: int = -600, ww: int = 1500) -> np.ndarray:
    """单张HU切片转8-bit PNG（应用肺窗）"""
    return apply_lung_window(hu_slice, wc, ww)


def segment_lung_mask(ct_slice: np.ndarray, threshold: int = -400) -> np.ndarray:
    """阈值法+形态学操作提取肺实质mask。

    Args:
        ct_slice: 2D CT切片 (HU值或已归一化)
        threshold: HU阈值，-400HU以下为空气/肺组织
    """
    binary = (ct_slice < threshold).astype(np.uint8)

    try:
        import cv2
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        areas = [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_labels)]
        areas.sort(key=lambda x: x[1], reverse=True)
        keep_ids = {i for i, a in areas[:2]}
        mask = np.zeros_like(binary)
        for lid in keep_ids:
            mask[labels == lid] = 1
        return mask.astype(bool)
    except ImportError:
        return binary.astype(bool)


def extract_nodule_candidate(volume_3d: np.ndarray,
                              centroid_xyz: Tuple[int, int, int],
                              cube_size: int = 64) -> np.ndarray:
    """从3D体积提取以结节为中心的cube（参考mr-mukherjee03处理方式）。

    Args:
        volume_3d: 3D CT体积 [D, H, W]
        centroid_xyz: 结节中心坐标 (x, y, z)
        cube_size: 输出cube尺寸
    """
    D, H, W = volume_3d.shape
    cx, cy, cz = centroid_xyz
    half = cube_size // 2

    x1 = max(0, cx - half)
    x2 = min(W, cx + half)
    y1 = max(0, cy - half)
    y2 = min(H, cy + half)
    z1 = max(0, cz - half)
    z2 = min(D, cz + half)

    cube = volume_3d[z1:z2, y1:y2, x1:x2]

    if cube.shape != (cube_size, cube_size, cube_size):
        result = np.zeros((cube_size, cube_size, cube_size), dtype=cube.dtype)
        dz, dy, dx = cube.shape
        result[:dz, :dy, :dx] = cube
        return result
    return cube


def extract_slices(volume_3d: np.ndarray,
                   slice_indices: Optional[List[int]] = None) -> List[np.ndarray]:
    """从3D体积提取指定切片，默认提取全部。"""
    if slice_indices is None:
        slice_indices = list(range(volume_3d.shape[0]))
    return [volume_3d[i] for i in slice_indices if 0 <= i < volume_3d.shape[0]]


def process_dicom_to_png(dicom_dir: str, output_dir: str, window_type: str = "lung"):
    """批量DICOM → PNG转换（肺窗/纵隔窗/骨窗）"""
    volume = load_ct_scan(dicom_dir)
    os.makedirs(output_dir, exist_ok=True)

    window_funcs = {
        "lung": lambda v: apply_lung_window(v),
        "mediastinal": lambda v: apply_mediastinal_window(v),
        "bone": lambda v: apply_bone_window(v),
    }
    apply_window = window_funcs.get(window_type, window_funcs["lung"])

    for i in range(volume.shape[0]):
        slice_uint8 = apply_window(volume[i])
        img = Image.fromarray(slice_uint8)
        img.save(os.path.join(output_dir, f"{i+1:04d}.png"))

    print(f"已转换 {volume.shape[0]} 张切片到 {output_dir}")


def load_png_volume(png_dir: str) -> np.ndarray:
    """加载PNG切片目录为3D体积 [D, H, W]。

    供关卡1切片浏览器使用。
    """
    png_files = sorted([
        f for f in os.listdir(png_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])
    slices = []
    for f in png_files:
        img = Image.open(os.path.join(png_dir, f)).convert("L")
        slices.append(np.array(img))
    if not slices:
        raise ValueError(f"在 {png_dir} 中未找到PNG切片")
    return np.stack(slices, axis=0)


WINDOW_PRESETS = {
    "肺窗": (-600, 1500),
    "纵隔窗": (50, 350),
    "骨窗": (400, 1800),
}
