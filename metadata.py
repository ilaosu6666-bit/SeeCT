"""LIDC-IDRI / LUNA16 元数据解析模块。

解析放射科医生的结节标注，提供良恶性标签转换。
"""

import csv
import xml.etree.ElementTree as ET
from typing import Dict, List


def parse_lidc_xml(xml_path: str) -> Dict:
    """解析 LIDC-IDRI XML 标注文件。

    Returns:
        {
            'patient_id': str,
            'nodules': [{
                'nodule_id': str,
                'malignancy': int,     # 1-5
                'subtlety': int,       # 1-5
                'spiculation': int,    # 1-5
                'texture': int,        # 1-5
                'calcification': int,  # 1-6
                'lobulation': int,     # 1-5
                'margin': int,         # 1-5
                'sphericity': int,     # 1-5
                'diameter': float,     # mm
            }, ...]
        }
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    ns = {"lidc": "http://www.nih.gov"}
    patient_id = root.findtext(".//{http://www.nih.gov}PatientID", "")

    nodules = []
    for session in root.findall(".//{http://www.nih.gov}readingSession"):
        for nodule in session.findall(".//{http://www.nih.gov}unblindedReadNodule"):
            nodule_id = nodule.findtext("{http://www.nih.gov}noduleID", "")

            characteristics = nodule.find("{http://www.nih.gov}characteristics")
            if characteristics is None:
                continue

            malignancy = int(float(characteristics.findtext(
                "{http://www.nih.gov}malignancy", "3"
            )))
            subtlety = int(float(characteristics.findtext(
                "{http://www.nih.gov}subtlety", "3"
            )))
            spiculation = int(float(characteristics.findtext(
                "{http://www.nih.gov}spiculation", "3"
            )))
            texture = int(float(characteristics.findtext(
                "{http://www.nih.gov}texture", "3"
            )))
            calcification = int(float(characteristics.findtext(
                "{http://www.nih.gov}calcification", "3"
            )))
            lobulation = int(float(characteristics.findtext(
                "{http://www.nih.gov}lobulation", "3"
            )))
            margin = int(float(characteristics.findtext(
                "{http://www.nih.gov}margin", "3"
            )))
            sphericity = int(float(characteristics.findtext(
                "{http://www.nih.gov}sphericity", "3"
            )))
            diameter = float(characteristics.findtext(
                "{http://www.nih.gov}diameter", "0"
            ))

            nodules.append({
                "nodule_id": nodule_id,
                "malignancy": malignancy,
                "subtlety": subtlety,
                "spiculation": spiculation,
                "texture": texture,
                "calcification": calcification,
                "lobulation": lobulation,
                "margin": margin,
                "sphericity": sphericity,
                "diameter": diameter,
            })

    return {"patient_id": patient_id, "nodules": nodules}


def malignancy_to_label(malignancy_ratings: List[int], threshold: int = 3) -> str:
    """LIDC 恶性评分(1-5) → Benign / Malignant。

    Args:
        malignancy_ratings: 多位放射科医生的恶性评分
        threshold: 平均分 >= threshold → Malignant, < threshold → Benign
    """
    if not malignancy_ratings:
        return "Unknown"
    avg = sum(malignancy_ratings) / len(malignancy_ratings)
    return "Malignant" if avg >= threshold else "Benign"


def parse_luna_csv(csv_path: str) -> List[Dict]:
    """解析 LUNA16 candidates.csv 或 annotations.csv。

    Returns:
        [{
            'seriesuid': str,
            'coordX': float, 'coordY': float, 'coordZ': float,
            'diameter_mm': float,
            'class': int  (0=非结节, 1=结节)
        }, ...]
    """
    results = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                "seriesuid": row.get("seriesuid", ""),
                "coordX": float(row.get("coordX", 0)),
                "coordY": float(row.get("coordY", 0)),
                "coordZ": float(row.get("coordZ", 0)),
                "diameter_mm": float(row.get("diameter_mm", 0)),
                "class": int(row.get("class", 0)),
            })
    return results


LIDC_FEATURE_NAMES_CN = {
    "subtlety": "显著性",
    "spiculation": "毛刺征",
    "texture": "纹理",
    "calcification": "钙化",
    "lobulation": "分叶征",
    "margin": "边缘",
    "sphericity": "球形度",
    "malignancy": "恶性度",
}
