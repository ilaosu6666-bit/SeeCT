"""病例库构建脚本。

将现有 cases/Lesion/ 和 cases/Normal/ 的平面PNG数据转换为
游戏所需的 cases/case_XXX/image.png + slices/ + label.json 格式。

本地运行一次后，生成的病例库随代码 Push 到 HF Spaces。
"""

import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image
import numpy as np

ROOT_DIR = Path(r"D:\CTapp")
LESION_DIR = ROOT_DIR / "cases" / "Lesion"
NORMAL_DIR = ROOT_DIR / "cases" / "Normal" / "Normal"
OUTPUT_DIR = ROOT_DIR / "cases"

CASE_METADATA_TEMPLATES: List[Dict] = [
    {
        "patient_id": "3000522",
        "title": "周围型肺结节 - 右肺上叶",
        "class": "Lesion",
        "difficulty": "easy",
        "description": "右肺上叶周围型结节，边缘清晰，直径约15mm。结节呈类圆形，密度均匀，邻近胸膜无牵拉。",
        "medical_knowledge": {
            "imaging_features": ["类圆形结节", "边缘清晰", "密度均匀"],
            "differential_diagnosis": ["炎性假瘤", "结核球"],
            "lidc_characteristics": {"subtlety": 3, "spiculation": 2, "texture": 3, "malignancy_avg": 2.5},
        },
        "teaching_points": [
            "边缘清晰的类圆形结节需结合临床判断",
            "Grad-CAM关注区域集中在结节实质部分",
        ],
    },
    {
        "patient_id": "3000566",
        "title": "肺结节伴毛刺征 - 右肺中叶",
        "class": "Lesion",
        "difficulty": "medium",
        "description": "右肺中叶结节，直径约18mm，边缘可见短毛刺征，邻近胸膜轻度凹陷。毛刺征是恶性结节的重要影像学特征。",
        "medical_knowledge": {
            "imaging_features": ["毛刺征", "胸膜凹陷征", "分叶状边缘"],
            "differential_diagnosis": ["周围型肺腺癌", "炎性假瘤"],
            "lidc_characteristics": {"subtlety": 4, "spiculation": 4, "texture": 5, "malignancy_avg": 4.2},
        },
        "teaching_points": [
            "毛刺征是恶性结节的典型特征之一",
            "Grad-CAM高亮区域与结节毛刺边缘高度吻合",
        ],
    },
    {
        "patient_id": "3000611",
        "title": "磨玻璃密度结节 - 左肺上叶",
        "class": "Lesion",
        "difficulty": "hard",
        "description": "左肺上叶磨玻璃密度影，边界模糊，直径约12mm。磨玻璃结节诊断难度大，需结合随访变化判断。",
        "medical_knowledge": {
            "imaging_features": ["磨玻璃密度", "边界模糊", "无实性成分"],
            "differential_diagnosis": ["非典型腺瘤样增生(AAH)", "局灶性炎症"],
            "lidc_characteristics": {"subtlety": 5, "spiculation": 1, "texture": 1, "malignancy_avg": 3.0},
        },
        "teaching_points": [
            "磨玻璃结节的AI诊断准确率通常低于实性结节",
            "边界模糊导致Grad-CAM热力图分散",
        ],
    },
    {
        "patient_id": "3000631",
        "title": "分叶状肺结节 - 右肺下叶",
        "class": "Lesion",
        "difficulty": "medium",
        "description": "右肺下叶分叶状结节，直径约22mm，边缘呈分叶状，内部密度不均。分叶征提示结节生长不均匀。",
        "medical_knowledge": {
            "imaging_features": ["分叶征", "密度不均", "边缘不规则"],
            "differential_diagnosis": ["肺腺癌", "结核球"],
            "lidc_characteristics": {"subtlety": 3, "spiculation": 3, "texture": 4, "malignancy_avg": 3.8},
        },
        "teaching_points": [
            "AI对分叶状边缘的敏感度高于光滑边缘",
            "分叶征提示结节各方向生长速度不一",
        ],
    },
    {
        "patient_id": "normal_001",
        "title": "正常肺实质 - 清晰透亮",
        "class": "Normal",
        "difficulty": "easy",
        "description": "双肺透亮度正常，肺纹理清晰，未见异常密度影。气管支气管通畅，纵隔居中。",
        "medical_knowledge": {
            "imaging_features": ["肺纹理清晰", "透亮度正常", "未见异常密度"],
            "differential_diagnosis": [],
            "lidc_characteristics": {},
        },
        "teaching_points": [
            "正常肺CT的关键识别点：肺纹理走行自然、无异常密度影",
            "AI对正常肺组织的Grad-CAM热力图应均匀分布",
        ],
    },
    {
        "patient_id": "normal_002",
        "title": "正常肺实质 - 纵隔窗",
        "class": "Normal",
        "difficulty": "easy",
        "description": "肺实质未见结节或肿块。纵隔结构正常，大血管走行自然，未见淋巴结肿大。",
        "medical_knowledge": {
            "imaging_features": ["肺实质清晰", "纵隔结构正常", "无淋巴结肿大"],
            "differential_diagnosis": [],
            "lidc_characteristics": {},
        },
        "teaching_points": [
            "初学者易将正常血管断面误认为结节",
            "连续浏览相邻切片有助于区分血管和结节",
        ],
    },
]


def find_patient_dir(lesion_dir: Path, patient_id: str) -> Optional[Path]:
    for item in sorted(lesion_dir.iterdir()):
        if item.is_dir() and patient_id in item.name:
            return item
    return None


def find_representative_slice(slices_dir: Path, num_slices: int) -> Optional[Path]:
    png_files = sorted([f for f in slices_dir.iterdir()
                        if f.suffix.lower() == ".png"])
    if not png_files:
        return None
    middle_idx = len(png_files) // 2
    return png_files[middle_idx]


def build_case_library():
    if not LESION_DIR.exists():
        print(f"Lesion目录不存在: {LESION_DIR}")
        return

    case_index = 1

    for template in CASE_METADATA_TEMPLATES:
        patient_id = template["patient_id"]
        case_name = f"case_{case_index:03d}"
        case_dir = OUTPUT_DIR / case_name
        slices_dir = case_dir / "slices"

        if patient_id.startswith("normal"):
            normal_dirs = [d for d in NORMAL_DIR.iterdir() if d.is_dir()]
            if not normal_dirs:
                print(f"跳过: Normal目录无患者子目录 {NORMAL_DIR}")
                continue
            source_slices = normal_dirs[0]
            if patient_id == "normal_002" and len(normal_dirs) > 1:
                source_slices = normal_dirs[1]
        else:
            patient_dir = find_patient_dir(LESION_DIR, patient_id)
            if patient_dir is None:
                print(f"跳过: 未找到患者目录包含 {patient_id}")
                continue
            source_slices = patient_dir

        if not source_slices.exists():
            print(f"跳过: 源目录不存在 {source_slices}")
            continue

        case_dir.mkdir(parents=True, exist_ok=True)
        slices_dir.mkdir(parents=True, exist_ok=True)

        png_files = sorted([f for f in source_slices.iterdir()
                            if f.suffix.lower() == ".png"])

        if not png_files:
            print(f"跳过: 源目录无PNG文件 {source_slices}")
            shutil.rmtree(case_dir)
            continue

        for png_file in png_files:
            shutil.copy2(png_file, slices_dir / png_file.name)

        representative = png_files[len(png_files) // 2]
        shutil.copy2(representative, case_dir / "image.png")

        meta = {
            "id": case_name,
            "title": template["title"],
            "class": template["class"],
            "difficulty": template["difficulty"],
            "description": template["description"],
            "image": "image.png",
            "answer_mask": "",
            "slice_count": len(png_files),
            "medical_knowledge": template["medical_knowledge"],
            "teaching_points": template["teaching_points"],
        }

        meta_path = case_dir / "label.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print(f"已创建 {case_name}: {template['title']} ({len(png_files)} 张切片)")
        case_index += 1

    print(f"\n病例库构建完成，共 {case_index - 1} 个病例")


if __name__ == "__main__":
    build_case_library()
