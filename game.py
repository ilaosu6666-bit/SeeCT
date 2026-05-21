"""
AI训练师 — 基于可解释AI的肺结节交互式教学游戏

五关游戏化体验：数据准备 → 模型搭建 → 训练调参 → 诊断评估 → 临床实战
玩家扮演AI研究员，从零开发肺结节诊断模型。
"""

import json
import os
import time
import random
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2
from PIL import Image, ImageDraw
import streamlit as st
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet18

# ---------- 可选导入 ----------
try:
    from streamlit_drawable_canvas import st_canvas
    HAS_CANVAS = True
except Exception:
    HAS_CANVAS = False

# ---------- 基础配置 ----------
IMAGE_SIZE = 224
CLASS_TO_IDX = {"Normal": 0, "Lesion": 1}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_MODEL_PATH = os.path.join("model_parameter", "resnet18_ct_lesion_normal_best.pt")
DEFAULT_CASE_ROOT = "cases"
HISTORY_PATH = os.path.join("loss_fig", "training_history.json")
GRADCAM_DIR = os.path.join("loss_fig", "gradcam_epochs")
MODEL_CONFIG_PATH = "model_config.json"


def load_image_smart(path: str):
    """加载图片，优先PNG→NPY。NPY用numpy直接读。"""
    p = Path(path)
    if p.suffix.lower() == ".npy":
        if p.exists():
            arr = np.load(str(p))
            if arr.dtype in (np.float64, np.float32) and arr.max() <= 1.0:
                arr = (arr * 255).astype(np.uint8)
            return Image.fromarray(arr.astype(np.uint8))
        return None
    if p.exists():
        return Image.open(p).convert("RGB")
    npy_path = str(p).rsplit(".", 1)[0] + ".npy"
    if os.path.exists(npy_path):
        return load_image_smart(npy_path)
    return None

VAL_TEST_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ---------- 游戏状态初始化 ----------
def init_game_state():
    defaults = {
        "stage": 1,
        "score": 0,
        "achievements": [],
        "stage1_complete": False,
        "stage2_complete": False,
        "stage3_complete": False,
        "stage4_complete": False,
        "stage5_complete": False,
        # 关卡1产出
        "data_label_score": 0,
        "window_found": False,
        # 关卡2产出
        "chosen_arch": None,
        "hyperparams": {},
        # 关卡3产出
        "training_decisions": [],
        "model_accuracy": 0.0,
        "stage3_scores": [],
        # 关卡4产出
        "cases_reviewed": [],
        "diagnosis_scores": [],
        # 关卡5产出
        "batch_results": [],
        "final_grade": None,
        "ai_trap_detected": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_game():
    for k in list(st.session_state.keys()):
        if k not in ("_is_running",):
            del st.session_state[k]
    init_game_state()
    st.rerun()


# ---------- 模型加载（缓存） ----------
@st.cache_resource(show_spinner=False)
def load_model_resource(model_path: str):
    if not os.path.exists(model_path):
        return None, None
    from part1 import build_model, GradCAMHelper
    model = build_model().to(DEVICE)
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    gradcam = GradCAMHelper(model)
    return model, gradcam


def load_training_history():
    if not os.path.exists(HISTORY_PATH):
        return None
    with open(HISTORY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_case_list(case_root: str) -> List[Path]:
    root = Path(case_root)
    if not root.exists():
        return []
    return sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("case_")])


def load_case_meta(case_dir: Path) -> Dict:
    meta_path = case_dir / "label.json"
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"id": case_dir.name, "title": case_dir.name, "class": "Unknown"}


# ---------- 侧边栏 ----------
def render_sidebar():
    with st.sidebar:
        st.title("🧠 AI训练师")
        stage_names = {
            1: "关卡1: 数据猎人",
            2: "关卡2: 架构师",
            3: "关卡3: 训练大师",
            4: "关卡4: 诊断专家",
            5: "关卡5: 临床实战",
        }
        stage = st.session_state.stage
        st.subheader(stage_names.get(stage, ""))

        progress = (stage - 1) / 4
        st.progress(min(progress, 1.0), text=f"进度: {stage}/5 关")

        st.metric("总分", st.session_state.score)
        st.metric("模型准确率", f"{st.session_state.model_accuracy:.1%}"
                  if st.session_state.model_accuracy else "未训练")

        st.markdown("---")
        st.markdown("### 🏆 成就")
        if st.session_state.achievements:
            for ach in st.session_state.achievements:
                st.success(f"{ach}")
        else:
            st.caption("尚未解锁任何成就")

        st.markdown("---")
        if st.button("🔄 重新开始", width="stretch"):
            reset_game()


# ---------- 关卡1: 数据猎人 ----------
def render_stage1():
    st.title("🔍 关卡1: 数据猎人")
    st.caption("目标：理解CT影像数据，收集并准备训练样本")
    st.markdown("---")

    # 追踪标签页浏览状态
    if "s1_tab1_visited" not in st.session_state:
        st.session_state.s1_tab1_visited = False
    if "s1_tab2_visited" not in st.session_state:
        st.session_state.s1_tab2_visited = False
    if "s1_tab3_complete" not in st.session_state:
        st.session_state.s1_tab3_complete = False

    tab1, tab2, tab3 = st.tabs(["📖 认识CT数据", "🎚️ 肺窗调节实验", "🏷️ 标记训练数据"])

    with tab1:
        st.subheader("CT是什么？")
        st.write("""
        **CT (Computed Tomography)** 是一系列X射线从不同角度扫描人体后重建的断面图像。
        你可以把它理解为"将人体切成很多薄片，然后一张张拍照"。
        """)

        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("""
            ### CT值的概念
            CT值以 **HU (Hounsfield Unit)** 为单位，表示组织对X射线的吸收程度：

            | 组织 | CT值 (HU) |
            |------|-----------|
            | 空气 | ≈ -1000 |
            | 肺组织 | ≈ -700 ~ -800 |
            | 脂肪 | ≈ -100 ~ -50 |
            | 水 | 0 |
            | 软组织 | +20 ~ +50 |
            | 骨骼 | +400 ~ +1000 |
            """)
        with col2:
            st.info(
                "💡 **关键洞察**\n\n"
                "肺部CT的特殊之处在于：肺组织充满空气，HU值很低(≈-800)。"
                "而结节是软组织，HU值较高(≈+20~+50)。\n\n"
                "这种密度差异正是AI识别结节的基础！"
            )

        st.markdown("### 3D体积切片浏览")
        st.write("CT数据本质上是3D体积(一堆切片堆叠)。拖动滑块浏览不同层面的CT图像：")

        demo_case = Path("cases/case_001/slices")
        if demo_case.exists():
            slices = sorted([f for f in demo_case.iterdir()
                           if f.suffix.lower() in (".png", ".npy") and "image" not in f.name])
            if len(slices) > 0:
                slice_idx = st.slider(
                    "切片编号 (Z轴)", 0, len(slices) - 1,
                    len(slices) // 2, key="s1_slice_browser",
                    on_change=lambda: st.session_state.update({"s1_tab1_visited": True}))
                slice_img = load_image_smart(str(slices[slice_idx]))
                if slice_img:
                    st.image(slice_img, caption=f"切片 #{slice_idx + 1} / {len(slices)}",
                             width=350)
                    st.caption(f"当前层面位置：第 {slice_idx + 1}/{len(slices)} 层 "
                              f"（约 {(slice_idx + 1) / len(slices) * 100:.0f}% 位置）")
                else:
                    st.info("切片可通过本地运行 build_case_library.py 生成。")
                    st.session_state.s1_tab1_visited = True  # 已阅读但无数据
            else:
                st.info("切片数据未上传（PNG在线不可用，仅本地可见）。游戏其他功能不受影响。")
                st.session_state.s1_tab1_visited = True
        else:
            st.info("切片浏览器仅在本地完整版可用。在线版继续体验其他功能。")
            st.session_state.s1_tab1_visited = True

    with tab2:
        st.subheader("肺窗调节实验")
        st.write("""
        原始CT的HU值范围很宽(-1000~+1000)，但人眼只能分辨有限的灰阶。
        **窗技术** 通过设置窗宽(WW)和窗位(WC)来选择显示特定HU范围，突出观察目标。
        """)

        col1, col2 = st.columns([1, 1])
        with col1:
            wc = st.slider("窗位 WC (Window Center)", -1000, 500, -600, 10,
                           help="窗位决定了显示的HU中心值。肺窗通常设在-600HU。",
                           on_change=lambda: st.session_state.update({"s1_tab2_visited": True}))
            ww = st.slider("窗宽 WW (Window Width)", 100, 3000, 1500, 10,
                           help="窗宽决定了显示的HU范围。肺窗通常为1500HU宽度。",
                           on_change=lambda: st.session_state.update({"s1_tab2_visited": True}))

        with col2:
            st.markdown("### 快速预设")
            presets = {"肺窗 (-600, 1500)": (-600, 1500),
                       "纵隔窗 (50, 350)": (50, 350),
                       "骨窗 (400, 1800)": (400, 1800)}
            for name, (p_wc, p_ww) in presets.items():
                if st.button(name, key=f"s1_preset_{name}"):
                    st.session_state["s1_wc"] = p_wc
                    st.session_state["s1_ww"] = p_ww
                    st.session_state.s1_tab2_visited = True
                    st.rerun()

        demo_image = None
        demo_case = Path("cases/case_001")
        img_path = demo_case / "image.png"
        loaded = load_image_smart(str(img_path))
        if loaded:
            demo_image = loaded.convert("L")
        else:
            # fallback: create a demo image
            demo_image = Image.new("L", (512, 512), 128)

        img_np = np.array(demo_image, dtype=np.float32)
        if img_np.max() <= 255:
            img_np = (img_np / 255.0) * 2000 - 1000  # 模拟HU范围

        low = wc - ww / 2
        high = wc + ww / 2
        if abs(high - low) < 1e-6:
            windowed = img_np.astype(np.uint8)
        else:
            windowed = np.clip((img_np - low) / (high - low) * 255, 0, 255).astype(np.uint8)

        col1, col2 = st.columns(2)
        with col1:
            st.image(demo_image, caption="原始HU值图像（未经窗处理）", width=350,
                     clamp=True)
        with col2:
            st.image(windowed, caption=f"窗处理后 (WC={wc}, WW={ww})",
                     width=350, clamp=True)

        if abs(wc - (-600)) <= 50 and abs(ww - 1500) <= 100:
            st.success("✅ 你已经接近临床标准肺窗参数！WC≈-600, WW≈1500 是观察肺实质的最佳设置。")
            if not st.session_state.window_found:
                st.session_state.window_found = True
                st.session_state.score += 15
                st.balloons()
                st.success("成就解锁: 肺窗大师 (+15分)")
        else:
            st.info("💡 提示：尝试 WC=-600, WW=1500（临床标准肺窗），观察肺纹理是否变清晰。")

    with tab3:
        st.subheader("训练数据探秘")
        st.write("""
        AI不是天生就会判断结节的——它需要从大量**已标注**的CT图像中学习。
        下面来看看专家是怎么标注训练数据的，体验一下这张CT的"标签"从何而来。
        """)

        # 准备预标注样本（全部来自病例库）
        if "s1_explore_samples" not in st.session_state:
            case_dirs = [p for p in Path("cases").iterdir()
                        if p.is_dir() and p.name.startswith("case_")]
            explore = []
            for c in case_dirs:
                meta = load_case_meta(c)
                img_path = c / meta.get("image", "image.png")
                # load_image_smart 内部有 .png→.npy 回退逻辑
                test_load = load_image_smart(str(img_path))
                if test_load is not None:
                    explore.append({
                        "path": str(img_path),
                        "label": meta.get("class", "Lesion"),
                        "title": meta.get("title", c.name),
                        "desc": meta.get("description", ""),
                    })
            random.shuffle(explore)
            st.session_state["s1_explore_samples"] = explore[:6]
            st.session_state["s1_explore_viewed"] = 0

        explore_samples = st.session_state["s1_explore_samples"]

        if len(explore_samples) == 0:
            st.warning("病例库为空，请先运行 build_case_library.py 构建病例数据。")
            st.session_state.s1_tab3_complete = True
        else:
            st.caption(f"共 {len(explore_samples)} 张已标注样本，请逐张浏览观察。")

            for i, sample in enumerate(explore_samples):
                st.markdown(f"### 样本 {i+1}: {sample['title']}")
                img = load_image_smart(sample["path"])
                if img is None:
                    continue

                col1, col2 = st.columns(2)
                with col1:
                    st.image(img, caption="原始CT图像", width=350)
                with col2:
                    # 展示专家标注信息卡片
                    label_class = sample["label"]
                    label_color = "red" if label_class == "Lesion" else "green"
                    st.markdown(f"**专家标注:** :{label_color}[{label_class}]")
                    if sample["desc"]:
                        st.caption(f"描述: {sample['desc']}")
                    st.markdown("---")
                    st.write("**这张图的特征:**")
                    if label_class == "Lesion":
                        st.write("🔴 肺实质内可见**异常密度影**（结节/病灶）")
                        st.write("📌 专家已在结节区域做标记")
                    else:
                        st.write("🟢 肺实质清晰，**无明显异常密度影**")
                        st.write("📌 专家确认此切片可用于正常对照")
                    st.write("---")
                    st.caption("💡 AI就是通过成千上万张这样的标注图像，"
                              "学会区分'有结节'和'无结节'的。")

            st.markdown("---")
            st.subheader("📊 你的观察报告")
            st.write("你已经浏览了全部样本。回顾一下你看到了什么：")

            lesion_count = len([s for s in explore_samples if s["label"] == "Lesion"])
            normal_count = len(explore_samples) - lesion_count

            col1, col2 = st.columns(2)
            with col1:
                st.metric("🔴 有结节样本", f"{lesion_count} 张")
            with col2:
                st.metric("🟢 正常样本", f"{normal_count} 张")

            st.info("""
            ### 💡 数据标注的重要性

            训练AI的第一步就是准备**标注数据**。AI学到的所有知识都来自于这些专家标注：

            - **标注质量差** → AI学到错误判断（把血管当结节）
            - **标注数据少** → AI没见过足够多的病例，泛化能力弱
            - **标注偏差** → 如果只标注大结节，AI会漏掉小结节

            在现实中，需要**多位放射科医生**共同标注同一张CT，才能保证标签的准确性。
            这正是医学AI开发中最昂贵、最耗时的环节！
            """)

            st.session_state.s1_tab3_complete = True

    # --- 关卡1进度追踪 & 进入下一关 ---
    st.markdown("---")
    st.subheader("📋 关卡进度")

    visited_count = (
        int(st.session_state.s1_tab1_visited) +
        int(st.session_state.s1_tab2_visited) +
        int(st.session_state.s1_tab3_complete)
    )
    all_visited = visited_count == 3

    col1, col2, col3 = st.columns(3)
    with col1:
        icon = "✅" if st.session_state.s1_tab1_visited else "⬜"
        st.markdown(f"{icon} 认识CT数据")
    with col2:
        icon = "✅" if st.session_state.s1_tab2_visited else "⬜"
        st.markdown(f"{icon} 肺窗调节实验")
    with col3:
        icon = "✅" if st.session_state.s1_tab3_complete else "⬜"
        st.markdown(f"{icon} 标记训练数据")

    st.progress(visited_count / 3)

    if not all_visited:
        st.info("👆 请依次浏览上方三个标签页，每完成一个会打勾。全部完成后解锁下一关。")
    else:
        st.success("🎉 所有模块已完成！可以进入下一关。")
        if st.button("进入下一关 →", type="primary", use_container_width=True):
            st.session_state.stage = 2
            if "初识CT" not in st.session_state.achievements:
                st.session_state.achievements.append("初识CT")
            st.session_state.stage1_complete = True
            st.rerun()


# ---------- 关卡2: 架构师 ----------
def render_stage2():
    st.title("🏗️ 关卡2: 架构师")
    st.caption("目标：选择神经网络架构并设置训练超参数")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("选择模型架构")
        arch_choice = st.radio(
            "你希望使用哪种架构？",
            ["2D ResNet18 (推荐)", "3D CNN (进阶)"],
            key="s2_arch",
            index=0 if st.session_state.chosen_arch is None
            else (0 if st.session_state.chosen_arch == "resnet18" else 1)
        )
        st.session_state.chosen_arch = "resnet18" if "ResNet18" in arch_choice else "3dcnn"

        if "ResNet18" in arch_choice:
            st.success("⭐ 新手友好")
            st.markdown("""
            **ResNet18** 是经典的2D图像分类网络：
            - 参数量: ~11M
            - 输入: 224×224 RGB图像
            - 特点: 残差连接防止梯度消失
            - 训练速度: 快（~2分钟/轮 CPU）
            - 适用: 大多数2D医学图像分类任务
            """)
        else:
            st.warning("🔥 进阶挑战")
            st.markdown("""
            **3D CNN** 直接处理3D CT体积块：
            - 参数量: ~5M (简化版)
            - 输入: 64×64×64 体积块
            - 特点: 利用空间上下文信息
            - 训练速度: 较慢（~8分钟/轮 CPU）
            - 来源: 参考 mr-mukherjee03/CT-Scan-Nodule-Detection
            """)

    with col2:
        st.subheader("超参数配置")
        lr = st.select_slider(
            "学习率 Learning Rate",
            options=[1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
            value=1e-4,
            format_func=lambda x: f"{x:.0e}",
            key="s2_lr"
        )
        st.caption("太高→震荡不收敛  太低→收敛太慢  1e-4是常用起点")

        batch_size = st.selectbox(
            "批次大小 Batch Size",
            [4, 8, 16, 32, 64],
            index=2,
            key="s2_batch"
        )
        st.caption("较大→训练稳定但显存高  较小→更新频繁但震荡")

        epochs = st.slider("训练轮数 Epochs", 10, 100, 50, 5, key="s2_epochs")
        st.caption("太少→欠拟合  太多→可能过拟合")

        st.session_state.hyperparams = {
            "learning_rate": lr,
            "batch_size": batch_size,
            "num_epochs": epochs,
        }

    st.markdown("---")
    st.subheader("📊 模型参数统计")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.session_state.chosen_arch == "resnet18":
            params = 11177538
        else:
            params = 4820000
        st.metric("总参数量", f"{params/1e6:.1f}M")
    with col2:
        est_time = epochs * (2 if st.session_state.chosen_arch == "resnet18" else 8)
        st.metric("预计训练时间", f"~{est_time}分钟 (CPU)")
    with col3:
        mem_est = 0.5 if st.session_state.chosen_arch == "resnet18" else 2.0
        st.metric("显存需求", f"~{mem_est:.1f}GB")

    # 架构可视化
    st.markdown("### 架构预览")
    if st.session_state.chosen_arch == "resnet18":
        st.code("""
ResNet18 结构:
├─ Conv2d(3→64, k=7, s=2) → BN → ReLU → MaxPool
├─ Layer1: [Conv(64→64, k=3)] ×2  (残差块)
├─ Layer2: [Conv(64→128, k=3)] ×2 (下采样)
├─ Layer3: [Conv(128→256, k=3)] ×2 (下采样)
├─ Layer4: [Conv(256→512, k=3)] ×2 (下采样)
├─ AdaptiveAvgPool → FC(512→2)
└─ Softmax → [Normal, Lesion]
        """)
    else:
        st.code("""
3D CNN 结构 (简化版, 参考 mr-mukherjee03):
├─ Conv3d(1→32, k=3) → ReLU → MaxPool3d(2)
├─ Conv3d(32→64, k=3) → ReLU → MaxPool3d(2)
├─ Conv3d(64→128, k=3) → ReLU → MaxPool3d(2)
├─ AdaptiveAvgPool3d → Flatten
├─ FC(128→64) → ReLU → Dropout(0.5)
└─ FC(64→2) → Softmax → [No Nodule, Nodule]
        """)

    if st.button("确认并进入训练 →", type="primary", width="stretch"):
        st.session_state.stage = 3
        st.session_state.score += 20
        if "架构师" not in st.session_state.achievements:
            st.session_state.achievements.append("架构师")
        if st.session_state.chosen_arch == "3dcnn":
            if "冒险家" not in st.session_state.achievements:
                st.session_state.achievements.append("冒险家")
                st.session_state.score += 10
        st.session_state.stage2_complete = True
        st.rerun()

    render_stage_nav(2)


# ---------- 关卡3: 训练大师 (核心) ----------
def render_stage3():
    st.title("🎯 关卡3: 训练大师 ⭐")
    st.caption("目标：观察AI学习过程，在关键时刻做出正确决策")
    st.markdown("---")

    history = load_training_history()

    if "s3_watched_epoch" not in st.session_state:
        st.session_state.s3_watched_epoch = 0
    if "s3_decision_made" not in st.session_state:
        st.session_state.s3_decision_made = {}
    if "s3_phase" not in st.session_state:
        st.session_state.s3_phase = "intro"

    if history is None and st.session_state.s3_phase == "intro":
        st.warning("""
        ⚠️ 未找到训练历史文件 (`loss_fig/training_history.json`)。

        请先在本地运行 `python model.py` 生成训练数据。

        **当前将使用模拟数据进行演示。**
        """)
        history = _generate_demo_history()

    total_epochs = len(history["epochs"]) if history else 50

    # --- 阶段：训练启动 ---
    if st.session_state.s3_phase == "intro":
        st.subheader("训练启动 (Epoch 1-4)")
        st.write("AI开始第一轮训练。观察损失和准确率的变化趋势......")

        _plot_training_curves(history, highlight_epoch=4)
        _show_gradcam_snapshot(1, "初始状态")
        _show_gradcam_snapshot(4, "4轮训练后")

        st.write("损失下降到 **{:.3f}**，准确率上升到 **{:.1f}%**".format(
            history["val_loss"][min(3, len(history["val_loss"])-1)] if history else 0.7,
            history["val_acc"][min(3, len(history["val_acc"])-1)] * 100 if history else 45
        ))

        if st.button("继续观察 →", type="primary", width="stretch"):
            st.session_state.s3_phase = "decision1"
            st.rerun()

    # --- 决策点1 ---
    elif st.session_state.s3_phase == "decision1":
        st.subheader("⚠️ 决策点 1: 损失下降变慢了")
        st.write("训练进行到第5轮，损失下降速度明显放缓。你注意到验证集准确率增长也开始平缓。")

        _plot_training_curves(history, highlight_epoch=5,
                              annotations=[(5, "损失下降放缓")])
        _show_gradcam_snapshot(5, "第5轮热力图")

        if 1 not in st.session_state.s3_decision_made:
            choice = st.radio(
                "作为AI研究员，你该如何应对？",
                ["A. 保持当前学习率继续训练（耐心等待收敛）",
                 "B. 大幅降低学习率（0.001→0.00001）",
                 "C. 提高学习率（加速训练）"],
                key="s3_d1",
                index=None
            )
            if choice and st.button("确认决策", key="s3_d1_confirm"):
                st.session_state.s3_decision_made[1] = choice
                if choice.startswith("A"):
                    st.session_state.score += 20
                    st.session_state.training_decisions.append("correct1")
                    st.success("✅ 正确！损失下降放缓是正常的收敛行为，保持学习率让模型稳步优化。")
                else:
                    st.error("❌ 不太合适。训练初期损失下降放缓是正常的，贸然改变学习率可能适得其反。")
                st.rerun()
        else:
            st.info(f"你的选择: {st.session_state.s3_decision_made[1]}")
            if st.button("继续 →"):
                st.session_state.watched_epoch = 15
                st.session_state.s3_phase = "decision2"
                st.rerun()

    # --- 决策点2 ---
    elif st.session_state.s3_phase == "decision2":
        st.subheader("⚠️ 决策点 2: 验证准确率停滞")
        st.write("训练到第15轮，验证准确率连续3轮没有提升，似乎遇到了瓶颈。")

        _plot_training_curves(history, highlight_epoch=15,
                              annotations=[(15, "验证集停滞")])
        _show_gradcam_snapshot(15, "第15轮热力图")

        st.markdown("### 过拟合 vs 欠拟合")
        col1, col2 = st.columns(2)
        with col1:
            st.warning("**欠拟合**: 训练和验证都很差 → 模型太简单/训练不足")
        with col2:
            st.error("**过拟合**: 训练很好但验证差 → 模型记住了训练数据但不会泛化")

        if "d2" not in st.session_state.s3_decision_made:
            choice = st.radio(
                "你的策略是什么？",
                ["A. 继续训练，模型还在学习（耐心等待）",
                 "B. 降低学习率（如 1e-4 → 3e-5）并继续训练",
                 "C. 立即停止训练（早停）"],
                key="s3_d2",
                index=None
            )
            if choice and st.button("确认决策", key="s3_d2_confirm"):
                st.session_state.s3_decision_made["d2"] = choice
                if choice.startswith("B"):
                    st.session_state.score += 20
                    st.session_state.training_decisions.append("correct2")
                    st.success("✅ 正确！降低学习率让模型进行更精细的调整，可能突破平台期。")
                elif choice.startswith("C"):
                    st.error("❌ 现在停止太早了！模型可能还在突破平台期，应该给更多机会。")
                else:
                    st.warning("⚠️ 继续同学习率也可以，但不是最优策略。降低学习率有助于精细调整。")
                st.rerun()
        else:
            st.info(f"你的选择: {st.session_state.s3_decision_made['d2']}")
            if st.button("继续 →"):
                st.session_state.s3_phase = "decision3"
                st.rerun()

    # --- 决策点3 ---
    elif st.session_state.s3_phase == "decision3":
        st.subheader("⚠️ 决策点 3: 过拟合警报")
        st.write("第25轮：训练准确率90%，验证准确率仅78% —— 差距拉大，出现**过拟合**！")

        _plot_training_curves(history, highlight_epoch=25,
                              annotations=[(25, "过拟合出现")])
        _show_gradcam_snapshot(25, "第25轮热力图")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("训练准确率", "90.2%")
        with col2:
            st.metric("验证准确率", "78.1%", delta="-12.1%")

        if "d3" not in st.session_state.s3_decision_made:
            choice = st.radio(
                "过拟合了！怎么办？",
                ["A. 应该增加正则化（Dropout/数据增强），减少过拟合",
                 "B. 继续训练更多轮，验证准确率可能会回升",
                 "C. 增加更多网络层数，提高模型容量"],
                key="s3_d3",
                index=None
            )
            if choice and st.button("确认决策", key="s3_d3_confirm"):
                st.session_state.s3_decision_made["d3"] = choice
                if choice.startswith("A"):
                    st.session_state.score += 20
                    st.session_state.training_decisions.append("correct3")
                    st.success("✅ 正确！正则化是应对过拟合的首选策略。")
                else:
                    st.error("❌ 错误！继续训练或增加层数会进一步加剧过拟合。")
                st.rerun()
        else:
            st.info(f"你的选择: {st.session_state.s3_decision_made['d3']}")

    # --- 训练完成 ---
    if st.session_state.s3_phase == "decision3" and "d3" in st.session_state.s3_decision_made:
        if st.button("完成训练 →", type="primary", width="stretch"):
            st.session_state.s3_phase = "final"
            st.session_state.model_accuracy = 0.823
            st.rerun()

    if st.session_state.s3_phase == "final":
        st.balloons()
        st.subheader("🎉 训练完成！")
        st.write("AI已经学会了基本的肺结节判断能力。让我们看看最终成果：")

        _plot_training_curves(history, highlight_epoch=total_epochs,
                              annotations=[
                                  (5, "决策1"), (15, "决策2"), (25, "决策3")
                              ])

        st.markdown("### 📈 Grad-CAM 演变回顾")
        st.write("观察AI注意力在整个训练过程中如何从随机到精准定位：")
        _show_gradcam_evolution()

        correct_decisions = len([d for d in st.session_state.training_decisions
                                 if d.startswith("correct")])
        st.session_state.score += correct_decisions * 5
        final_acc = 0.82 + correct_decisions * 0.03
        st.session_state.model_accuracy = min(final_acc, 0.92)

        st.metric("最终验证准确率", f"{st.session_state.model_accuracy:.1%}")

        if correct_decisions == 3:
            st.success("🏆 成就解锁: 稳扎稳打（全部决策正确）")
            if "稳扎稳打" not in st.session_state.achievements:
                st.session_state.achievements.append("稳扎稳打")
                st.session_state.score += 15

        if "洞察本质" not in st.session_state.achievements:
            st.session_state.achievements.append("洞察本质")

        if st.button("进入诊断评估 →", type="primary", width="stretch"):
            st.session_state.stage = 4
            st.session_state.stage3_complete = True
            st.rerun()

    render_stage_nav(3)


# ---------- 关卡4: 诊断专家 ----------
def render_stage4():
    st.title("🩺 关卡4: 诊断专家")
    st.caption("目标：用你训练的AI辅助诊断真实CT病例")
    st.markdown("---")

    case_list = load_case_list(DEFAULT_CASE_ROOT)
    if not case_list:
        st.warning("病例库为空。请先运行 build_case_library.py")
        render_stage_nav(4)
        return

    model_path = os.path.exists(DEFAULT_MODEL_PATH)
    model, gradcam = load_model_resource(DEFAULT_MODEL_PATH) if model_path else (None, None)

    if "s4_current_case_idx" not in st.session_state:
        st.session_state.s4_current_case_idx = 0
    if "s4_case_done" not in st.session_state:
        st.session_state.s4_case_done = set()
    if "s4_diagnosis_made" not in st.session_state:
        st.session_state.s4_diagnosis_made = False
    if "s4_total_cases" not in st.session_state:
        st.session_state.s4_total_cases = 4

    done_count = len(st.session_state.s4_case_done)
    total_cases = st.session_state.s4_total_cases
    st.progress(done_count / total_cases, text=f"已完成: {done_count}/{total_cases} 个病例")

    if done_count >= total_cases:
        st.success("🎉 所有病例诊断完成！")
        avg_score = (sum(st.session_state.diagnosis_scores) /
                     len(st.session_state.diagnosis_scores)) if st.session_state.diagnosis_scores else 0
        st.metric("诊断平均分", f"{avg_score:.0f}/100")

        if avg_score >= 80:
            if "诊断专家" not in st.session_state.achievements:
                st.session_state.achievements.append("诊断专家")
                st.session_state.score += 15

        if st.button("进入临床实战 →", type="primary", width="stretch"):
            st.session_state.stage = 5
            st.session_state.stage4_complete = True
            st.rerun()
        render_stage_nav(4)
        return

    case_idx = st.session_state.s4_current_case_idx
    if case_idx >= len(case_list):
        case_idx = 0

    case_dir = case_list[case_idx]
    meta = load_case_meta(case_dir)
    img_path_png = case_dir / meta.get("image", "image.png")
    img_path_npy = case_dir / (meta.get("image", "image.png").rsplit(".", 1)[0] + ".npy"
                               if "." in meta.get("image", "image.png") else meta.get("image", "image.png") + ".npy")
    image_path = img_path_png
    if not img_path_png.exists() and not img_path_npy.exists():
        st.error(f"图片未找到: {image_path}")
        render_stage_nav(4)
        return

    image = load_image_smart(str(image_path))
    if image is None:
        st.error(f"无法加载图片: {image_path}")
        render_stage_nav(4)
        return

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image, caption="CT图像", width=400)
    with col2:
        st.markdown("### 病例信息")
        st.write(f"**难度:** {meta.get('difficulty', 'unknown')}")
        st.write(f"**描述:** {meta.get('description', '无')}")

        if meta.get("medical_knowledge"):
            with st.expander("📚 医学知识点（诊断后查看）"):
                mk = meta["medical_knowledge"]
                if mk.get("imaging_features"):
                    st.write("**影像学特征:**")
                    for f in mk["imaging_features"]:
                        st.write(f"- {f}")
                if mk.get("differential_diagnosis"):
                    st.write("**鉴别诊断:**")
                    for d in mk["differential_diagnosis"]:
                        st.write(f"- {d}")

    st.markdown("---")
    st.subheader("你的独立诊断")

    col1, col2 = st.columns(2)
    with col1:
        user_has_nodule = st.radio("是否有结节？", ["有结节", "无结节"],
                                    key=f"s4_nodule_{case_idx}",
                                    index=None)
    with col2:
        if user_has_nodule == "有结节":
            st.radio("良恶性判断？", ["倾向良性", "倾向恶性"],
                     key=f"s4_mal_{case_idx}", index=None)

    if user_has_nodule and not st.session_state.s4_diagnosis_made:
        st.markdown("### 圈选结节区域")
        w, h = image.size
        col1, col2 = st.columns(2)
        with col1:
            x = st.slider("左上角 X", 0, max(0, w - 1), int(w * 0.25),
                          key=f"s4_x_{case_idx}")
            rect_w = st.slider("框宽度", 1, max(1, w), int(w * 0.25),
                               key=f"s4_rw_{case_idx}")
        with col2:
            y = st.slider("左上角 Y", 0, max(0, h - 1), int(h * 0.25),
                          key=f"s4_y_{case_idx}")
            rect_h = st.slider("框高度", 1, max(1, h), int(h * 0.25),
                               key=f"s4_rh_{case_idx}")

        user_mask = _rectangle_mask((w, h), x, y, rect_w, rect_h)
        outlined = _draw_mask_outline(image, user_mask, (0, 255, 0))
        st.image(outlined, caption="你的圈选（绿色轮廓）", width=400)

    if (user_has_nodule is not None and
            not st.session_state.s4_diagnosis_made and
            st.button("启动AI分析 🚀", type="primary", width="stretch")):
        st.session_state.s4_diagnosis_made = True
        st.rerun()

    if st.session_state.s4_diagnosis_made:
        st.markdown("---")
        st.subheader("🤖 AI分析结果")

        with st.spinner("AI正在分析..."):
            if model is not None and gradcam is not None:
                result = _predict_with_model(model, gradcam, image)
            else:
                result = _predict_with_rule_demo(image)

            cam = result["cam"]
            overlay_img = _overlay_heatmap(image, cam)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("AI预测类别", result["pred_class"])
        with col2:
            st.metric("病变概率", f"{result.get('lesion_prob', 0):.3f}")
        with col3:
            st.metric("正常概率", f"{result.get('normal_prob', 0):.3f}")

        col1, col2 = st.columns(2)
        with col1:
            st.image(overlay_img, caption="AI 热力图叠加", width=400)
        with col2:
            ai_mask = (cam > 0.55).astype(np.uint8)
            ai_outline = _draw_mask_outline(image, ai_mask, (255, 0, 0))
            st.image(ai_outline, caption="AI 关注区域（红色轮廓）", width=400)

        st.markdown("### 诊断评分")
        true_class = meta.get("class", "Unknown")
        case_score = 50
        user_pred = "Lesion" if user_has_nodule == "有结节" else "Normal"
        if user_pred == true_class:
            case_score += 30
        if model is not None:
            if result["pred_class"] == true_class:
                case_score += 20
        st.session_state.score += case_score
        st.session_state.diagnosis_scores.append(case_score)
        st.metric("本病例得分", f"{case_score}/100")

        if true_class != "Unknown" and case_score >= 80:
            st.success("出色的诊断判断！")

        if st.button("下一个病例 →", key=f"s4_next_{case_idx}"):
            st.session_state.s4_current_case_idx = (case_idx + 1) % len(case_list)
            st.session_state.s4_case_done.add(case_idx)
            st.session_state.s4_diagnosis_made = False
            st.rerun()

    render_stage_nav(4)


# ---------- 关卡5: 临床实战 ----------
def render_stage5():
    st.title("🏥 关卡5: 临床实战")
    st.caption("目标：模拟放射科医生，完成批量CT筛查")
    st.markdown("---")

    model_path = os.path.exists(DEFAULT_MODEL_PATH)
    model, gradcam = load_model_resource(DEFAULT_MODEL_PATH) if model_path else (None, None)

    # 准备筛查队列
    if "s5_queue" not in st.session_state:
        case_list = load_case_list(DEFAULT_CASE_ROOT)
        queue = []
        for case_dir in case_list[:8]:
            meta = load_case_meta(case_dir)
            img_path = case_dir / meta.get("image", "image.png")
            if img_path.exists():
                queue.append({
                    "path": str(img_path),
                    "meta": meta,
                    "ai_label": None,
                    "ai_conf": None,
                })
        random.shuffle(queue)

        # 预先计算AI结果
        for item in queue:
            img = load_image_smart(item["path"])
            if img is None:
                continue
            result = _predict_with_model(model, gradcam, img)
            item["ai_label"] = result["pred_class"]
            item["ai_conf"] = max(result.get("lesion_prob", 0),
                                  result.get("normal_prob", 0))

        # AI陷阱：找2个模型confidence低的case，故意标记错误
        low_conf = sorted(queue, key=lambda x: x["ai_conf"])[:2]
        for item in low_conf:
            item["ai_label"] = "Lesion" if item["ai_label"] == "Normal" else "Normal"
            item["ai_trap"] = True

        st.session_state.s5_queue = queue
        st.session_state.s5_current = 0
        st.session_state.s5_answers = {}
        st.session_state.s5_start_time = time.time()
        st.session_state.s5_finished = False

    if st.session_state.s5_finished:
        _render_s5_results()
        render_stage_nav(5)
        return

    queue = st.session_state.s5_queue
    current = st.session_state.s5_current

    st.markdown(f"### 待筛查: {len(queue) - current} / {len(queue)} 张剩余")
    st.progress(current / len(queue))

    if current >= len(queue):
        st.session_state.s5_finished = True
        st.rerun()

    item = queue[current]

    col1, col2 = st.columns([3, 2])
    with col1:
        st.image(item["path"], caption=f"CT #{current + 1}", width=350)

    with col2:
        st.markdown("### AI辅助")
        ai_on = st.toggle("显示AI分析", value=True, key=f"s5_ai_{current}")

        if ai_on:
            if model is not None and gradcam is not None:
                img = load_image_smart(item["path"])
                if img:
                    result = _predict_with_model(model, gradcam, img)
                    overlay_img = _overlay_heatmap(img, result["cam"])
                st.image(overlay_img, caption="AI热力图", width=300)
                st.metric("AI判断", item["ai_label"])
            else:
                st.info("AI判断: " + item["ai_label"])
                st.caption(f"(置信度: {item['ai_conf']:.2%})")
        else:
            st.caption("AI辅助已关闭 — 独立判断模式")

        st.markdown("### 你的判断")
        user_label = st.radio("作出诊断", ["Normal (正常)", "Lesion (病变)"],
                              key=f"s5_label_{current}", index=None)

        if user_label and st.button("确认并下一张 →", key=f"s5_next_{current}"):
            predicted = "Lesion" if "Lesion" in user_label else "Normal"
            st.session_state.s5_answers[current] = {
                "user_label": predicted,
                "ai_label": item["ai_label"],
                "true_label": item["meta"].get("class", "Unknown"),
                "ai_trap": item.get("ai_trap", False),
            }
            st.session_state.s5_current = current + 1
            st.rerun()

    elapsed = time.time() - st.session_state.s5_start_time
    st.caption(f"已用时: {int(elapsed // 60)}分{int(elapsed % 60)}秒")

    render_stage_nav(5)


def _render_s5_results():
    st.balloons()
    st.subheader("📊 最终结算")

    answers = st.session_state.s5_answers
    total = len(answers)
    if total == 0:
        st.warning("未完成任何筛查")
        return

    correct = len([a for a in answers.values()
                   if a["user_label"] == a["true_label"]])
    ai_correct = len([a for a in answers.values()
                      if a["ai_label"] == a["true_label"]])
    traps_found = len([a for a in answers.values()
                       if a.get("ai_trap") and a["user_label"] != a["ai_label"]
                       and a["user_label"] == a["true_label"]])
    total_traps = len([a for a in answers.values() if a.get("ai_trap")])

    elapsed = time.time() - st.session_state.s5_start_time
    avg_time = elapsed / total if total > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("筛查准确率", f"{correct/total:.0%}")
    col2.metric("平均耗时", f"{avg_time:.0f}秒/张")
    col3.metric("AI依赖度", f"{sum(1 for a in answers.values() if a['user_label'] == a['ai_label'])/total:.0%}")
    col4.metric("发现AI陷阱", f"{traps_found}/{total_traps}")

    # 雷达图（用文本模拟）
    st.markdown("### 综合能力评估")
    dimensions = {
        "诊断准确率": min(100, int(correct / total * 100)),
        "阅片速度": max(30, min(100, int(100 - avg_time / 30 * 100))),
        "独立判断力": min(100, int((1 - sum(1 for a in answers.values()
                         if a['user_label'] == a['ai_label']) / total) * 100)),
        "AI协作能力": min(100, int(80 + traps_found * 10)),
    }

    cols = st.columns(len(dimensions))
    for i, (name, value) in enumerate(dimensions.items()):
        with cols[i]:
            st.metric(name, f"{value}/100")
            st.progress(value / 100)

    total_score = sum(dimensions.values()) / len(dimensions)
    if total_score >= 90:
        grade = "S"
        grade_text = "完美！你已是一名出色的AI放射科医生"
    elif total_score >= 75:
        grade = "A"
        grade_text = "优秀！你的诊断能力很强"
    elif total_score >= 60:
        grade = "B"
        grade_text = "良好！继续积累经验会更出色"
    else:
        grade = "C"
        grade_text = "需要更多练习，但这是正常的起点"

    st.session_state.final_grade = grade
    st.markdown(f"## 总评级: **{grade}**")
    st.success(grade_text)

    st.session_state.score += int(total_score)

    if traps_found >= total_traps and total_traps > 0:
        if "超越AI" not in st.session_state.achievements:
            st.session_state.achievements.append("超越AI")
            st.session_state.score += 15
            st.success("🏆 成就解锁: 超越AI — 你发现了AI的错误！")

    if grade in ("S", "A"):
        if "完美收官" not in st.session_state.achievements:
            st.session_state.achievements.append("完美收官")
            st.session_state.score += 20

    st.markdown("---")
    st.markdown("### 🏆 本次训练总结")
    st.write(f"- 总得分: **{st.session_state.score}**")
    st.write(f"- 模型最终准确率: **{st.session_state.model_accuracy:.1%}**")
    st.write(f"- 解锁成就: {len(st.session_state.achievements)} 个")
    st.write(f"- 最终评级: **{grade}**")

    st.info("""
    ### 🎓 你学到了什么？

    通过这五关，你体验了一个完整的AI开发流程：
    1. **数据准备** → 理解医学图像的特性
    2. **模型搭建** → 选择架构和超参数
    3. **训练调参** → 观察学习过程，应对过拟合
    4. **诊断评估** → AI+人类协作诊断
    5. **临床实战** → 批判性使用AI辅助

    AI不是万能的，关键是与人类专业判断互补配合。
    """)

    if st.button("🔄 重新挑战", width="stretch"):
        reset_game()


# ---------- 辅助工具函数 ----------

def render_stage_nav(stage: int):
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if stage > 1:
            if st.button("← 上一关", key=f"prev_{stage}"):
                st.session_state.stage = stage - 1
                st.rerun()
    with col3:
        if stage < 5:
            if st.button("下一关 →", key=f"skip_{stage}"):
                st.session_state.stage = stage + 1
                st.rerun()


def _plot_training_curves(history, highlight_epoch=None, annotations=None):
    if history is None:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
    epochs = history["epochs"]

    ax1.plot(epochs, history["train_loss"], "b-", label="Train Loss", alpha=0.7)
    ax1.plot(epochs, history["val_loss"], "r-", label="Val Loss", alpha=0.7)
    if highlight_epoch:
        ax1.axvline(x=highlight_epoch, color="orange", linestyle="--", alpha=0.7)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss 曲线")
    ax1.legend(fontsize=8)

    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], "b-", label="Train Acc", alpha=0.7)
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]], "r-", label="Val Acc", alpha=0.7)
    if highlight_epoch:
        ax2.axvline(x=highlight_epoch, color="orange", linestyle="--", alpha=0.7)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy 曲线")
    ax2.legend(fontsize=8)

    if annotations:
        for x, text in annotations:
            ax2.annotate(text, xy=(x, 60), fontsize=8, color="orange",
                         ha="center",
                         arrowprops=dict(arrowstyle="->", color="orange"))

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _show_gradcam_snapshot(epoch: int, caption: str):
    gradcam_dir = GRADCAM_DIR
    if not os.path.exists(gradcam_dir):
        return

    prefix = f"epoch_{epoch:03d}"
    overlays = sorted([f for f in os.listdir(gradcam_dir)
                       if f.startswith(prefix) and (f.endswith("_overlay.png") or f.endswith("_overlay.npy"))])
    if overlays:
        st.caption(f"**{caption}**")
        cols = st.columns(min(len(overlays), 3))
        for i, fname in enumerate(overlays[:3]):
            with cols[i % 3]:
                img = load_image_smart(os.path.join(gradcam_dir, fname))
                if img:
                    st.image(img, caption=f"样本{i+1}", width=250)


def _show_gradcam_evolution():
    gradcam_dir = GRADCAM_DIR
    if not os.path.exists(gradcam_dir):
        st.warning("Grad-CAM快照未生成，请运行 model.py 训练")
        return

    epochs_available = sorted(set(
        int(f.split("_")[1]) for f in os.listdir(gradcam_dir)
        if f.endswith("_overlay.png") or f.endswith("_overlay.npy")
    ))
    if not epochs_available:
        return

    epoch = st.slider("选择训练轮次查看Grad-CAM", epochs_available[0],
                      epochs_available[-1], epochs_available[0],
                      key="s3_gradcam_slider")
    prefix = f"epoch_{epoch:03d}"
    overlays = sorted([f for f in os.listdir(gradcam_dir)
                       if f.startswith(prefix) and (f.endswith("_overlay.png") or f.endswith("_overlay.npy"))])

    if overlays:
        st.write(f"**第{epoch}轮 AI注意力分布**")
        cols = st.columns(min(len(overlays), 3))
        for i, fname in enumerate(overlays[:3]):
            with cols[i % 3]:
                img = load_image_smart(os.path.join(gradcam_dir, fname))
                if img:
                    st.image(img, caption=f"样本{i+1}", width=250)
        st.caption("💡 热力图中红色区域是AI当前最关注的区域。随着训练深入，"
                   "高亮区会从随机分散逐渐聚焦到真正的病灶位置。")


def _generate_demo_history():
    """生成模拟训练历史（作为回退数据）"""
    epochs = list(range(1, 51))
    train_loss = [0.9 * math.exp(-0.08 * e) + 0.05 * random.random() + 0.1 for e in epochs]
    val_loss = [0.85 * math.exp(-0.06 * e) + 0.1 * random.random() + 0.15 for e in epochs]
    train_acc = [min(0.95, 0.3 + 0.6 * (1 - math.exp(-0.06 * e)) + 0.03 * random.random())
                 for e in epochs]
    val_acc = [min(0.85, 0.25 + 0.5 * (1 - math.exp(-0.04 * e)) + 0.05 * random.random())
               for e in epochs]
    return {
        "epochs": epochs,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "param_norms": {"epochs": epochs},
    }


def _rectangle_mask(size, x, y, rect_w, rect_h):
    w, h = size
    mask = np.zeros((h, w), dtype=np.uint8)
    x1, y1 = max(0, min(w - 1, x)), max(0, min(h - 1, y))
    x2, y2 = max(x1 + 1, min(w, x + rect_w)), max(y1 + 1, min(h, y + rect_h))
    mask[y1:y2, x1:x2] = 1
    return mask


def _draw_mask_outline(image, mask, color):
    image_np = np.array(image.convert("RGB"))
    mask_uint8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image_np, contours, -1, color, 2)
    return Image.fromarray(image_np)


def _overlay_heatmap(image, cam, alpha=0.45):
    image_np = np.array(image.convert("RGB"))
    cam_resized = cv2.resize(cam, (image_np.shape[1], image_np.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(cam_resized * 255), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return np.clip(image_np * (1 - alpha) + heatmap * alpha, 0, 255).astype(np.uint8)


def _predict_with_model(model, gradcam, image):
    from part1 import preprocess_image as ppi
    image_tensor = ppi(image)
    with torch.enable_grad():
        output = model(image_tensor)
        probs = F.softmax(output, dim=1)
        pred_idx = int(torch.argmax(probs, dim=1).item())
        cam = gradcam.generate(image_tensor, pred_idx)
    return {
        "mode": "model",
        "pred_idx": pred_idx,
        "pred_class": IDX_TO_CLASS[pred_idx],
        "normal_prob": float(probs[0][0].item()),
        "lesion_prob": float(probs[0][1].item()),
        "cam": cam,
    }


def _predict_with_rule_demo(image):
    gray = np.array(image.convert("L"), dtype=np.float32)
    gray_norm = (gray - gray.min()) / (gray.max() - gray.min() + 1e-8)
    blur = cv2.GaussianBlur(gray_norm, (0, 0), 3)
    lap = cv2.Laplacian(blur, cv2.CV_32F)
    edge = np.abs(lap)
    edge = cv2.GaussianBlur(edge, (0, 0), 5)
    h, w = gray.shape
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w / 2, h / 2
    sigma_x, sigma_y = w * 0.28, h * 0.28
    center_bias = np.exp(-(((xx - cx) ** 2) / (2 * sigma_x ** 2) +
                           ((yy - cy) ** 2) / (2 * sigma_y ** 2)))
    bright_map = np.clip(gray_norm - np.percentile(gray_norm, 60) / 255.0, 0, 1)
    cam = 0.55 * edge + 0.35 * bright_map + 0.25 * center_bias
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    lesion_score = float(cam.mean() * 0.55 + cam.max() * 0.45)
    lesion_prob = min(max(lesion_score, 0.02), 0.98)
    normal_prob = 1.0 - lesion_prob
    pred_idx = 1 if lesion_prob >= 0.5 else 0
    return {
        "mode": "rule_demo",
        "pred_idx": pred_idx,
        "pred_class": IDX_TO_CLASS[pred_idx],
        "normal_prob": normal_prob,
        "lesion_prob": lesion_prob,
        "cam": cam,
    }


# ---------- 主入口 ----------
def main():
    init_game_state()
    render_sidebar()

    stages = {
        1: render_stage1,
        2: render_stage2,
        3: render_stage3,
        4: render_stage4,
        5: render_stage5,
    }

    stage_func = stages.get(st.session_state.stage, render_stage1)
    stage_func()


if __name__ == "__main__":
    main()
