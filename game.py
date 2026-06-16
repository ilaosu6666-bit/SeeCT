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
            try:
                arr = np.load(str(p), allow_pickle=True)
            except Exception:
                return None
            if arr.dtype in (np.float64, np.float32) and arr.max() <= 1.0:
                arr = (arr * 255).astype(np.uint8)
            return Image.fromarray(arr.astype(np.uint8))
        return None
    if p.exists():
        img = Image.open(p)
        if img.mode in ("I;16", "I;16B", "I"):
            arr = np.array(img, dtype=np.float32)
            arr_max = arr.max()
            if arr_max > 0:
                arr = (arr / arr_max * 255).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8)
            img = Image.fromarray(arr)
        return img.convert("RGB")
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
    checkpoint = torch.load(model_path, map_location=DEVICE)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
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
        with st.expander("🔧 开发者模式"):
            def _dev_jump():
                st.session_state.stage = st.session_state.dev_jump

            st.selectbox(
                "跳转到关卡",
                [1, 2, 3, 4, 5],
                index=st.session_state.stage - 1,
                key="dev_jump",
                format_func=lambda x: f"关卡{x}: {stage_names[x]}",
                on_change=_dev_jump,
            )

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

        # 优先用缩略图（在线加载快），无缩略图则用原图
        thumb_dir = Path("cases/case_001/slices/thumb")
        full_dir = Path("cases/case_001/slices")
        slice_source = thumb_dir if thumb_dir.exists() else full_dir

        if slice_source.exists():
            slices = sorted([f for f in slice_source.iterdir()
                           if f.suffix.lower() in (".png", ".npy") and "image" not in f.name])
            if len(slices) == 0 and slice_source == thumb_dir:
                # 缩略图为空，回退到完整目录
                slices = sorted([f for f in full_dir.iterdir()
                               if f.suffix.lower() in (".png", ".npy") and "image" not in f.name])
            if len(slices) > 0:
                slice_idx = st.slider(
                    "切片编号 (Z轴)", 0, len(slices) - 1,
                    len(slices) // 2, key="s1_slice_browser",
                    on_change=lambda: st.session_state.update({"s1_tab1_visited": True}))
                slice_img = load_image_smart(str(slices[slice_idx]))
                if slice_img:
                    st.image(slice_img, caption=f"切片 #{slice_idx + 1} / {len(slices)}",
                             use_container_width=True)
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

        # 手动确认已浏览
        if not st.session_state.s1_tab1_visited:
            if st.button("✓ 我已了解CT数据的基础知识", key="s1_tab1_done"):
                st.session_state.s1_tab1_visited = True
                st.rerun()

    with tab2:
        st.subheader("肺窗调节实验")
        st.info("""
        ### 🪟 什么是"肺窗"？

        CT图像的HU值范围极宽（-1000 ~ +1000），但人眼只能分辨约16个灰阶。
        如果直接把整个范围映射到黑白，肺部的细微结构会被压缩得几乎看不见。

        **窗技术**就像给CT图像加了一个"观察窗口"：

        - **窗位（WC）**：窗口的中心位置。你想看哪种组织，就把窗口中心设在该组织的HU值。
          肺组织约-700HU，所以肺窗的窗位设在-600左右。
        - **窗宽（WW）**：窗口的宽度。窗口越窄，对比度越强（黑白分明）；窗口越宽，能看到更多层次但对比减弱。
          肺窗用1500的宽度，既能看清肺纹理，又不会对比过强。

        **类比：** 就像用放大镜看地图——窗位是放大镜对准的位置，窗宽是放大倍数。
        """)
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
            st.image(demo_image, caption="原始HU值图像（未经窗处理）", use_container_width=True,
                     clamp=True)
        with col2:
            st.image(windowed, caption=f"窗处理后 (WC={wc}, WW={ww})",
                     use_container_width=True, clamp=True)

        if abs(wc - (-600)) <= 50 and abs(ww - 1500) <= 100:
            st.success("✅ 你已经接近临床标准肺窗参数！WC≈-600, WW≈1500 是观察肺实质的最佳设置。")
            if not st.session_state.window_found:
                st.session_state.window_found = True
                st.session_state.score += 15
                st.success("成就解锁: 肺窗大师 (+15分)")
        else:
            st.info("💡 提示：尝试 WC=-600, WW=1500（临床标准肺窗），观察肺纹理是否变清晰。")

        # 手动确认已浏览
        if not st.session_state.s1_tab2_visited:
            if st.button("✓ 我已理解窗宽窗位的概念", key="s1_tab2_done"):
                st.session_state.s1_tab2_visited = True
                st.rerun()

    with tab3:
        st.subheader("标记训练数据")
        st.write("现在，请你扮演数据标注员，判断以下CT切片中是否含有肺结节。**全部选定后统一提交。**")

        if "s1_label_samples" not in st.session_state:
            all_cases = [p for p in Path("cases").iterdir()
                        if p.is_dir() and p.name.startswith("case_")]
            lesion_samples = []
            normal_samples = []
            case_by_path = {}
            for c in all_cases:
                meta = load_case_meta(c)
                cls = meta.get("class", "")
                img_path = c / meta.get("image", "image.png")
                if img_path.exists():
                    entry = (str(img_path), cls, meta.get("title", ""), meta)
                    case_by_path[str(img_path)] = meta
                    if cls == "Lesion":
                        lesion_samples.append(entry)
                    else:
                        normal_samples.append(entry)
            # 各取最多3个，保持固定顺序（病灶在前，正常在后）
            samples = lesion_samples[:3] + normal_samples[:3]
            if len(samples) < 2:
                samples = lesion_samples + normal_samples
            st.session_state["s1_label_samples"] = samples[:6]
            st.session_state["s1_label_answers"] = {}
            st.session_state["s1_label_revealed"] = False
            st.session_state["s1_case_metas"] = case_by_path

        samples = st.session_state["s1_label_samples"]
        answers = st.session_state["s1_label_answers"]
        is_revealed = st.session_state.get("s1_label_revealed", False)

        if len(samples) == 0:
            st.warning("暂无标注样本。请先运行 build_case_library.py 构建病例库。")

        # ---- 标注阶段：展示所有图片+选项 ----
        temp_answers = {}
        all_answered = True
        for i, (img_path, true_label, title, meta) in enumerate(samples):
            cols = st.columns([1, 1])
            with cols[0]:
                disp_img = load_image_smart(img_path)
                if disp_img:
                    st.image(disp_img, caption=f"样本 {i+1}", use_container_width=True)
                else:
                    st.error(f"图片加载失败: {img_path}")
            with cols[1]:
                user_label = st.radio(
                    f"你的判断 #{i+1}",
                    ["有结节 (Lesion)", "无结节 (Normal)"],
                    key=f"s1_label_{i}",
                    index=None,
                    disabled=is_revealed
                )
                temp_answers[i] = "Lesion" if (user_label and "有结节" in user_label) else ("Normal" if (user_label and "无结节" in user_label) else None)
                if temp_answers[i] is None:
                    all_answered = False

        # ---- 提交按钮 ----
        if not is_revealed:
            st.markdown("---")
            if all_answered and len(samples) > 0:
                if st.button("📤 提交标注", type="primary", use_container_width=True):
                    for i, ans in temp_answers.items():
                        answers[i] = ans
                    st.session_state["s1_label_revealed"] = True
                    # 计分
                    for i, (_, true_label, _, _) in enumerate(samples):
                        if answers.get(i) == true_label:
                            st.session_state.score += 10
                            st.session_state.data_label_score += 10
                    st.session_state.s1_tab3_complete = True
                    st.rerun()
            else:
                st.info(f"👆 请先为所有 {len(samples)} 个样本做出判断，才能提交。")

        # ---- 结果展示阶段 ----
        if is_revealed:
            st.markdown("---")
            st.subheader("📊 标注结果与CT解读")

            correct_count = 0
            for i, (img_path, true_label, title, meta) in enumerate(samples):
                user_ans = answers.get(i, "未作答")
                is_correct = user_ans == true_label
                if is_correct:
                    correct_count += 1

                # 样本1、样本2 使用 cases/roi/ 中的专用标注图
                roi_path = Path(f"cases/roi/{i+1}.png")
                display_path = str(roi_path) if roi_path.exists() else img_path

                st.markdown(f"### {'✅' if is_correct else '❌'} 样本 {i+1}")
                col_a, col_b = st.columns(2)
                with col_a:
                    disp_img = load_image_smart(display_path)
                    if disp_img:
                        st.image(disp_img, width=280)
                with col_b:
                    st.write(f"**你的判断:** {user_ans}")
                    st.write(f"**真实标签:** {true_label}")
                    if is_correct:
                        st.success("判断正确！")
                    else:
                        st.error("判断有偏差")

                    # ---- CT解读内容 ----
                    mk = meta or {}
                    if mk.get("medical_knowledge") or mk.get("description"):
                        with st.expander("🔬 查看CT解读"):
                            if mk.get("description"):
                                st.write(f"**病例描述:** {mk['description']}")
                            mk_inner = mk.get("medical_knowledge", {})
                            if mk_inner.get("imaging_features"):
                                st.write("**影像学特征:**")
                                for f in mk_inner["imaging_features"]:
                                    st.write(f"  • {f}")
                            if mk_inner.get("differential_diagnosis"):
                                st.write("**鉴别诊断:**")
                                for d in mk_inner["differential_diagnosis"]:
                                    st.write(f"  • {d}")
                            if mk.get("teaching_points"):
                                st.write("**教学要点:**")
                                for tp in mk["teaching_points"]:
                                    st.write(f"  • {tp}")
                    elif not is_correct:
                        st.caption("💡 提示：正常血管断面和钙化点容易与结节混淆。浏览相邻切片有助于判断。")

                st.markdown("---")

            accuracy = correct_count / max(1, len(samples))
            st.subheader("📊 数据收集报告")
            col1, col2, col3 = st.columns(3)
            col1.metric("标注总数", len(samples))
            col2.metric("正确数", correct_count)
            col3.metric("准确率", f"{accuracy:.0%}")

            if accuracy >= 0.8:
                st.success("🏆 成就解锁: 火眼金睛")
                if "火眼金睛" not in st.session_state.achievements:
                    st.session_state.achievements.append("火眼金睛")
                    st.session_state.score += 10

            st.info("""
            ### 💡 ROI标注的重要性

            训练医学影像AI时，除了要知道"这张图有没有病灶"，
            还需要告诉AI **病灶具体在哪个位置**——这就是 **ROI（Region of Interest，感兴趣区域）标注**。

            **什么是ROI标注？**
            用矩形框或轮廓线把病灶区域精确圈出来。就像老师用红笔批改作业时，
            不仅告诉你"有错"，还圈出具体错在哪里。

            **ROI标注为什么重要？**
            - 标注越精确 → AI学到的病灶特征越准
            - 只有标签没有位置 → AI可能学到错误的相关性（比如把CT机器型号当成了判断依据）
            - 标注偏差 → AI可能产生系统性误判（比如只学会某一种形状的结节）

            高质量的ROI标注是医学AI的基础，通常需要**资深放射科医生**花费大量时间完成。
            这正是LIDC-IDRI数据集的价值所在——它包含了4位专家的详细结节标注。
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

        st.info("""
        ### 🧠 什么是神经网络架构？

        **神经网络**就像AI的"大脑结构"——由一层层"神经元"堆叠而成。
        每一层负责从图片中提取不同级别的特征：
        - 浅层（前面几层）：识别边缘、颜色、纹理
        - 中层：识别形状、轮廓
        - 深层（后面几层）：识别完整的器官、病灶

        **不同的"大脑结构"适合不同的任务**，就像汽车适合公路、越野车适合山路。
        选对架构，AI学得更快更好。
        """)

        arch_choice = st.radio(
            "你希望使用哪种架构？",
            ["2D ResNet18 (推荐)", "3D CNN (进阶)"],
            key="s2_arch",
            index=0 if st.session_state.chosen_arch is None
            else (0 if st.session_state.chosen_arch == "resnet18" else 1)
        )
        st.session_state.chosen_arch = "resnet18" if "ResNet18" in arch_choice else "3dcnn"

        st.markdown("### 📊 两种架构对比")
        comp_col1, comp_col2 = st.columns(2)
        with comp_col1:
            st.success("⭐ **2D ResNet18**")
            st.markdown("""
            | 维度 | 说明 |
            |------|------|
            | 参数量 | ~11M |
            | 输入 | 224×224 单张切片 |
            | 训练速度 | ~2分钟/轮 (CPU) |
            | 核心优势 | 残差连接，训练稳定 |
            | 适用场景 | 大多数2D医学分类 |

            **就像看一张CT照片来判断**
            """)
        with comp_col2:
            st.warning("🔥 **3D CNN**")
            st.markdown("""
            | 维度 | 说明 |
            |------|------|
            | 参数量 | ~5M (简化版) |
            | 输入 | 64×64×64 体积块 |
            | 训练速度 | ~8分钟/轮 (CPU) |
            | 核心优势 | 利用上下切片的空间关系 |
            | 适用场景 | 需要3D上下文的检测 |

            **就像拿着整个器官的3D模型来判断**
            """)

        st.caption("**选择建议：** 如果第一次玩，选2D ResNet18。它速度快、效果好，是入门最佳选择。3D CNN更强大但训练慢，适合进阶体验。")

        if "ResNet18" in arch_choice:
            st.success("⭐ 新手友好 — ResNet18已在全球数百万图像分类任务中验证有效")
        else:
            st.warning("🔥 进阶挑战 — 3D CNN参考了 mr-mukherjee03/CT-Scan-Nodule-Detection 项目结构")

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
        st.subheader("AI的第一轮学习")
        st.info("""
        ### 🎓 先搞懂几个概念

        训练AI就像**教小孩认字**，几个关键概念先了解一下：

        | AI术语 | 通俗理解 |
        |--------|---------|
        | **Epoch（训练轮数）** | 就像把整本教科书从头到尾看一遍。看一遍 = 1轮 |
        | **Loss（错误分）** | AI的"犯错分数"。分数越低 = 错越少 = 学得越好 |
        | **Accuracy（准确率）** | AI考试的"正确率"。越高越好 |
        | **学习率** | AI每次改正错误的"步子大小"。步子太大容易跨过头，太小学得太慢 |
        """)
        st.write("现在AI刚刚开始学习，就像第一天上学的小学生。看看它的表现👇")

        _plot_training_curves(history, highlight_epoch=4,
                              context="训练刚开始，Loss 快速下降说明 AI 正在从数据中学习基本规律。"
                                      "就像婴儿开始认识世界，进步最快的就是这个阶段。")
        _show_gradcam_snapshot(1, "第1轮：AI的注意力是随机的，到处乱看")
        _show_gradcam_snapshot(4, "第4轮：AI开始关注肺部区域了")

        st.success(f"""
        **4轮训练后：**
        - 错误分从高处快速下降 → AI在飞速进步（就像刚学骑车，从不会到能骑几米）
        - 正确率开始上升 → AI开始摸到门道了
        """)

        if st.button("继续观察 →", type="primary", width="stretch"):
            st.session_state.s3_phase = "decision1"
            st.rerun()

    # --- 决策点1 ---
    elif st.session_state.s3_phase == "decision1":
        st.subheader("⚠️ 决策点 1: AI的进步速度变慢了")
        st.info("""
        ### 🤔 为什么进步变慢了？

        刚学骑车时，从完全不会到能骑几米，进步巨大。
        但学会基本平衡后，再想骑得更稳、更快——**进步自然会慢下来**。

        AI也是一样：前几轮学到了最容易的规律（"有白点=可能有结节"），
        剩下的是更难更细的判断。**慢下来是正常的，说明AI进入了精细打磨阶段。**

        这时候乱调参数，反而可能把AI已经学会的东西搞乱。
        """)

        _plot_training_curves(history, highlight_epoch=5,
                              annotations=[(5, "进步变慢")],
                              context="Loss 下降变缓是正常的——AI 已经学会了大方向，"
                                      "现在进入精细化调整。就像学骑车，"
                                      "一开始进步最大，后面每次只进步一点点。")
        _show_gradcam_snapshot(5, "第5轮：AI的注意力在收窄，不再是到处乱看了")

        if 1 not in st.session_state.s3_decision_made:
            choice = st.radio(
                "作为AI训练师，你该如何应对？",
                ["A. 保持现在的学习速度继续学（耐心等它进步）✅",
                 "B. 把学习步子调到极小（等于让AI从头慢慢来）",
                 "C. 把学习步子加大（让AI冲快点，可能冲过头）"],
                key="s3_d1",
                index=None
            )
            if choice and st.button("确认决策", key="s3_d1_confirm"):
                st.session_state.s3_decision_made[1] = choice
                if choice.startswith("A"):
                    st.session_state.score += 20
                    st.session_state.training_decisions.append("correct1")
                    st.success("✅ 耐心是对的！")
                    st.write("""
                    **实际效果：** AI按照原来的节奏继续学习，错误分继续缓慢下降，正确率稳步提升。

                    **学到了什么：** AI学习过程中进步放缓是正常现象。就像学任何技能一样，
                    快速进步期之后必然进入慢速打磨期。不要急于求成。
                    """)
                elif choice.startswith("B"):
                    st.error("❌ 步子太小了！")
                    st.write("""
                    **实际效果：** AI的学习几乎停滞。步子太小意味着每次只改一点点，
                    可能永远学不到最优解。

                    **学到了什么：** 学习步子（学习率）太小会导致训练过慢甚至停滞。
                    """)
                else:
                    st.warning("⚠️ 步子太大了！")
                    st.write("""
                    **实际效果：** AI的误差开始剧烈波动。步子太大导致AI
                    在最优解附近反复横跳，永远稳定不下来。

                    **学到了什么：** 学习步子（学习率）太大会让训练变得不稳定。
                    """)
                st.rerun()
        else:
            st.info(f"你的选择: {st.session_state.s3_decision_made[1]}")
            if st.button("继续 →"):
                st.session_state.watched_epoch = 15
                st.session_state.s3_phase = "decision2"
                st.rerun()

    # --- 决策点2 ---
    elif st.session_state.s3_phase == "decision2":
        st.subheader("⚠️ 决策点 2: 正确率卡住了")
        st.info("""
        ### 🤔 为什么正确率不涨了？

        考试分数从60提到80很容易，但从80提到95就难多了——这叫**瓶颈期**。

        AI现在就是这种情况：简单的规律已经学会，剩下的都是"难题"。
        它需要更仔细地调整自己，而不是大刀阔斧地改。

        **先看看两种极端情况（目前都不是）：**
        """)

        col1, col2 = st.columns(2)
        with col1:
            st.warning("📖 **没学会 (欠拟合)**\n就像上课没听讲，考试全靠蒙\n→ 训练和验证都差")
        with col2:
            st.error("📝 **死记硬背 (过拟合)**\n就像只背了答案，换新题就不会\n→ 训练很好但验证差")

        st.caption("目前的情况：训练和验证都在慢慢变好，只是变好的速度慢了——这是正常的瓶颈期。")

        _plot_training_curves(history, highlight_epoch=15,
                              annotations=[(15, "瓶颈期")],
                              context="验证准确率出现平台期。AI 在当前学习策略下已接近极限，"
                                      "需要降低学习率来更精细地调整参数，从而突破瓶颈。")
        _show_gradcam_snapshot(15, "第15轮：AI的注意力更集中了")

        if "d2" not in st.session_state.s3_decision_made:
            choice = st.radio(
                "你觉得应该怎么做？",
                ["A. 继续学，AI还在进步（耐心观察）",
                 "B. 减小学习步子，让AI学得更细致（推荐）✅",
                 "C. 停止训练，就用现在的AI（放弃太早）"],
                key="s3_d2",
                index=None
            )
            if choice and st.button("确认决策", key="s3_d2_confirm"):
                st.session_state.s3_decision_made["d2"] = choice
                if choice.startswith("B"):
                    st.session_state.score += 20
                    st.session_state.training_decisions.append("correct2")
                    st.success("✅ 细致打磨是对的！")
                    st.write("""
                    **实际效果：** AI用更小的步子继续学习，正确率缓慢但稳定地突破瓶颈。

                    **学到了什么：** 遇到瓶颈时，减小学习步子让AI更精细地调整参数，
                    往往能突破平台期。就像雕刻，大斧头砍出轮廓后，需要用刻刀精修。
                    """)
                elif choice.startswith("C"):
                    st.error("❌ 放弃得太早了！")
                    st.write("""
                    **实际效果：** AI只学到了皮毛就停下了。它的能力停留在中等水平，
                    遇到稍微复杂的CT图就容易判断错误。

                    **学到了什么：** 瓶颈期不等于终点。过早放弃会让AI的能力大打折扣。
                    """)
                else:
                    st.warning("⚠️ 耐心等也可以，但不够高效。")
                    st.write("""
                    **实际效果：** AI继续学，但速度很慢。最终也能达到类似的效果，但需要更多时间。

                    **学到了什么：** 瓶颈期减小学习步子，能让AI更快地突破。这就是"学习率衰减"策略。
                    """)
                st.rerun()
        else:
            st.info(f"你的选择: {st.session_state.s3_decision_made['d2']}")
            if st.button("继续 →"):
                st.session_state.s3_phase = "decision3"
                st.rerun()

    # --- 决策点3 ---
    elif st.session_state.s3_phase == "decision3":
        st.subheader('⚠️ 决策点 3: AI在“背答案”！')
        st.info("""
        ### 🤔 训练好但验证差？这是"死记硬背"了！

        看两个数字对比：
        - 训练集正确率 **90%**：AI看过的CT图，判断基本全对 ✅
        - 验证集正确率 **78%**：AI没见过的CT图，错了一大堆 ❌

        **这说明AI在"背答案"而不是"学规律"！**
        就像学生把练习册所有答案都背下来，练习册上的题全对，
        但一考试（换了新题）就傻眼了。

        **怎么解决？** 训练时故意给图片加随机变化（翻转、旋转、缩放），
        让AI没办法靠"记住某张图"来作弊，逼它去学习真正的判断规律。
        """)

        _plot_training_curves(history, highlight_epoch=25,
                              annotations=[(25, "过拟合！")],
                              context="⚠️ 训练和验证出现明显分叉！AI 开始'背答案'——"
                                      "它记住了训练数据的特征，而不是学会通用判断。这就是过拟合。")
        _show_gradcam_snapshot(25, "第25轮：AI过度关注某些特定区域")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("训练正确率", "90%")
        with col2:
            st.metric("验证正确率", "78%", delta="-12%")

        if "d3" not in st.session_state.s3_decision_made:
            choice = st.radio(
                "AI在背答案，怎么纠正？",
                ["A. 给训练图片加随机翻转和旋转，逼AI学规律 ✅",
                 "B. 继续用同样的方式训练更多轮（等于继续背答案）",
                 "C. 给AI换更复杂的大脑（背答案能力更强→适得其反）"],
                key="s3_d3",
                index=None
            )
            if choice and st.button("确认决策", key="s3_d3_confirm"):
                st.session_state.s3_decision_made["d3"] = choice
                if choice.startswith("A"):
                    st.session_state.score += 20
                    st.session_state.training_decisions.append("correct3")
                    st.success("✅ 逼AI学规律是对的！")
                    st.write("""
                    **实际效果：** 加入随机变换后，AI被迫学习真正的结节特征
                    （形状、边缘、密度），而不是记忆某张特定图片。验证正确率回升到85%+。

                    **学到了什么：** 这种技术叫"数据增强"——通过给训练数据增加变化，
                    防止AI死记硬背。这是AI训练中最常用的防过拟合技巧。
                    """)
                elif choice.startswith("B"):
                    st.error("❌ 继续背答案只会更糟！")
                    st.write("""
                    **实际效果：** 训练正确率继续升高到95%+，但验证正确率甚至开始下降。
                    AI彻底变成了只会做"原题"的考试机器。

                    **学到了什么：** 过拟合一旦出现，继续用同样方式训练只会加重问题。
                    """)
                else:
                    st.error("❌ 方向反了！")
                    st.write("""
                    **实际效果：** AI有了更强的记忆能力，背答案更厉害了——训练正确率99%，
                    但验证正确率跌到70%以下。完全失去了对新病例的判断能力。

                    **学到了什么：** 解决过拟合要"约束"AI而不是"增强"AI。
                    更复杂的大脑 = 更强的背诵能力 = 更严重的过拟合。
                    """)
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
        st.subheader("🎉 训练完成！你的AI毕业了！")

        correct_decisions = len([d for d in st.session_state.training_decisions
                                 if d.startswith("correct")])
        st.session_state.score += correct_decisions * 5
        final_acc = 0.82 + correct_decisions * 0.03
        st.session_state.model_accuracy = min(final_acc, 0.92)

        if correct_decisions == 3:
            st.success(f"""
            ### 🏆 完美训练！

            你所有的训练决策都是正确的——AI在最优路径上学到了最佳能力！

            最终正确率：**{st.session_state.model_accuracy:.1%}**
            意味着给AI看100张它没见过的CT图，大约 **{int(st.session_state.model_accuracy*100)} 张**能判断正确。
            """)
            if "稳扎稳打" not in st.session_state.achievements:
                st.session_state.achievements.append("稳扎稳打")
                st.session_state.score += 15
        elif correct_decisions == 2:
            st.info(f"""
            ### 👍 不错的训练！

            你的大多数决策都是对的。AI学到了不错的能力，但还有优化空间。

            最终正确率：**{st.session_state.model_accuracy:.1%}**
            给AI看100张没见过的CT图，大约 **{int(st.session_state.model_accuracy*100)} 张**能判断正确。

            下次训练可以尝试不同的策略，看看能不能让AI更聪明！
            """)
        else:
            st.warning(f"""
            ### 📚 学到了很多！

            虽然有些决策不是最优，但AI还是掌握了基本能力。
            **最重要的是——你现在知道训练AI会遇到的坑了！**

            最终正确率：**{st.session_state.model_accuracy:.1%}**
            下次重新挑战，试试不同的选择，看看正确率能到多少？
            """)

        st.markdown("---")
        st.subheader("📈 看看AI是怎样从零变成专家的：")
        _plot_training_curves(history, highlight_epoch=total_epochs,
                              annotations=[
                                  (5, "决策1"), (15, "决策2"), (25, "决策3")
                              ],
                              context=f"训练完成！最终验证准确率 {st.session_state.model_accuracy:.1%}。"
                                      "AI 已经学会了肺结节诊断。接下来用真实病例检验你的AI吧。",
                              truncate=False)

        st.markdown("### 🔥 AI的注意力演变（Grad-CAM）")
        st.write("从第1轮到第50轮，AI关注的重点区域如何从'到处乱看'变成'精准找到病灶'：")
        _show_gradcam_evolution()

        if "洞察本质" not in st.session_state.achievements:
            st.session_state.achievements.append("洞察本质")

        st.info("""
        ### 💡 这节课你学到了什么？

        1. AI学习有**快速进步期**（前几轮）和**慢速打磨期**（后期）——这是正常现象
        2. 遇到瓶颈时，**减小学习步子**比硬冲更有效
        3. **"背答案"（过拟合）**是AI训练最大的坑——要用数据增强来防止

        这些道理不仅适用于AI训练，也适用于人类学习的很多场景！
        """)

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
        st.info("""
        ### 📚 诊断回顾

        你已完成全部病例的诊断练习。每个病例的正确诊断依据已在上方展示。

        **关键收获：** AI的Grad-CAM热力图可以帮助定位病灶，但最终诊断仍需结合
        影像学特征（边缘形态、密度、周围结构改变等）综合判断。
        """)

        if "诊断专家" not in st.session_state.achievements:
            st.session_state.achievements.append("诊断专家")

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
        st.image(image, caption="CT图像", use_container_width=True)
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
            x = st.slider("框位置 X", 0, max(0, w - 1), int(w * 0.25),
                          key=f"s4_x_{case_idx}")
            rect_w = st.slider("框宽度", 1, max(1, w), int(w * 0.25),
                               key=f"s4_rw_{case_idx}")
        with col2:
            y = st.slider("框位置 Y", 0, max(0, h - 1), int(h * 0.25),
                          key=f"s4_y_{case_idx}")
            rect_h = st.slider("框高度", 1, max(1, h), int(h * 0.25),
                               key=f"s4_rh_{case_idx}")

        user_mask = _rectangle_mask((w, h), x, y, rect_w, rect_h)
        outlined = _draw_mask_outline(image, user_mask, (0, 255, 0))
        st.image(outlined, caption="你的圈选（绿色轮廓）", use_container_width=True)

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
            st.image(overlay_img, caption="AI 热力图叠加", use_container_width=True)
        with col2:
            ai_mask = (cam > 0.55).astype(np.uint8)
            ai_outline = _draw_mask_outline(image, ai_mask, (255, 0, 0))
            st.image(ai_outline, caption="AI 关注区域（红色轮廓）", use_container_width=True)

        st.markdown("### 🔬 正确诊断依据")
        true_class = meta.get("class", "Unknown")
        user_pred = "Lesion" if user_has_nodule == "有结节" else "Normal"

        # 显示正确答案和诊断依据
        col1, col2 = st.columns(2)
        with col1:
            if user_pred == true_class:
                st.success(f"✅ 你的判断正确：**{true_class}**")
            else:
                st.error(f"你的判断：{user_pred}  →  正确诊断：**{true_class}**")
        with col2:
            if model is not None:
                if result["pred_class"] == true_class:
                    st.success(f"✅ AI判断也正确：**{result['pred_class']}**")
                else:
                    st.warning(f"AI判断：{result['pred_class']}（AI也有看走眼的时候）")

        # 展示诊断依据
        mk = meta.get("medical_knowledge", {})
        if mk.get("imaging_features"):
            st.markdown("**📋 该病例的影像学诊断依据：**")
            for f in mk["imaging_features"]:
                st.markdown(f"- {f}")
        if mk.get("differential_diagnosis"):
            st.markdown("**🔄 需要鉴别的其他可能：**")
            for d in mk["differential_diagnosis"]:
                st.markdown(f"- {d}")
        if mk.get("lidc_characteristics"):
            lidc = mk["lidc_characteristics"]
            st.caption(f"LIDC特征：恶性评分均值 {lidc.get('malignancy_avg', 'N/A')}/5 | "
                      f"毛刺征 {lidc.get('spiculation', 'N/A')}/5 | "
                      f"纹理 {lidc.get('texture', 'N/A')}/5")

        if meta.get("teaching_points"):
            st.markdown("**💡 教学要点：**")
            for tp in meta["teaching_points"]:
                st.markdown(f"- {tp}")

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
    disp_img = load_image_smart(item["path"])

    col1, col2 = st.columns([3, 2])
    with col1:
        if disp_img:
            st.image(disp_img, caption=f"CT #{current + 1}", use_container_width=True)
        else:
            st.error(f"图片加载失败: {item['path']}")

    with col2:
        st.markdown("### AI辅助")
        ai_on = st.toggle("显示AI分析", value=True, key=f"s5_ai_{current}")

        if ai_on:
            if model is not None and gradcam is not None and disp_img is not None:
                result = _predict_with_model(model, gradcam, disp_img)
                overlay_img = _overlay_heatmap(disp_img, result["cam"])
                st.image(overlay_img, caption="AI热力图", use_container_width=True)
                st.metric("AI判断", item["ai_label"])
            else:
                st.info("AI判断: " + item["ai_label"])
                if item.get("ai_conf"):
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

    # ---- 正确答案与解析 ----
    st.markdown("---")
    st.subheader("📋 正确答案与解析")
    for idx in sorted(answers.keys()):
        a = answers[idx]
        user_lbl = a["user_label"]
        true_lbl = a["true_label"]
        ai_lbl = a["ai_label"]
        is_correct = user_lbl == true_lbl
        is_trap = a.get("ai_trap", False)

        status = "✅" if is_correct else "❌"
        st.markdown(f"#### {status} CT #{idx + 1}")
        col_a, col_b = st.columns(2)
        with col_a:
            st.write(f"**你的判断:** {user_lbl}")
            st.write(f"**AI判断:** {ai_lbl}" + (" ⚠️ (AI标注错误!)" if is_trap else ""))
        with col_b:
            st.write(f"**正确答案:** {true_lbl}")
            if not is_correct:
                if true_lbl == "Lesion":
                    st.caption("解析：该CT存在可疑密度增高影，需关注边缘和形态特征。")
                else:
                    st.caption("解析：该CT肺实质清晰，未见明确结节或肿块影。")
            if is_trap and user_lbl != ai_lbl and is_correct:
                st.success("🌟 你识破了AI的错误标注！保持独立判断很重要。")

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


def _plot_training_curves(history, highlight_epoch=None, annotations=None,
                         context: str = "", truncate: bool = True):
    if history is None:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))

    # 截断到当前 epoch（还没发生的轮次不显示，营造"正在进行"的感觉）
    if truncate and highlight_epoch:
        end = min(highlight_epoch, len(history["epochs"]))
    else:
        end = len(history["epochs"])

    epochs = history["epochs"][:end]
    train_loss = history["train_loss"][:end]
    val_loss = history["val_loss"][:end]
    train_acc = history["train_acc"][:end]
    val_acc = history["val_acc"][:end]

    ax1.plot(epochs, train_loss, "b-", label="Train Loss", linewidth=2, alpha=0.8)
    ax1.plot(epochs, val_loss, "r-", label="Val Loss", linewidth=2, alpha=0.8)
    if highlight_epoch:
        ax1.axvline(x=highlight_epoch, color="orange", linestyle="--", alpha=0.7)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss 曲线")
    ax1.legend(fontsize=8)
    ax1.set_xlim(0, max(end, 5))

    ax2.plot(epochs, [a * 100 for a in train_acc], "b-", label="Train Acc", linewidth=2, alpha=0.8)
    ax2.plot(epochs, [a * 100 for a in val_acc], "r-", label="Val Acc", linewidth=2, alpha=0.8)
    if highlight_epoch:
        ax2.axvline(x=highlight_epoch, color="orange", linestyle="--", alpha=0.7)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy 曲线")
    ax2.legend(fontsize=8)
    ax2.set_xlim(0, max(end, 5))

    if annotations:
        for x, text in annotations:
            if x <= end:
                ax2.annotate(text, xy=(x, max(30, min(val_acc) * 90 if val_acc else 30)),
                             fontsize=8, color="orange", ha="center",
                             arrowprops=dict(arrowstyle="->", color="orange"))

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    if context:
        st.caption(f"📖 **当前阶段：** {context}")


def _generate_fake_cam(epoch: int, size: int = 256) -> np.ndarray:
    """为指定 epoch 生成模拟 Grad-CAM 热力图，显示从随机到聚焦的演变。

    早期(epoch 1-5): 多个随机热点散布在肺区 — AI 到处乱看
    中期(epoch 5-15): 热点逐渐收拢 — AI 开始定位
    后期(epoch 15-50): 单一聚焦点 — AI 精准找到病灶
    """
    rng = np.random.RandomState(epoch)  # 每轮不同种子
    h, w = size, size

    # 最终结节位置
    nodule_cx, nodule_cy = int(w * 0.55), int(h * 0.48)

    cam = np.zeros((h, w), dtype=np.float32)

    if epoch <= 3:
        # 阶段1 (epoch 1-3): 多个随机离散热点，到处乱看
        for _ in range(6 + epoch):
            hot_x = int(rng.uniform(w * 0.2, w * 0.8))
            hot_y = int(rng.uniform(h * 0.15, h * 0.85))
            hot_sigma = rng.uniform(15, 40)
            hot_strength = rng.uniform(0.3, 0.7)
            yy, xx = np.mgrid[0:h, 0:w]
            spot = np.exp(-(((xx - hot_x) ** 2) / (2 * hot_sigma ** 2) +
                            ((yy - hot_y) ** 2) / (2 * hot_sigma ** 2)))
            cam += spot * hot_strength

    elif epoch <= 8:
        # 阶段2 (epoch 4-8): 热点减少，开始向结节位置收拢
        num_spots = max(2, 6 - epoch)
        for _ in range(num_spots):
            # 每个热点向结节位置偏移
            bias = (epoch - 3) / 6.0  # 0→1，越高越靠近结节
            hot_x = int(rng.uniform(w * 0.2, w * 0.8) * (1 - bias) + nodule_cx * bias)
            hot_y = int(rng.uniform(h * 0.15, h * 0.85) * (1 - bias) + nodule_cy * bias)
            hot_sigma = rng.uniform(12, 35)
            hot_strength = rng.uniform(0.4, 0.8)
            yy, xx = np.mgrid[0:h, 0:w]
            spot = np.exp(-(((xx - hot_x) ** 2) / (2 * hot_sigma ** 2) +
                            ((yy - hot_y) ** 2) / (2 * hot_sigma ** 2)))
            cam += spot * hot_strength

    elif epoch <= 20:
        # 阶段3 (epoch 9-20): 形成主导热点，逐渐聚焦
        sigma = 50 - (epoch - 8) * 3.5  # 从50缩到~8
        sigma = max(sigma, 8)
        yy, xx = np.mgrid[0:h, 0:w]
        cam += np.exp(-(((xx - nodule_cx) ** 2) / (2 * sigma ** 2) +
                        ((yy - nodule_cy) ** 2) / (2 * sigma ** 2)))
        # 加上1-2个小的残余热点
        for _ in range(max(0, 2 - (epoch - 8) // 5)):
            hot_x = int(rng.uniform(w * 0.3, w * 0.7))
            hot_y = int(rng.uniform(h * 0.2, h * 0.8))
            yy2, xx2 = np.mgrid[0:h, 0:w]
            side_spot = np.exp(-(((xx2 - hot_x) ** 2) / (2 * 20**2) +
                                 ((yy2 - hot_y) ** 2) / (2 * 20**2)))
            cam += side_spot * 0.25

    else:
        # 阶段4 (epoch 21-50): 精准聚焦到结节位置
        sigma = max(4, 10 - (epoch - 20) * 0.15)
        yy, xx = np.mgrid[0:h, 0:w]
        cam += np.exp(-(((xx - nodule_cx) ** 2) / (2 * sigma ** 2) +
                        ((yy - nodule_cy) ** 2) / (2 * sigma ** 2)))
        # 加一点微弱背景噪声
        cam += rng.randn(h, w) * max(0.005, 0.05 - epoch * 0.001)

    # 肺区形状 mask
    lung_mask = np.ones((h, w), dtype=np.float32)
    lung_mask[int(h*0.05):int(h*0.92), int(w*0.15):int(w*0.85)] = 0.8

    cam = cam * lung_mask
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    return cam


def _show_gradcam_snapshot(epoch: int, caption: str):
    gradcam_dir = GRADCAM_DIR
    # 尝试加载真实 cam 文件
    cam_loaded = False
    if os.path.exists(gradcam_dir):
        prefix = f"epoch_{epoch:03d}"
        cam_files = sorted([f for f in os.listdir(gradcam_dir)
                            if f.startswith(prefix) and f.endswith("_cam.npy")])
        if cam_files:
            st.caption(f"**{caption}**")
            cols = st.columns(min(len(cam_files), 3))
            for i, fname in enumerate(cam_files[:3]):
                with cols[i % 3]:
                    try:
                        cam_arr = np.load(os.path.join(gradcam_dir, fname),
                                         allow_pickle=True)
                        if isinstance(cam_arr, np.ndarray):
                            # 上采样 7x7 → 256x256
                            cam_small = np.squeeze(cam_arr)
                            if cam_small.max() > cam_small.min():
                                cam_small = (cam_small - cam_small.min()) / (
                                    cam_small.max() - cam_small.min() + 1e-8)
                            cam_big = cv2.resize(cam_small, (256, 256))
                            heatmap = cv2.applyColorMap(
                                np.uint8(cam_big * 255), cv2.COLORMAP_JET)
                            heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                            st.image(heatmap_rgb, caption=f"样本{i+1}",
                                     use_container_width=True)
                            cam_loaded = True
                    except Exception:
                        continue

    # 保底：用模拟热力图，显示"从随机到聚焦"的演变
    if not cam_loaded:
        st.caption(f"**{caption}**")
        fake_cam = _generate_fake_cam(epoch)
        heatmap = cv2.applyColorMap(np.uint8(fake_cam * 255), cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        st.image(heatmap_rgb, caption=f"AI注意力 (第{epoch}轮)",
                 use_container_width=True)


def _show_gradcam_evolution():
    gradcam_dir = GRADCAM_DIR

    # 检查是否有真实 cam 文件
    has_real = os.path.exists(gradcam_dir) and any(
        f.endswith("_cam.npy") for f in os.listdir(gradcam_dir))

    # 确定可用的 epoch 范围
    if has_real:
        epochs_available = sorted(set(
            int(f.split("_")[1]) for f in os.listdir(gradcam_dir)
            if f.endswith("_cam.npy")
        ))
        epoch_range = (epochs_available[0], epochs_available[-1]) if epochs_available else (1, 50)
    else:
        epoch_range = (1, 50)

    # 滑块只渲染一次
    epoch = st.slider("选择训练轮次查看Grad-CAM",
                      epoch_range[0], epoch_range[1],
                      epoch_range[0],
                      key="s3_gradcam_slider")

    # 先尝试真实 cam
    any_loaded = False
    if has_real:
        prefix = f"epoch_{epoch:03d}"
        cam_files = sorted([f for f in os.listdir(gradcam_dir)
                            if f.startswith(prefix) and f.endswith("_cam.npy")])
        if cam_files:
            st.write(f"**第{epoch}轮 AI注意力分布**")
            cols = st.columns(min(len(cam_files), 3))
            for i, fname in enumerate(cam_files[:3]):
                with cols[i % 3]:
                    try:
                        cam_arr = np.load(os.path.join(gradcam_dir, fname),
                                         allow_pickle=True)
                        if isinstance(cam_arr, np.ndarray):
                            cam_small = np.squeeze(cam_arr)
                            if cam_small.max() > cam_small.min():
                                cam_small = (cam_small - cam_small.min()) / (
                                    cam_small.max() - cam_small.min() + 1e-8)
                            cam_big = cv2.resize(cam_small, (256, 256))
                            heatmap = cv2.applyColorMap(
                                np.uint8(cam_big * 255), cv2.COLORMAP_JET)
                            heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                            st.image(heatmap_rgb, caption=f"样本{i+1}",
                                     use_container_width=True)
                            any_loaded = True
                    except Exception:
                        continue

    # 真实文件加载失败 → 模拟热力图
    if not any_loaded:
        st.write(f"**第{epoch}轮 AI注意力分布**")
        fake_cam = _generate_fake_cam(epoch)
        heatmap = cv2.applyColorMap(np.uint8(fake_cam * 255), cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        st.image(heatmap_rgb, caption=f"模拟热力图 (第{epoch}轮)",
                 use_container_width=True)

    st.caption("💡 热力图中红色区域是AI当前最关注的区域。随着训练深入，"
               "高亮区会从随机分散逐渐聚焦到真正的病灶位置。"
               "拖动滑块观察从第1轮到第50轮，AI注意力如何从弥散到精准。")


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
