import time
import random
import pandas as pd
import streamlit as st


def init_learning_state():
    if "learn_step" not in st.session_state:
        st.session_state.learn_step = 0

    if "demo_params" not in st.session_state:
        st.session_state.demo_params = {
            "参数A": round(random.uniform(-1.0, 1.0), 2),
            "参数B": round(random.uniform(-1.0, 1.0), 2),
            "参数C": round(random.uniform(-1.0, 1.0), 2),
            "参数D": round(random.uniform(-1.0, 1.0), 2),
        }

    if "demo_loss_history" not in st.session_state:
        st.session_state.demo_loss_history = [0.92]

    if "demo_acc_history" not in st.session_state:
        st.session_state.demo_acc_history = [0.28]


def reset_learning_demo():
    st.session_state.learn_step = 0
    st.session_state.demo_params = {
        "参数A": round(random.uniform(-1.0, 1.0), 2),
        "参数B": round(random.uniform(-1.0, 1.0), 2),
        "参数C": round(random.uniform(-1.0, 1.0), 2),
        "参数D": round(random.uniform(-1.0, 1.0), 2),
    }
    st.session_state.demo_loss_history = [0.92]
    st.session_state.demo_acc_history = [0.28]


def render_param_bars(params: dict):
    cols = st.columns(len(params))
    for i, (name, value) in enumerate(params.items()):
        with cols[i]:
            st.metric(name, f"{value:.2f}")
            # 用进度条模拟参数位置（映射到 0~100）
            progress_value = int((value + 1.0) / 2.0 * 100)
            progress_value = max(0, min(100, progress_value))
            st.progress(progress_value)


def simulate_param_update():
    new_params = {}
    for k, v in st.session_state.demo_params.items():
        delta = random.uniform(-0.18, 0.18)
        nv = v + delta

        # 让参数逐渐趋于“稳定”
        nv = nv * 0.85
        nv = max(-1.0, min(1.0, nv))
        new_params[k] = round(nv, 2)

    st.session_state.demo_params = new_params

    last_loss = st.session_state.demo_loss_history[-1]
    last_acc = st.session_state.demo_acc_history[-1]

    new_loss = max(0.06, last_loss - random.uniform(0.06, 0.16))
    new_acc = min(0.96, last_acc + random.uniform(0.06, 0.14))

    st.session_state.demo_loss_history.append(round(new_loss, 3))
    st.session_state.demo_acc_history.append(round(new_acc, 3))


def render_learning_stage_header():
    st.title("AI是怎么学会看CT的？")
    st.caption("用大众能看懂的方式，模拟展示 AI 模型参数是如何一步步学习出来的。")


def render_learning_demo():
    init_learning_state()
    render_learning_stage_header()

    top1, top2 = st.columns([1, 1])
    with top1:
        if st.button("重新开始学习演示", width="stretch"):
            reset_learning_demo()
            st.rerun()
    with top2:
        if st.button("下一步", width="stretch"):
            st.session_state.learn_step = min(st.session_state.learn_step + 1, 5)
            st.rerun()

    st.markdown("---")

    step = st.session_state.learn_step
    params = st.session_state.demo_params

    if step == 0:
        st.subheader("第 1 步：AI刚开始什么都不懂")
        st.write("AI 一开始并不会识别病灶，它的内部参数是随机的。")
        st.write("这些参数决定了 AI 看图时更重视哪些特征，但一开始它们没有任何经验。")

        render_param_bars(params)

        st.info("你现在看到的不是全部参数，只是几个“代表性参数”，用来模拟 AI 内部经验的初始状态。")

    elif step == 1:
        st.subheader("第 2 步：AI先猜一次")
        st.write("AI 先根据当前参数看一张 CT 图，然后做出第一次判断。")
        st.write("这时它并不可靠，更像是“凭当前经验先猜一个答案”。")

        c1, c2 = st.columns([1.1, 1])
        with c1:
            st.image("https://dummyimage.com/600x380/eeeeee/666666&text=CT%E5%9B%BE%E5%83%8F%E8%BE%93%E5%85%A5", caption="训练样本（示意）")
        with c2:
            st.metric("AI判断：正常", "48%")
            st.metric("AI判断：病变", "52%")
            st.success("真实答案：病变")

        st.warning("这一步说明：AI不是直接知道答案，而是先根据当前参数做预测。")

    elif step == 2:
        st.subheader("第 3 步：AI发现自己判断得不够准确")
        st.write("系统会把 AI 的预测结果和正确答案进行比较，算出当前的“错误程度”。")
        st.write("错误越大，说明它的内部参数越不合理。")

        loss_value = st.session_state.demo_loss_history[-1]
        st.metric("当前错误程度", f"{loss_value:.2f}")

        error_progress = int(loss_value * 100)
        st.progress(error_progress)

        st.info("你可以把“错误程度”理解成：AI这次离正确答案还有多远。")

    elif step == 3:
        st.subheader("第 4 步：AI开始微调内部参数")
        st.write("如果 AI 猜错了，系统就会轻微调整它的内部参数。")
        st.write("这样下次再看到类似图像时，它的判断就会更接近正确答案。")

        old_params = params.copy()
        simulate_param_update()
        new_params = st.session_state.demo_params

        cols = st.columns(len(new_params))
        for i, key in enumerate(new_params.keys()):
            with cols[i]:
                st.write(f"**{key}**")
                st.write(f"{old_params[key]:.2f} → {new_params[key]:.2f}")

        st.success("这一步就是“参数学习”的核心：AI根据错误反馈，不断修正自己。")

    elif step == 4:
        st.subheader("第 5 步：AI重复学习很多轮")
        st.write("这个过程不会只发生一次，而是会重复很多次。")
        st.write("每一轮，AI都会继续看图、猜答案、发现误差、调整参数。")

        # 再模拟几轮
        for _ in range(4):
            simulate_param_update()

        loss_df = pd.DataFrame({
            "学习轮数": list(range(1, len(st.session_state.demo_loss_history) + 1)),
            "错误程度": st.session_state.demo_loss_history
        })

        acc_df = pd.DataFrame({
            "学习轮数": list(range(1, len(st.session_state.demo_acc_history) + 1)),
            "判断准确率": st.session_state.demo_acc_history
        })

        c1, c2 = st.columns(2)
        with c1:
            st.write("**错误程度在下降**")
            st.line_chart(loss_df.set_index("学习轮数"))
        with c2:
            st.write("**判断准确率在上升**")
            st.line_chart(acc_df.set_index("学习轮数"))

        st.info("大众可以把它理解为：AI在反复练习，慢慢总结经验。")

    elif step == 5:
        st.subheader("第 6 步：AI学会了基本判断")
        st.write("经过很多轮训练后，AI的内部参数逐渐稳定。")
        st.write("这时它已经不是随机猜测，而是形成了较稳定的判断能力。")

        render_param_bars(st.session_state.demo_params)

        final_loss = st.session_state.demo_loss_history[-1]
        final_acc = st.session_state.demo_acc_history[-1]

        c1, c2 = st.columns(2)
        with c1:
            st.metric("最终错误程度", f"{final_loss:.2f}")
        with c2:
            st.metric("最终判断准确率", f"{final_acc:.0%}")

        st.success("现在，这组“学出来的参数”就可以被用来分析新的 CT 图像了。")
        st.markdown("### 你可以这样理解")
        st.write("AI参数不是人工一个个写进去的，而是通过大量样本训练，一步步调整出来的。")
import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import base64
import io
import cv2
import numpy as np
from PIL import Image, ImageDraw
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet18

# 可选依赖：用于网页端手绘框选
# streamlit 1.57 移除了 image_to_url，导致 streamlit-drawable-canvas 崩溃。
# 在导入 canvas 前 monkey-patch 恢复该函数。
def _patched_image_to_url(
    image,
    width: int = 0,
    clamp: bool = False,
    channels: str = "RGB",
    output_format: str = "JPEG",
    image_id: str = "",
    allow_emoji: bool = False,
):
    import base64 as _b64
    from io import BytesIO as _BytesIO
    from PIL.Image import Image as _PILImage

    if isinstance(image, _PILImage):
        buf = _BytesIO()
        fmt = "PNG" if output_format.upper() == "JPEG" else output_format.upper()
        image.save(buf, format=fmt)
        return f"data:image/{fmt.lower()};base64,{_b64.b64encode(buf.getvalue()).decode()}"
    return ""

try:
    st.elements.image.image_to_url = _patched_image_to_url
except Exception:
    pass

try:
    from streamlit_drawable_canvas import st_canvas
    HAS_CANVAS = True
except Exception:
    HAS_CANVAS = False

# =========================
# 基础配置
# =========================
# st.set_page_config is called by streamlit_app.py — do not call here to avoid React errors

APP_TITLE = "智影溯源"
IMAGE_SIZE = 224
CLASS_TO_IDX = {"Normal": 0, "Lesion": 1}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_MODEL_PATH = os.path.join("model_parameter", "resnet18_ct_lesion_normal_best.pt")
DEFAULT_CASE_ROOT = "cases"

VAL_TEST_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# =========================
# 模型构建
# =========================
def build_model() -> nn.Module:
    model = resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2)
    return model


class GradCAMHelper:
    def __init__(self, model: nn.Module):
        self.model = model
        self.feature_maps = []
        self.gradients = []
        self._register_hooks()

    def _register_hooks(self):
        target_layer = self.model.layer4[-1]

        def forward_hook(module, inputs, output):
            self.feature_maps.clear()
            self.feature_maps.append(output)

        def backward_hook(module, grad_input, grad_output):
            self.gradients.clear()
            self.gradients.append(grad_output[0])

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def generate(self, image_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        output = self.model(image_tensor)
        score = output[:, class_idx]
        score.backward(retain_graph=True)

        fmap = self.feature_maps[-1][0]
        grad = self.gradients[-1][0]
        weights = torch.mean(grad, dim=(1, 2))
        cam = torch.zeros(fmap.shape[1:], dtype=torch.float32, device=fmap.device)

        for i, w in enumerate(weights):
            cam += w * fmap[i]

        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.detach().cpu().numpy()


class GuidedBackpropHelper:
    """Guided Backpropagation: 展示模型关注的细粒度纹理/边缘特征。

    与 Grad-CAM 互补——Grad-CAM 回答"哪里重要"，
    Guided Backprop 回答"什么样的纹理/边缘重要"。
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.hook_handles = []
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def relu_hook(module, grad_input, grad_output):
            # 只传递正值梯度 (Guided Backprop 核心)
            if isinstance(grad_input, tuple):
                gradient = grad_input[0]
            else:
                gradient = grad_input
            gradient = torch.clamp(gradient, min=0)
            return (gradient,)

        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
                handle = module.register_full_backward_hook(relu_hook)
                self.hook_handles.append(handle)

    def generate(self, image_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        image_tensor = image_tensor.detach().clone().requires_grad_(True)

        self.model.zero_grad(set_to_none=True)
        output = self.model(image_tensor)
        score = output[:, class_idx]
        score.backward()

        guided_grad = image_tensor.grad[0]
        guided_grad = guided_grad.abs().max(dim=0)[0]
        guided_grad = guided_grad - guided_grad.min()
        if guided_grad.max() > 0:
            guided_grad = guided_grad / guided_grad.max()

        return guided_grad.detach().cpu().numpy()

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()


@st.cache_resource(show_spinner=False)
def load_model(model_path: str) -> Tuple[nn.Module, GradCAMHelper]:
    model = build_model().to(DEVICE)
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    gradcam = GradCAMHelper(model)
    return model, gradcam


# =========================
# 图像与病例库
# =========================
def list_case_items(case_root: str) -> List[Path]:
    root = Path(case_root)
    if not root.exists():
        return []
    items = [p for p in root.iterdir() if p.is_dir()]
    return sorted(items)


VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy"}


def find_first_image(case_dir: Path) -> Optional[Path]:
    for p in sorted(case_dir.iterdir()):
        if p.suffix.lower() in VALID_EXTS:
            return p
    return None


def _load_image_smart(path: str):
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
    npy_path = p.with_suffix(".npy")
    if npy_path.exists():
        return _load_image_smart(str(npy_path))
    return None


def load_case_meta(case_dir: Path) -> Dict:
    meta_path = case_dir / "label.json"
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    image_path = find_first_image(case_dir)
    return {
        "id": case_dir.name,
        "title": case_dir.name,
        "class": "Unknown",
        "difficulty": "unknown",
        "description": "未提供病例描述",
        "image": image_path.name if image_path else "",
        "answer_mask": ""
    }


# =========================
# 推理与可视化
# =========================
def preprocess_image(image: Image.Image) -> torch.Tensor:
    image_rgb = image.convert("RGB")
    tensor = VAL_TEST_TRANSFORM(image_rgb).unsqueeze(0).to(DEVICE)
    return tensor


def overlay_heatmap(image: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    image_rgb = image.convert("RGB")
    image_np = np.array(image_rgb)
    cam_resized = cv2.resize(cam, (image_np.shape[1], image_np.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(cam_resized * 255), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = np.clip(image_np * (1 - alpha) + heatmap * alpha, 0, 255).astype(np.uint8)
    return overlay


def cam_to_mask(cam: np.ndarray, out_size: Tuple[int, int], threshold: float) -> np.ndarray:
    w, h = out_size
    cam_resized = cv2.resize(cam, (w, h))
    mask = (cam_resized >= threshold).astype(np.uint8)
    return mask


def predict_with_model(model: nn.Module, gradcam: GradCAMHelper, image: Image.Image) -> Dict:
    image_tensor = preprocess_image(image)
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
        "summary": "使用真实模型完成分类与 Grad-CAM 可视化。"
    }


def predict_with_rule_demo(image: Image.Image) -> Dict:
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
    center_bias = np.exp(-(((xx - cx) ** 2) / (2 * sigma_x ** 2) + ((yy - cy) ** 2) / (2 * sigma_y ** 2)))

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
        "summary": "当前未加载真实 .pt 模型，正在使用规则演示模式，仅用于跑通展示流程。"
    }
def build_demo_lung_mask(image_shape):
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    center = (w // 2, int(h * 0.52))
    axes = (int(w * 0.34), int(h * 0.38))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 1, -1)
    return mask.astype(bool)


def normalize_map(att_map):
    att_map = att_map.astype(np.float32)
    att_map = att_map - att_map.min()
    if att_map.max() > 0:
        att_map = att_map / att_map.max()
    return att_map


def keep_top_components(binary_mask, top_k=2, min_area=300):
    binary_mask = binary_mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

    components = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            components.append((i, area))

    components = sorted(components, key=lambda x: x[1], reverse=True)[:top_k]

    result = np.zeros_like(binary_mask)
    for label_id, _ in components:
        result[labels == label_id] = 1

    return result.astype(bool)


def smooth_binary_mask(binary_mask, kernel_size=9):
    mask = (binary_mask.astype(np.uint8) * 255)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return (mask > 0)


def refine_heatmap(cam, lung_mask):
    cam = normalize_map(cam)
    cam[cam < 0.25] = 0  # 去掉弱响应
    cam = cam * lung_mask.astype(np.float32)
    return cam

# =========================
# 用户标注与 IoU
# =========================
def create_empty_mask(size: Tuple[int, int]) -> np.ndarray:
    w, h = size
    return np.zeros((h, w), dtype=np.uint8)


def rectangle_mask(size: Tuple[int, int], x: int, y: int, rect_w: int, rect_h: int) -> np.ndarray:
    w, h = size
    mask = np.zeros((h, w), dtype=np.uint8)
    x1 = max(0, min(w - 1, x))
    y1 = max(0, min(h - 1, y))
    x2 = max(x1 + 1, min(w, x + rect_w))
    y2 = max(y1 + 1, min(h, y + rect_h))
    mask[y1:y2, x1:x2] = 1
    return mask


def draw_mask_outline(image: Image.Image, mask: np.ndarray, color: Tuple[int, int, int]) -> Image.Image:
    image_np = np.array(image.convert("RGB"))
    mask_uint8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image_np, contours, -1, color, 2)
    return Image.fromarray(image_np)


def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    mask_a = mask_a.astype(bool)
    mask_b = mask_b.astype(bool)
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(intersection / union) if union > 0 else 0.0


def score_text(iou: float) -> str:
    if iou >= 0.7:
        return "非常接近 AI 关注区域"
    if iou >= 0.4:
        return "有一定重合，还可以继续优化"
    return "与 AI 关注区域差异较大"


# =========================
# UI 辅助
# =========================
def render_case_info(meta: Dict):   
    st.markdown("### 病例信息")
    st.write(f"**病例ID：** {meta.get('id', '-')}")
    st.write(f"**标题：** {meta.get('title', '-')}")
    st.write(f"**建议难度：** {meta.get('difficulty', '-')}")
    st.write(f"**描述：** {meta.get('description', '-')}")


def get_user_mask(image: Image.Image) -> np.ndarray:
    """获取用户圈选区域。优先画布拖框，不可用时回退滑块。"""
    w, h = image.size

    st.markdown("### 用户圈选区域")

    if HAS_CANVAS:
        st.caption("在图上直接拖拽绘制矩形框，框选你关注的区域。")
        try:
            canvas_result = st_canvas(
                fill_color="rgba(0, 255, 0, 0.12)",
                stroke_width=2,
                stroke_color="#00ff00",
                background_image=np.array(image.convert("RGB")),
                update_streamlit=True,
                height=h,
                width=w,
                drawing_mode="rect",
                key="canvas",
            )

            if canvas_result.json_data and len(canvas_result.json_data.get("objects", [])) > 0:
                obj = canvas_result.json_data["objects"][-1]
                x = int(obj.get("left", 0))
                y = int(obj.get("top", 0))
                rect_w = int(obj.get("width", 1) * obj.get("scaleX", 1))
                rect_h = int(obj.get("height", 1) * obj.get("scaleY", 1))
                return rectangle_mask((w, h), x, y, rect_w, rect_h)

            st.caption("👆 请在图上拖框，或使用下方滑块替代。")
        except Exception:
            st.caption("画布加载失败，请使用滑块模式圈选。")

    c1, c2 = st.columns(2)
    with c1:
        x = st.slider("左上角 X", 0, max(0, w - 1), int(w * 0.25))
        rect_w = st.slider("框宽度", 1, max(1, w), int(w * 0.25))
    with c2:
        y = st.slider("左上角 Y", 0, max(0, h - 1), int(h * 0.25))
        rect_h = st.slider("框高度", 1, max(1, h), int(h * 0.25))
    return rectangle_mask((w, h), x, y, rect_w, rect_h)


# =========================
# 主界面
# =========================
def main(standalone: bool = True):
    if standalone:
        st.title(APP_TITLE)
        st.caption("适合演示：内置病例库、用户圈选、AI热力图、结论输出、IoU 对比。")

        with st.sidebar:
            st.subheader("页面导航")
            page_mode = st.radio(
                "选择页面",
                ["AI学习过程演示", "CT分析演示"],
                index=0
            )
        if page_mode == "AI学习过程演示":
            render_learning_demo()
            return
    st.markdown("---")
    st.subheader("参数设置")
    case_root = st.text_input("病例库目录", value=DEFAULT_CASE_ROOT)
    data_mode = st.radio("图像来源", ["内置病例库", "手动上传"], index=0)
    model_mode = st.radio("AI 模式", ["自动判断（有模型就用模型）", "强制规则演示模式"], index=0)
    model_path = st.text_input("模型路径（.pt）", value=DEFAULT_MODEL_PATH)
    alpha = st.slider("热力图透明度", 0.1, 0.9, 0.45, 0.05)
    mask_threshold = st.slider("AI 关注区域阈值", 0.1, 0.9, 0.55, 0.05)
    st.markdown("---")
    st.write(f"当前设备：`{DEVICE}`")
    st.write(f"可直接拖框：{'是' if HAS_CANVAS else '否'}")

    image: Optional[Image.Image] = None
    meta: Dict = {}

    if data_mode == "内置病例库":
        case_items = list_case_items(case_root)
        if not case_items:
            st.warning(
                "还没有找到病例库。请在项目目录下创建 `cases/` 文件夹，例如：\n"
                "cases/case_001/image.png 和 cases/case_001/label.json"
            )
        else:
            selected_name = st.selectbox("选择病例", [p.name for p in case_items])
            selected_case = next(p for p in case_items if p.name == selected_name)
            meta = load_case_meta(selected_case)
            image_name = meta.get("image") or (find_first_image(selected_case).name if find_first_image(selected_case) else "")
            image_path = selected_case / image_name
            image = _load_image_smart(str(image_path))
            if image:
                render_case_info(meta)
            else:
                st.error(f"没有找到病例图片：{image_path}")

    else:
        uploaded = st.file_uploader("上传 CT 图片", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])
        if uploaded is not None:
            image = _load_image_smart(uploaded)
            meta = {
                "id": "upload_case",
                "title": uploaded.name,
                "difficulty": "custom",
                "description": "用户手动上传的图像"
            }
            render_case_info(meta)

    if image is None:
        st.info("先选择一个内置病例，或上传一张图片。")
        st.markdown("### 推荐目录结构")
        st.code(
            """your_project/
├─ streamlit_ct_demo_complete.py
├─ model_parameter/
│  └─ resnet18_ct_lesion_normal_best.pt
└─ cases/
   ├─ case_001/
   │  ├─ image.png
   │  └─ label.json
   └─ case_002/
      ├─ image.png
      └─ label.json
""",
            language="text"
        )
        st.stop()

    st.markdown("---")
    left, right = st.columns([1.1, 1])
    with left:
        st.image(image, caption="当前图像", width=400)
    with right:
        st.markdown("### 演示流程")
        st.write("1. 先看图，自己判断哪里可能有异常")
        st.write("2. 画一个矩形框，代表你的判断区域")
        st.write("3. 点击分析，查看 AI 结论和热力图")
        st.write("4. 观察你的框与 AI 关注区域的 IoU 分数")

    user_mask = get_user_mask(image)
    outlined_user = draw_mask_outline(image, user_mask, (0, 255, 0))
    st.image(outlined_user, caption="你的圈选结果（绿色轮廓）", width=400)

    if st.button("开始分析"):

        use_real_model = (model_mode == "自动判断（有模型就用模型）") and os.path.exists(model_path)

        with st.spinner("AI 正在分析中..."):
            if use_real_model:
                model, gradcam = load_model(model_path)
                result = predict_with_model(model, gradcam, image)
            else:
                result = predict_with_rule_demo(image)

            cam = result["cam"]
            img_shape = np.array(image).shape

            # Resize CAM to match image size (model outputs 7x7, image is e.g. 512x512)
            if cam.shape[:2] != img_shape[:2]:
                cam = cv2.resize(cam, (img_shape[1], img_shape[0]))

            # 1. 肺区限制 + 热图优化
            lung_mask = build_demo_lung_mask(img_shape)
            cam_refined = refine_heatmap(cam, lung_mask)

            # 2. CAM -> mask
            ai_mask = (cam_refined > mask_threshold).astype(np.uint8)

            # 3. 形态学优化
            kernel = np.ones((7, 7), np.uint8)
            ai_mask = cv2.morphologyEx(ai_mask, cv2.MORPH_CLOSE, kernel)
            ai_mask = cv2.morphologyEx(ai_mask, cv2.MORPH_OPEN, kernel)

            # 4. 平滑边界
            ai_mask = smooth_binary_mask(ai_mask, kernel_size=9).astype(np.uint8)

            # 5. 热力图叠加
            overlay = overlay_heatmap(image, cam_refined, alpha)

            # 6. IoU
            iou = compute_iou(user_mask, ai_mask)

            # 7. 轮廓
            ai_outline = draw_mask_outline(image, ai_mask, (255, 0, 0))

            # 8. 对比图
            compare_img = np.array(image.convert("RGB"))
            user_bool = user_mask.astype(bool)
            ai_bool = ai_mask.astype(bool)
            overlap = np.logical_and(user_bool, ai_bool)

            compare_img[user_bool] = [0, 255, 0]
            compare_img[ai_bool] = [255, 0, 0]
            compare_img[overlap] = [255, 255, 0]

        st.success("分析完成")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("预测类别", result["pred_class"])
        m2.metric("Lesion 概率", f"{result['lesion_prob']:.3f}")
        m3.metric("Normal 概率", f"{result['normal_prob']:.3f}")
        m4.metric("IoU 分数", f"{iou:.3f}")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.image(overlay, caption="AI 热力图叠加", width=350)
        with c2:
            st.image(ai_outline, caption="AI 关注区域（红色轮廓）", width=350)
        with c3:
            st.image(compare_img, caption="对比图（黄=重合，绿=用户，红=AI）", width=350)

        st.markdown("### AI 结论")
        if result["pred_class"] == "Lesion":
            st.write(
                f"AI判断：该图像/区域存在 **异常可能**，当前病灶倾向概率为 **{result['lesion_prob']:.2%}**。"
            )
        else:
            st.write(
                f"AI判断：该图像当前更接近 **正常组织**，Normal 概率为 **{result['normal_prob']:.2%}**。"
            )

        st.markdown("### 过程解释")
        st.write(result["summary"])
        st.write("红色越明显，表示模型或规则在给出结论时越关注该区域。")
        st.write(f"你的圈选与 AI 关注区域的重合度为 **{iou:.3f}**，评价：**{score_text(iou)}**。")

        if meta.get("class") and meta.get("class") != "Unknown":
            st.markdown("### 病例库参考标签")
            st.write(f"病例库标注类别：**{meta.get('class')}**")



if __name__ == "__main__":
    main()
