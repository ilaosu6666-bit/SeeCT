---
title: 智影溯源 - AI肺结节教学平台
emoji: 🫁
colorFrom: indigo
colorTo: blue
sdk: streamlit
sdk_version: "1.28.0"
app_file: app.py
pinned: false
---

# seeCT 智影溯源 - 基于可解释AI的肺结节交互式教学平台

## 项目简介

本项目构建了一个面向医学影像教学的交互式平台，通过**游戏化体验**让用户从零开始"开发"一个肺结节诊断AI模型。

核心特色：
- **AI训练师游戏**：扮演AI研究员，经历数据准备→模型搭建→训练调参→诊断评估→临床实战五关
- **Grad-CAM可解释性**：热力图直观展示AI关注区域，理解模型决策依据
- **在线访问**：部署于 Streamlit Cloud，浏览器打开即用

## 在线访问

👉 **[https://zhiyingsuyuan.streamlit.app](https://zhiyingsuyuan.streamlit.app)**

## 本地开发

```bash
# 安装依赖
pip install -r requirements.txt

# 启动本地服务
streamlit run streamlit_app.py --server.port 8501

# 训练模型（本地GPU）
python model.py
```

## 目录结构

```
CTapp/
├─ streamlit_app.py          # 在线入口
├─ game.py                   # AI训练师游戏系统
├─ part1.py                  # CT分析演示工具
├─ model.py                  # ResNet18训练脚本
├─ data_preprocessing.py     # DICOM/.mhd预处理
├─ build_case_library.py     # 病例库构建
├─ model_parameter/          # 模型文件
├─ cases/                    # 教学病例库
└─ loss_fig/                 # 训练历史数据
```

## 技术栈

- **深度学习**: PyTorch + ResNet18 / 3D CNN
- **可解释AI**: Grad-CAM + Guided Backpropagation
- **Web框架**: Streamlit
- **部署**: Streamlit Cloud
- **数据集**: LIDC-IDRI, LUNA16

## 功能列表

### AI训练师游戏（5关）
1. **数据猎人** — 肺窗调节、切片浏览、结节标记
2. **架构师** — 选择ResNet/3D CNN、配置超参数
3. **训练大师** — 观察训练+决策点+大模型演变（核心）
4. **诊断专家** — AI辅助诊断、圈选对比、IoU评分
5. **临床实战** — 批量筛查、AI陷阱、综合评价

### CT分析演示
- 病例库浏览与选择
- Grad-CAM热力图叠加
- 用户圈选与AI关注区域IoU对比
- 手动上传CT图像分析

## License

MIT
