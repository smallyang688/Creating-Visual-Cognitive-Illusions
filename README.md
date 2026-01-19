# 🎨 北京大学2025秋-计算机视觉大作业-视觉错觉
# PKU2025Fall-Computer Vision-Final Project-Creating Visual Cognitive Illusions

[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-blue.svg)](https://github.com/smallyang688/Creating-Visual-Cognitive-Illusions)
[![PDF Report](https://img.shields.io/badge/Report-PDF-red.svg)](https://github.com/smallyang688/Creating-Visual-Cognitive-Illusions/blob/master/report/report.pdf)
[![Project Page](https://img.shields.io/badge/Project-Page-green.svg)](https://smallyang688.github.io/Creating-Visual-Cognitive-Illusions_homepage/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

本项目基于[Diffusion-Illusions](https://github.com/RyannDaGreat/Diffusion-Illusions)。我们首先复现了作者实现的三种视觉错觉的代码，并总结和抽象出了一个全新的框架**Unified Multi-View Optimization Framework**，并在此框架的基础上实现了八种新的视觉错觉：

1. **Cross-Domain Luminance Decoupling** (跨域亮度解耦错觉)
2. **Differentiable Cylindrical Anamorphosis** (可微柱面变形错觉)
3. **Distance-Dependent Spectral Hybridization** (距离相关的频谱混合错觉)
4. **Image-Driven Hard Constraint Optimization** (图像驱动的硬约束优化)
5. **Intra-Channel Frequency Splitting** (通道内频域分离)
6. **Motion Integration Steganography** (运动积分隐写)
7. **Multi-Angle Moire Cryptography** (多角度莫尔密码学)
8. **Orthogonal Voxel Projection Synthesis** (正交体素投影合成)

我们将这八种创新的代码打包成了八个ipynb文件，放在了Notebooks文件夹里，直接在Colab上运行即可复现我们的结果。（详细的复现说明可见supplementary material/supplementary material.md）

我们还设计了一个全自动的流水线，可以批量生成视觉错觉并且由VLM自动为生成的图片打分。以此可以评价生成视觉错觉的算法的效果，也可以便于筛选出效果最好的图片。在本次大作业中，我们使用的VLM是Qwen3-VL-4B-Instruct，对一种文字的Illusion和一种图片的Illusion进行了评估。这部分所有的代码都在Experiments文件夹里，详细的复现说明可见/Experiments/blur_illusion/README.md和/Experiments/word_illusion/README.md

关于我们的创新点进一步解释和分析可参考我们的论文/report/report，关于更多可视化结果和交互游戏可以访问我们的项目homepage：[https://smallyang688.github.io/Creating-Visual-Cognitive-Illusions_homepage/](https://smallyang688.github.io/Creating-Visual-Cognitive-Illusions_homepage/)

## 项目结构

```
Creating-Visual-Cognitive-Illusion/
├── Experiments/                       # 主要的实验流水线，具有自动化评估
│   ├── blur_illusion/                 # 运动模糊隐写实验
│   │   ├── auto_closed_loop.py
│   │   ├── closed_loop_results/       # 生成结果（200+ PNG文件）
│   │   ├── README.md                  # 详细的环境配置和复现指南
│   │   ├── requirements.txt
│   │   ├── source/                    # 核心实现模块
│   │   ├── summary.json
│   │   └── ...
│   └── word_illusion/                 # 文字翻转文字图实验
│       ├── analysis/                  # 分析和评估脚本
│       ├── docs/                      # 文档和设置指南
│       ├── environments/              # Conda环境配置
│       ├── LICENSE
│       ├── notebooks/                 # Colab兼容的notebook
│       ├── readme.md                  # 主要的复现指南
│       ├── results/                   # 实验结果和数据集
│       ├── scripts/                   # 流水线执行脚本
│       ├── setup.py
│       └── src/                       # 源代码包
├── Notebooks/                         # 各种错觉类型的notebook实现
│   ├── badcases/                      # 失败测试案例和调试示例
│   ├── Cross-Domain Luminance Decoupling/ # 跨域亮度解耦错觉
│   ├── Differentiable Cylindrical Anamorphosis/ # 可微柱面变形错觉
│   ├── Distance-Dependent Spectral Hybridization/ # 距离相关的频谱混合错觉
│   ├── Image-Driven Hard Constraint Optimization/ # 图像驱动的硬约束优化
│   ├── Intra-Channel Frequency Splitting/ # 通道内频域分离
│   ├── Motion Integration Steganography/ # 运动积分隐写
│   ├── Multi-Angle Moire Cryptography/ # 多角度莫尔密码学
│   ├── notebooks_from_original_author/ # 原作者的基线实现
│   └── Orthogonal Voxel Projection Synthesis/ # 正交体素投影合成
├── report/                            # 项目报告文件
│   ├── report.pdf                     # 编译完成的PDF报告
│   └── report_source/                 # LaTeX源码和资源文件
├── requirements.txt                   # Python依赖项
├── readme.md                          # 项目概述
└── supplementary material/            # 补充材料
    └── supplementary material.md
```

## 🚀 快速开始

### 环境要求

- **Python**: 3.8+
- **GPU**: NVIDIA GPU (推荐RTX 30系列或更高)
- **内存**: ≥16GB RAM
- **平台**: Google Colab Pro / AutoDL RTX 4090

### 安装依赖

```bash
# 克隆仓库
git clone https://github.com/smallyang688/Creating-Visual-Cognitive-Illusions.git
cd Creating-Visual-Cognitive-Illusions

# 安装依赖
pip install -r requirements.txt
```

### 运行实验

#### 1. 单个错觉生成 (推荐新手)
打开 `Notebooks/` 中对应的notebook，在Google Colab中运行即可。

#### 2. 文字翻转错觉实验
```bash
cd Experiments/word_illusion
python scripts/run_ambigram_pipeline.py
```

#### 3. 运动模糊隐写实验
```bash
cd Experiments/blur_illusion
python auto_closed_loop.py
```

## 📊 实验结果

### 文字翻转错觉 (Text Flip Ambigrams)
- **数据集规模**: EECS/SMS (3333样本), ICS/LOVE (1923样本)
- **评估方法**: VLM盲测OCR + 语义相似度评分
- **质量指标**: 编辑距离评分 + 语义相似度
- **最佳性能**: 完全匹配率 > 85%

### 运动模糊隐写 (Motion Blur Steganography)
- **实验规模**: 100组独立实验
- **模糊强度**: 45度角35x35像素核
- **平均得分**: 6.8分 (满分10分)
- **最高得分**: 10分 (完美隐藏)

## 🎨 错觉类型展示

| 错觉类型 | 技术原理 | Colab链接 |
|---------|---------|----------|
| 跨域亮度解耦 | RGB↔灰度域转换 | [运行](https://colab.research.google.com/) |
| 可微柱面变形 | 可微几何投影 | [运行](https://colab.research.google.com/) |
| 距离相关频谱混合 | 频域金字塔融合 | [运行](https://colab.research.google.com/) |
| 图像驱动硬约束优化 | QR码嵌入优化 | [运行](https://colab.research.google.com/) |
| 通道内频域分离 | RGB通道频域操作 | [运行](https://colab.research.google.com/) |
| 运动积分隐写 | 模糊核卷积编码 | [运行](https://colab.research.google.com/) |
| 多角度莫尔密码学 | 多角度干涉图案 | [运行](https://colab.research.google.com/) |
| 正交体素投影合成 | 3D网格正交投影 | [运行](https://colab.research.google.com/) |

## 📈 性能基准

### 计算资源使用
- **简单错觉**: 5-15分钟 (Colab Pro)
- **复杂错觉**: 20-45分钟 (Colab Pro)
- **批量实验**: 10-12小时 (RTX 4090)

### 质量评估
- **主观评分**: 人工评估 (0-10分)
- **客观评分**: VLM模型评估 (Qwen-VL)
- **自动化评估**: 编辑距离 + 语义相似度

## 🔧 技术栈

- **深度学习框架**: PyTorch, Diffusers (用于扩散模型)
- **图像处理**: OpenCV, Matplotlib, imageio
- **机器学习**: Transformers, einops
- **数据处理**: Pandas, NumPy, SciPy
- **可视化**: Matplotlib, tensorboardX
- **开发工具**: icecream (调试), tqdm (进度条), rich (美化输出)
- **VLM评估**: Qwen3-VL-4B-Instruct (用于自动评分)

## 贡献与支持

⭐ **如果这个项目对你有帮助，请给我们一个star！** ⭐

本项目由北京大学计算机视觉课程学生完成，致力于探索视觉认知错觉的生成与应用。

Made with ❤️ for Computer Vision</content>
</xai:function_call">撰写一个完整的README.md文件
