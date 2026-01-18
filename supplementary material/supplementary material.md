# 补充材料：基于扩散模型的视觉错觉生成系统

## 项目概述

本补充材料为我们的计算机视觉期末大作业提供了更多详细信息，该项目专注于使用扩散模型生成视觉认知错觉。该项目实现了并扩展了Diffusion Illusion框架，通过Score Distillation Sampling (SDS)优化增强了生成各种类型视觉错觉的能力。

我们项目的所有代码都开源在GitHub上：https://github.com/smallyang688/Creating-Visual-Cognitive-Illusions ，我们还搭建了炫酷的项目主页 https://smallyang688.github.io/Creating-Visual-Cognitive-Illusions_homepage/ ，包括更多示例以及交互玩法，欢迎来玩！

## 项目结构

项目代码库的组织结构如下：

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
│   │   ├── fold_target.jpg
│   │   ├── fold.ipynb
│   │   ├── words.ipynb
│   │   ├── your_name_clear.ipynb
│   │   └── your_name_target.jpg
│   ├── Cross-Domain Luminance Decoupling/ # 跨域亮度解耦错觉
│   │   └── color_hybrid.ipynb
│   ├── Differentiable Cylindrical Anamorphosis/ # 可微柱面变形错觉
│   │   ├── glass_monster.ipynb
│   │   ├── glass_your_name.ipynb
│   │   ├── monster_target.jpg
│   │   └── your_name_target.jpg
│   ├── Distance-Dependent Spectral Hybridization/ # 距离相关的频谱混合错觉
│   │   └── multiscale_hybrid.ipynb
│   ├── Image-Driven Hard Constraint Optimization/ # 图像驱动的硬约束优化
│   │   ├── QRCode.ipynb
│   │   └── target.png
│   ├── Intra-Channel Frequency Splitting/ # 通道内频域分离
│   │   └── color_channel_hybrid.ipynb
│   ├── Motion Integration Steganography/ # 运动积分隐写
│   │   ├── blur_panda.ipynb
│   │   ├── blur_rose.ipynb
│   │   ├── panda_target.jpg
│   │   └── rose_target.jpg
│   ├── Multi-Angle Moire Cryptography/ # 多角度莫尔密码学
│   │   ├── love_target.png
│   │   ├── rotation_love.ipynb
│   │   ├── rotation_triple.ipynb
│   │   ├── triple_target1.jpg
│   │   ├── triple_target2.jpg
│   │   └── triple_target3.jpg
│   ├── notebooks_from_original_author/ # 原作者的基线实现
│   │   ├── flippy_illusions_for_colab.ipynb
│   │   ├── hidden_characters_for_colab.ipynb
│   │   ├── parker_puzzle_colab.ipynb
│   │   ├── rotation_overlays_for_colab.ipynb
│   │   └── twisting_squares_colab.ipynb
│   └── Orthogonal Voxel Projection Synthesis/ # 正交体素投影合成
│       └── cube.ipynb
├── report/                            # LaTeX报告和编译文件
│   ├── report.aux
│   ├── report.fdb_latexmk
│   ├── report.fls
│   ├── report.log
│   └── report.tex
├── requirements.txt                   # Python依赖项
├── readme.md                          # 项目概述
└── supplementary material/            # 本补充材料
    └── supplementary material.md
```

## 复现指南

### 硬件要求

我们使用两个高性能计算平台进行实验：

1. **Google Colab Pro**：配备A100 GPU和高RAM配置
   - 3000次优化迭代约需10分钟
   - 适用于大多数notebook实验

2. **AutoDL云端**：RTX 4090 GPU实例
   - PyTorch 2.0.0
   - Python 3.8 (Ubuntu 20.04)
   - CUDA 11.8
   - 推荐用于大规模批量实验

### 实验复现

#### 1. Experiments文件夹（自动化流水线）

`Experiments/`文件夹包含两个主要的实验流水线，具有全面的自动化评估系统。每个子文件夹都包含详细的README文件，提供完整的环境配置和复现说明：

**A. 模糊错觉流水线** (`Experiments/blur_illusion/`)
- **环境配置**：按照`Experiments/blur_illusion/README.md`中的详细环境设置说明进行操作
- **功能特性**：自动生成运动模糊隐写错觉，并使用VLM进行评分
- **系统要求**：兼容CUDA的GPU，DashScope API密钥用于Qwen-VL模型
- **运行时间**：A100 GPU上每个实验约需6-7分钟

**B. 文字错觉流水线** (`Experiments/word_illusion/`)
- **环境配置**：按照`Experiments/word_illusion/readme.md`中的conda环境设置说明进行操作
- **功能特性**：文字翻转文字图生成与自动化VLM评分流水线
- **系统要求**：双conda环境（visual_anagrams + qwen3vl）
- **数据集**：EECS/SMS（3333个样本）和ICS/LOVE（1923个样本）实验

两个流水线都包含完整的环境配置、依赖安装和逐步复现过程文档。

#### 2. Notebooks文件夹中的单个错觉Notebook

对于`Notebooks/`文件夹下包含`.ipynb`文件的所有子文件夹，复现过程非常简单：

**基本步骤**：
1. 将`.ipynb`文件上传到Google Colab
2. 如果文件夹包含图片文件（`.jpg`、`.png`），也将它们上传到Colab
3. 按顺序运行notebook单元格
4. Notebook将自动处理依赖安装和执行

**文件夹特定说明**：

- **badcases/**：各种失败模式的调试示例
- **Cross-Domain Luminance Decoupling/**：RGB到灰度转换错觉
- **Differentiable Cylindrical Anamorphosis/**：需要目标图片进行柱面投影
- **Distance-Dependent Spectral Hybridization/**：多尺度频域操作
- **Image-Driven Hard Constraint Optimization/**：使用`target.png`进行约束优化
- **Intra-Channel Frequency Splitting/**：RGB通道级频域分离
- **Motion Integration Steganography/**：需要目标图片进行模糊编码
- **Multi-Angle Moire Cryptography/**：多个目标图片用于不同旋转角度
- **Orthogonal Voxel Projection Synthesis/**：3D体素网格优化（计算量最大）
- **notebooks_from_original_author/**：原Diffusion Illusions论文的基线实现

### 运行时间估算

- **简单错觉**（颜色混合、基本变换）：Colab Pro上5-15分钟
- **复杂错觉**（3D体素投影、柱面变形）：Colab Pro上20-45分钟
- **批量实验**（Experiments/文件夹流水线）：RTX 4090上完整复现需10-12小时

### 依赖项

所有必需的Python包都在根目录的`requirements.txt`中列出。单个notebook会自动处理额外依赖。对于实验流水线，请遵循每个`Experiments/`子文件夹中的特定环境设置指南。

### 输出和评估

- 生成的错觉会自动保存为图片文件
- 实验流水线包含自动化的VLM评分
- 结果包括视觉输出和定量评估指标
- 所有输出都与主报告中的分析兼容

### 故障排除

1. **GPU内存问题**：在notebook参数中减少批次大小或图像分辨率
2. **Colab超时**：使用Colab Pro进行长时间运行的实验
3. **依赖冲突**：按照README文件中指定的方式创建新的conda环境
4. **API速率限制**：对于VLM评估，确保有足够的API配额

详细的技术问题请参考`Experiments/`文件夹中相应README文件中的故障排除部分。

---

本补充材料提供了复现我们的视觉错觉生成结果所需的所有必要信息。自动化流水线和单个notebook的结合确保了大尺度实验的可扩展性和探索特定错觉类型的灵活性。
