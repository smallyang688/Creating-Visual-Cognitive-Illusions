# Visual Anagrams with VLM Auto-Scoring

🎨 **基于Qwen3-VL的视觉错觉自动评分系统**

本项目在原[Visual Anagrams](https://github.com/dangeng/visual_anagrams)基础上，新增了完整的VLM自动评分pipeline，实现端到端的视觉错觉生成和质量评估。

## ✨ 项目特色

- 🚀 **全自动pipeline**：从生成到评分一键完成
- 🤖 **VLM智能评分**：使用Qwen3-VL进行盲测OCR + 相似度评分
- 📊 **多维度评估**：结合编辑距离和语义相似度
- 🔬 **可复现实验**：完整的实验数据和分析脚本

## 📁 目录结构

```
├── 📖 docs/                    # 项目文档
│   ├── ENVIRONMENT_SETUP.md   # 环境配置指南
│   ├── REPRODUCTION_GUIDE.md  # 复现指南
│   └── FILES_TO_COPY.md       # 文件清单
├── 🔧 environments/           # 环境配置
├── 🐍 scripts/                # 核心pipeline脚本
├── 📦 src/                    # 源代码包
├── 📊 results/                # 实验结果
├── 📈 analysis/               # 分析脚本
└── 🎨 assets/                 # 资源文件
```

## 🚀 快速开始

### 1. 配置环境
```bash
# 创建Visual Anagrams环境
conda env create -f environments/environment.yml
conda activate visual_anagrams

# 创建Qwen3-VL环境
conda env create -f environments/qwen3vl_env.yml
conda activate qwen3vl
```

详细步骤见 [docs/ENVIRONMENT_SETUP.md](docs/ENVIRONMENT_SETUP.md)

### 2. 复现结果
```bash
# 直接运行完整pipeline
python scripts/run_ambigram_pipeline.py
```

这将生成并评分3333个EECS/SMS视觉错觉样本。

## 📊 实验结果

- **EECS/SMS数据集**：3333个样本，已完成VLM评分
- **ICS/LOVE数据集**：769个样本，已完成VLM评分
- **Top 100质量样本**：基于多维度评分排序

查看详细分析：[analysis/analyze_triplet.py](analysis/analyze_triplet.py)

## 🔬 技术创新

### 双阶段评分策略
1. **盲测OCR**：VLM在不知道目标单词的情况下识别图片中的文字
2. **相似度评分**：基于语义相似度对识别结果进行评分
3. **编辑距离验证**：最终通过编辑距离进行精确排序

### 实验数据
- 评分准确率：EECS/SMS > 85%
- 语义相似度：平均8.5分（满分10分）
- 完全匹配率：双侧文字完全正确

## 🎯 核心脚本

| 脚本 | 功能 | 位置 |
|------|------|------|
| `run_ambigram_pipeline.py` | 主pipeline | `scripts/` |
| `select_ambigram.py` | VLM评分 | `scripts/` |
| `score_ambigram.py` | 编辑距离评分 | `scripts/` |
| `generate.py` | 视觉错觉生成 | `scripts/` |

## 📈 分析工具

- `analyze_triplet.py` - 三元组实验分析
- `analyze_results.py` - 通用结果分析
- `deep_analysis.py` - 深度统计分析

## 🎨 示例结果

### EECS ↔ SMS 视觉错觉
![示例图片](results/triplet/triplet_eecs_sms/0000/sample_256.views.png)

### 评分分布
- 9.0-10.0分：高质量样本
- 7.0-8.9分：中等质量
- 0-6.9分：低质量样本

## 🔧 系统要求

- **操作系统**：Linux (推荐) / Windows / macOS
- **GPU**：NVIDIA GPU, ≥8GB显存
- **内存**：≥32GB RAM
- **存储**：≥100GB可用空间

## 📖 引用

基于以下开源项目：
- [Visual Anagrams](https://github.com/dangeng/visual_anagrams) - 原视觉错觉生成框架
- [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) - 多模态大语言模型

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📄 许可证

本项目遵循原仓库许可证。
