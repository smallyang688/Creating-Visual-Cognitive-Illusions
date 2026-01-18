# 环境配置指南 - 从零开始复现Visual Anagrams + VLM评分结果

本文档说明如何从零开始配置环境来复现本项目的视觉错觉生成和VLM自动评分结果。

## 系统要求

- Linux操作系统（原项目在Linux上开发）
- NVIDIA GPU（推荐至少8GB显存）
- 至少64GB RAM（用于模型加载）
- 至少500GB存储空间（用于模型和生成结果）

## 环境配置步骤

### 1. 克隆仓库并安装依赖

```bash
git clone [你的仓库地址]
cd [你的仓库目录]
```

### 2. 创建Visual Anagrams环境（用于生成视觉错觉）

```bash
# 创建conda环境
conda env create -f environment.yml

# 激活环境
conda activate visual_anagrams

# 安装项目包
pip install -e .
```

### 3. 配置Hugging Face访问权限

本项目使用DeepFloyd IF模型，需要Hugging Face账户和访问权限：

1. 访问 [Hugging Face](https://huggingface.co/join) 创建账户
2. 登录账户后访问 [DeepFloyd/IF-I-XL-v1.0](https://huggingface.co/DeepFloyd/IF-I-XL-v1.0)
3. 点击"Accept repository"接受使用条款

### 4. 登录Hugging Face

```bash
# 在visual_anagrams环境中运行
python huggingface_login.py
# 按照提示输入Hugging Face访问token
```

### 5. 创建Qwen3-VL环境（用于VLM自动评分）

```bash
# 创建新的conda环境用于VLM评分
conda env create -f qwen3vl_env.yml

# 激活环境
conda activate qwen3vl
```

### 6. 下载模型

首次运行时会自动下载所需的模型：
- DeepFloyd IF系列模型（约20GB）
- Qwen3-VL-4B-Instruct模型（约8GB）

## 环境验证

### 测试Visual Anagrams环境

```bash
conda activate visual_anagrams
python generate.py --help
```

### 测试Qwen3-VL环境

```bash
conda activate qwen3vl
python select_ambigram.py --help
```

## 常见问题

### 1. CUDA版本不兼容

如果遇到CUDA版本问题，检查你的CUDA版本：

```bash
nvidia-smi
nvcc --version
```

确保安装的PyTorch版本与你的CUDA版本兼容。

### 2. 内存不足

如果显存不足，可以在`run_ambigram_pipeline.py`中调整参数：
- 减小`QWEN_MAX_GPU_MEM`
- 使用`--cpu_offload`选项
- 使用`--load_in_4bit`进行量化

### 3. Hugging Face下载缓慢

如果模型下载缓慢，可以：
- 使用代理
- 使用hf-mirror等镜像源
- 设置环境变量：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## 环境路径配置

在`run_ambigram_pipeline.py`中，需要根据你的系统配置修改Python路径：

```python
# 修改为你的conda环境路径
GENERATE_PYTHON = "/path/to/your/conda/envs/visual_anagrams/bin/python"
QWEN_PYTHON = "/path/to/your/conda/envs/qwen3vl/bin/python"
```

## 下一步

配置完成后，参考`REPRODUCTION_GUIDE.md`开始复现结果。
