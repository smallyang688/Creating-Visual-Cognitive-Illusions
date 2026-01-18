# 复现指南 - 生成Visual Anagrams + VLM评分结果

本文档说明如何从零开始复现本项目的视觉错觉生成和VLM自动评分结果。

## 目录结构

确保你的仓库包含以下关键文件：

```
your-repo/
├── environment.yml              # Visual Anagrams环境配置
├── qwen3vl_env.yml             # Qwen3-VL环境配置
├── run_ambigram_pipeline.py    # 主pipeline脚本
├── select_ambigram.py          # VLM评分脚本
├── score_ambigram.py           # 编辑距离评分脚本
├── generate.py                 # 视觉错觉生成脚本（来自原仓库）
├── visual_anagrams/            # 原项目核心代码
├── results/                    # 生成的结果文件夹
└── ENVIRONMENT_SETUP.md        # 环境配置指南
```

## 复现步骤

### 步骤1：环境配置

按照`ENVIRONMENT_SETUP.md`中的说明配置两个conda环境：
- `visual_anagrams`：用于生成视觉错觉
- `qwen3vl`：用于VLM自动评分

### 步骤2：修改环境路径

在`run_ambigram_pipeline.py`中修改Python路径为你的系统路径：

```python
# 修改为你的实际conda环境路径
GENERATE_PYTHON = "/path/to/your/conda/envs/visual_anagrams/bin/python"
QWEN_PYTHON = "/path/to/your/conda/envs/qwen3vl/bin/python"
```

### 步骤3：运行完整pipeline

```bash
# 确保你在项目根目录
cd your-repo-directory

# 直接运行主pipeline（会自动调用两个环境）
python run_ambigram_pipeline.py
```

这个脚本会：
1. 使用`visual_anagrams`环境生成视觉错觉图片
2. 使用`qwen3vl`环境进行VLM自动评分
3. 生成编辑距离评分
4. 复制top结果到指定文件夹

### 步骤4：验证结果

运行完成后，检查`results/`文件夹，应该包含：

```
results/
├── triplet/
│   ├── triplet_eecs_sms/          # 生成的EECS/SMS视觉错觉
│   └── triplet_ics_love/          # 生成的ICS/LOVE视觉错觉
├── triplet_eecs_sms_select.json   # VLM评分结果
├── triplet_eecs_sms_scored.json   # 编辑距离评分结果
├── triplet_eecs_sms_top100/       # Top 100结果
└── ...
```

## 参数说明

### 主要参数（在run_ambigram_pipeline.py中修改）

```python
# 生成参数
NUM_INFERENCE_STEPS = 70          # 推理步数，越高质量越好但速度越慢
GUIDANCE_SCALE = 9.5              # 引导尺度，影响生成质量
SEED = 0                          # 随机种子，保证可复现

# VLM评分参数
QWEN_MODEL = "Qwen/Qwen3-VL-4B-Instruct"  # 使用的VLM模型
QWEN_MAX_NEW_TOKENS = 128         # 最大生成token数
TOP_K = 100                       # 保留top K结果
```

### 作业配置

在`JOBS`列表中定义要生成的视觉错觉任务：

```python
JOBS = [
    {
        "name": "triplet_eecs_sms",
        "samples": 3333,                    # 生成样本数量
        "prompts": [
            "the word eecs, cursive writing",
            "the word sms, cursive writing"
        ],
        "pair": ("eecs", "sms"),            # 目标单词对
    },
    # 更多作业...
]
```

## 运行时间估计

- 生成3333个EECS/SMS样本：约24-48小时（取决于GPU性能）
- VLM评分：约2-4小时
- 编辑距离评分：几秒钟

## 中断和恢复

如果生成过程被中断，可以修改`run_ambigram_pipeline.py`中的`RESUME = True`，脚本会自动跳过已完成的作业。

## 输出文件说明

### 1. 生成的图片文件夹
- `results/triplet/[job_name]/[sample_id]/sample_256.views.png`：生成的视觉错觉图片
- `results/triplet/[job_name]/metadata.pkl`：生成参数和metadata

### 2. VLM评分结果
- `[job_name]_select.json`：VLM识别和评分结果
- 包含每个样本的OCR识别结果和相似度评分

### 3. 编辑距离评分结果
- `[job_name]_scored.json`：基于编辑距离的最终评分和排序
- `[job_name]_top100/`：复制的top 100结果

## 质量验证

运行分析脚本验证结果质量：

```bash
python analyze_triplet.py    # 分析EECS/SMS结果
python analyze_results.py    # 通用分析脚本
```

## 故障排除

### 1. 环境路径错误
确保`GENERATE_PYTHON`和`QWEN_PYTHON`指向正确的conda环境python可执行文件。

### 2. 模型下载失败
- 检查网络连接
- 验证Hugging Face token权限
- 考虑使用代理或镜像

### 3. 内存不足
- 减小批次大小
- 使用CPU offload
- 使用4-bit量化

### 4. 生成质量不佳
- 增加`NUM_INFERENCE_STEPS`
- 调整`GUIDANCE_SCALE`
- 检查prompt质量

## 扩展使用

### 生成其他单词对

在`JOBS`列表中添加新的作业：

```python
{
    "name": "triplet_your_words",
    "samples": 1000,
    "prompts": ["the word AAA, cursive writing", "the word BBB, cursive writing"],
    "pair": ("aaa", "bbb"),
}
```

### 自定义评分标准

修改`select_ambigram.py`中的评分逻辑来自定义VLM评分标准。

## 引用和致谢

本项目基于：
- [Visual Anagrams](https://github.com/dangeng/visual_anagrams) - 原视觉错觉生成代码
- [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) - 用于自动评分的VLM模型
