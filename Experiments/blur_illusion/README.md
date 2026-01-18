# 自动生成视觉错觉与VLM评估系统 (Auto Closed-Loop Visual Illusions)

本项目基于Diffusion-Illusions实现了自动化生成视觉错觉并使用VLM进行客观评估的完整系统。通过Stable Diffusion生成隐藏在模糊图像中的目标物体，并使用Qwen-VL模型进行自动评分。

## 项目特色

- 🤖 **全自动流程**: 从目标生成到VLM评估的闭环系统
- 🎯 **批量实验**: 支持运行100组独立实验
- 📊 **客观评分**: 使用VLM进行0-10分的标准化评估
- 🔄 **实时保存**: 实验过程实时保存，避免数据丢失

## 实验结果

基于100组实验的数据统计：
- 平均得分: 6.8分
- 最高得分: 10分 (完美隐藏)
- 匹配率: 87% (VLM能正确识别目标物体)
- 平均耗时: 6.3分钟/实验

## 环境要求

- Python 3.8+
- CUDA兼容GPU (推荐RTX 30系列或更高)
- 至少16GB RAM
- 阿里云DashScope API密钥 (用于Qwen-VL模型)

## 安装步骤

### 1. 克隆原仓库并复制核心文件

```bash
# 克隆Diffusion-Illusions仓库
git clone https://github.com/RyannDaGreat/Diffusion-Illusions.git
cd Diffusion-Illusions

# 复制必要的源代码文件到你的项目目录
mkdir -p source
cp -r Diffusion-Illusions/source/* source/
cp Diffusion-Illusions/requirements.txt .
cp Diffusion-Illusions/auto_closed_loop.py .
```

### 2. 安装依赖

```bash
# 安装基础依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install openai pillow numpy tqdm

# 安装项目依赖
pip install -r requirements.txt

# 安装额外的diffusers依赖
pip install diffusers transformers accelerate
```

### 3. 配置API密钥

在系统环境变量中设置阿里云API密钥：

```bash
export DASHSCOPE_API_KEY="your_api_key_here"
```

或者在Python脚本中直接设置：
```python
import os
os.environ['DASHSCOPE_API_KEY'] = "your_api_key_here"
```

> **注意**: 获取API密钥需要注册阿里云账号并开通DashScope服务。Qwen-VL-Plus模型按调用次数收费。

## 文件结构

```
your_project/
├── auto_closed_loop.py          # 主脚本
├── source/                      # 核心模块
│   ├── stable_diffusion.py      # Stable Diffusion封装
│   ├── learnable_textures.py    # 可学习纹理类
│   ├── stable_diffusion_labels.py # 标签处理
│   └── ...
├── requirements.txt             # 依赖列表
├── closed_loop_results/         # 实验结果目录 (自动生成)
│   ├── exp_000_teapot_static.png
│   ├── exp_000_teapot_shaken.png
│   ├── evaluation_log_100.json  # 评估日志
│   └── ...
└── README.md                    # 本文档
```

## 使用方法

### 基本运行

```bash
python auto_closed_loop.py
```

脚本将自动：
1. 创建`closed_loop_results`目录
2. 运行100组实验
3. 实时保存结果到JSON文件
4. 生成static(静态)和shaken(模糊)图像对

### 参数说明

脚本中的关键参数：

```python
BLUR_STRENGTH = 35       # 模糊核大小
GUIDANCE_STRENGTH = 4000 # 引导强度
TOTAL_STEPS = 2500       # 总训练步数
WARMUP_STEPS = 500       # 预热步数
```

### 目标物体池

脚本随机从以下物体中选择目标：

```python
objects_pool = [
    "teapot", "banana", "heart", "umbrella", "apple",
    "star", "flower", "car", "tree", "cat",
    "dog", "book", "chair", "shoe", "cup",
    "bottle", "key", "lamp", "sun", "moon"
]
```

## 输出结果

### 图像文件

每组实验生成两张图片：
- `exp_XXX_target_static.png`: 原始生成的错觉图像
- `exp_XXX_target_shaken.png`: 应用模糊后的效果图

### 评估日志

`evaluation_log_100.json` 包含每组实验的详细信息：

```json
{
    "id": 0,
    "target": "teapot",
    "score": 7,
    "prediction": "teapot",
    "is_match": true,
    "time_cost": 379.0,
    "paths": {
        "static": "closed_loop_results/exp_000_teapot_static.png",
        "shaken": "closed_loop_results/exp_000_teapot_shaken.png"
    },
    "raw_response": "...Qwen-VL的原始响应..."
}
```

## 技术原理

### 训练流程

1. **引导图生成**: 使用Stable Diffusion生成目标物体的彩色图像
2. **纹理初始化**: 创建可学习的Fourier域图像表示
3. **双重损失优化**:
   - Surface Loss: 使图像符合艺术风格
   - Secret Loss: 确保模糊后显现目标物体
4. **VLM评估**: 使用Qwen-VL-Plus进行客观评分

### 模糊机制

采用45度角的运动模糊核，大小为35x35像素，通过卷积实现：

```python
kernel = get_blur_kernel(BLUR_STRENGTH, 45, device)
img_blur = F.conv2d(img_clean, kernel, padding=kernel.shape[2]//2, groups=3)
```

## 注意事项

### 硬件要求
- GPU内存至少8GB
- 单个实验耗时约6-7分钟
- 100组实验总耗时约10-12小时

### API费用
- 每次VLM调用约消耗0.01-0.02元人民币
- 100组实验总费用约2-4元

### 稳定性
- 实验结果有随机性，可能因GPU/种子不同而异
- 建议在相同环境下复现以保证一致性

### 故障排除

**CUDA内存不足**:
```python
# 在脚本开头添加
torch.cuda.empty_cache()
```

**API调用失败**:
- 检查DASHSCOPE_API_KEY是否正确设置
- 确认网络连接正常
- 检查API余额是否充足

**导入错误**:
- 确保所有source/目录下的文件都已正确复制
- 检查Python路径设置

## 扩展开发

### 修改目标物体
在`objects_pool`列表中添加新的物体名称。

### 调整参数
可以修改模糊强度、训练步数等参数来调整效果。

### 自定义风格
修改`style_prompt`和`neg_prompt`来改变艺术风格。

## 引用

本项目基于以下开源项目：
- [Diffusion-Illusions](https://github.com/RyannDaGreat/Diffusion-Illusions)
- [Stable Diffusion](https://github.com/CompVis/stable-diffusion)
- [Qwen-VL](https://dashscope.aliyuncs.com/)

## 许可证

请遵循原Diffusion-Illusions项目的许可证。
