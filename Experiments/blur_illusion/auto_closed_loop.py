import os, sys, json, torch, random, warnings, time, re, shutil
import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image
from openai import OpenAI
import torch.nn.functional as F

# === 环境配置 ===
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
sys.path.insert(0, os.getcwd())

try:
    import source.stable_diffusion as sd
    from source.learnable_textures import LearnableImageFourier
    from source.stable_diffusion_labels import NegativeLabel
    print("✅ 核心模块加载成功")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

# === API 配置 ===
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# === 核心工具 ===
def get_blur_kernel(size, angle, device):
    k = torch.zeros((size, size), device=device)
    k[size // 2, :] = 1.0
    k = TF.rotate(k.unsqueeze(0).unsqueeze(0), angle).squeeze()
    k = k / k.sum()
    return k.view(1, 1, size, size).repeat(3, 1, 1, 1)

def extract_json_from_text(text):
    try: return json.loads(text)
    except:
        match = re.search(r"```json\s*(.*?)```", text, re.DOTALL)
        if match: return json.loads(match.group(1))
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match: return json.loads(match.group(0))
    return None

def vlm_verify_and_score(img_path, target_word):
    import base64
    with open(img_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode('utf-8')
    
    prompt = f"""
    You are an expert in evaluating optical illusions. 
    The user is trying to hide the object "{target_word}" in a blurry image.
    
    Please analyze the image and output a JSON object with the following keys:
    - "detected_object": What object do you clearly see in this image? (Answer with a single noun or short phrase)
    - "score": An integer from 0 to 10. (10 means the object "{target_word}" is extremely clear. 0 means invisible.)
    - "reason": A brief explanation.
    
    Output ONLY the JSON object.
    """

    try:
        res = client.chat.completions.create(
            model="qwen-vl-plus",
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
            ]}]
        )
        raw = res.choices[0].message.content.strip()
        return extract_json_from_text(raw) or {"score": 0, "detected_object": "parse_error"}, raw
    except Exception as e:
        return {"score": 0, "detected_object": "api_error", "reason": str(e)}, str(e)

def main():
    device = torch.device("cuda")
    print("🚀 加载模型...")
    model_sd = sd.StableDiffusion(device, "CompVis/stable-diffusion-v1-4")
    
    # === 1. 自动清理逻辑 (按照你的要求新增) ===
    output_dir = "closed_loop_results"
    
    print(f"\n🧹 正在检查输出目录: {output_dir} ...")
    if os.path.exists(output_dir):
        print(f"⚠️ 发现旧目录，正在强制删除所有旧图片和JSON...")
        shutil.rmtree(output_dir) # 递归删除整个文件夹
        print("✅ 旧数据已清除！")
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"✅ 已创建全新的空目录: {output_dir}\n")

    # === 参数配置 ===
    BLUR_STRENGTH = 35       
    GUIDANCE_STRENGTH = 4000 
    TOTAL_STEPS = 2500       
    WARMUP_STEPS = 500       
    
    style_prompt = "Children's crayon drawing of a city street, colorful buildings, red houses, green trees, textured paper, rough sketch, messy cute style, vibrant colors"
    neg_prompt = "blur, smooth, photo, realistic, 3d render, low quality, flat color, grayscale"

    # 目标池
    objects_pool = [
        "teapot", "banana", "heart", "umbrella", "apple", 
        "star", "flower", "car", "tree", "cat", 
        "dog", "book", "chair", "shoe", "cup", 
        "bottle", "key", "lamp", "sun", "moon"
    ]
    
    TOTAL_EXPERIMENTS = 100
    log_file = os.path.join(output_dir, "evaluation_log_100.json")
    evaluation_log = []

    print(f"🎯 计划运行 {TOTAL_EXPERIMENTS} 组全新实验...")

    for i in range(TOTAL_EXPERIMENTS):
        target = random.choice(objects_pool)
        print(f"\n=== [实验 {i+1}/{TOTAL_EXPERIMENTS}] 目标: {target} ===")
        start_time = time.time()

        # 1. 生成引导图
        with torch.no_grad():
            raw = model_sd.prompt_to_img(f"A colorful {target} on white background")
            target_t = torch.from_numpy(raw).permute(2,0,1).float() if isinstance(raw, np.ndarray) else raw.clone()
            target_t = F.interpolate(target_t.unsqueeze(0), size=(256, 256)).to(device)
            if target_t.max() > 1.1: target_t /= 255.0

        # 2. 初始化
        image_maker = LearnableImageFourier(height=256, width=256).to(device)
        optim = torch.optim.SGD(image_maker.parameters(), lr=0.001) 
        kernel = get_blur_kernel(BLUR_STRENGTH, 45, device)
        label = NegativeLabel(style_prompt, neg_prompt)

        # 3. 训练循环
        for step in range(TOTAL_STEPS):
            img = image_maker()
            img_input = img.unsqueeze(0)

            # A. Surface Loss
            try:
                model_sd.train_step(label.embedding, img_input, guidance_scale=100)
            except RuntimeError: optim.zero_grad(); pass

            # B. Secret Loss (Warm-up)
            if step < WARMUP_STEPS:
                current_strength = 0
            else:
                progress = (step - WARMUP_STEPS) / (TOTAL_STEPS - WARMUP_STEPS)
                current_strength = GUIDANCE_STRENGTH * min(1.0, progress * 1.5)

            if current_strength > 0:
                img_clean = image_maker().unsqueeze(0)
                img_blur = F.conv2d(img_clean, kernel, padding=kernel.shape[2]//2, groups=3)
                loss_secret = torch.mean((img_blur - target_t)**2) * current_strength
                loss_secret.backward()
            
            optim.step()
            optim.zero_grad()
            
            if (step+1) % 1000 == 0:
                print(f"  Step {step+1}/{TOTAL_STEPS}")

        # 4. 保存与打分
        h_path = f"{output_dir}/exp_{i:03d}_{target}_shaken.png"
        s_path = f"{output_dir}/exp_{i:03d}_{target}_static.png"
        
        TF.to_pil_image(img_blur.squeeze().detach().cpu().clamp(0,1)).save(h_path)
        TF.to_pil_image(img.detach().cpu().clamp(0,1)).save(s_path)
        
        print(f"🤖 Qwen 评分中...")
        score_data, raw_resp = vlm_verify_and_score(h_path, target)
        
        is_match = target.lower() in score_data.get("detected_object", "").lower()
        
        log_entry = {
            "id": i,
            "target": target,
            "score": score_data.get("score", 0),
            "prediction": score_data.get("detected_object", "unknown"),
            "is_match": is_match,
            "time_cost": round(time.time() - start_time, 2),
            "paths": {"static": s_path, "shaken": h_path},
            "raw_response": raw_resp
        }
        
        evaluation_log.append(log_entry)
        print(f"📊 结果: {log_entry['score']}分 | 识别: {log_entry['prediction']} | 匹配: {'✅' if is_match else '❌'}")

        # 5. 实时保存
        with open(log_file, "w") as f:
            json.dump(evaluation_log, f, indent=4)

    print(f"\n✅ 100组实验全部完成！数据已保存至: {log_file}")

if __name__ == "__main__":
    main()