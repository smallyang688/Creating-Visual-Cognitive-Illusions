import argparse
import json
from pathlib import Path

from PIL import Image
import torch

# ----------------------------------------------------------------------
# 保持你原始的 Import，绝对不改
# ----------------------------------------------------------------------
try:
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
except Exception as exc:
    raise RuntimeError(
        "Missing transformers with Qwen3-VL support. "
    ) from exc


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score ambigram images with Qwen3-VL (Blind OCR + LLM Match)."
    )
    # --- 原有参数 ---
    parser.add_argument("--input_dir", default="results/ambigram_eecs_sms_cursive")
    parser.add_argument("--pattern", default="sample_256.views.png")
    parser.add_argument("--mode", choices=["views", "rotate"], default="views")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--output_json", default="results/ambigram_eecs_sms_cursive_scores.json")
    
    # --- 评分用参数 ---
    parser.add_argument("--match_left", default="EECS", help="Target word (Hidden from OCR stage)")
    parser.add_argument("--match_right", default="SMS", help="Target word (Hidden from OCR stage)")
    parser.add_argument("--copy_dir", default=None)
    parser.add_argument("--stream", action="store_true")
    
    # --- 兼容性参数 (你原本命令里有 blind_prompt，这里留着防止报错，虽然逻辑已经全盲了) ---
    parser.add_argument("--blind_prompt", action="store_true")
    parser.add_argument("--only_matches", action="store_true")

    # --- 硬件加载参数 (保持不变) ---
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--flash_attn", action="store_true")
    parser.add_argument("--gpu_only", action="store_true")
    parser.add_argument("--cpu_offload", action="store_true")
    parser.add_argument("--max_gpu_mem", default="14GiB")
    parser.add_argument("--max_cpu_mem", default="64GiB")
    parser.add_argument("--offload_dir", default=None)
    parser.add_argument("--dtype", choices=["auto", "fp16", "bf16"], default="auto")
    
    return parser.parse_args()


# ----------------------------------------------------------------------
# load_model 完全保持你最开始发给我的样子
# ----------------------------------------------------------------------
def load_model(
    model_id,
    load_in_4bit=False,
    flash_attn=False,
    cpu_offload=False,
    max_gpu_mem="14GiB",
    max_cpu_mem="64GiB",
    device_map="auto",
    offload_dir=None,
    dtype="auto",
):
    model_kwargs = {
        "device_map": device_map,
        "dtype": "auto" if dtype == "auto" else (torch.float16 if dtype == "fp16" else torch.bfloat16),
    }
    if flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    if load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
        except Exception as exc:
            raise RuntimeError(
                "bitsandbytes not available; install it or drop --load_in_4bit"
            ) from exc
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_enable_fp32_cpu_offload=cpu_offload,
        )
        model_kwargs["quantization_config"] = quant_config
        if cpu_offload:
            model_kwargs["max_memory"] = {0: max_gpu_mem, "cpu": max_cpu_mem}
            if offload_dir:
                model_kwargs["offload_folder"] = offload_dir
                model_kwargs["offload_state_dict"] = True
    elif cpu_offload:
        model_kwargs["max_memory"] = {0: max_gpu_mem, "cpu": max_cpu_mem}
        if offload_dir:
            model_kwargs["offload_folder"] = offload_dir
            model_kwargs["offload_state_dict"] = True

    model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, **model_kwargs)
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


def collect_images(root, pattern, limit):
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Input directory not found: {root}")
    paths = sorted(root.rglob(pattern))
    if limit and limit > 0:
        paths = paths[:limit]
    return paths


def parse_json_block(text):
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except Exception:
        return None


# ==============================================================================
# 阶段 1：盲测 OCR (Blind OCR) - 只看图，不知道目标
# ==============================================================================
def run_blind_ocr(model, processor, image, rotated, mode, max_new_tokens):
    """
    不给 AI 任何关于 'EECS' 或 'SMS' 的提示。
    只让它识别文字。
    """
    if mode == "views":
        prompt = (
            "Look at the image which has a left panel and a right panel. "
            "Identify the text written in the left panel and the text in the right panel. "
            "Return JSON only: {\"left\": \"...\", \"right\": \"...\"}"
        )
        content = [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]
    else:
        prompt = (
            "Look at Image A (upright) and Image B (rotated). "
            "Identify the text in each. "
            "Return JSON only: {\"upright\": \"...\", \"rotated\": \"...\"}"
        )
        content = [
            {"type": "image", "image": image},
            {"type": "image", "image": rotated},
            {"type": "text", "text": prompt},
        ]
        
    messages = [{"role": "user", "content": content}]
    
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    parsed = parse_json_block(text)
    return text, parsed


# ==============================================================================
# 阶段 2：文本相似度评分 (LLM Scoring) - 只看字，不看图
# ==============================================================================
def run_llm_similarity(model, processor, found_text, target_text):
    """
    这是一个纯文本任务。让 AI 判断 OCR 结果和目标有多像。
    """
    if not found_text:
        return 0.0

    # 这里的 prompt 引导 AI 进行“模糊匹配”打分
    prompt = (
        f"Compare the text '{found_text}' with the target word '{target_text}'.\n"
        "Rate their visual or structural similarity on a scale from 0 to 10.\n"
        "- 10: Perfect match.\n"
        "- 8-9: Very close (e.g. '5' looks like 'S', or 'EES' vs 'EECS').\n"
        "- 5-7: Somewhat recognizable.\n"
        "- 0-4: Completely different.\n"
        "Return JSON only: {\"score\": 0}"
    )
    
    # 这里不需要 image，只传 text
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    
    generated_ids = model.generate(**inputs, max_new_tokens=64) # 短一点即可
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    parsed = parse_json_block(text)
    if parsed and "score" in parsed:
        try:
            return float(parsed["score"])
        except:
            return 0.0
    return 0.0


def main():
    args = parse_args()
    paths = collect_images(args.input_dir, args.pattern, args.limit)
    if not paths:
        raise SystemExit(f"No images found under {args.input_dir} with pattern {args.pattern}")

    cpu_offload = args.cpu_offload and not args.gpu_only
    device_map = "cuda:0" if args.gpu_only else "auto"
    
    print(f"Loading model: {args.model}...")
    model, processor = load_model(
        args.model,
        load_in_4bit=args.load_in_4bit,
        flash_attn=args.flash_attn,
        cpu_offload=cpu_offload,
        max_gpu_mem=args.max_gpu_mem,
        max_cpu_mem=args.max_cpu_mem,
        device_map=device_map,
        offload_dir=args.offload_dir,
        dtype=args.dtype,
    )

    results = []
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.stream:
        output_path.write_text("[]")
        
    print(f"Start Processing. Targets: Left='{args.match_left}', Right='{args.match_right}'")
    print("Strategy: 1. Blind OCR (Vision) -> 2. Similarity Check (Text)")

    for path in paths:
        try:
            image = Image.open(path).convert("RGB")
            rotated = image.rotate(180, expand=True)
            
            # 1. 盲识别 (Vision)
            raw_ocr_text, ocr_json = run_blind_ocr(
                model, processor, image, rotated, args.mode, args.max_new_tokens
            )
            
            # 提取识别结果
            left_found = ""
            right_found = ""
            if ocr_json:
                if args.mode == "views":
                    left_found = ocr_json.get("left", "")
                    right_found = ocr_json.get("right", "")
                else:
                    left_found = ocr_json.get("upright", "")
                    right_found = ocr_json.get("rotated", "")
            
            # 2. 相似度打分 (Text Only) - 两次独立的推理
            # 如果 OCR 是空的，直接给 0 分，省一次推理
            score_left = 0.0
            score_right = 0.0
            
            if left_found:
                score_left = run_llm_similarity(model, processor, left_found, args.match_left)
            
            if right_found:
                score_right = run_llm_similarity(model, processor, right_found, args.match_right)
            
            final_score = (score_left + score_right) / 2
            
            record = {
                "path": str(path),
                "score": final_score,
                "left_found": left_found,
                "right_found": right_found,
                "score_left": score_left,
                "score_right": score_right,
                "raw_ocr": raw_ocr_text
            }
            results.append(record)

            if args.stream:
                print(f"{final_score:.1f}\t{path}\t'{left_found}' / '{right_found}'")
                output_path.write_text(json.dumps(results, ensure_ascii=True, indent=2))
                
                # 复制逻辑
                if args.copy_dir and final_score >= 8.5: # 门槛可以自己调
                    copy_dir = Path(args.copy_dir)
                    copy_dir.mkdir(parents=True, exist_ok=True)
                    dst = copy_dir / path.name
                    dst.write_bytes(path.read_bytes())
        
        except Exception as e:
            print(f"Error processing {path}: {e}")
            continue

    # 最终排序
    results_sorted = sorted(results, key=lambda x: x["score"], reverse=True)
    
    print("\nTop candidates:")
    for item in results_sorted[: args.top_k]:
        print(f"{item['score']}\t{item['path']}\t{item['left_found']} / {item['right_found']}")

    output_path.write_text(json.dumps(results_sorted, ensure_ascii=True, indent=2))
    print(f"Saved full results to {output_path}")


if __name__ == "__main__":
    main()