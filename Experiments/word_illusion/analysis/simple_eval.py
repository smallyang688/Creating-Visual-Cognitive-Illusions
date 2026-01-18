"""
简单的图像评估脚本，用于初步筛选生成的图像
此脚本将评估图像的清晰度、对比度等基本特征
"""
import os
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import argparse


def calculate_sharpness(image):
    """计算图像的清晰度（拉普拉斯方差）"""
    img_array = np.array(image.convert('L'))  # 转换为灰度图
    laplacian_var = np.var(img_array)
    return laplacian_var


def calculate_contrast(image):
    """计算图像对比度（标准差）"""
    img_array = np.array(image.convert('L'))
    contrast = np.std(img_array)
    return contrast


def calculate_brightness(image):
    """计算图像平均亮度"""
    img_array = np.array(image.convert('L'))
    brightness = np.mean(img_array)
    return brightness


def evaluate_image_quality(image_path):
    """评估单个图像的质量"""
    try:
        image = Image.open(image_path)
        
        sharpness = calculate_sharpness(image)
        contrast = calculate_contrast(image)
        brightness = calculate_brightness(image)
        
        # 综合评分（可以根据需要调整权重）
        # 对比度和清晰度通常更重要
        score = 0.4 * contrast + 0.6 * sharpness
        
        return {
            'sharpness': sharpness,
            'contrast': contrast,
            'brightness': brightness,
            'score': score
        }
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Simple evaluation of generated images')
    parser.add_argument('--input_dir', type=str, 
                       default='results/triplet/triplet_eecs_sms',
                       help='Directory containing generated images')
    parser.add_argument('--pattern', type=str, 
                       default='sample_256.views.png',
                       help='Pattern for selecting images')
    parser.add_argument('--top_k', type=int, default=100,
                       help='Number of top images to select')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    image_paths = list(input_dir.rglob(args.pattern))
    
    print(f"Found {len(image_paths)} images to evaluate")
    
    results = []
    for img_path in tqdm(image_paths, desc="Evaluating images"):
        quality_metrics = evaluate_image_quality(img_path)
        if quality_metrics:
            results.append({
                'path': str(img_path),
                'quality': quality_metrics
            })
    
    # 根据综合评分排序
    results.sort(key=lambda x: x['quality']['score'], reverse=True)
    
    # 选择前top_k个图像
    top_results = results[:args.top_k]
    
    print(f"\nTop {args.top_k} images by quality score:")
    for i, result in enumerate(top_results):
        q = result['quality']
        print(f"{i+1:3d}. {result['path']} - Score: {q['score']:.2f}, "
              f"Sharpness: {q['sharpness']:.2f}, Contrast: {q['contrast']:.2f}")
    
    # 将结果写入文件
    output_file = input_dir / f"simple_evaluation_top_{args.top_k}.txt"
    with open(output_file, 'w') as f:
        f.write(f"Top {args.top_k} images by quality score:\n")
        for i, result in enumerate(top_results):
            q = result['quality']
            f.write(f"{i+1:3d}. {result['path']} - Score: {q['score']:.2f}, "
                    f"Sharpness: {q['sharpness']:.2f}, Contrast: {q['contrast']:.2f}\n")
    
    print(f"\nResults saved to {output_file}")
    
    # 创建一个目录存放选中的图像
    selected_dir = input_dir / f"selected_top_{args.top_k}"
    selected_dir.mkdir(exist_ok=True)
    
    for i, result in enumerate(top_results):
        src_path = Path(result['path'])
        dst_path = selected_dir / f"{i:03d}_{src_path.name}"
        dst_path.symlink_to(src_path.resolve())
    
    print(f"Symlinks to top {args.top_k} images created in {selected_dir}")


if __name__ == "__main__":
    main()