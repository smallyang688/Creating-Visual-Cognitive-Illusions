import json
from collections import Counter

# 读取评估结果
with open('results/ambigram_eecs_sms_cursive_scored.json', 'r') as f:
    data = json.load(f)

# 统计编辑距离分布
dists = [item['total_dist'] for item in data]
dist_counts = Counter(dists)

print('编辑距离分布:')
for dist in sorted(dist_counts.keys()):
    print(f'距离 {dist}: {dist_counts[dist]} 个样本')

print(f'\n总样本数: {len(data)}')
print(f'最小编辑距离: {min(dists)}')
print(f'最大编辑距离: {max(dists)}')
print(f'平均编辑距离: {sum(dists)/len(dists):.2f}')

# 统计高质量样本 (编辑距离 <= 3)
high_quality = len([d for d in dists if d <= 3])
print(f'\n高质量样本 (编辑距离<=3): {high_quality} 个 ({high_quality/len(dists)*100:.1f}%)')

# 查看top 5结果
print('\nTop 5 样本:')
for i, item in enumerate(data[:5]):
    print(f'{i+1}. 距离: {item["total_dist"]}, 路径: {item["path"]}')
    print(f'   识别结果: 左="{item["left"]}", 右="{item["right"]}"')
    print(f'   标准化: 左="{item["left_norm"]}", 右="{item["right_norm"]}"')
