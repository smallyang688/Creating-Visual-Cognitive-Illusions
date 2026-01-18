import json
from collections import Counter

# 读取triplet_eecs_sms_select_blind_test.json数据
with open('results/triplet_eecs_sms_select_blind_test.json', 'r') as f:
    data = json.load(f)

print("triplet_eecs_sms_select_blind_test.json 分析")
print("=" * 50)
print(f"总样本数: {len(data)}")

# 统计综合评分分布
scores = []
for item in data:
    if 'score' in item and item['score'] is not None:
        scores.append(item['score'])

score_counts = Counter(scores)
print(f"\n评分范围: {min(scores):.1f} - {max(scores):.1f}")
print(f"平均评分: {sum(scores)/len(scores):.2f}")

print("\n综合评分分布:")
print("评分 | 数量 | 百分比")
print("-" * 20)
for score in sorted(score_counts.keys()):
    count = score_counts[score]
    pct = count / len(data) * 100
    print("4.1f")

# 统计识别准确性
perfect_left = 0
perfect_right = 0
both_perfect = 0

for item in data:
    left_match = item.get('left_found', '').upper() == 'EECS'
    right_match = item.get('right_found', '').upper() == 'SMS'

    if left_match:
        perfect_left += 1
    if right_match:
        perfect_right += 1
    if left_match and right_match:
        both_perfect += 1

print("\n识别准确性分析:")
print(f"左侧EECS准确率: {perfect_left}/{len(data)} = {perfect_left/len(data)*100:.1f}%")
print(f"右侧SMS准确率: {perfect_right}/{len(data)} = {perfect_right/len(data)*100:.1f}%")
print(f"双侧完全匹配: {both_perfect}/{len(data)} = {both_perfect/len(data)*100:.1f}%")

# 高分样本分析 (评分>=9.0)
high_score_samples = [item for item in data if item.get('score', 0) >= 9.0]
print(f"\n高分样本 (评分≥9.0): {len(high_score_samples)}/{len(data)} = {len(high_score_samples)/len(data)*100:.1f}%")

if high_score_samples:
    high_perfect_left = sum(1 for item in high_score_samples if item.get('left_found', '').upper() == 'EECS')
    high_perfect_right = sum(1 for item in high_score_samples if item.get('right_found', '').upper() == 'SMS')
    print(f"高分样本左侧准确率: {high_perfect_left}/{len(high_score_samples)} = {high_perfect_left/len(high_score_samples)*100:.1f}%")
    print(f"高分样本右侧准确率: {high_perfect_right}/{len(high_score_samples)} = {high_perfect_right/len(high_score_samples)*100:.1f}%")

# 分析最常见的识别错误
left_recognitions = Counter(item.get('left_found', '') for item in data)
right_recognitions = Counter(item.get('right_found', '') for item in data)

print("\n最常见的左侧识别结果 (前5):")
for word, count in left_recognitions.most_common(5):
    print(f"  '{word}': {count} 次")

print("\n最常见的右侧识别结果 (前5):")
for word, count in right_recognitions.most_common(5):
    print(f"  '{word}': {count} 次")
