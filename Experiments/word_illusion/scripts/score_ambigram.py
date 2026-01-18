import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score ambigram results by edit distance to target words."
    )
    parser.add_argument(
        "--input_json",
        default="results/ambigram_eecs_sms_cursive_scores.json",
        help="Path to results JSON from select_ambigram.py.",
    )
    parser.add_argument(
        "--match_left",
        default="EECS",
        help="Target word for left panel.",
    )
    parser.add_argument(
        "--match_right",
        default="SMS",
        help="Target word for right panel.",
    )
    parser.add_argument(
        "--pairs",
        default=None,
        help="Comma-separated pairs like eecs:sms,ics:love,pku:thu. Overrides match_left/right.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=15,
        help="How many top candidates to print.",
    )
    parser.add_argument(
        "--output_json",
        default="results/ambigram_eecs_sms_cursive_scored.json",
        help="Where to write scored results JSON.",
    )
    parser.add_argument(
        "--copy_dir",
        default=None,
        help="If set, copy the top_k images into this directory.",
    )
    return parser.parse_args()


def normalize(text):
    if not isinstance(text, str):
        return ""
    return "".join(ch for ch in text.upper() if ch.isalnum())


def edit_distance(a, b):
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ch_a in enumerate(a, start=1):
        curr = [i]
        for j, ch_b in enumerate(b, start=1):
            cost = 0 if ch_a == ch_b else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def parse_pairs(pairs_arg, match_left, match_right):
    if not pairs_arg:
        return [(match_left, match_right)]
    pairs = []
    for raw in pairs_arg.split(","):
        raw = raw.strip()
        if not raw:
            continue
        if ":" in raw:
            left, right = raw.split(":", 1)
        elif "/" in raw:
            left, right = raw.split("/", 1)
        else:
            raise ValueError(f"Invalid pair format: {raw}")
        pairs.append((left.strip(), right.strip()))
    if not pairs:
        return [(match_left, match_right)]
    return pairs


def main():
    args = parse_args()
    input_path = Path(args.input_json)
    if not input_path.exists():
        raise SystemExit(f"Input JSON not found: {input_path}")

    data = json.loads(input_path.read_text())
    raw_pairs = parse_pairs(args.pairs, args.match_left, args.match_right)
    pairs = [(normalize(l), normalize(r)) for l, r in raw_pairs]

    scored = []
    for item in data:
        left_raw = item.get("left")
        right_raw = item.get("right")
        left = normalize(left_raw)
        right = normalize(right_raw)
        best_total = None
        best_pair = None
        best_left = None
        best_right = None
        for pair_left, pair_right in pairs:
            if not left:
                dist_left = len(pair_left) + 5
            else:
                dist_left = edit_distance(left, pair_left)
            if not right:
                dist_right = len(pair_right) + 5
            else:
                dist_right = edit_distance(right, pair_right)
            total = dist_left + dist_right
            if best_total is None or total < best_total:
                best_total = total
                best_pair = f"{pair_left}:{pair_right}"
                best_left = dist_left
                best_right = dist_right
        scored.append(
            {
                **item,
                "left_norm": left,
                "right_norm": right,
                "dist_left": best_left,
                "dist_right": best_right,
                "total_dist": best_total,
                "best_pair": best_pair,
            }
        )

    scored_sorted = sorted(scored, key=lambda x: x["total_dist"])
    top_k = scored_sorted[: args.top_k]

    print("Top candidates by edit distance:")
    for item in top_k:
        print(
            f"{item['total_dist']}\t{item.get('left')} / {item.get('right')}\t{item['path']}"
        )

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(scored_sorted, ensure_ascii=True, indent=2))
    print(f"Saved scored results to {output_path}")

    if args.copy_dir:
        copy_dir = Path(args.copy_dir)
        copy_dir.mkdir(parents=True, exist_ok=True)
        for item in top_k:
            src = Path(item["path"])
            dst = copy_dir / src.name
            dst.write_bytes(src.read_bytes())


if __name__ == "__main__":
    main()
