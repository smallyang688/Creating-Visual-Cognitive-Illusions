import subprocess
import sys
from pathlib import Path


# Update these two paths if your envs live elsewhere.
GENERATE_PYTHON = "/home/geming/miniforge3/envs/visual_anagrams/bin/python"
QWEN_PYTHON = "/home/geming/miniforge3/envs/qwen3vl/bin/python"

SAVE_DIR = "results"
STYLE = "black ink on white paper, centered, high contrast, clean calligraphy, no extra elements"
VIEWS = ["identity", "rotate_180"]
NUM_INFERENCE_STEPS = 70
GUIDANCE_SCALE = 9.5
SEED = 0
DEVICE = "cuda"
GENERATE_1024 = False
SKIP_STAGE_2 = False
RESUME = True

JOBS = [
    {
        "name": "triplet_eecs_sms",
        "samples": 3333,
        "prompts": ["the word eecs, cursive writing", "the word sms, cursive writing"],
        "pair": ("eecs", "sms"),
    },
    {
        "name": "triplet_ics_love",
        "samples": 3333,
        "prompts": ["the word ics, cursive writing", "the word love, cursive writing"],
        "pair": ("ics", "love"),
    },
    {
        "name": "triplet_pku_thu",
        "samples": 3333,
        "prompts": ["the word pku, cursive writing", "the word thu, cursive writing"],
        "pair": ("pku", "thu"),
    },
]

SELECT_PATTERN = "sample_256.views.png"
QWEN_MODEL = "Qwen/Qwen3-VL-4B-Instruct"
QWEN_DTYPE = "fp16"
QWEN_GPU_ONLY = True
QWEN_CPU_OFFLOAD = False
QWEN_MAX_GPU_MEM = "14GiB"
QWEN_MAX_CPU_MEM = "64GiB"
QWEN_MAX_NEW_TOKENS = 128
BLIND_PROMPT = True
STREAM = True

TOP_K = 100


def run_cmd(cmd):
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    save_dir = Path(SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)

    output_root = save_dir / "triplet"
    output_root.mkdir(parents=True, exist_ok=True)

    for idx, job in enumerate(JOBS):
        seed = SEED + idx * 10000
        name = job["name"]
        samples = job["samples"]
        prompts = job["prompts"]
        first_sample_dir = output_root / name / f"{seed:04d}"
        if RESUME and first_sample_dir.exists():
            print(f"Skipping existing job: {first_sample_dir}")
            continue
        cmd = [
            GENERATE_PYTHON,
            "generate.py",
            "--name",
            name,
            "--save_dir",
            str(output_root),
            "--prompts",
            *prompts,
            "--style",
            STYLE,
            "--views",
            *VIEWS,
            "--num_samples",
            str(samples),
            "--num_inference_steps",
            str(NUM_INFERENCE_STEPS),
            "--guidance_scale",
            str(GUIDANCE_SCALE),
            "--seed",
            str(seed),
            "--device",
            DEVICE,
        ]
        if GENERATE_1024:
            cmd.append("--generate_1024")
        if SKIP_STAGE_2:
            cmd.append("--skip_stage_2")
        run_cmd(cmd)

    for job in JOBS:
        job_name = job["name"]
        job_dir = output_root / job_name
        left, right = job["pair"]
        select_json = f"results/{job_name}_select.json"
        scored_json = f"results/{job_name}_scored.json"
        top_dir = f"results/{job_name}_top100"

        cmd = [
            QWEN_PYTHON,
            "select_ambigram.py",
            "--input_dir",
            str(job_dir),
            "--pattern",
            SELECT_PATTERN,
            "--model",
            QWEN_MODEL,
            "--dtype",
            QWEN_DTYPE,
            "--max_new_tokens",
            str(QWEN_MAX_NEW_TOKENS),
            "--output_json",
            select_json,
        ]
        if QWEN_GPU_ONLY:
            cmd.append("--gpu_only")
        if QWEN_CPU_OFFLOAD:
            cmd.append("--cpu_offload")
            cmd += ["--max_gpu_mem", QWEN_MAX_GPU_MEM, "--max_cpu_mem", QWEN_MAX_CPU_MEM]
        if BLIND_PROMPT:
            cmd.append("--blind_prompt")
        if STREAM:
            cmd.append("--stream")
        run_cmd(cmd)

        cmd = [
            QWEN_PYTHON,
            "score_ambigram.py",
            "--input_json",
            select_json,
            "--pairs",
            f"{left}:{right}",
            "--top_k",
            str(TOP_K),
            "--output_json",
            scored_json,
            "--copy_dir",
            top_dir,
        ]
        run_cmd(cmd)


if __name__ == "__main__":
    main()
