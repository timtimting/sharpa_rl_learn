#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source /home/tim/miniconda3/etc/profile.d/conda.sh
conda activate sharpa-rl
cd "$repository_root"

experiment_root="logs/rotation_robust"
teacher_checkpoint="logs/rotation_finetune/2026-09-25_23-24-22/stage1_nn/best.pth"
mkdir -p "$experiment_root" outputs

python -u rl_isaaclab/scripts/train.py \
  --task Isaac-Inhand-Rotate-Sharpa-Wave-v0 \
  --num_envs 1024 --headless --device cuda:0 --seed 2026 \
  --resume --load_path "$teacher_checkpoint" --max_agent_steps 30000000 \
  'env.sim.gravity=[0.0,0.0,-9.81]' \
  env.gravity_curriculum=false \
  env.randomize_pd_gains=true \
  env.randomize_com=true \
  env.randomize_mass=true \
  env.randomize_friction=true \
  agent.algorithm.learning_rate=0.0001 \
  agent.algorithm.experiment_name=rotation_robust \
  2>&1 | tee "$experiment_root/train_console.log"

best_checkpoint="$(find "$experiment_root" -mindepth 3 -maxdepth 3 -path '*/stage1_nn/best.pth' -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f 2-)"
if [[ -z "$best_checkpoint" ]]; then
  echo "Training completed without a best checkpoint" >&2
  exit 1
fi

experiment_dir="$(dirname "$(dirname "$best_checkpoint")")"
last_checkpoint="$experiment_dir/stage1_nn/last.pth"

for evaluation_seed in 42 123; do
  python -u rl_isaaclab/scripts/evaluate.py \
    --task Isaac-Inhand-Rotate-Sharpa-Wave-v0 \
    --num_envs 64 --steps 1000 --headless --device cuda:0 \
    --seed "$evaluation_seed" --robust \
    --checkpoints "$teacher_checkpoint" "$best_checkpoint" "$last_checkpoint" \
    --output "$experiment_dir/robust_comparison_seed_${evaluation_seed}.json" \
    "hydra.run.dir=/tmp/sharpa-robust-evaluation-${evaluation_seed}" \
    2>&1 | tee "$experiment_dir/robust_evaluation_seed_${evaluation_seed}.log"
done

python -u rl_isaaclab/scripts/evaluate.py \
  --task Isaac-Inhand-Rotate-Sharpa-Wave-v0 \
  --num_envs 64 --steps 1000 --headless --device cuda:0 \
  --seed 42 \
  --checkpoints "$teacher_checkpoint" "$best_checkpoint" \
  --output "$experiment_dir/nominal_comparison_seed_42.json" \
  hydra.run.dir=/tmp/sharpa-robust-nominal-evaluation-42 \
  2>&1 | tee "$experiment_dir/nominal_evaluation_seed_42.log"

python - "$experiment_dir" <<'PY'
import json
import sys

experiment_dir = sys.argv[1]
summary = {}
for seed in (42, 123):
    with open(f"{experiment_dir}/robust_comparison_seed_{seed}.json", encoding="utf-8") as result_file:
        summary[f"robust_seed_{seed}"] = json.load(result_file)
with open(f"{experiment_dir}/nominal_comparison_seed_42.json", encoding="utf-8") as result_file:
    summary["nominal_seed_42"] = json.load(result_file)
with open(f"{experiment_dir}/summary.json", "w", encoding="utf-8") as summary_file:
    json.dump(summary, summary_file, indent=2)
print(f"Saved final summary to {experiment_dir}/summary.json", flush=True)
PY
