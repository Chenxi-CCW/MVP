from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run lmms-eval for MVP-Token+LLaVA under multiple pruning settings.")
    parser.add_argument("--tasks", type=str, required=True, help="Comma-separated lmms-eval tasks, e.g. pope,gqa,mme")
    parser.add_argument("--pretrained", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--adapter-ckpt", type=str, default="outputs/mvp_pruner/checkpoints/best.pt")
    parser.add_argument("--output-root", type=str, default="outputs/lmms_mvp_eval")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--cuda-visible-devices", type=str, default=None, help="e.g. 0,1")
    parser.add_argument("--lmms-eval-repo", type=str, default=None, help="Optional external lmms-eval checkout.")
    parser.add_argument("--llava-repo", type=str, default=None, help="Optional external LLaVA checkout.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-processes", type=int, default=1, help="Number of GPU processes for accelerate launch.")
    parser.add_argument("--limit", type=str, default=None)
    parser.add_argument("--log-samples", action="store_true")
    parser.add_argument("--extra-model-args", type=str, default="")
    parser.add_argument("--extra-eval-args", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    lmms_eval_repo = Path(args.lmms_eval_repo).resolve() if args.lmms_eval_repo else root / "lmmseval"
    llava_repo = Path(args.llava_repo).resolve() if args.llava_repo else root / "LLaVA"
    env["LMMS_EVAL_REPO"] = str(lmms_eval_repo)
    env["LLAVA_REPO"] = str(llava_repo)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(lmms_eval_repo), str(llava_repo), str(root), env.get("PYTHONPATH", "")]
    ).strip(os.pathsep)
    env["LMMS_EVAL_PLUGINS"] = "qvts_lmms_plugin"
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    settings = [
        ("topk_32", "selection_mode=topk,topk=32"),
        ("topk_64", "selection_mode=topk,topk=64"),
        ("topk_128", "selection_mode=topk,topk=128"),
        # ("topk_192", "selection_mode=topk,topk=192"),
        # ("threshold_0p5", "selection_mode=threshold,threshold=0.5"),
    ]

    for name, setting_args in settings:
        run_output = output_root / name
        run_output.mkdir(parents=True, exist_ok=True)

        model_args = [
            f"pretrained={args.pretrained}",
            f"adapter_ckpt={Path(args.adapter_ckpt).resolve()}",
            f"device={args.device}",
            f"device_map={args.device}",
            setting_args,
        ]
        if args.extra_model_args:
            model_args.append(args.extra_model_args)

        cmd = [
            sys.executable,
            "-m",
            "accelerate.commands.launch",
        ]
        if args.num_processes > 1:
            cmd.extend(["--num_processes", str(args.num_processes)])
        cmd.extend(
            [
                "-m",
                "lmms_eval",
                "--model",
                "MVP_llava",
                "--model_args",
                ",".join(model_args),
                "--tasks",
                args.tasks,
                "--batch_size",
                str(args.batch_size),
                "--output_path",
                str(run_output),
            ]
        )
        if args.limit is not None:
            cmd.extend(["--limit", str(args.limit)])
        if args.log_samples:
            cmd.append("--log_samples")
        if args.extra_eval_args:
            cmd.extend(args.extra_eval_args.split())

        print(f"\n[RUN] {name}")
        print(" ".join(cmd))
        subprocess.run(cmd, check=True, env=env, cwd=root)


if __name__ == "__main__":
    main()
