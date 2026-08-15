# SPDX-License-Identifier: CC-BY-NC-4.0
"""Run matched VALPA imaginations and Isaac ground truth for random actions.

Start ``valpa/inference/serve_policy_quant.py`` first. For every transition this
script requests a config-bounded random action and one-step imagination,
executes the action in the registered task scene, and immediately reports the
resulting observation.
"""

from __future__ import annotations

import argparse
import sys
import traceback

import cv2  # Must be imported before isaaclab.
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="AngledReachDrillTask")
parser.add_argument("--task-dirs", nargs="+", default=["wm_tasks/angledreach"])
parser.add_argument("--remote-host", default="localhost")
parser.add_argument("--remote-port", type=int, default=8000)
parser.add_argument("--max-steps", type=int, default=100)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
args_cli.livestream = 0

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

from robolab.core.environments.factory import get_envs
from robolab.core.environments.runtime import create_env
from robolab.registrations.droid_ee.auto_env_registrations_angled import (
    auto_register_droid_ee_envs,
)
from robolab.eval import create_client


def _wire_observation(client, obs: dict) -> dict:
    extracted = client._extract_observation(obs, env_id=0)
    return {
        "external_image": extracted["external_image"],
        "wrist_image": extracted["wrist_image"],
        "ee_pose": extracted["ee_pose"],
    }


def main() -> None:
    auto_register_droid_ee_envs(task_dirs=args_cli.task_dirs, task=[args_cli.task])
    env_names = get_envs(task=args_cli.task)
    if len(env_names) != 1:
        raise RuntimeError(
            f"Expected exactly one registered scene for {args_cli.task}, got {env_names}"
        )

    env, env_cfg = create_env(
        env_names[0], device=args_cli.device, num_envs=1,
        use_fabric=True, policy="valpa",
    )
    client = create_client(
        "valpa",
        remote_host=args_cli.remote_host, remote_port=args_cli.remote_port
    )

    try:
        obs, _ = env.reset()
        client.reset()
        metadata = client.metadata()
        print(
            f"[patch-loss] task={args_cli.task} scene={env_names[0]} "
            f"output={metadata.get('analysis_output_dir')}",
            flush=True,
        )

        for step in range(args_cli.max_steps):
            response = client._request({
                "method": "infer",
                "obs": _wire_observation(client, obs),
                "instruction": env_cfg.instruction,
                "env_id": 0,
            })
            action = np.asarray(response["action"], dtype=np.float32)
            obs, _, terminated, truncated, _ = env.step(
                torch.from_numpy(action).to(env.device).unsqueeze(0)
            )
            report = client.report_ground_truth(obs, env_id=0)
            losses = ", ".join(
                f"{stream}={values['mean_patch_l1']:.6f}"
                for stream, values in report["loss"].items()
            )
            print(f"[patch-loss] step={step:03d} action={action.tolist()} {losses}", flush=True)

            if bool(terminated[0]) or bool(truncated[0]):
                raise RuntimeError(
                    "Simulator terminated before the requested random transitions completed"
                )

        summary = client.finish_rollout(env_id=0)
        print(
            f"[patch-loss] complete: {summary['steps']} matched transitions saved to "
            f"{summary['output_dir']}",
            flush=True,
        )
    finally:
        client.close()
        env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        simulation_app.close()
        sys.exit(1)
    simulation_app.close()
