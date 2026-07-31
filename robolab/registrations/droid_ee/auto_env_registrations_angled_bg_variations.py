# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Droid-EE registration for angled-task background x clutter experiments.

Generated USD/task files live in a temporary cache. The source angled registrar,
task files, and scene assets are never modified.
"""

import os
import random
import re
import tempfile
from pathlib import Path

import robolab.constants
from robolab.constants import BACKGROUND_ASSET_DIR, PACKAGE_DIR, TASK_DIR


ANGLED_REACH_TASK_SUBFOLDERS = ["wm_tasks/bg_distractor"]
DEFAULT_NUM_BACKGROUNDS = 5
DEFAULT_OBJECT_COUNTS = (1, 2, 3, 4, 5)

# A fixed bank makes clutter levels comparable across tasks. Z is the stable
# tabletop center height for the upright asset; XY is relative to the target.
DISTRACTOR_SPECS = (
    ("clutter_soup_can", "assets/objects/hot3d/soup_can.usd", (-0.20, 0.00, 0.085)),
    ("clutter_milk_carton", "assets/objects/hope/milk_carton.usd", (0.20, 0.00, 0.095)),
    ("clutter_mug", "assets/objects/hot3d/mug.usd", (0.00, -0.20, 0.046)),
    ("clutter_mustard", "assets/objects/ycb/mustard.usd", (0.00, 0.20, 0.096)),
)


def _resolve_background_paths(backgrounds):
    """Resolve background filenames, asset-relative paths, or absolute paths."""
    from robolab.variations.backgrounds import find_background_files

    resolved = []
    for background in backgrounds:
        if os.path.isabs(background):
            path = background
        else:
            asset_relative_path = os.path.join(BACKGROUND_ASSET_DIR, background)
            path = asset_relative_path if os.path.isfile(asset_relative_path) else find_background_files(
                folder_path=BACKGROUND_ASSET_DIR,
                filename=os.path.basename(background),
            )
        if path is None or not os.path.isfile(path):
            raise FileNotFoundError(f"Background '{background}' was not found in '{BACKGROUND_ASSET_DIR}'.")
        resolved.append(os.path.abspath(path))
    return resolved


def _background_env_name(background_path):
    name = os.path.splitext(os.path.basename(background_path))[0]
    name = re.sub(r"_2k$", "", name)
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_").lower()


def _select_backgrounds(backgrounds, num_backgrounds, background_seed):
    from robolab.variations.backgrounds import find_background_files

    if backgrounds:
        return _resolve_background_paths(backgrounds)

    default_background = find_background_files(
        folder_path=BACKGROUND_ASSET_DIR,
        filename="home_office.exr",
    )
    if default_background is None:
        raise FileNotFoundError("Default background 'home_office.exr' was not found.")
    default_background = os.path.abspath(default_background)
    candidates = [
        os.path.abspath(path)
        for path in find_background_files(BACKGROUND_ASSET_DIR)
        if os.path.abspath(path) != default_background
    ]
    if not 1 <= num_backgrounds <= len(candidates):
        raise ValueError(
            f"num_backgrounds must be between 1 and {len(candidates)}; got {num_backgrounds}."
        )
    return random.Random(background_seed).sample(candidates, num_backgrounds)


def _task_files(task_dirs, tasks):
    """Resolve requested tasks, or discover every Python task in task_dirs."""
    from robolab.core.task.task_utils import resolve_task_path

    if tasks:
        return [resolve_task_path(task, TASK_DIR)[0] for task in tasks]

    files = []
    for task_dir in task_dirs:
        files.extend((Path(TASK_DIR) / task_dir).glob("*.py"))
    return [str(path) for path in sorted(files) if path.name != "__init__.py"]


def _generated_cache_dir():
    return Path(tempfile.mkdtemp(prefix="robolab_angledreach_clutter_"))


def _create_clutter_scene(original_scene_path, target_object, original_objects, object_count, output_path):
    """Create a non-destructive USD overlay with target + 0..4 distractors."""
    from pxr import Gf, Sdf, Usd
    from robolab.core.utils.usd_utils import get_usd_objects_info

    if output_path.exists():
        output_path.unlink()
    root_layer = Sdf.Layer.CreateNew(str(output_path))
    if root_layer is None:
        raise RuntimeError(f"Could not create clutter overlay layer: {output_path}")
    root_layer.subLayerPaths = [os.path.abspath(original_scene_path)]
    root_layer.defaultPrim = "world"
    stage = Usd.Stage.Open(root_layer)
    if stage is None:
        raise RuntimeError(f"Could not create clutter overlay: {output_path}")

    # Remove all task-authored objects except the table and reach target.
    for object_name in original_objects:
        if object_name not in {"table", target_object}:
            stage.OverridePrim(f"/world/{object_name}").SetActive(False)

    scene_objects = get_usd_objects_info(original_scene_path)
    target_info = next((obj for obj in scene_objects if obj["name"] == target_object), None)
    if target_info is None:
        raise ValueError(f"Target '{target_object}' not found in scene '{original_scene_path}'.")
    target_x, target_y, _ = target_info["position"]

    for distractor_name, asset_path, (dx, dy, z) in DISTRACTOR_SPECS[:object_count - 1]:
        absolute_asset_path = os.path.join(PACKAGE_DIR, asset_path)
        if not os.path.isfile(absolute_asset_path):
            raise FileNotFoundError(f"Distractor asset not found: {absolute_asset_path}")
        prim = stage.DefinePrim(f"/world/{distractor_name}", "Xform")
        prim.GetPayloads().AddPayload(absolute_asset_path)
        prim.CreateAttribute("xformOp:translate", Sdf.ValueTypeNames.Double3).Set(
            Gf.Vec3d(float(target_x + dx), float(target_y + dy), z)
        )
        prim.CreateAttribute("xformOp:orient", Sdf.ValueTypeNames.Quatf).Set(
            Gf.Quatf(1.0, 0.0, 0.0, 0.0)
        )
        prim.CreateAttribute("xformOp:scale", Sdf.ValueTypeNames.Float3).Set(
            Gf.Vec3f(1.0, 1.0, 1.0)
        )
        prim.CreateAttribute("xformOpOrder", Sdf.ValueTypeNames.TokenArray).Set(
            ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
        )

    stage.GetRootLayer().Save()


def _create_clutter_task(task_file, object_count, cache_dir):
    """Create a temporary Task subclass pointing at a generated clutter USD."""
    from robolab.core.task.task_utils import load_task_from_file

    task_class = load_task_from_file(task_file)
    target_object = task_class.goal["object"]
    original_scene_path = task_class.scene.scene.spawn.usd_path
    variant_name = f"{task_class.__name__.removesuffix('Task')}Objects{object_count}Task"
    scene_path = cache_dir / f"{variant_name}.usda"
    generated_task_path = cache_dir / f"{variant_name}.py"

    _create_clutter_scene(
        original_scene_path=original_scene_path,
        target_object=target_object,
        original_objects=task_class.contact_object_list,
        object_count=object_count,
        output_path=scene_path,
    )

    distractor_names = [spec[0] for spec in DISTRACTOR_SPECS[:object_count - 1]]
    contact_objects = ["table", target_object, *distractor_names]
    relative_module = Path(task_file).resolve().relative_to(Path(TASK_DIR).resolve()).with_suffix("")
    module_name = ".".join(("robolab", "tasks", *relative_module.parts))
    source = (
        "from dataclasses import dataclass\n"
        "from robolab.core.scenes.utils import import_scene\n"
        f"from {module_name} import {task_class.__name__} as _BaseTask\n\n"
        "@dataclass\n"
        f"class {variant_name}(_BaseTask):\n"
        f"    scene = import_scene({str(scene_path)!r}, {contact_objects!r})\n"
        f"    contact_object_list = {contact_objects!r}\n"
        f"    task_name = {task_class.__name__!r}\n\n"
        "del _BaseTask\n"
    )
    generated_task_path.write_text(source, encoding="utf-8")
    return str(generated_task_path)


def auto_register_droid_ee_envs_bg_variations(
    task_dirs=ANGLED_REACH_TASK_SUBFOLDERS,
    backgrounds=None,
    num_backgrounds=DEFAULT_NUM_BACKGROUNDS,
    background_seed=1,
    tasks=None,
    cameras=None,
    object_counts=DEFAULT_OBJECT_COUNTS,
):
    """Register angled-task x background x total-object-count environments."""
    from robolab.core.environments.factory import auto_discover_and_create_cfgs, get_envs_by_tag
    from robolab.core.observations.observation_utils import generate_image_obs_from_cameras, generate_obs_cfg
    from robolab.registrations.droid_jointpos.camera_presets import WRIST_RIGHT
    from robolab.robots.droid import (
        DroidCfg,
        DroidIKActionCfg,
        ProprioceptionObservationCfg,
        WristCameraCfg,
        contact_gripper,
    )
    from robolab.variations.backgrounds import generate_background_config

    object_counts = tuple(object_counts)
    if not object_counts or any(count not in DEFAULT_OBJECT_COUNTS for count in object_counts):
        raise ValueError(f"object_counts must contain values from {DEFAULT_OBJECT_COUNTS}; got {object_counts}.")

    if cameras is None:
        cameras = WRIST_RIGHT
    ImageObsCfg = generate_image_obs_from_cameras(cameras)
    ObservationCfg = generate_obs_cfg({
        "image_obs": ImageObsCfg(),
        "proprio_obs": ProprioceptionObservationCfg(),
    })
    scene_cameras = [camera for camera in cameras if camera is not WristCameraCfg]

    background_paths = _select_backgrounds(backgrounds, num_backgrounds, background_seed)
    background_names = [_background_env_name(path) for path in background_paths]
    if len(background_names) != len(set(background_names)):
        raise ValueError("Selected backgrounds produce duplicate environment suffixes.")

    source_task_files = _task_files(task_dirs, tasks)
    cache_dir = _generated_cache_dir()
    generated_tasks = {
        object_count: [
            _create_clutter_task(task_file, object_count, cache_dir)
            for task_file in source_task_files
        ]
        for object_count in object_counts
    }

    expected = len(source_task_files) * len(background_paths) * len(object_counts)
    print(
        f"\033[96m[RoboLab] Registering {expected} angled variants: "
        f"{len(source_task_files)} tasks x {len(background_paths)} backgrounds x "
        f"{len(object_counts)} object counts\033[0m"
    )

    for background_path, background_name in zip(background_paths, background_names):
        background_cfg = generate_background_config(background_path)
        for object_count in object_counts:
            auto_discover_and_create_cfgs(
                task_dir=TASK_DIR,
                tasks=generated_tasks[object_count],
                add_tags=["background_variations", "angled_reach_background_variations"],
                env_postfix=f"_bg_{background_name}",
                observations_cfg=ObservationCfg(),
                actions_cfg=DroidIKActionCfg(),
                robot_cfg=DroidCfg,
                camera_cfg=[*scene_cameras],
                background_cfg=background_cfg,
                contact_gripper=contact_gripper,
                dt=1 / (60 * 2),
                render_interval=8,
                decimation=8,
                seed=1,
            )

    registered_envs = get_envs_by_tag("angled_reach_background_variations")
    print(f"\033[96m[RoboLab] Registered {len(registered_envs)} angled variants.\033[0m")
    if len(registered_envs) != expected:
        raise RuntimeError(f"Expected {expected} registered environments, found {len(registered_envs)}.")

    if robolab.constants.VERBOSE:
        from robolab.core.environments.factory import print_env_table
        print_env_table()
    return registered_envs
