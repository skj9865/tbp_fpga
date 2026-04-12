#!/usr/bin/env python3
"""Standalone Monty inference script for FPGA platform.

Bypasses Hydra/experiment framework and directly uses tbp.monty classes.
Reproduces the world_image_on_scanned_model benchmark (~66-68% accuracy).
"""
import argparse
import csv
import logging
import os
import sys
import time
from pathlib import Path

# Must set before any matplotlib import (two_d_data.py imports plt)
os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import torch

from tbp.monty.context import RuntimeContext
from tbp.monty.frameworks.actions.action_samplers import ConstantSampler
from tbp.monty.frameworks.actions.actions import (
    LookDown,
    LookUp,
    TurnLeft,
    TurnRight,
)
from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.environments.embodied_data import (
    SaccadeOnImageEnvironmentInterface,
)
from tbp.monty.frameworks.environments.two_d_data import SaccadeOnImageEnvironment
from tbp.monty.frameworks.experiments.mode import ExperimentMode
from tbp.monty.frameworks.experiments.seed import episode_seed
from tbp.monty.frameworks.models.evidence_matching.burst_sampling import (
    BurstSamplingHypothesesUpdater,
)
from tbp.monty.frameworks.models.evidence_matching.learning_module import (
    EvidenceGraphLM,
)
from tbp.monty.frameworks.models.goal_state_generation import (
    EvidenceGoalStateGenerator,
)
from tbp.monty.frameworks.models.graph_matching import MontyForGraphMatching
from tbp.monty.frameworks.models.motor_policies import InformedPolicy
from tbp.monty.frameworks.models.motor_system import MotorSystem
from tbp.monty.frameworks.models.sensor_modules import CameraSM, Probe

logger = logging.getLogger(__name__)

# ============================================================================
# Section 2: Configuration constants
# ============================================================================

SEED = 42
MAX_EVAL_STEPS = 500
MAX_TOTAL_STEPS = 6000
MIN_EVAL_STEPS = 20
MIN_TRAIN_STEPS = 3
NUM_EXPLORATORY_STEPS = 1000
MAX_TOTAL_STEPS_MONTY = 2500

# 12 objects × 4 versions = 48 episodes
SCENES = [i for i in range(12) for _ in range(4)]     # [0,0,0,0,1,1,1,1,...,11,11,11,11]
VERSIONS = list(range(4)) * 12                         # [0,1,2,3,0,1,2,3,...,0,1,2,3]

SENSOR_FEATURES = [
    "pose_vectors",
    "pose_fully_defined",
    "on_object",
    "object_coverage",
    "rgba",
    "hsv",
    "principal_curvatures",
    "principal_curvatures_log",
    "gaussian_curvature",
    "mean_curvature",
    "gaussian_curvature_sc",
    "mean_curvature_sc",
]


# ============================================================================
# Section 3: Component factory functions
# ============================================================================

def create_sensor_modules():
    """Create patch and view_finder sensor modules."""
    patch_sm = CameraSM(
        sensor_module_id="patch",
        features=SENSOR_FEATURES,
        save_raw_obs=True,
    )
    view_finder_sm = Probe(
        sensor_module_id="view_finder",
        save_raw_obs=True,
    )
    return [patch_sm, view_finder_sm]


def create_learning_module():
    """Create EvidenceGraphLM with benchmark configuration."""
    gsg = EvidenceGoalStateGenerator(
        goal_tolerances=dict(location=0.015),
        elapsed_steps_factor=10,
        min_post_goal_success_steps=5,
        x_percent_scale_factor=0.75,
        desired_object_distance=0.03,
    )

    lm = EvidenceGraphLM(
        max_match_distance=0.01,
        tolerances=dict(
            patch=dict(
                hsv=np.array([0.1, 0.2, 0.2]),
                principal_curvatures_log=np.ones(2),
            )
        ),
        feature_weights=dict(
            patch=dict(
                hsv=np.array([1.0, 0.5, 0.5]),
                pose_vectors=np.ones(3),
                principal_curvatures_log=np.ones(2),
            )
        ),
        x_percent_threshold=20,
        evidence_threshold_config="all",
        max_graph_size=0.3,
        num_model_voxels_per_dim=100,
        gsg=gsg,
        hypotheses_updater_class=BurstSamplingHypothesesUpdater,
        hypotheses_updater_args=dict(
            max_nneighbors=10,
            deletion_trigger_slope=0.2,
        ),
    )
    lm.learning_module_id = "learning_module_0"
    return lm


def create_motor_system():
    """Create MotorSystem with InformedPolicy and ConstantSampler."""
    agent_id = AgentID("agent_id_0")
    action_sampler = ConstantSampler(
        actions=[LookUp, LookDown, TurnLeft, TurnRight],
        rotation_degrees=20.0,
    )
    policy = InformedPolicy(
        use_goal_state_driven_actions=False,
        action_sampler=action_sampler,
        agent_id=agent_id,
    )
    return MotorSystem(policy=policy)


def create_model(sensor_modules, learning_modules, motor_system):
    """Create MontyForGraphMatching model."""
    agent_id = AgentID("agent_id_0")
    model = MontyForGraphMatching(
        sensor_modules=sensor_modules,
        learning_modules=learning_modules,
        motor_system=motor_system,
        sm_to_agent_dict=dict(patch=agent_id, view_finder=agent_id),
        sm_to_lm_matrix=[[0]],
        lm_to_lm_matrix=None,
        lm_to_lm_vote_matrix=None,
        min_eval_steps=MIN_EVAL_STEPS,
        min_train_steps=MIN_TRAIN_STEPS,
        num_exploratory_steps=NUM_EXPLORATORY_STEPS,
        max_total_steps=MAX_TOTAL_STEPS_MONTY,
    )
    model.min_lms_match = 1
    return model


def load_pretrained_model(model, model_path):
    """Load pretrained weights into the model."""
    model_path = Path(model_path)
    if model_path.is_dir():
        model_path = model_path / "model.pt"
    logger.info(f"Loading pretrained model from {model_path}")
    state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict)
    logger.info("Model loaded successfully")


# ============================================================================
# Section 4: Episode loop
# ============================================================================

def run_inference(data_path, model_path, max_eval_steps, seed, log_level,
                  output_csv=None, max_episodes=None):
    """Run the evaluation loop.

    Returns:
        List of dicts with per-episode results.
    """
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.WARNING),
        format="%(levelname)s:%(name)s:%(funcName)s:%(lineno)d:%(message)s",
        stream=sys.stderr,
    )

    # Create components
    sensor_modules = create_sensor_modules()
    lm = create_learning_module()
    motor_system = create_motor_system()
    model = create_model(sensor_modules, [lm], motor_system)

    # Set experiment mode to EVAL
    model.set_experiment_mode(ExperimentMode.EVAL)

    # Load pretrained model
    load_pretrained_model(model, model_path)

    # Create environment
    env = SaccadeOnImageEnvironment(data_path=data_path)

    # Create environment interface (shares motor_system with model)
    rng = np.random.RandomState(seed)
    env_interface = SaccadeOnImageEnvironmentInterface(
        scenes=SCENES,
        versions=VERSIONS,
        env=env,
        motor_system=model.motor_system,
        rng=rng,
        transform=None,
        experiment_mode=ExperimentMode.EVAL,
        seed=seed,
    )

    # Verify motor_system sharing
    assert env_interface.motor_system is model.motor_system

    # Pre-epoch: set up first object
    env_interface.pre_epoch()

    num_episodes = len(SCENES)
    if max_episodes is not None:
        num_episodes = min(max_episodes, num_episodes)
    results = []
    rng_seed_history = []

    # Open CSV for incremental writing
    csv_file = None
    csv_writer = None
    csv_fieldnames = [
        "episode", "target_object", "detected_object",
        "terminal_state", "primary_performance", "steps",
        "correct", "correct_mlh", "elapsed_sec",
    ]
    if output_csv:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        csv_file = open(output_csv, "w", newline="")
        csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fieldnames)
        csv_writer.writeheader()
        csv_file.flush()

    logger.info(f"Starting evaluation: {num_episodes} episodes")

    for ep in range(num_episodes):
        ep_start = time.time()

        # Reset RNG for this episode
        ep_seed = episode_seed(seed, ExperimentMode.EVAL, ep)
        if ep_seed in rng_seed_history:
            logger.warning(f"RNG seed {ep_seed} was used in a previous episode")
        rng_seed_history.append(ep_seed)
        rng = np.random.RandomState(ep_seed)

        # Pre-episode
        target = env_interface.primary_target
        model.pre_episode(target)
        env_interface.pre_episode(rng)

        # Run episode steps
        ctx = RuntimeContext(rng=rng)
        step = 0
        while True:
            observations = env_interface.step(ctx, first=(step == 0))

            if model.check_reached_max_matching_steps(max_eval_steps):
                logger.info(
                    f"Ep {ep}: terminated due to max matching steps: {max_eval_steps}"
                )
                break

            if step >= MAX_TOTAL_STEPS:
                logger.info(f"Ep {ep}: terminated due to max episode steps: {step}")
                model.deal_with_time_out()
                break

            if model.is_motor_only_step:
                model.pass_features_directly_to_motor_system(ctx, observations)
            else:
                model.step(ctx, observations)

            if model.is_done:
                break

            step += 1

        # Post-episode
        model.post_episode()

        # Collect results — replicate logic from logging_utils.calculate_performance
        target_object = target["object"] if target else "unknown"
        terminal_state = lm.terminal_state
        detected_object = lm.detected_object
        possible_matches = lm.get_possible_matches()

        # Determine primary_performance (mirrors logging_utils.py:558-621)
        primary_performance = terminal_state
        if terminal_state == "match" and detected_object is not None:
            target_to_graph = lm.graph_id_to_target.get(detected_object, set())
            if target_object in target_to_graph:
                primary_performance = "correct"
            else:
                primary_performance = "confused"

        elif terminal_state == "time_out":
            if len(possible_matches) == 1:
                primary_performance = "pose_time_out"
            # Check MLH (mirrors logging_utils.py:832-837)
            mlh = lm.get_current_mlh()
            mlh_graph_id = mlh.get("graph_id")
            if mlh_graph_id is not None:
                target_to_graph = lm.graph_id_to_target.get(mlh_graph_id, set())
                if target_object in target_to_graph:
                    primary_performance = "correct_mlh"
                else:
                    primary_performance = "confused_mlh"
            detected_object = mlh_graph_id  # For CSV output

        is_correct = primary_performance == "correct"
        is_correct_mlh = primary_performance == "correct_mlh"
        status = primary_performance.upper()

        result = dict(
            episode=ep,
            target_object=target_object,
            detected_object=detected_object,
            terminal_state=terminal_state,
            primary_performance=primary_performance,
            steps=step,
            correct=is_correct,
            correct_mlh=is_correct_mlh,
            elapsed_sec=round(time.time() - ep_start, 1),
        )
        results.append(result)

        logger.info(
            f"Ep {ep:2d} [{status:11s}]: target={target_object}, "
            f"detected={detected_object}, state={terminal_state}, "
            f"steps={step}, time={result['elapsed_sec']}s"
        )
        # Always print a brief summary line to stdout
        print(
            f"[{ep+1:2d}/{num_episodes}] {status:11s} "
            f"target={target_object:30s} detected={str(detected_object):30s} "
            f"state={str(terminal_state):12s} steps={step:4d} "
            f"time={result['elapsed_sec']}s"
        )

        # Write to CSV immediately
        if csv_writer:
            csv_writer.writerow(result)
            csv_file.flush()

        # post_episode loads next scene; skip on last episode to avoid
        # FileNotFoundError when dataset has fewer scenes/versions than SCENES.
        if ep < num_episodes - 1:
            env_interface.post_episode()

    if csv_file:
        csv_file.close()

    return results


# ============================================================================
# Section 5: Results collection and output
# ============================================================================

def print_results(results):
    """Print summary statistics."""
    total = len(results)
    n_correct = sum(1 for r in results if r["correct"])
    n_correct_mlh = sum(1 for r in results if r["correct_mlh"])
    n_combined = n_correct + n_correct_mlh
    n_wrong = total - n_combined

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"Total episodes:      {total}")
    print(f"Correct (match):     {n_correct} ({100*n_correct/total:.1f}%)")
    print(f"Correct (MLH):       {n_correct_mlh} ({100*n_correct_mlh/total:.1f}%)")
    print(f"Combined correct:    {n_combined} ({100*n_combined/total:.1f}%)")
    print(f"Wrong:               {n_wrong} ({100*n_wrong/total:.1f}%)")

    # Per-object breakdown
    objects = {}
    for r in results:
        obj = r["target_object"]
        if obj not in objects:
            objects[obj] = dict(total=0, correct=0, correct_mlh=0)
        objects[obj]["total"] += 1
        if r["correct"]:
            objects[obj]["correct"] += 1
        if r["correct_mlh"]:
            objects[obj]["correct_mlh"] += 1

    print(f"\n{'Object':<30s} {'Total':>5s} {'Match':>5s} {'MLH':>5s} {'Acc%':>6s}")
    print("-" * 52)
    for obj in sorted(objects.keys()):
        s = objects[obj]
        acc = 100 * (s["correct"] + s["correct_mlh"]) / s["total"]
        print(f"{obj:<30s} {s['total']:5d} {s['correct']:5d} {s['correct_mlh']:5d} {acc:5.1f}%")
    print("=" * 70)


def save_csv(results, csv_path):
    """Save results to CSV file."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "episode", "target_object", "detected_object",
        "terminal_state", "primary_performance", "steps",
        "correct", "correct_mlh", "elapsed_sec",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {csv_path}")


# ============================================================================
# Section 6: argparse main
# ============================================================================

def default_model_path():
    """Resolve default pretrained model path from environment."""
    monty_models = os.environ.get(
        "MONTY_MODELS",
        os.path.expanduser("~/tbp/results/monty/pretrained_models"),
    )
    return os.path.join(
        monty_models,
        "pretrained_ycb_v12",
        "surf_agent_1lm_numenta_lab_obj",
        "pretrained",
    )


def default_data_path():
    """Resolve default data path from environment."""
    monty_data = os.environ.get("MONTY_DATA", os.path.expanduser("~/tbp/data"))
    return os.path.join(monty_data, "worldimages", "standard_scenes")


def main():
    parser = argparse.ArgumentParser(
        description="Standalone Monty inference (world_image_on_scanned_model)"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=default_data_path(),
        help="Path to world image dataset",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=default_model_path(),
        help="Path to pretrained model directory or model.pt file",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Path to save results CSV",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--max-eval-steps",
        type=int,
        default=MAX_EVAL_STEPS,
        help="Maximum matching steps per episode (default: 500)",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Run only first N episodes (default: all 48)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Python logging level (default: WARNING)",
    )

    args = parser.parse_args()

    print(f"Data path:  {args.data_path}")
    print(f"Model path: {args.model_path}")
    print(f"Seed: {args.seed}, Max eval steps: {args.max_eval_steps}")
    print()

    results = run_inference(
        data_path=args.data_path,
        model_path=args.model_path,
        max_eval_steps=args.max_eval_steps,
        seed=args.seed,
        log_level=args.log_level,
        output_csv=args.output_csv,
        max_episodes=args.max_episodes,
    )

    print_results(results)

    if args.output_csv and not Path(args.output_csv).exists():
        save_csv(results, args.output_csv)


if __name__ == "__main__":
    main()
