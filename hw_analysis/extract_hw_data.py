#!/usr/bin/env python3
"""Extract intermediate data from Monty inference for standalone HW ops testing.

Hooks into Monty's pipeline, runs one matching step, and saves all intermediate
values needed by hw_ops.py to a .npz file.

Usage:
    python hw_analysis/extract_hw_data.py
    # Output: hw_analysis/hw_test_data.npz
"""
import sys
import os
import copy
import logging
import numpy as np

# Add scripts/ to path for monty_inference imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from monty_inference import (
    create_sensor_modules,
    create_learning_module,
    create_motor_system,
    create_model,
    load_pretrained_model,
    default_data_path,
    default_model_path,
    SaccadeOnImageEnvironment,
    SaccadeOnImageEnvironmentInterface,
    ExperimentMode,
    SCENES, VERSIONS, MAX_TOTAL_STEPS,
)
from tbp.monty.context import RuntimeContext


def main():
    logging.basicConfig(level=logging.WARNING)

    # Build model
    sensor_modules = create_sensor_modules()
    lm = create_learning_module()
    motor_system = create_motor_system()
    model = create_model(sensor_modules, [lm], motor_system)
    model.set_experiment_mode(ExperimentMode.EVAL)
    lm.use_multithreading = False

    load_pretrained_model(model, default_model_path())

    env = SaccadeOnImageEnvironment(data_path=default_data_path())
    rng = np.random.RandomState(42)
    env_interface = SaccadeOnImageEnvironmentInterface(
        scenes=SCENES, versions=VERSIONS, env=env,
        motor_system=model.motor_system, rng=rng,
        transform=None, experiment_mode=ExperimentMode.EVAL, seed=42,
    )
    env_interface.pre_epoch()

    # Get the displacer to hook into
    updater = lm.hypotheses_updater
    displacer = updater.hypotheses_displacer

    captured = {}

    # Monkey-patch to capture intermediate values
    orig_calc_evidence = displacer._calculate_evidence_for_new_locations

    def patched_calc_evidence(graph_id, input_channel, search_locations,
                              channel_possible_poses, channel_features):
        captured["graph_id"] = graph_id
        captured["input_channel"] = input_channel
        captured["search_locations"] = search_locations.copy()
        captured["channel_possible_poses"] = channel_possible_poses.copy()
        captured["channel_features"] = copy.deepcopy(channel_features)

        graph = displacer.graph_memory.get_graph(graph_id, input_channel)
        captured["node_positions"] = graph.pos.copy()

        result = orig_calc_evidence(
            graph_id, input_channel, search_locations,
            channel_possible_poses, channel_features,
        )
        captured["evidence_result"] = result.copy()
        return result

    displacer._calculate_evidence_for_new_locations = patched_calc_evidence

    orig_displace = displacer.displace_hypotheses_and_compute_evidence

    def patched_displace(channel_displacement, channel_features,
                         evidence_update_threshold, graph_id,
                         possible_hypotheses, total_hypotheses_count):
        captured["displacement"] = channel_displacement.copy()
        captured["old_locations"] = possible_hypotheses.locations.copy()
        captured["old_poses"] = possible_hypotheses.poses.copy()
        captured["old_evidence"] = possible_hypotheses.evidence.copy()
        captured["pose_vectors_input"] = copy.deepcopy(channel_features)

        result = orig_displace(
            channel_displacement, channel_features,
            evidence_update_threshold, graph_id,
            possible_hypotheses, total_hypotheses_count,
        )
        captured["new_evidence"] = result[0].evidence.copy()
        captured["ready"] = True
        return result

    displacer.displace_hypotheses_and_compute_evidence = patched_displace

    # Run until we capture data
    ctx = RuntimeContext(rng=rng)
    target = env_interface.primary_target
    model.pre_episode(target)
    env_interface.pre_episode(rng)

    captured["ready"] = False
    step = 0
    while step < MAX_TOTAL_STEPS:
        observations = env_interface.step(ctx, first=(step == 0))
        if model.is_motor_only_step:
            model.pass_features_directly_to_motor_system(ctx, observations)
        else:
            model.step(ctx, observations)

        if captured.get("ready"):
            break
        if model.is_done:
            break
        step += 1

    if not captured.get("ready"):
        print("ERROR: Could not capture data")
        sys.exit(1)

    # Extract all data needed for standalone HW ops
    graph_id = captured["graph_id"]
    input_channel = captured["input_channel"]
    channel_features = captured["channel_features"]

    from tbp.monty.frameworks.utils.graph_matching_utils import get_relevant_curvature
    from tbp.monty.frameworks.utils.spatial_arithmetics import get_angles_for_all_hypotheses

    curvature = float(get_relevant_curvature(channel_features))
    K = displacer.max_nneighbors
    max_match_distance = displacer.max_match_distance
    past_weight = displacer.past_weight
    present_weight = displacer.present_weight

    # Get KDTree results
    graph = displacer.graph_memory.get_graph(graph_id, input_channel)
    search_locs = captured["search_locations"]
    nn_dists, nn_ids = graph._location_tree.query(search_locs, k=K, p=2, workers=1)

    # Node locations and features at NN indices
    all_locs = displacer.graph_memory.get_locations_in_graph(graph_id, input_channel)
    nn_locs = all_locs[nn_ids]

    # Node pose features
    node_pose_features = displacer.graph_memory.get_features_at_node(
        graph_id, input_channel, nn_ids,
        feature_keys=["pose_vectors", "pose_fully_defined"],
    )
    node_pv = node_pose_features["pose_vectors"]  # (H, K, 9)

    # Feature evidence data
    feat_array = displacer.graph_memory.get_feature_array(graph_id)[input_channel]
    feat_order = displacer.graph_memory.get_feature_order(graph_id)[input_channel]
    tols = displacer.tolerances[input_channel]
    fweights = displacer.feature_weights[input_channel]

    # Build feature arrays
    F = feat_array.shape[1]
    tol_list = np.zeros(F)
    weight_list = np.zeros(F)
    query_list = np.zeros(F)
    circular = np.zeros(F, dtype=bool)

    start_idx = 0
    for feat_name in feat_order:
        if feat_name in ["pose_vectors", "pose_fully_defined"]:
            continue
        feat_val = channel_features[feat_name]
        flen = len(feat_val) if hasattr(feat_val, "__len__") else 1
        end_idx = start_idx + flen
        query_list[start_idx:end_idx] = feat_val
        tol_list[start_idx:end_idx] = tols[feat_name]
        weight_list[start_idx:end_idx] = fweights[feat_name]
        circular[start_idx:end_idx] = (
            [True, False, False] if feat_name == "hsv" else False
        )
        start_idx = end_idx

    # Feature weights for pose evidence
    fw = displacer.feature_weights[input_channel]
    w_sn = fw["pose_vectors"][0]
    w_cd = fw["pose_vectors"][1] if channel_features.get("pose_fully_defined", False) else 0.0
    pose_fully_defined = bool(channel_features.get("pose_fully_defined", False))
    feature_evidence_increment = displacer.feature_evidence_increment

    # Per-node pose_fully_defined mask (used for CD evidence)
    node_pose_fully_defined = node_pose_features["pose_fully_defined"]  # (H, K, 1)

    # Monty's feature evidence (ground truth)
    monty_feat_ev = displacer.feature_evidence_calculator.calculate(
        channel_feature_array=feat_array,
        channel_feature_order=feat_order,
        channel_feature_weights=fweights,
        channel_query_features=channel_features,
        channel_tolerances=tols,
        input_channel=input_channel,
    )

    # Save everything
    out_path = os.path.join(os.path.dirname(__file__), "hw_test_data.npz")
    np.savez(
        out_path,
        # Step 6 inputs
        poses=captured["old_poses"],
        locations=captured["old_locations"],
        displacement=captured["displacement"],
        pose_vectors=channel_features["pose_vectors"],
        # Step 7 inputs
        search_locations=search_locs,
        node_positions=captured["node_positions"],
        K=K,
        # Step 7 outputs (ground truth)
        nn_distances=nn_dists,
        nn_indices=nn_ids,
        nn_locations=nn_locs,
        # Step 8 inputs
        curvature=curvature,
        max_match_distance=max_match_distance,
        node_pose_vectors=node_pv,
        surface_normals=captured["channel_possible_poses"].dot(
            channel_features["pose_vectors"].T
        ).transpose((0, 2, 1))[:, 0],
        pose_fully_defined=pose_fully_defined,
        node_pose_fully_defined=node_pose_fully_defined,
        w_sn=w_sn,
        w_cd=w_cd,
        feature_evidence_increment=feature_evidence_increment,
        # Step 9 inputs
        feature_array=feat_array,
        query_features=query_list,
        tolerances=tol_list,
        feature_weights=weight_list,
        circular_mask=circular,
        # Step 10 inputs
        old_evidence=captured["old_evidence"],
        past_weight=past_weight,
        present_weight=present_weight,
        # Ground truth outputs
        monty_evidence_result=captured["evidence_result"],
        monty_new_evidence=captured["new_evidence"],
        monty_feature_evidence=monty_feat_ev,
        # Metadata
        graph_id=graph_id,
        input_channel=input_channel,
    )

    H = captured["old_poses"].shape[0]
    N = captured["node_positions"].shape[0]
    print(f"Data extracted and saved to {out_path}")
    print(f"  Graph: {graph_id}, Channel: {input_channel}")
    print(f"  H={H}, N={N}, K={K}, F={F}")
    print(f"  pose_fully_defined={pose_fully_defined}")
    print(f"  File size: {os.path.getsize(out_path) / 1024:.0f} KB")


if __name__ == "__main__":
    main()
