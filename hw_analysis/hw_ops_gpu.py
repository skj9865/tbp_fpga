#!/usr/bin/env python3
"""GPU-accelerated HW operations using PyTorch.

hw_ops.py와 동일한 연산을 GPU(MPS/CUDA)에서 병렬 수행합니다.
결과를 hw_ops.py(CPU)와 비교하여 정확성을 검증합니다.

Usage:
    python hw_analysis/hw_ops_gpu.py
"""
import os
import time
import numpy as np
import torch


def select_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def sync(device):
    """Synchronize GPU to get accurate timing."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        if hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()
        else:
            # PyTorch < 2.0: force sync by transferring a small tensor
            torch.tensor([0.0], device=device).cpu()


def load_data(path=None):
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "hw_test_data.npz")
    if not os.path.exists(path):
        print(f"ERROR: {path} not found.")
        print("Run extract_hw_data.py first to generate test data.")
        raise SystemExit(1)
    return np.load(path, allow_pickle=True)


# ============================================================================
# Step 6. Rotation (GPU)
# ============================================================================

def step6a_rotate_displacement(poses, displacement):
    """poses[H,3,3] @ displacement[3] → [H,3]"""
    return torch.matmul(poses, displacement)


def step6b_search_locations(locations, rotated_disp):
    """locations[H,3] + rotated_disp[H,3] → [H,3]"""
    return locations + rotated_disp


def step6c_rotate_pose_vectors(poses, pose_vectors):
    """poses[H,3,3] @ pose_vectors.T[3,3] → transpose → [H,3,3]"""
    pv_T = pose_vectors.T
    result = torch.matmul(poses, pv_T)
    return result.permute(0, 2, 1)


# ============================================================================
# Step 7. Nearest Neighbor Search (GPU brute-force)
# ============================================================================

def step7_nn_search(search_locations, node_positions, K):
    """Brute-force NN: all pairwise distances, then top-K.

    search_locations[H,3], node_positions[N,3] → distances[H,K], indices[H,K]
    """
    # [H,1,3] - [1,N,3] → [H,N,3] → sum → [H,N]
    diff = search_locations.unsqueeze(1) - node_positions.unsqueeze(0)
    dist_sq = (diff * diff).sum(dim=2)

    # Top-K smallest (sorted=True returns in ascending order of -dist_sq,
    # i.e. descending dist_sq, so we negate)
    topk = torch.topk(-dist_sq, k=K, dim=1, sorted=True)
    nn_indices = topk.indices  # [H, K]
    nn_distances = torch.sqrt(-topk.values)  # [H, K]

    return nn_distances, nn_indices


# ============================================================================
# Step 8. Custom Distance + Pose Evidence (GPU)
# ============================================================================

def step8a_custom_distances(nn_locs, search_locs, surface_normals, curvature):
    """Custom distance = euclidean + |dot(diff, normal)| / (|curv|+0.5)"""
    diff = nn_locs - search_locs.unsqueeze(1)  # [H,K,3]
    eucl = torch.sqrt((diff * diff).sum(dim=2))  # [H,K]
    dot = (diff * surface_normals.unsqueeze(1)).sum(dim=2)  # [H,K]
    curv_factor = 1.0 / (abs(curvature) + 0.5)
    return eucl + torch.abs(dot) * curv_factor


def step8b_distance_weights(custom_dists, max_match_distance):
    """weights = (max_dist - dist) / max_dist, mask = weights <= 0"""
    weights = (max_match_distance - custom_dists) / max_match_distance
    mask = weights <= 0
    return weights, mask


def step8c_surface_normal_evidence(node_pv, query_pv):
    """dot → arccos → -(sin(angle/2) - 0.5)"""
    dot = (node_pv * query_pv.unsqueeze(1)).sum(dim=2)
    dot = torch.clamp(dot, -1.0, 1.0)
    angles = torch.arccos(dot)
    evidence = -(torch.sin(angles / 2.0) - 0.5)
    return angles, evidence


def step8d_curvature_direction_evidence(node_pv, query_pv):
    """dot → arccos → error = pi/2 - |angle - pi/2| → -(sin(error) - 0.5)"""
    dot = (node_pv * query_pv.unsqueeze(1)).sum(dim=2)
    dot = torch.clamp(dot, -1.0, 1.0)
    angles = torch.arccos(dot)
    half_pi = torch.pi / 2.0
    error = half_pi - torch.abs(angles - half_pi)
    evidence = -(torch.sin(error) - 0.5)
    return angles, evidence


def step8e_pose_evidence(sn_evidence, cd_evidence, w_sn, w_cd, use_cd_mask=None):
    """Weighted sum: pose_ev = sn_ev * w_sn + cd_ev * w_cd"""
    if use_cd_mask is not None:
        cd_evidence = cd_evidence * use_cd_mask.float()
    return sn_evidence * w_sn + cd_evidence * w_cd


# ============================================================================
# Step 9. Feature Evidence (GPU)
# ============================================================================

def step9_feature_evidence(feature_array, query_features, tolerances, weights,
                           circular_mask):
    """Feature comparison → clamp → weighted average."""
    N, F = feature_array.shape

    # Compute differences
    diff = torch.abs(feature_array - query_features.unsqueeze(0))  # [N, F]

    # Circular (hue) handling
    if circular_mask.any():
        d1 = torch.abs(1.0 + feature_array - query_features.unsqueeze(0))
        d3 = torch.abs(feature_array - (query_features.unsqueeze(0) + 1.0))
        circular_diff = torch.minimum(torch.minimum(d1, diff), d3)
        diff = torch.where(circular_mask.unsqueeze(0), circular_diff, diff)

    # Evidence: clamp(tol - diff, 0) / tol
    evidence = torch.clamp(tolerances.unsqueeze(0) - diff, min=0) / tolerances.unsqueeze(0)

    # Weighted average
    result = (evidence * weights.unsqueeze(0)).sum(dim=1) / weights.sum()
    return result


# ============================================================================
# Step 10. Evidence Accumulation (GPU)
# ============================================================================

def step10_accumulate_evidence(pose_evidence, feature_evidence, nn_ids,
                                dist_mask, old_evidence,
                                past_weight, present_weight,
                                feat_evidence_increment=1.0):
    """Combine pose + feature, max over K, weighted accumulation."""
    # Feature evidence at NN indices: [H, K]
    feat_at_nn = feature_evidence[nn_ids]

    # Radius evidence
    radius_evidence = pose_evidence + feat_at_nn * feat_evidence_increment
    radius_evidence[dist_mask] = -1.0

    # Max over K
    evidence_to_add = radius_evidence.max(dim=1).values  # [H]

    # Weighted accumulation
    new_evidence = old_evidence * past_weight + evidence_to_add * present_weight

    return radius_evidence, evidence_to_add, new_evidence


# ============================================================================
# Main
# ============================================================================

def main():
    device = select_device()
    print(f"Device: {device}")
    print()

    data = load_data()

    H = data["poses"].shape[0]
    N = data["node_positions"].shape[0]
    K = int(data["K"])
    F = data["feature_array"].shape[1]

    print("=" * 70)
    print(f"HW Ops — GPU ({device}) vs CPU Comparison")
    print("=" * 70)
    print(f"  H={H}, N={N}, K={K}, F={F}")
    print()

    # Load ground truth from Monty
    gt_search_locs = data["search_locations"]
    gt_nn_dists = data["nn_distances"]
    gt_nn_ids = data["nn_indices"]
    gt_feat_ev = data["monty_feature_evidence"]
    gt_ev_result = data["monty_evidence_result"]

    # Transfer data to GPU
    # MPS doesn't support float64; use float32 for GPU, float64 for CUDA/CPU
    dtype = torch.float32 if device.type == "mps" else torch.float64

    poses = torch.tensor(data["poses"], dtype=dtype, device=device)
    locations = torch.tensor(data["locations"], dtype=dtype, device=device)
    displacement = torch.tensor(data["displacement"], dtype=dtype, device=device)
    pose_vectors = torch.tensor(data["pose_vectors"], dtype=dtype, device=device)
    node_positions = torch.tensor(data["node_positions"], dtype=dtype, device=device)
    curvature = float(data["curvature"])
    max_match_distance = float(data["max_match_distance"])
    node_pv = torch.tensor(data["node_pose_vectors"], dtype=dtype, device=device)
    pose_fully_defined = bool(data["pose_fully_defined"])
    w_sn = float(data["w_sn"])
    w_cd = float(data["w_cd"])
    node_pose_fd = data["node_pose_fully_defined"]
    use_cd_mask = torch.tensor(node_pose_fd[:, :, 0], dtype=torch.bool, device=device) if pose_fully_defined else None
    feat_ev_increment = float(data["feature_evidence_increment"])
    feature_array = torch.tensor(data["feature_array"], dtype=dtype, device=device)
    query_features = torch.tensor(data["query_features"], dtype=dtype, device=device)
    tolerances = torch.tensor(data["tolerances"], dtype=dtype, device=device)
    feat_weights = torch.tensor(data["feature_weights"], dtype=dtype, device=device)
    circular_mask = torch.tensor(data["circular_mask"], dtype=torch.bool, device=device)
    old_evidence = torch.tensor(data["old_evidence"], dtype=dtype, device=device)
    past_weight = float(data["past_weight"])
    present_weight = float(data["present_weight"])

    # Warm up GPU
    _ = torch.matmul(poses[:10], displacement)
    sync(device)

    atol = 1e-4 if dtype == torch.float32 else 1e-10
    all_ok = True

    # =========================================================================
    # Step 6
    # =========================================================================
    print("Step 6. Rotation")
    sync(device)
    t0 = time.time()

    rotated_disp = step6a_rotate_displacement(poses, displacement)
    search_locs = step6b_search_locations(locations, rotated_disp)
    rotated_pv = step6c_rotate_pose_vectors(poses, pose_vectors)

    sync(device)
    t6 = time.time() - t0

    diff = np.max(np.abs(search_locs.cpu().numpy() - gt_search_locs))
    ok = diff < atol
    print(f"  Verify: max_diff={diff:.2e} {'OK' if ok else 'FAIL'} ({t6:.4f}s)")
    all_ok = all_ok and ok

    surface_normals = rotated_pv[:, 0]

    # =========================================================================
    # Step 7
    # =========================================================================
    print("Step 7. NN Search")
    sync(device)
    t0 = time.time()

    nn_dists, nn_ids = step7_nn_search(search_locs, node_positions, K)

    sync(device)
    t7 = time.time() - t0

    nn_dists_np = nn_dists.cpu().numpy()
    nn_ids_np = nn_ids.cpu().numpy()
    dist_diff = np.max(np.abs(nn_dists_np - gt_nn_dists))
    ids_match = np.all(nn_ids_np == gt_nn_ids)
    ok = dist_diff < atol and ids_match
    print(f"  Verify: dist_diff={dist_diff:.2e}, ids_match={ids_match} {'OK' if ok else 'FAIL'} ({t7:.4f}s)")
    all_ok = all_ok and ok

    # =========================================================================
    # Step 8
    # =========================================================================
    print("Step 8. Pose Evidence")
    nn_locs = node_positions[nn_ids]

    sync(device)
    t0 = time.time()

    custom_dists = step8a_custom_distances(nn_locs, search_locs, surface_normals, curvature)
    dist_weights, dist_mask = step8b_distance_weights(custom_dists, max_match_distance)

    sn_node_pv = node_pv[:, :, :3].reshape(H, K, 3)
    sn_query_pv = rotated_pv[:, 0]
    sn_angles, sn_evidence = step8c_surface_normal_evidence(sn_node_pv, sn_query_pv)

    if pose_fully_defined:
        cd_node_pv = node_pv[:, :, 3:6].reshape(H, K, 3)
        cd_query_pv = rotated_pv[:, 1]
        cd_angles, cd_evidence = step8d_curvature_direction_evidence(cd_node_pv, cd_query_pv)
    else:
        cd_evidence = torch.zeros_like(sn_evidence)

    pose_evidence = step8e_pose_evidence(sn_evidence, cd_evidence, w_sn, w_cd, use_cd_mask)

    sync(device)
    t8 = time.time() - t0
    print(f"  Pose evidence range: [{pose_evidence.min().item():.4f}, {pose_evidence.max().item():.4f}] ({t8:.4f}s)")

    # =========================================================================
    # Step 9
    # =========================================================================
    print("Step 9. Feature Evidence")
    sync(device)
    t0 = time.time()

    feat_evidence = step9_feature_evidence(
        feature_array, query_features, tolerances, feat_weights, circular_mask
    )

    sync(device)
    t9 = time.time() - t0

    diff = np.max(np.abs(feat_evidence.cpu().numpy() - gt_feat_ev))
    ok = diff < atol
    print(f"  Verify: max_diff={diff:.2e} {'OK' if ok else 'FAIL'} ({t9:.4f}s)")
    all_ok = all_ok and ok

    # =========================================================================
    # Step 10
    # =========================================================================
    print("Step 10. Evidence Accumulation")
    sync(device)
    t0 = time.time()

    radius_ev, ev_to_add, new_evidence = step10_accumulate_evidence(
        pose_evidence, feat_evidence, nn_ids, dist_mask,
        old_evidence, past_weight, present_weight, feat_ev_increment,
    )

    sync(device)
    t10 = time.time() - t0

    diff = np.max(np.abs(ev_to_add.cpu().numpy() - gt_ev_result))
    ok = diff < atol
    print(f"  Verify: max_diff={diff:.2e} {'OK' if ok else 'FAIL'} ({t10:.4f}s)")
    all_ok = all_ok and ok

    # =========================================================================
    # Summary
    # =========================================================================
    total = t6 + t7 + t8 + t9 + t10

    # Run CPU version for comparison
    print()
    print("Running CPU (hw_ops.py) for comparison...")
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "hw_ops", os.path.join(os.path.dirname(__file__), "hw_ops.py")
    )
    hw_ops_cpu = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hw_ops_cpu)

    cpu_data = hw_ops_cpu.load_data()
    t0 = time.time()
    cpu_rd = hw_ops_cpu.step6a_rotate_displacement(cpu_data["poses"], cpu_data["displacement"])
    cpu_sl = hw_ops_cpu.step6b_search_locations(cpu_data["locations"], cpu_rd)
    cpu_rpv = hw_ops_cpu.step6c_rotate_pose_vectors(cpu_data["poses"], cpu_data["pose_vectors"])
    t6_cpu = time.time() - t0

    t0 = time.time()
    cpu_nn_d, cpu_nn_i = hw_ops_cpu.step7_nn_search(cpu_sl, cpu_data["node_positions"], K)
    t7_cpu = time.time() - t0

    t0 = time.time()
    cpu_nn_locs = cpu_data["node_positions"][cpu_nn_i]
    cpu_sn = cpu_rpv[:, 0]
    cpu_cd = hw_ops_cpu.step8a_custom_distances(cpu_nn_locs, cpu_sl, cpu_sn, float(cpu_data["curvature"]))
    cpu_dw, cpu_dm = hw_ops_cpu.step8b_distance_weights(cpu_cd, float(cpu_data["max_match_distance"]))
    cpu_sn_pv = cpu_data["node_pose_vectors"][:, :, :3].reshape(H, K, 3)
    _, cpu_sn_ev = hw_ops_cpu.step8c_surface_normal_evidence(cpu_sn_pv, cpu_rpv[:, 0])
    if pose_fully_defined:
        cpu_cd_pv = cpu_data["node_pose_vectors"][:, :, 3:6].reshape(H, K, 3)
        _, cpu_cd_ev = hw_ops_cpu.step8d_curvature_direction_evidence(cpu_cd_pv, cpu_rpv[:, 1])
    else:
        cpu_cd_ev = np.zeros_like(cpu_sn_ev)
    cpu_use_cd = np.array(cpu_data["node_pose_fully_defined"][:, :, 0], dtype=bool) if pose_fully_defined else None
    cpu_pe = hw_ops_cpu.step8e_pose_evidence(cpu_sn_ev, cpu_cd_ev, w_sn, w_cd, cpu_use_cd)
    t8_cpu = time.time() - t0

    t0 = time.time()
    cpu_fe = hw_ops_cpu.step9_feature_evidence(
        cpu_data["feature_array"], cpu_data["query_features"],
        cpu_data["tolerances"], cpu_data["feature_weights"],
        cpu_data["circular_mask"]
    )
    t9_cpu = time.time() - t0

    t0 = time.time()
    _, cpu_eta, _ = hw_ops_cpu.step10_accumulate_evidence(
        cpu_pe, cpu_fe, cpu_nn_i, cpu_dw, cpu_dm,
        cpu_data["old_evidence"], past_weight, present_weight, feat_ev_increment
    )
    t10_cpu = time.time() - t0

    total_cpu = t6_cpu + t7_cpu + t8_cpu + t9_cpu + t10_cpu

    print()
    print("=" * 70)
    print("COMPARISON: GPU vs CPU")
    print("=" * 70)
    print(f"  {'Step':20s} {'GPU':>10s} {'CPU':>10s} {'Speedup':>10s}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10}")
    for name, tg, tc in [
        ("6. Rotation", t6, t6_cpu),
        ("7. NN Search", t7, t7_cpu),
        ("8. Pose Evidence", t8, t8_cpu),
        ("9. Feature Evidence", t9, t9_cpu),
        ("10. Accumulation", t10, t10_cpu),
    ]:
        speedup = tc / tg if tg > 0 else float('inf')
        print(f"  {name:20s} {tg:>9.4f}s {tc:>9.4f}s {speedup:>9.1f}x")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10}")
    speedup_total = total_cpu / total if total > 0 else float('inf')
    print(f"  {'TOTAL':20s} {total:>9.4f}s {total_cpu:>9.4f}s {speedup_total:>9.1f}x")
    print()
    print(f"  All verifications: {'PASSED' if all_ok else 'FAILED'}")
    print()


if __name__ == "__main__":
    main()
