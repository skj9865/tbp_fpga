#!/usr/bin/env python3
"""Standalone HW operations script — Monty 의존성 없이 numpy만 사용.

Monty 추론 파이프라인에서 FPGA로 가속할 연산(Steps 6-10)만 분리하여 실행합니다.
extract_hw_data.py로 추출한 테스트 데이터를 로드하여 각 단계를 수행합니다.

Usage:
    python scripts/extract_hw_data.py   # 먼저 데이터 추출 (1회만)
    python scripts/hw_ops.py            # HW 연산 실행
"""
import os
import time
import numpy as np


def load_data(path=None):
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "hw_test_data.npz")
    if not os.path.exists(path):
        print(f"ERROR: {path} not found.")
        print("Run extract_hw_data.py first to generate test data.")
        raise SystemExit(1)
    return np.load(path, allow_pickle=True)


# ============================================================================
# Step 6. Rotation
# ============================================================================

def step6a_rotate_displacement(poses, displacement):
    """각 가설의 pose 행렬로 displacement 벡터를 회전.

    poses[H, 3, 3] × displacement[3] → rotated[H, 3]

    연산: 각 가설 h, 각 축 i에 대해:
      rotated[h, i] = poses[h,i,0]*disp[0] + poses[h,i,1]*disp[1] + poses[h,i,2]*disp[2]

    총: 9H 곱셈, 6H 덧셈
    """
    H = poses.shape[0]
    result = np.zeros((H, 3))
    for h in range(H):
        for i in range(3):
            result[h, i] = (poses[h, i, 0] * displacement[0]
                          + poses[h, i, 1] * displacement[1]
                          + poses[h, i, 2] * displacement[2])
    return result


def step6b_search_locations(locations, rotated_disp):
    """현재 위치에 회전된 displacement를 더하여 탐색 위치 계산.

    locations[H, 3] + rotated_disp[H, 3] → search_locations[H, 3]

    총: 3H 덧셈
    """
    H = locations.shape[0]
    result = np.zeros((H, 3))
    for h in range(H):
        for i in range(3):
            result[h, i] = locations[h, i] + rotated_disp[h, i]
    return result


def step6c_rotate_pose_vectors(poses, pose_vectors):
    """각 가설의 pose로 query pose_vectors를 회전.

    poses[H, 3, 3] × pose_vectors.T[3, 3] → rotated_pv[H, 3, 3]

    총: 27H 곱셈, 18H 덧셈
    """
    H = poses.shape[0]
    pv_T = pose_vectors.T  # [3, 3]
    result = np.zeros((H, 3, 3))
    for h in range(H):
        for i in range(3):
            for j in range(3):
                result[h, i, j] = (poses[h, i, 0] * pv_T[0, j]
                                 + poses[h, i, 1] * pv_T[1, j]
                                 + poses[h, i, 2] * pv_T[2, j])
    # Transpose: 각 행이 pose vector가 되도록
    return result.transpose((0, 2, 1))


# ============================================================================
# Step 7. Nearest Neighbor Search (Brute-force)
# ============================================================================

def step7_nn_search(search_locations, node_positions, K):
    """각 가설의 탐색 위치에서 가장 가까운 K개 노드 찾기.

    search_locations[H, 3], node_positions[N, 3] → distances[H,K], indices[H,K]

    각 가설 h, 각 노드 n에 대해:
      diff = search_loc[h] - node_pos[n]        (3 뺄셈)
      dist_sq = diff[0]² + diff[1]² + diff[2]²  (3 곱셈 + 2 덧셈)

    총: 3HN 곱셈, 5HN 덧셈/뺄셈, HN 비교 (top-K)
    """
    H = search_locations.shape[0]
    N = node_positions.shape[0]
    nearest_ids = np.zeros((H, K), dtype=int)
    distances = np.zeros((H, K))

    for h in range(H):
        dist_sq = np.zeros(N)
        for n in range(N):
            dx = search_locations[h, 0] - node_positions[n, 0]
            dy = search_locations[h, 1] - node_positions[n, 1]
            dz = search_locations[h, 2] - node_positions[n, 2]
            dist_sq[n] = dx * dx + dy * dy + dz * dz

        # Top-K selection (partial sort)
        idx = np.argpartition(dist_sq, K)[:K]
        idx = idx[np.argsort(dist_sq[idx])]
        nearest_ids[h] = idx
        distances[h] = np.sqrt(dist_sq[idx])

    return distances, nearest_ids


# ============================================================================
# Step 8. Custom Distance + Pose Evidence
# ============================================================================

def step8a_custom_distances(nn_locs, search_locs, surface_normals, curvature):
    """Custom distance = Euclidean + |dot(diff, normal)| / (|curv| + 0.5)

    nn_locs[H, K, 3], search_locs[H, 3], surface_normals[H, 3] → dists[H, K]

    각 (h, k)에 대해:
      diff = nn_loc - search_loc                (3 뺄셈)
      eucl = sqrt(diff²)                        (3 곱셈 + 2 덧셈 + 1 sqrt)
      dot = diff · normal                       (3 곱셈 + 2 덧셈)
      custom = eucl + |dot| / (|curv| + 0.5)    (1 abs + 1 add + 1 div + 1 mul + 1 add)

    총 per HK: 6 mul, 4 add, 1 sqrt, 1 div, 1 abs
    """
    H, K, _ = nn_locs.shape
    result = np.zeros((H, K))
    curv_factor = 1.0 / (abs(curvature) + 0.5)

    for h in range(H):
        for k in range(K):
            dx = nn_locs[h, k, 0] - search_locs[h, 0]
            dy = nn_locs[h, k, 1] - search_locs[h, 1]
            dz = nn_locs[h, k, 2] - search_locs[h, 2]

            eucl = np.sqrt(dx * dx + dy * dy + dz * dz)
            dot = (dx * surface_normals[h, 0]
                 + dy * surface_normals[h, 1]
                 + dz * surface_normals[h, 2])

            result[h, k] = eucl + abs(dot) * curv_factor

    return result


def step8b_distance_weights(custom_dists, max_match_distance):
    """거리 기반 가중치: 가까울수록 높은 weight, 임계값 초과 시 0 이하.

    weights[h,k] = (max_dist - custom_dist[h,k]) / max_dist

    총 per HK: 1 sub, 1 div, 1 cmp
    """
    H, K = custom_dists.shape
    weights = np.zeros((H, K))
    for h in range(H):
        for k in range(K):
            weights[h, k] = (max_match_distance - custom_dists[h, k]) / max_match_distance

    mask = weights <= 0
    return weights, mask


def step8c_surface_normal_evidence(node_pv, query_pv):
    """Surface normal dot product → angle → evidence.

    node_pv[H, K, 3] · query_pv[H, 3] → dot → arccos → angle
    evidence = -(sin(angle/2) - 0.5)

    총 per HK: 3 mul, 2 add, 1 arccos, 1 sin, 1 div, 1 sub, 1 neg
    """
    H, K, _ = node_pv.shape
    angles = np.zeros((H, K))
    evidence = np.zeros((H, K))

    for h in range(H):
        for k in range(K):
            dot = (node_pv[h, k, 0] * query_pv[h, 0]
                 + node_pv[h, k, 1] * query_pv[h, 1]
                 + node_pv[h, k, 2] * query_pv[h, 2])

            # Clamp to [-1, 1]
            if dot > 1.0:
                dot = 1.0
            elif dot < -1.0:
                dot = -1.0

            angle = np.arccos(dot)
            angles[h, k] = angle
            evidence[h, k] = -(np.sin(angle / 2.0) - 0.5)

    return angles, evidence


def step8d_curvature_direction_evidence(node_pv, query_pv):
    """Curvature direction: dot → angle → error = pi/2 - |angle - pi/2| → evidence.

    총 per HK: 3 mul, 2 add, 1 arccos, 1 sin, 2 sub, 1 abs, 1 neg
    """
    H, K, _ = node_pv.shape
    half_pi = np.pi / 2.0
    angles = np.zeros((H, K))
    evidence = np.zeros((H, K))

    for h in range(H):
        for k in range(K):
            dot = (node_pv[h, k, 0] * query_pv[h, 0]
                 + node_pv[h, k, 1] * query_pv[h, 1]
                 + node_pv[h, k, 2] * query_pv[h, 2])

            if dot > 1.0:
                dot = 1.0
            elif dot < -1.0:
                dot = -1.0

            angle = np.arccos(dot)
            error = half_pi - abs(angle - half_pi)
            angles[h, k] = angle
            evidence[h, k] = -(np.sin(error) - 0.5)

    return angles, evidence


def step8e_pose_evidence(sn_evidence, cd_evidence, w_sn, w_cd, use_cd_mask=None):
    """가중 합산: pose_ev = sn_ev * w_sn + cd_ev * w_cd

    use_cd_mask[H, K]: 노드별 pose_fully_defined. False인 노드는 cd_evidence=0.

    총 per HK: 2 mul, 1 add
    """
    H, K = sn_evidence.shape
    result = np.zeros((H, K))
    for h in range(H):
        for k in range(K):
            cd = cd_evidence[h, k]
            if use_cd_mask is not None and not use_cd_mask[h, k]:
                cd = 0.0
            result[h, k] = sn_evidence[h, k] * w_sn + cd * w_cd
    return result


# ============================================================================
# Step 9. Feature Evidence
# ============================================================================

def step9_feature_evidence(feature_array, query_features, tolerances, weights,
                           circular_mask):
    """각 노드의 feature와 query feature 비교 → evidence.

    feature_array[N, F], query[F] → evidence[N]

    각 노드 n, feature f에 대해:
      - Hue (circular): diff = min(|1+stored-query|, |stored-query|, |stored-query-1|)
      - Others:         diff = |stored - query|
      - ev = clamp(tolerance - diff, 0) / tolerance
      - result = weighted average over features

    총 per N: ~3F sub + F div + F mul + F cmp
    """
    N, F = feature_array.shape

    # Compute per-feature differences
    diffs = np.zeros((N, F))
    for n in range(N):
        for f in range(F):
            if circular_mask[f]:
                d1 = abs(1.0 + feature_array[n, f] - query_features[f])
                d2 = abs(feature_array[n, f] - query_features[f])
                d3 = abs(feature_array[n, f] - (query_features[f] + 1.0))
                diffs[n, f] = min(d1, d2, d3)
            else:
                diffs[n, f] = abs(feature_array[n, f] - query_features[f])

    # Clamp and normalize
    evidence = np.zeros((N, F))
    for n in range(N):
        for f in range(F):
            ev = tolerances[f] - diffs[n, f]
            if ev < 0:
                ev = 0
            evidence[n, f] = ev / tolerances[f]

    # Weighted average
    weight_sum = 0.0
    for f in range(F):
        weight_sum += weights[f]

    result = np.zeros(N)
    for n in range(N):
        s = 0.0
        for f in range(F):
            s += evidence[n, f] * weights[f]
        result[n] = s / weight_sum

    return result


# ============================================================================
# Step 10. Evidence Accumulation
# ============================================================================

def step10_accumulate_evidence(pose_evidence, feature_evidence, nn_ids,
                                dist_weights, dist_mask,
                                old_evidence, past_weight, present_weight,
                                feat_evidence_increment=1.0):
    """Pose + Feature evidence를 합산, K개 중 최대값, 가중 누적.

    각 가설 h에 대해:
      radius_evidence[h, k] = pose_ev[h, k] + feat_ev[nn_ids[h, k]] * feat_weight
      (mask가 True이면 -1)
      evidence_to_add[h] = max(radius_evidence[h, :])
      new_evidence[h] = old_evidence[h] * past_weight + evidence_to_add[h] * present_weight

    총 per H: K mul + K add + (K-1) cmp + 2 mul + 1 add
    """
    H, K = pose_evidence.shape

    feat_increment = feat_evidence_increment

    # Combine pose + feature evidence per (h, k)
    radius_evidence = np.zeros((H, K))
    for h in range(H):
        for k in range(K):
            feat_ev = feature_evidence[nn_ids[h, k]]
            radius_evidence[h, k] = pose_evidence[h, k] + feat_ev * feat_increment

            # Apply distance mask: if too far, set to -1
            if dist_mask[h, k]:
                radius_evidence[h, k] = -1.0

    # Max over K neighbors
    evidence_to_add = np.zeros(H)
    for h in range(H):
        best = radius_evidence[h, 0]
        for k in range(1, K):
            if radius_evidence[h, k] > best:
                best = radius_evidence[h, k]
        evidence_to_add[h] = best

    # Weighted accumulation
    new_evidence = np.zeros(H)
    for h in range(H):
        new_evidence[h] = (old_evidence[h] * past_weight
                         + evidence_to_add[h] * present_weight)

    return radius_evidence, evidence_to_add, new_evidence


# ============================================================================
# Main: Load data and run all steps
# ============================================================================

def main():
    data = load_data()

    # --- Metadata ---
    H = data["poses"].shape[0]
    N = data["node_positions"].shape[0]
    K = int(data["K"])
    F = data["feature_array"].shape[1]
    graph_id = str(data["graph_id"])
    input_channel = str(data["input_channel"])

    print("=" * 70)
    print("HW Accelerated Operations — Standalone Execution")
    print("=" * 70)
    print(f"  Graph: {graph_id}")
    print(f"  Channel: {input_channel}")
    print(f"  H (hypotheses): {H}")
    print(f"  N (graph nodes): {N}")
    print(f"  K (neighbors):   {K}")
    print(f"  F (features):    {F}")
    print()

    all_ok = True

    # ==========================================================================
    # Step 6: Rotation
    # ==========================================================================
    print("=" * 70)
    print("Step 6. Rotation")
    print("-" * 70)

    poses = data["poses"]
    locations = data["locations"]
    displacement = data["displacement"]
    pose_vectors = data["pose_vectors"]

    t0 = time.time()

    print(f"\n  6a. Rotate displacement")
    print(f"      Input:  poses[{H}, 3, 3], displacement[3]")
    print(f"      displacement = [{displacement[0]:.6f}, {displacement[1]:.6f}, {displacement[2]:.6f}]")
    rotated_disp = step6a_rotate_displacement(poses, displacement)
    print(f"      Output: rotated_disp[{H}, 3]")
    print(f"      Sample [0]: [{rotated_disp[0,0]:.6f}, {rotated_disp[0,1]:.6f}, {rotated_disp[0,2]:.6f}]")
    print(f"      Ops: {9*H:,} mul + {6*H:,} add = {15*H:,}")

    print(f"\n  6b. Search locations")
    print(f"      Input:  locations[{H}, 3] + rotated_disp[{H}, 3]")
    search_locs = step6b_search_locations(locations, rotated_disp)
    print(f"      Output: search_locations[{H}, 3]")
    print(f"      Sample [0]: [{search_locs[0,0]:.6f}, {search_locs[0,1]:.6f}, {search_locs[0,2]:.6f}]")
    print(f"      Ops: {3*H:,} add")

    print(f"\n  6c. Rotate pose vectors")
    print(f"      Input:  poses[{H}, 3, 3], pose_vectors[3, 3]")
    rotated_pv = step6c_rotate_pose_vectors(poses, pose_vectors)
    print(f"      Output: rotated_pv[{H}, 3, 3]")
    print(f"      Ops: {27*H:,} mul + {18*H:,} add = {45*H:,}")

    surface_normals = rotated_pv[:, 0]  # First row = surface normal

    t6 = time.time() - t0
    step6_ops = 36 * H + 27 * H
    print(f"\n  Step 6 total: {step6_ops:,} ops ({t6:.3f}s)")

    # Verify against ground truth
    gt_search = data["search_locations"]
    diff = np.max(np.abs(search_locs - gt_search))
    ok = diff < 1e-10
    print(f"  Verify vs Monty: max_diff={diff:.2e} {'OK' if ok else 'FAIL'}")
    all_ok = all_ok and ok

    # ==========================================================================
    # Step 7: Nearest Neighbor Search
    # ==========================================================================
    print(f"\n{'=' * 70}")
    print("Step 7. Nearest Neighbor Search (Brute-force)")
    print("-" * 70)

    node_positions = data["node_positions"]
    print(f"  Input:  search_locations[{H}, 3], node_positions[{N}, 3]")
    print(f"  Params: K={K}")

    t0 = time.time()
    nn_dists, nn_ids = step7_nn_search(search_locs, node_positions, K)
    t7 = time.time() - t0

    print(f"  Output: distances[{H}, {K}], indices[{H}, {K}]")
    print(f"  Sample [0]: nearest nodes = {nn_ids[0].tolist()}")
    print(f"              distances     = [{', '.join(f'{d:.6f}' for d in nn_dists[0])}]")

    step7_ops = 3 * H * N + 5 * H * N + H * N  # mul + add + cmp
    print(f"\n  Step 7 total: {step7_ops:,} ops ({t7:.3f}s)")
    print(f"    - {3*H*N:,} mul + {5*H*N:,} add + {H*N:,} cmp")

    gt_dists = data["nn_distances"]
    gt_ids = data["nn_indices"]
    dist_diff = np.max(np.abs(nn_dists - gt_dists))
    ids_match = np.all(nn_ids == gt_ids)
    ok = dist_diff < 1e-10 and ids_match
    print(f"  Verify vs Monty: dist_diff={dist_diff:.2e}, ids_match={ids_match} {'OK' if ok else 'FAIL'}")
    all_ok = all_ok and ok

    # ==========================================================================
    # Step 8: Custom Distance + Pose Evidence
    # ==========================================================================
    print(f"\n{'=' * 70}")
    print("Step 8. Custom Distance + Pose Evidence")
    print("-" * 70)

    nn_locs = node_positions[nn_ids]
    curvature = float(data["curvature"])
    max_match_distance = float(data["max_match_distance"])
    node_pv = data["node_pose_vectors"]
    pose_fully_defined = bool(data["pose_fully_defined"])
    w_sn = float(data["w_sn"])
    w_cd = float(data["w_cd"])
    node_pose_fully_defined = data["node_pose_fully_defined"]  # (H, K, 1)
    use_cd_mask = np.array(node_pose_fully_defined[:, :, 0], dtype=bool) if pose_fully_defined else None
    feat_ev_increment = float(data["feature_evidence_increment"])

    t0 = time.time()

    # 8a
    print(f"\n  8a. Custom distance")
    print(f"      curvature = {curvature:.6f}")
    print(f"      curv_factor = 1/(|{curvature:.4f}| + 0.5) = {1.0/(abs(curvature)+0.5):.6f}")
    custom_dists = step8a_custom_distances(nn_locs, search_locs, surface_normals, curvature)
    print(f"      Output: custom_dists[{H}, {K}]")
    print(f"      Range: [{custom_dists.min():.6f}, {custom_dists.max():.6f}]")

    # 8b
    print(f"\n  8b. Distance weights")
    print(f"      max_match_distance = {max_match_distance}")
    dist_weights, dist_mask = step8b_distance_weights(custom_dists, max_match_distance)
    n_masked = dist_mask.sum()
    print(f"      Masked (too far): {n_masked}/{H*K} ({n_masked/(H*K)*100:.1f}%)")

    # 8c
    print(f"\n  8c. Surface normal evidence")
    sn_node_pv = node_pv[:, :, :3].reshape(H, K, 3)
    sn_query_pv = rotated_pv[:, 0]  # surface normal
    sn_angles, sn_evidence = step8c_surface_normal_evidence(sn_node_pv, sn_query_pv)
    print(f"      Angles range:   [{np.degrees(sn_angles.min()):.1f}, {np.degrees(sn_angles.max()):.1f}] deg")
    print(f"      Evidence range: [{sn_evidence.min():.4f}, {sn_evidence.max():.4f}]")

    # 8d
    if pose_fully_defined:
        print(f"\n  8d. Curvature direction evidence")
        cd_node_pv = node_pv[:, :, 3:6].reshape(H, K, 3)
        cd_query_pv = rotated_pv[:, 1]
        cd_angles, cd_evidence = step8d_curvature_direction_evidence(cd_node_pv, cd_query_pv)
        print(f"      Angles range:   [{np.degrees(cd_angles.min()):.1f}, {np.degrees(cd_angles.max()):.1f}] deg")
        print(f"      Evidence range: [{cd_evidence.min():.4f}, {cd_evidence.max():.4f}]")
    else:
        print(f"\n  8d. Curvature direction: SKIP (pose_fully_defined=False)")
        cd_evidence = np.zeros((H, K))

    # 8e
    print(f"\n  8e. Weighted pose evidence")
    print(f"      w_sn={w_sn}, w_cd={w_cd}")
    if use_cd_mask is not None:
        n_use_cd = use_cd_mask.sum()
        print(f"      use_cd nodes: {n_use_cd}/{H*K} ({n_use_cd/(H*K)*100:.1f}%)")
    pose_evidence = step8e_pose_evidence(sn_evidence, cd_evidence, w_sn, w_cd, use_cd_mask)
    print(f"      Pose evidence range: [{pose_evidence.min():.4f}, {pose_evidence.max():.4f}]")

    t8 = time.time() - t0
    step8_ops = 15 * H * K + 10 * H * K  # approximate
    print(f"\n  Step 8 total: ~{step8_ops:,} ops ({t8:.3f}s)")

    # ==========================================================================
    # Step 9: Feature Evidence
    # ==========================================================================
    print(f"\n{'=' * 70}")
    print("Step 9. Feature Evidence")
    print("-" * 70)

    feature_array = data["feature_array"]
    query_features = data["query_features"]
    tolerances = data["tolerances"]
    feat_weights = data["feature_weights"]
    circular_mask = data["circular_mask"]

    print(f"  Input:  feature_array[{N}, {F}], query[{F}]")
    print(f"  Query features: [{', '.join(f'{q:.4f}' for q in query_features)}]")
    print(f"  Tolerances:     [{', '.join(f'{t:.4f}' for t in tolerances)}]")
    print(f"  Weights:        [{', '.join(f'{w:.1f}' for w in feat_weights)}]")
    print(f"  Circular (hue): [{', '.join(str(bool(c)) for c in circular_mask)}]")

    t0 = time.time()
    feat_evidence = step9_feature_evidence(
        feature_array, query_features, tolerances, feat_weights, circular_mask
    )
    t9 = time.time() - t0

    print(f"  Output: feature_evidence[{N}]")
    print(f"  Range: [{feat_evidence.min():.4f}, {feat_evidence.max():.4f}]")
    print(f"  Mean:  {feat_evidence.mean():.4f}")

    step9_ops = 5 * N * F
    print(f"\n  Step 9 total: ~{step9_ops:,} ops ({t9:.3f}s)")

    gt_feat = data["monty_feature_evidence"]
    diff = np.max(np.abs(feat_evidence - gt_feat))
    ok = diff < 1e-10
    print(f"  Verify vs Monty: max_diff={diff:.2e} {'OK' if ok else 'FAIL'}")
    all_ok = all_ok and ok

    # ==========================================================================
    # Step 10: Evidence Accumulation
    # ==========================================================================
    print(f"\n{'=' * 70}")
    print("Step 10. Evidence Accumulation")
    print("-" * 70)

    old_evidence = data["old_evidence"]
    past_weight = float(data["past_weight"])
    present_weight = float(data["present_weight"])

    print(f"  past_weight={past_weight}, present_weight={present_weight}")
    print(f"  Old evidence range: [{old_evidence.min():.4f}, {old_evidence.max():.4f}]")

    t0 = time.time()
    print(f"  feature_evidence_increment={feat_ev_increment}")

    radius_ev, ev_to_add, new_evidence = step10_accumulate_evidence(
        pose_evidence, feat_evidence, nn_ids,
        dist_weights, dist_mask,
        old_evidence, past_weight, present_weight,
        feat_evidence_increment=feat_ev_increment,
    )
    t10 = time.time() - t0

    print(f"\n  Radius evidence [H, K]: range [{radius_ev.min():.4f}, {radius_ev.max():.4f}]")
    print(f"  Evidence to add [H]:    range [{ev_to_add.min():.4f}, {ev_to_add.max():.4f}]")
    print(f"  New evidence [H]:       range [{new_evidence.min():.4f}, {new_evidence.max():.4f}]")

    step10_ops = 12 * H + 12 * H + 10 * H
    print(f"\n  Step 10 total: ~{step10_ops:,} ops ({t10:.3f}s)")

    gt_ev = data["monty_evidence_result"]
    diff = np.max(np.abs(ev_to_add - gt_ev))
    ok = diff < 1e-10
    print(f"  Verify evidence_to_add vs Monty: max_diff={diff:.2e} {'OK' if ok else 'FAIL'}")
    all_ok = all_ok and ok

    # ==========================================================================
    # Summary
    # ==========================================================================
    total_time = t6 + t7 + t8 + t9 + t10
    total_ops = step6_ops + step7_ops + step8_ops + step9_ops + step10_ops

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)
    print(f"  {'Step':20s} {'Ops':>12s} {'Time':>10s} {'%':>6s}")
    print(f"  {'-'*20} {'-'*12} {'-'*10} {'-'*6}")
    for name, ops, t in [
        ("6. Rotation", step6_ops, t6),
        ("7. NN Search", step7_ops, t7),
        ("8. Pose Evidence", step8_ops, t8),
        ("9. Feature Evidence", step9_ops, t9),
        ("10. Accumulation", step10_ops, t10),
    ]:
        pct = t / total_time * 100
        print(f"  {name:20s} {ops:>12,} {t:>9.3f}s {pct:>5.1f}%")

    print(f"  {'-'*20} {'-'*12} {'-'*10} {'-'*6}")
    print(f"  {'TOTAL':20s} {total_ops:>12,} {total_time:>9.3f}s")
    print()
    print(f"  All verifications: {'PASSED' if all_ok else 'FAILED'}")
    print()


if __name__ == "__main__":
    main()
