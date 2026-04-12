# HW 가속 대상 연산 상세 분석

## 변수 정의

| 변수 | 의미 | 일반 범위 |
|------|------|----------|
| H | 가설(hypothesis) 수 | 수십~수백 (step마다 변동) |
| N | 그래프 노드 수 (오브젝트당) | 1280~3194 |
| K | 최대 이웃 수 (max_nneighbors) | 10 |
| G | 학습된 오브젝트 수 | 12 |
| F | feature 수 (hsv + curvature_log) | ~5 |

모든 연산은 **매 스텝 × G 오브젝트** 반복 (48 steps × 12 objects = 576회/에피소드)

---

## Step 6. 회전 적용 (Rotation)

### 6a. Displacement 회전
```
입력:  poses[H, 3, 3], displacement[3]
출력:  rotated_displacements[H, 3]

연산:  각 가설 h에 대해:
  rotated[h, i] = Σ(j=0..2) poses[h, i, j] × displacement[j]

  → H × 3 × 3 = 9H 곱셈, 6H 덧셈
```

### 6b. 새 위치 계산
```
입력:  locations[H, 3], rotated_displacements[H, 3]
출력:  search_locations[H, 3]

연산:  search_locations[h, i] = locations[h, i] + rotated_displacements[h, i]

  → 3H 덧셈
```

### 6c. Pose vector 회전
```
입력:  poses[H, 3, 3], pose_vectors[3, 3]
출력:  rotated_pv[H, 3, 3]

연산:  각 가설 h에 대해:
  rotated_pv[h] = poses[h] × pose_vectors.T   (3x3 × 3x3 행렬곱)

  → H × 27 곱셈, H × 18 덧셈
```

**Step 6 총계: 36H 곱셈, 27H 덧셈**

---

## Step 7. KDTree 탐색 (Nearest Neighbor Search)

### 현재 구현: scipy KDTree
```
입력:  search_locations[H, 3], tree (N개 노드의 3D 좌표)
출력:  nearest_node_ids[H, K], distances[H, K]
파라미터: K=10, p=2 (L2 distance)
```

### Brute-force 대체 시 사칙연산:
```
각 가설 h, 각 노드 n에 대해:
  diff[i] = search_locations[h, i] - node_pos[n, i]    (3 뺄셈)
  sq[i] = diff[i] × diff[i]                            (3 곱셈)
  dist_sq = sq[0] + sq[1] + sq[2]                      (2 덧셈)

  → H × N × (3 곱셈 + 5 덧셈/뺄셈)

이후 각 가설별 top-K 선택: H × N 비교 연산 (partial sort)
```

**Step 7 총계: 3HN 곱셈, 5HN 덧셈, HN 비교**
- H=100, N=2000 → 600K 곱셈, 1M 덧셈

---

## Step 8. Custom Distance + Pose Evidence

### 8a. Custom Distance 계산
```
입력:  nearest_node_locs[H, K, 3], search_locations[H, 3],
       surface_normal[H, 3], curvature[1]
출력:  custom_dists[H, K]

각 가설 h, 이웃 k에 대해:
  diff[i] = node_locs[h,k,i] - search_locs[h,i]       (3 뺄셈)

  // Euclidean distance
  eucl = sqrt(diff[0]² + diff[1]² + diff[2]²)          (3 곱셈, 2 덧셈, 1 sqrt)

  // Dot product with surface normal
  dot = diff[0]×sn[h,0] + diff[1]×sn[h,1] + diff[2]×sn[h,2]  (3 곱셈, 2 덧셈)

  // Custom distance
  custom = eucl + |dot| × (1 / (|curvature| + 0.5))    (1 abs, 1 add, 1 div, 1 mul, 1 add)

  → HK × (6 곱셈 + 4 덧셈 + 1 sqrt + 1 div + 1 abs)
```

### 8b. Node Distance Weights
```
입력:  custom_dists[H, K], max_match_distance (상수 = 0.01)
출력:  weights[H, K], mask[H, K]

  weights[h,k] = (max_match_distance - custom_dists[h,k]) / max_match_distance
  mask[h,k] = (weights[h,k] <= 0)

  → HK × (1 뺄셈 + 1 나눗셈 + 1 비교)
```

### 8c. Surface Normal Evidence
```
입력:  node_pose_vectors[H, K, 3], query_pose_vectors[H, 3]
출력:  surface_normal_evidence[H, K]

각 가설 h, 이웃 k에 대해:
  // einsum: dot product
  dot = Σ(i=0..2) node_pv[h,k,i] × query_pv[h,i]      (3 곱셈, 2 덧셈)

  // Angle
  dot_clamped = clamp(dot, -1, 1)                       (2 비교)
  angle = arccos(dot_clamped)                           (1 arccos)

  // Evidence
  evidence = -(sin(angle / 2) - 0.5)                   (1 div, 1 sin, 1 sub, 1 neg)

  → HK × (3 곱셈 + 2 덧셈 + 1 arccos + 1 sin + 3 기타)
```

### 8d. Curvature Direction Evidence (pose_fully_defined=True일 때)
```
  8c와 동일 구조, node_pv[h,k,3:6]과 query_pv[h,1]에 대해:

  cd1_angle = arccos(clamp(dot, -1, 1))
  cd1_error = π/2 - |cd1_angle - π/2|                  (1 sub, 1 abs, 1 sub)
  cd1_evidence = -(sin(cd1_error) - 0.5)               (1 sin, 1 sub, 1 neg)

  → HK × (3 곱셈 + 2 덧셈 + 1 arccos + 1 sin + 5 기타)
```

### 8e. 가중 합산
```
  pose_evidence[h,k] = sn_evidence[h,k] × w_sn + cd1_evidence[h,k] × w_cd

  → HK × (2 곱셈 + 1 덧셈)
```

**Step 8 총계 (per H×K):**
- 8a: 6 mul, 4 add, 1 sqrt, 1 div
- 8b: 1 sub, 1 div, 1 cmp
- 8c: 3 mul, 2 add, 1 arccos, 1 sin
- 8d: 3 mul, 2 add, 1 arccos, 1 sin
- 8e: 2 mul, 1 add
- **합계: ~15 mul, ~10 add, 2 arccos, 2 sin, 1 sqrt, 2 div (per H×K)**

---

## Step 9. Feature Evidence

```
입력:  feature_array[N, F], query_features[F], tolerances[F], weights[F]
출력:  feature_evidence[N]

각 노드 n, 각 feature f에 대해:
  // 일반 feature (saturation, value, curvature 등)
  diff = |feature_array[n,f] - query[f]|                (1 sub, 1 abs)

  // Hue (circular, f=0)
  diff = min(|1+stored-query|, |stored-query|, |stored-(query+1)|)
                                                         (3 sub, 3 abs, 2 cmp)

  // Evidence 계산
  ev = clamp(tolerance[f] - diff, 0, inf)               (1 sub, 1 cmp)
  ev = ev / tolerance[f]                                 (1 div)

이후: weighted average over F features
  result[n] = Σ(ev[f] × weight[f]) / Σ(weight[f])      (F mul, F-1 add, 1 div)

→ N × (~3F add/sub + F div + F mul)
```

**참고:** 이 연산은 전체 N 노드에 대해 한 번 수행 후, nearest_node_ids[H, K]로
인덱싱하여 hypothesis별 evidence로 변환: `node_feature_evidence[nearest_node_ids]`

**Step 9 총계: ~5NF 연산 (N=2000, F=5 → ~50K 연산)**

---

## Step 10. Evidence 누적

```
입력:  pose_evidence[H, K], feature_evidence[H, K],
       old_evidence[H], past_weight, present_weight
출력:  new_evidence[H]

각 가설 h에 대해:
  // mask 적용: 거리 초과 → -1
  radius_evidence[h,k] = pose_evidence[h,k] + feat_evidence[h,k] × feat_increment
  radius_evidence[h,k] = -1  (if mask[h,k])

  // K개 이웃 중 최대값
  evidence_to_add[h] = max(radius_evidence[h, 0..K-1])  (K-1 비교)

  // 가중 평균 누적
  new_evidence[h] = old_evidence[h] × past_weight + evidence_to_add[h] × present_weight
                                                         (2 곱셈, 1 덧셈)

→ H × (K mul + K add + K cmp + 2 mul + 1 add)
```

**Step 10 총계: ~12H mul, ~12H add, ~10H cmp (K=10)**

---

## 전체 요약 (1 스텝 × 1 오브젝트)

| Step | 주요 연산 | 연산량 (H=100, N=2000, K=10) |
|------|----------|----------------------------|
| 6. Rotation | matmul 3x3 | ~6K ops |
| 7. NN Search | 거리 계산 + top-K | **~1.6M ops** |
| 8. Pose Evidence | arccos, sin, dot | ~30K ops |
| 9. Feature Evidence | abs diff, weighted avg | ~50K ops |
| 10. Accumulate | max, weighted sum | ~3K ops |
| **합계** | | **~1.7M ops/object/step** |

**1 스텝 전체 (G=12 오브젝트):** ~20M ops
**1 에피소드 (48 steps):** ~960M ops

Step 7 (NN Search)이 전체의 **~94%** 를 차지하며,
이를 FPGA에서 N개 노드를 병렬 처리하면 가장 큰 가속 효과.
