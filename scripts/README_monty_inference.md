# monty_inference.py — 실행 가이드 & 내부 구조

Monty(tbp.monty)로 worldimages 데이터셋을 추론하는 standalone 하네스.
Hydra/experiment 프레임워크를 우회하고 tbp.monty 클래스를 직접 조립해
`world_image_on_scanned_model` 벤치(~66–68%)를 재현한다.

---

## 1. 환경 셋업 (한 번)

코드는 GitHub 클론으로 받되, 파이썬 환경은 직접 구성해야 한다 (CPU 서버 기준,
wandb 미설치 = board식 최소셋).

```bash
git clone <this-repo> && cd tbp_fpga
conda create -n monty python=3.11 && conda activate monty

# torch CPU 빌드 먼저 (GPU면 cu 인덱스로)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r scripts/requirements-board.txt      # 최소셋
pip install --no-deps torch-geometric              # model.pt unpickle 필수
pip install "numpy==1.26.4"                         # skimage가 2.x로 올려서 재핀
pip install --no-deps -e tbp.monty/                # ← --no-deps 필수(torch 다운그레이드 방지)
python scripts/patch_lazy_imports.py               # wandb/pandas lazy 패치(최소셋일 때)

# 검증
python -c "import numpy,torch,tbp.monty,torch_geometric; print('OK')"
```

**빠지기 쉬운 3가지**: (1) torch를 requirements보다 먼저 원하는 빌드로,
(2) `--no-deps -e tbp.monty/`, (3) `patch_lazy_imports.py`(wandb 안 깔았을 때).

### 모델 · 데이터 (repo 밖, 별도 전송)

git에 없다. `~/tbp/` 아래에 둔다:
- 모델: `~/tbp/results/monty/pretrained_models/pretrained_ycb_v12/<MODEL>/pretrained/model.pt`
  - worldimages(12물체) = `surf_agent_1lm_numenta_lab_obj`
  - D405/로봇랩(10물체) = `surf_agent_1lm_tbp_robot_lab` (harness 기본값)
- 데이터: `~/tbp/data/worldimages/{standard,bright,dark}_scenes/` (각 12물체 × 4버전)

---

## 2. 실행법

```bash
conda activate monty

# 스모크 (1 에피소드)
python scripts/monty_inference.py --max-episodes 1

# 전체 벤치 (48 에피소드) + CSV
python scripts/monty_inference.py --all \
    --model-path ~/tbp/results/monty/pretrained_models/pretrained_ycb_v12/surf_agent_1lm_numenta_lab_obj/pretrained/model.pt \
    --data-path  ~/tbp/data/worldimages/standard_scenes \
    --output-csv eval_std.csv

# 단일 물체 (scene 2 = montys_brain, 4버전)
python scripts/monty_inference.py --scenes 2,2,2,2 --versions 0,1,2,3 --output-csv eval_brain.csv
```

### 경로/모델/데이터셋을 env로 (인자 생략)

| env | 기본값 | 역할 |
|---|---|---|
| `MONTY_MODELS` | `~/tbp/results/monty/pretrained_models` | 모델 base 디렉토리 |
| `MONTY_MODEL_NAME` | `surf_agent_1lm_tbp_robot_lab` | v12 아래 모델 디렉토리명 |
| `MONTY_DATA` | `~/tbp/data` | 데이터 base 디렉토리 |
| `MONTY_DATASET` | `standard_scenes` | worldimages 아래 데이터셋명 |
| `MONTY_UPDATER` | `burst` | 가설 updater (`burst` / `default`) |

```bash
# worldimages 서버: 모델/데이터셋을 env로 → 인자 없이 실행
export MONTY_MODEL_NAME=surf_agent_1lm_numenta_lab_obj
MONTY_DATASET=bright_scenes python scripts/monty_inference.py --all --output-csv eval_bright.csv
```

### 주요 CLI 옵션

| 옵션 | 기본 | 설명 |
|---|---|---|
| `--all` | — | data-path 아래 모든 (scene,version) 실행 |
| `--scenes` / `--versions` | 0,1,2,3 반복 | 실행할 scene/version 인덱스 (쉼표) |
| `--max-episodes N` | 전체 | 앞 N 에피소드만 |
| `--max-eval-steps` | 500 | 에피소드당 최대 **매칭** step |
| `--seed` | 42 | 난수 시드 (saccade 경로·가설 샘플 결정 → 결정론) |
| `--hfov` | 54.201 | depth→3D 투영 FOV (모델 학습값과 일치해야) |
| `--output-csv` | — | per-episode 결과 저장 |
| `--log-level` | WARNING | 로깅 레벨 |

---

## 3. 내부 함수 구조

`main()`이 컴포넌트를 조립해 `run_inference()`를 호출하는 단순 구조.

```
main()                      # 인자 파싱 → 조립 → 실행
 └─ run_inference()         # 모델 생성·로드 + 에피소드 루프 (핵심)
     ├─ create_sensor_modules()   # patch(CameraSM) + view_finder(Probe)
     ├─ create_learning_module()  # EvidenceGraphLM (매칭 파라미터)
     │    └─ _updater_config()    # MONTY_UPDATER → burst/default updater 선택
     ├─ create_motor_system()     # InformedPolicy + ConstantSampler (saccade)
     ├─ create_model()            # MontyForGraphMatching (위 3개 결합)
     ├─ load_pretrained_model()   # model.pt 가중치(그래프 메모리) 로드
     └─ [에피소드 루프] → print_results() + save_csv()
```

### 컴포넌트 팩토리

| 함수 | 만드는 것 | 핵심 설정 |
|---|---|---|
| `create_sensor_modules` | patch SM + view_finder | patch가 hsv/법선/곡률 특징 추출 |
| `create_learning_module` | `EvidenceGraphLM` | `x_percent_threshold=8`, tolerances(hsv/곡률), feature_weights, `**_updater_config()` |
| `_updater_config` | updater 인자 dict | `burst`(동적 H) / `default`(고정 H) — `MONTY_UPDATER` env |
| `create_motor_system` | `MotorSystem` | InformedPolicy, ConstantSampler(20° 고정 액션) |
| `create_model` | `MontyForGraphMatching` | SM↔LM 매트릭스 결선 |
| `load_pretrained_model` | — | model.pt → LM graph_memory 로드 |

### 핵심: `run_inference()` 에피소드 루프

```
for 에피소드 in 대상:
    RNG(seed) 리셋 → target 설정 → model.pre_episode(target)
    while:
        observations = env.step(...)              # saccade로 다음 패치 관측
        if check_reached_max_matching_steps(500): break   # 매칭 step 상한
        if step >= MAX_TOTAL_STEPS(6000): deal_with_time_out(); break
        model.step(ctx, observations)             # evidence 갱신(Step 6–10)
        if model.is_done: break                   # 후보 1개로 수렴 → match
    model.post_episode()
    → primary_performance 판정 → CSV 행 기록
```

- **env**: `SaccadeOnImageEnvironment` — 정지 이미지 하나 위를 패치가 saccade하며 수백 step 인식.
- **종료**: 후보(possible match)가 1개로 좁혀지면 `match`; 못 좁히고 상한 도달이면 `time_out`(이때 1위 = MLH).
- **결정론**: 같은 seed → saccade 경로·가설 샘플 동일 → 결과 비트 재현.

### 결과 (`save_csv` / `print_results`)

CSV 컬럼: `episode, target_object, detected_object, terminal_state,
primary_performance, steps, correct, correct_mlh, elapsed_sec`

- **correct** = 수렴(match)해서 맞음 / **correct_mlh** = time_out이지만 1위 가설이 정답
- **정확도**: `correct`만(엄격) 또는 `correct or correct_mlh`(MLH 포함). 벤치 보고엔 보통 MLH 포함.

---

## 4. 참고

- 카메라 캡쳐(D405/Femto/OAK)·isolate·라이브 데모는 별도 스크립트 (worldimages 추론엔 불필요).
- updater 파라미터·프로파일·HW 분석: `연구노트/`(FPGA_HW구현분석 등), `hw_analysis/`.
- tbp.monty 소스 수정 이력: `scripts/PATCHES.md`.
