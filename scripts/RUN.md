# RUN.md — 실행 명령 치트시트

카메라별 capture → preprocess → inference 파이프라인. 자세한 배경은
`../CLAUDE.md`, 패치 이력은 `PATCHES.md` 참고.

핵심 축 = **HFOV**. capture 카메라와 inference `--hfov`가 맞아야 함.

| 카메라 | HFOV | capture 스크립트 | inference `--hfov` |
|---|---|---|---|
| D405   | 54.201 | `rgbd_capture_d405.py`  | (생략=기본) |
| Femto  | 54.201 | `rgbd_capture_femto.py` | (생략=기본) |
| iPad   | 54.201 | (Robot Lab 학습값)      | (생략=기본) |
| OAK-D Pro | 63.75 | `rgbd_capture.py`     | `--hfov 63.75` |

---

## 0. 캡쳐+추론 한 번에 (Z8, OAK-D Pro, `tbp_fpga` env)

`capture_and_infer.py` — depthai 캡쳐 + Monty warm 모델을 **한 프로세스**에서.
모델 1회 로드 후 상주, `'c'` 누를 때마다 캡쳐→isolate(메모리)→추론→즉시 결과
(~2s). 별도 isolate/inference 실행 불필요. OAK는 depthai만 필요해서 depthai를
`tbp_fpga`에 설치해둠 (`pip install depthai`, env 분리 사유였던 pyorbbecsdk와 무관).

```bash
conda activate tbp_fpga
python scripts/capture_and_infer.py                       # object 미지 (detected만)
python scripts/capture_and_infer.py --object numenta_mug  # correct/wrong 채점까지
python scripts/capture_and_infer.py \                     # 캡쳐도 디스크에 보존
    --save-dir ~/tbp/data/worldimages/captured_scenes_oak
```
- HFOV 자동 63.75 (OAK). isolate 내장(손+배경 마스킹) — 별도 실행 불필요.
- Preview 3패널: **RGB(노란 박스=물체 놓을 위치) | depth raw | ISOLATED(=Monty가 실제로 받는 것)**.
- 키: `'c'`=캡쳐+추론, `'i'`=isolate 패널 토글, `'d'`=캡쳐 후 디버그창 토글, `'q'`=종료.
- **`cov%`가 캡쳐 품질 게이지.** 학습 매칭(iPad) 데이터는 13-22%. 3% 미만이면
  물체 표면 depth에 구멍이 너무 많은 것(OAK가 광택/흰색에 약함) → 거리·각도·조명 조정.
  `surf%`는 중앙 ROI depth 유효율(iPad 89-97% vs OAK 광택물체 55-60%).

#### SSH 에서 실행 (preview 는 Z8 물리 모니터에)

SSH 세션엔 `DISPLAY` 가 없어서 cv2 창이 안 뜸. Z8 물리 세션은 **`:1`**
(`ls /tmp/.X11-unix` → `X1`). 키(`c`/`q`/`i`/`d`)는 preview 창과 **터미널 양쪽**
에서 받으므로 SSH 만으로 전 과정 조작 가능.

```bash
conda activate tbp_fpga
scripts/ci.sh --object numenta_mug --extended-disparity   # DISPLAY=:1 자동 설정
# 또는 직접:
DISPLAY=:1 python scripts/capture_and_infer.py --object numenta_mug --extended-disparity
```
`conda run` 은 tty 를 안 넘겨서 터미널 키 입력이 죽음 — 반드시 env 를 먼저 activate.

#### 거리 — 가장 중요한 변수

측정값: **OAK-D Pro 기본 최소거리 = 0.316m** (캡쳐 4장 모두 min이 정확히 동일 =
하드웨어 바닥). 그보다 가까이 두면 물체에 depth가 아예 안 잡히고 `dist`가
배경값(수 m)으로 튐. 한편 Monty는 `MONTY_DEPTH_CLIP`(기본 **0.4m**) 너머를
off-object 처리하고, 학습 데이터(standard_scenes)의 물체 거리는 **0.17-0.26m**.

즉 기본 설정의 OAK는 [0.316m, 0.4m]라는 **8cm 창**에서만 동작 — 그래서 기존
캡쳐가 0.37-0.40m로 clip 경계에 걸려 물체의 5-28%가 잘려나갔음.

```bash
# 권장: extended disparity 로 최소거리를 낮춰 학습 거리대에서 촬영
python scripts/capture_and_infer.py --object numenta_mug --extended-disparity   # 물체 0.20-0.25m
# 대안: 기본 설정으로 0.35-0.40m 에서 찍고 clip 을 넓힘
python scripts/capture_and_infer.py --object numenta_mug --depth-clip 0.8
```
`--no-isolate` 는 raw depth 를 그대로 Monty 에 넘김. Monty 의 depth clip 이 배경을
이미 제거하므로 isolate 는 **손 제거**용으로만 유의미 (손은 물체와 같은 depth 라
clip 으로 안 걸러짐). 0.40m 기존 데이터 A/B 에서는 raw 가 나았음(2/4 vs 1/4).

---

## 1. Capture (Z8, `orbbec` env)

```bash
conda activate orbbec

# D405
python scripts/rgbd_capture_d405.py  --object numenta_mug --index 0 --num-captures 4

# Femto Bolt  (--align none, 물체를 RGB 프레임 중심에)
python scripts/rgbd_capture_femto.py --object numenta_mug --index 0 --stdin

# OAK-D Pro  (IR 프로젝터 ON = 기본, 출력 native 640x480 = 63.75 deg)
python scripts/rgbd_capture.py       --object numenta_mug --index 0 --output_dir ~/tbp/data/worldimages/captured_scenes_oak
```

### FPGA(Versal, USB2) — OAK-D Pro, 보드에서 실행

`rgbd_capture_oak_fpga.py` 하나만 보드로 scp. **OAK에 외부전원(Y-어댑터) 필수**
(bus 전원만으론 color+depth 동시 시 brown-out crash). 보드는 headless라
preview는 `/dev/fb0`(HDMI)로 나감. 자세한 배경은 memory/oak-fpga-usb2-sequential.

```bash
scp scripts/rgbd_capture_oak_fpga.py root@<board-ip>:/home/agi/rgbd_test/

# 기본 = 라이브 preview (보드 HDMI 모니터), 'c'=캡쳐 'q'=종료 (실행 터미널에서)
python3 rgbd_capture_oak_fpga.py --object numenta_mug --index 0

# 자동 N장 (외부전원)
python3 rgbd_capture_oak_fpga.py --object numenta_mug --index 0 --headless --num-captures 4

# 외부전원 없을 때 폴백 (depth/color 세션 분리, 느림)
python3 rgbd_capture_oak_fpga.py --object numenta_mug --index 0 --sequential --fps 5
```
- 색 뒤바뀌어 보이면 `--fb-rgb`, preview 크기는 `--fb-scale`.
- inference는 OAK와 동일하게 `--hfov 63.75`.

카메라 섞이면 안 되므로 카메라별로 **다른 output 디렉토리** 사용
(예: `captured_scenes_femto`, `captured_scenes_oak`).

## 2. Preprocess — isolate (손+배경 마스킹, 항상 실행)

```bash
python scripts/isolate_object.py \
    --input  ~/tbp/data/worldimages/captured_scenes_oak \
    --output ~/tbp/data/worldimages/captured_scenes_oak_isolated \
    --recursive --debug-dir /tmp/iso_debug
```

> smooth_depth.py 는 정확도 떨어뜨림 — raw + isolate 권장 (dead end).

## 3. Inference (Z8, `tbp_fpga` env)

```bash
conda activate tbp_fpga

# 스모크 (1 에피소드)
python scripts/monty_inference.py --max-episodes 1

# Femto / D405  (HFOV 기본 54.201)
python scripts/monty_inference.py --all \
    --data-path ~/tbp/data/worldimages/captured_scenes_femto_isolated \
    --output-csv eval_femto.csv

# OAK-D Pro  (--hfov 63.75 필수)
python scripts/monty_inference.py --all --hfov 63.75 \
    --data-path ~/tbp/data/worldimages/captured_scenes_oak_isolated \
    --output-csv eval_oak.csv

# 단일 물체
python scripts/monty_inference.py \
    --scenes 0,0,0,0 --versions 0,1,2,3 \
    --data-path ~/tbp/data/worldimages/captured_scenes_isolated \
    --output-csv eval_mug.csv
```

- `--hfov` 는 실행 시 `two_d_data.py` 의 HFOV를 런타임 설정 (sed 토글 불필요).
  생략 시 기본 54.201.
- `eval_*.csv` 는 gitignore (기존 baseline CSV만 tracked).

## 4. 검증 / 시각화

```bash
python scripts/rgbd_verify.py     <scene_dir>   # 포맷 체크
python scripts/visualize_depth.py <scene_dir>   # depth 뷰
python scripts/compare_depth.py   <a> <b>       # captured vs reference

# calibration sanity
python scripts/check_d405_calibration.py
python scripts/check_femto_calibration.py
python scripts/check_oak_calibration.py
```
