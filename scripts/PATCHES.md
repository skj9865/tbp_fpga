# tbp.monty 원본 수정 기록

이 문서는 FPGA(ARM/PetaLinux) 환경에서 monty_inference.py를 실행하기 위해
tbp.monty 원본 코드에 가한 모든 수정사항을 기록합니다.

모든 패치는 `scripts/patch_lazy_imports.py`로 자동 적용됩니다.

---

## 패치 1: torch_geometric lazy import

- **파일**: `src/tbp/monty/frameworks/models/object_model.py`
- **원본**:
  ```python
  import torch_geometric
  import torch_geometric.transforms as T
  from scipy.spatial import KDTree
  from sklearn.neighbors import kneighbors_graph
  from torch_geometric.data import Data
  ```
- **수정**:
  ```python
  from scipy.spatial import KDTree
  try:
      import torch_geometric
      import torch_geometric.transforms as T
      from sklearn.neighbors import kneighbors_graph
      from torch_geometric.data import Data
  except ImportError:
      torch_geometric = None
      T = None
      kneighbors_graph = None
      Data = None
  ```
- **이유**: torch_geometric이 설치되지 않은 환경에서 import 실패 방지.
  inference 실행 경로에서는 학습용 그래프 구축 코드가 호출되지 않음.
- **영향**: inference 동작에 영향 없음. 학습(training) 시에는 torch_geometric 설치 필요.

---

## 패치 2: wandb/pandas lazy import (graph_matching_loggers)

- **파일**: `src/tbp/monty/frameworks/loggers/graph_matching_loggers.py`
- **원본**:
  ```python
  import pandas as pd
  import wandb
  from sklearn.preprocessing import LabelEncoder
  ```
- **수정**:
  ```python
  try:
      import pandas as pd
  except ImportError:
      pd = None
  try:
      import wandb
  except ImportError:
      wandb = None
  try:
      from sklearn.preprocessing import LabelEncoder
  except ImportError:
      LabelEncoder = None
  ```
- **이유**: wandb/pandas가 미설치된 환경에서 import 실패 방지.
  inference 시 로깅 코드가 호출되지 않음.
- **영향**: inference 동작에 영향 없음. wandb 로깅 사용 시에는 설치 필요.

---

## 패치 3: wandb/pandas lazy import (wandb_handlers)

- **파일**: `src/tbp/monty/frameworks/loggers/wandb_handlers.py`
- **원본**:
  ```python
  import pandas as pd
  import wandb
  ```
- **수정**:
  ```python
  try:
      import pandas as pd
  except ImportError:
      pd = None
  try:
      import wandb
  except ImportError:
      wandb = None
  ```
- **이유**: 패치 2와 동일.
- **영향**: 패치 2와 동일.

---

## 패치 4: torch_geometric Data.keys 버전 호환

- **파일**: `src/tbp/monty/frameworks/utils/object_model_utils.py`
- **원본**:
  ```python
  for key in list(torch_graph.keys):
  ```
- **수정**:
  ```python
  _keys = torch_graph.keys() if callable(torch_graph.keys) else torch_graph.keys
  for key in list(_keys):
  ```
- **이유**: tbp.monty는 torch-geometric 2.1.0 기준으로 작성됨.
  해당 버전에서 `Data.keys`는 property였으나, 최신 버전에서는 method로 변경됨.
  FPGA에 설치된 torch-geometric이 신버전이므로 `keys()` 호출이 필요.
- **영향**: 구버전/신버전 모두 호환. 동작 로직 변경 없음.

---

## 요약

| # | 파일 | 수정 유형 | 동작 영향 |
|---|------|----------|----------|
| 1 | models/object_model.py | lazy import | 없음 |
| 2 | loggers/graph_matching_loggers.py | lazy import | 없음 |
| 3 | loggers/wandb_handlers.py | lazy import | 없음 |
| 4 | utils/object_model_utils.py | API 호환 | 없음 |

모든 패치는 inference 실행 경로의 알고리즘 로직을 변경하지 않습니다.
원본 동작을 복원하려면 tbp.monty를 다시 clone하거나 git checkout하면 됩니다.
