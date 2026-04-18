# DL mid-term — Hugging Face  노트북

`transformers`로 **SAM2**, **Depth Anything V2**, **SmolVLM2**를 각각 돌리는 예제 노트북(`code/`)과 의존성 목록(`requirements.txt`)입니다.

---

## 1. Conda 설치

아직 Conda가 없다면 다음 중 하나를 설치합니다.

- **Miniconda**(가벼움, 권장): [Miniconda 설치 페이지](https://docs.conda.io/en/latest/miniconda.html)에서 OS에 맞는 설치 프로그램을 받아 실행합니다.
- **Anaconda**(패키지 포함): [Anaconda 다운로드](https://www.anaconda.com/download)

설치 후 터미널을 다시 열고, `conda --version`이 출력되는지 확인합니다.

---

## 2. 가상환경 만들기·활성화

프로젝트 루트(이 `README.md`가 있는 폴더)에서 실행합니다.

```bash
cd /path/to/mid_term

# Python 3.10–3.12 권장 (requirements.txt 상단 주석과 동일)
conda create -n DL-mid-term python=3.11 -y
conda activate DL-mid-term
```

이후 프롬프트 앞에 `(DL-mid-term)`이 보이면 해당 환경이 활성화된 상태입니다. 비활성화는 `conda deactivate`입니다.

---

## 3. PyTorch 설치

GPU(CUDA)·Apple Silicon(MPS)·CPU에 맞게 **먼저** `torch`를 설치하는 것이 안전합니다. [PyTorch 시작하기](https://pytorch.org/get-started/locally/)에서 환경에 맞는 명령을 고릅니다.

예시(버전은 사이트 안내에 맞게 조정):

```bash
# CUDA 12.x 예시
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# CPU만
pip install torch torchvision
```

`requirements.txt`에는 `torch`가 **포함되어 있지 않습니다**. 반드시 위와 같이 별도 설치한 뒤 다음 단계로 넘어갑니다.

---

## 4. 나머지 패키지 설치 (`requirements.txt`)

프로젝트 루트에서:

```bash
pip install -r requirements.txt
```

`transformers`, `accelerate`, `matplotlib`, `PyQt5`, `scipy`, `num2words`, Jupyter(`notebook`, `ipykernel`) 등이 설치됩니다.

### 선택 패키지

| 용도 | 패키지 | 설치 예시 |
|------|--------|-----------|
| SmolVLM2 **로컬 mp4** 비디오 셀에서 디코딩이 필요할 때 | `torchvision` | `pip install torchvision` (보통 torch와 함께 설치됨) |
| 07 마크다운 기준 3D 뷰(Open3D 또는 matplotlib 경로) | `open3d` | `pip install open3d` |

---

## 5. Jupyter에서 노트북 실행

```bash
conda activate DL-mid-term
cd code
jupyter notebook
```

브라우저에서 아래 파일을 순서대로 열고 **위에서부터** 셀을 실행합니다.

- `01_hf_sam2_raw.ipynb`
- `07_hf_depth_anything_raw.ipynb`
- `09_hf_smolvlm_raw.ipynb`

**VS Code / Cursor**를 쓰는 경우: `code` 폴더를 연 뒤 `.ipynb`를 열고, 커널로 `DL-mid-term` 환경의 Python을 선택합니다.

### (선택) 커널을 목록에 고정 등록

```bash
conda activate DL-mid-term
python -m ipykernel install --user --name DL-mid-term --display-name "Python (DL-mid-term)"
```

---

## 6. `requirements.txt`와 세 노트북 대응 관계

| 구분 | 내용 |
|------|------|
| **공통** | `transformers`, `accelerate`, `huggingface-hub`, `safetensors`, `numpy`, `pillow`, `matplotlib` — 세 노트북 모두 사용 |
| **01 SAM2** | `scipy`(마스크 후처리), `PyQt5`(포인트 픽커). **데스크톱 GUI**가 필요하며, 원격·헤드리스면 노트북 내 대안 셀을 사용 |
| **07 Depth** | 공통만으로 충분. 마크다운에 Open3D·matplotlib 3D 대체가 언급되어 있으나, 깊이 맵·2D 시각화만 쓰면 `open3d` 없이 실행 가능 |
| **09 SmolVLM2** | `num2words`(프로세서 import 시 필요) |
| **실행 도구** | `notebook`, `ipykernel` — Jupyter에서 `.ipynb` 실행 |

모델 가중치는 Hugging Face Hub에 있으면 `from_pretrained` 시 **캐시**(`~/.cache/huggingface/hub/` 등)로 자동 내려받습니다. 비공개 모델이면 `huggingface-cli login`이 필요할 수 있습니다(본 예제는 공개 체크포인트 기준).

---

## 7. 노트북별 요약

### `01_hf_sam2_raw.ipynb` — SAM2 포인트 세그멘테이션

Hugging Face의 `Sam2Processor` / `Sam2Model`로 이미지에서 **클릭 기반** 마스크를 냅니다.

| | 설명 |
|---|------|
| **입력** | 로컬 RGB 이미지 경로, 전경(좌클릭)·배경(우클릭) 좌표, `MODEL_ID` |
| **출력** | 원본 해상도 `bool` 세그멘테이션 마스크, matplotlib 오버레이 시각화 |

---

### `07_hf_depth_anything_raw.ipynb` — Depth Anything V2

`AutoImageProcessor` + `AutoModelForDepthEstimation`으로 **단안 상대 깊이**를 추정합니다.

| | 설명 |
|---|------|
| **입력** | 로컬 RGB 이미지 경로, `MODEL_ID` |
| **출력** | H×W `float32` 깊이 맵 **[0,1]** (**0=가까움, 1=멀음**, 메트릭 거리 아님), 깊이·비교 플롯 등 |

---

### `09_hf_smolvlm_raw.ipynb` — SmolVLM2 (이미지·텍스트 / 비디오)

`AutoProcessor` + `AutoModelForImageTextToText`로 **HuggingFaceTB/SmolVLM2-500M-Video-Instruct** 체크포인트를 사용합니다.

| | 설명 |
|---|------|
| **입력** | 이미지 경로 + 자연어 질문(텍스트); 마지막 선택 셀에서는 `.mp4` 경로 + 텍스트 |
| **출력** | 모델이 생성한 **자연어 답변 문자열** |

---

## 8. 자주 겪는 이슈

- **SAM2 Qt 창이 안 뜸**: SSH·서버에서는 디스플레이가 없을 수 있습니다. 노트북의 수동 좌표 대안 셀을 사용하거나 로컬 데스크톱에서 실행하세요.
- **`transformers` 버전 오류**: `requirements.txt`의 `transformers>=4.50.0`을 만족하는지 확인하세요. SAM2·최신 API는 비교적 최근 버전이 필요합니다.
- **디바이스**: 각 노트북 마크다운에 `SAM2_DEVICE`, `DEPTH_ANYTHING_DEVICE`, `SMOLVLM_DEVICE` 환경 변수 안내가 있습니다.
