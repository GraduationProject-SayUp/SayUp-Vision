## SayUp Vision - 감정 인식 및 얼굴 랜드마크 시각화

이 저장소는 SayUp 프로젝트의 Vision 모듈로, 다음 두 가지 핵심 기능을 제공합니다:

- **얼굴 감정 인식**: 이미지 속 인물의 감정을 딥러닝 모델로 분류
- **입술 강조 얼굴 메쉬 시각화**: Mediapipe를 활용하여 얼굴 랜드마크, 특히 입술을 시각적으로 강조

<br>

### 📂 주요 파일 설명

| 파일명 | 설명 |
|--------|------|
| `face_emotion.py` | 이미지에서 얼굴을 감지하고 감정을 예측하는 코드입니다. huggingface의 사전 학습 모델을 사용합니다. |
| `face_mesh_lips.py` | Mediapipe Face Mesh를 활용하여 얼굴 랜드마크를 시각화하며, 입술 부분은 빨간색으로 강조합니다. |

<br>

### 감정 분류 모델 (face_emotion.py)

- 사용 모델: [`dima806/facial_emotions_image_detection`](https://huggingface.co/dima806/facial_emotions_image_detection)
- 감정 클래스:
  - 😢 `sad`
  - 🤢 `disgust`
  - 😠 `angry`
  - 😐 `neutral`
  - 😨 `fear`
  - 😲 `surprise`
  - 😄 `happy`

**예시 코드 실행**

```bash
python face_emotion.py
```

**출력 예시**
```text
Predicted emotion: happy
Bounding box: (x, y, w, h)
```

<br>

## 얼굴 메쉬 + 입술 강조 (face_mesh_lips.py)

- Mediapipe의 Face Mesh 기능 사용
- 전체 얼굴 랜드마크 표시 + 입술 영역은 빨간색으로 강조

**예시 코드 출력**

```bash
python face_mesh_lips.py
```

**출력**
```text
matplotlib으로 랜드마크가 시각화된 이미지 창 표시
```

<br>

### 📦 의존 라이브러리 설치

```bash
pip install -r requirements.txt
```

필요 패키지
- `mediapipe`
- `torch`
- `transformers`
- `opencv-python`
- `matplotlib`
- `numpy`

<br>

### 💡 참고

- `emotions/`, `__pycache__/`, `models/__pycache__/` 등은 `.gitignore`로 관리됩니다.
- 이미지는 `img.jpg` 파일로 테스트합니다. 원하는 이미지 경로로 변경하세요.

<br>

### 📸 결과 예시

| 얼굴 메쉬 예시 | 감정 예측 예시 |
|----------------|----------------|
| ![그림1](https://github.com/user-attachments/assets/279556ca-4e0b-4c70-944f-c76cbeb15a19)| ![화면_캡처_2024-12-17_145354](https://github.com/user-attachments/assets/67db33e8-65c1-4caa-9e00-f4ed5295964a)|


