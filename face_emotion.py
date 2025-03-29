import cv2
import matplotlib.pyplot as plt
import os
import math
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
import numpy as np
from typing import Optional, Tuple, Dict

device = 'cpu'  # 강제로 CPU 사용

# 이미지 로드
def load_image(image_path: str) -> Optional[np.ndarray]:
    """
    이미지를 로드하고 RGB로 변환하는 함수

    Args:
        image_path (str): 이미지 파일 경로

    Returns:
        Optional[np.ndarray]: RGB 형식의 이미지 배열, 실패시 None
    """
    image = cv2.imread(image_path)
    if image is None:
        return None

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def preprocess_image(image: np.ndarray) -> Tuple[Optional[np.ndarray], Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]]:
    """
    입력 이미지에서 얼굴을 검출하고 크기를 조정하는 함수

    Args:
        image (np.ndarray): 입력 이미지

    Returns:
        Tuple[Optional[np.ndarray], Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]]:
            - 검출된 얼굴 이미지 또는 None
            - 얼굴 bbox 좌표 (x,y,w,h) 또는 (None, None, None, None)
    """
    FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    if face_cascade.empty():
        print("XML 파일을 로드할 수 없습니다")
        return None, (None, None, None, None)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(image, 1.3, 5)

    if len(faces) > 0:
        (x,y,w,h) = faces[0]
        face = image[y:y+h, x:x+w]
        return face, (x,y,w,h)

    return None, (None, None, None, None)

def model_load() -> Tuple[AutoImageProcessor, AutoModelForImageClassification, str]:
    """
    감정 인식 모델과 프로세서를 로드하는 함수

    Returns:
        Tuple[AutoImageProcessor, AutoModelForImageClassification, str]: 
            이미지 프로세서, 모델, 디바이스
    """
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    MODEL_PATH = "dima806/facial_emotions_image_detection"

    processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
    model = AutoModelForImageClassification.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()

    return processor, model, device

def predict(
    image: np.ndarray, 
    processor: AutoImageProcessor, 
    model: AutoModelForImageClassification, 
    device: str
) -> int:
    """
    이미지의 감정을 예측하는 함수

    Args:
        image (np.ndarray): 입력 이미지
        processor (AutoImageProcessor): 이미지 프로세서
        model (AutoModelForImageClassification): 감정 인식 모델
        device (str): 연산 장치('cuda' 또는 'cpu')

    Returns:
        int: 예측된 감정 클래스 인덱스
    """
    inputs = processor(images=image, return_tensors="pt")
    inputs = inputs.to(device)
    with torch.no_grad():
        preds = model(inputs.pixel_values)
    preds = preds.logits.argmax(-1).item()
    return preds

def predict_probs(
    image: np.ndarray, 
    processor: AutoImageProcessor, 
    model: AutoModelForImageClassification, 
    device: str
) -> Dict[str, float]:
    """
    이미지의 각 감정 클래스별 확률을 예측하는 함수

    Returns:
        Dict[str, float]: 각 감정 레이블별 예측 확률
    """
    inputs = processor(images=image, return_tensors="pt")
    inputs = inputs.to(device)
    with torch.no_grad():
        preds = model(inputs.pixel_values)
        probabilities = torch.nn.functional.softmax(preds.logits, dim=-1)

    emotion_dict = {
        'sad': probabilities[0][0].item(),
        'disgust': probabilities[0][1].item(),
        'angry': probabilities[0][2].item(),
        'neutral': probabilities[0][3].item(),
        'fear': probabilities[0][4].item(),
        'surprise': probabilities[0][5].item(),
        'happy': probabilities[0][6].item()
    }
    return emotion_dict

def map_prediction(prediction: int) -> str:
    """
    예측된 클래스 인덱스를 감정 레이블로 변환하는 함수

    Args:
        prediction (int): 예측된 클래스 인덱스

    Returns:
        str: 감정 레이블
    """
    emotion_dict: Dict[int, str] = {
        0: 'sad', 
        1: 'disgust', 
        2: 'angry', 
        3: 'neutral', 
        4: 'fear', 
        5: 'surprise', 
        6: 'happy'
    }
    return emotion_dict[prediction]

def mozaic_face(
    face: np.ndarray, 
    mosaic_size: int = 15,
    method: str = 'pixelate',
    alpha: float = 0.8
) -> np.ndarray:
    """
    얼굴 이미지를 다양한 방식으로 모자이크 처리하는 함수

    Args:
        face (np.ndarray): 입력 얼굴 이미지
        mosaic_size (int): 모자이크 블록 크기
        method (str): 모자이크 방식 ('pixelate', 'blur', 'gradient')
        alpha (float): 블렌딩 강도 (0.0 ~ 1.0)

    Returns:
        np.ndarray: 모자이크 처리된 얼굴 이미지
    """
    h, w = face.shape[:2]

    if method == 'pixelate':
        # 기본 모자이크
        face_small = cv2.resize(face, (w//mosaic_size, h//mosaic_size), 
                              interpolation=cv2.INTER_LINEAR)
        face_mosaic = cv2.resize(face_small, (w, h), 
                               interpolation=cv2.INTER_NEAREST)

    elif method == 'blur':
        # 가우시안 블러 적용
        face_mosaic = cv2.GaussianBlur(face, (mosaic_size, mosaic_size), 0)

    elif method == 'gradient':
        # 그라데이션 모자이크
        face_blur = cv2.GaussianBlur(face, (mosaic_size, mosaic_size), 0)
        face_mosaic = cv2.addWeighted(face, 1-alpha, face_blur, alpha, 0)

    else:
        raise ValueError("지원하지 않는 모자이크 방식입니다.")

    return face_mosaic.astype(np.uint8)

def main(image_path: str) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[str], Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]]:
    """
    이미지의 감정을 예측하는 메인 함수

    Args:
        image_path (str): 입력 이미지 파일 경로

    Returns:
        Tuple[np.ndarray, Optional[np.ndarray], Optional[str], Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]]: 
            - 원본 이미지(bbox 포함)
            - 전처리된 얼굴 이미지 또는 None
            - 예측된 감정 레이블 또는 None  
            - 얼굴 bbox 좌표 (x,y,w,h) 또는 (None, None, None, None)
    """
    image = load_image(image_path)

    face, (x,y,w,h) = preprocess_image(image) 
    if face is None:
        return image, None, None, (None, None, None, None)

    image = cv2.rectangle(image, (x,y), (x+w, y+h), (255,0,0), 2)

    processor, model, device = model_load()
    prediction = predict(face, processor, model, device)

    return image, face, map_prediction(prediction), (x,y,w,h)

image_path = r"C:\\img.jpg"  # 파일 경로
image, face, emotion, bbox = main(image_path)

# 결과 확인
print(f"Predicted emotion: {emotion}")
print(f"Bounding box: {bbox}")

# 결과 시각화
if face is not None:
    plt.imshow(face)
    plt.title(f"Emotion: {emotion}")
    plt.axis("off")
    plt.show()
else:
    print("얼굴을 감지하지 못했습니다.")