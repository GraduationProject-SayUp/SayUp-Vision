import cv2
import mediapipe as mp
import matplotlib.pyplot as plt

# MediaPipe Face Mesh 초기화
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# 스타일 설정
mesh_drawing_spec = mp_drawing.DrawingSpec(thickness=1, color=(200, 200, 200))  # 메쉬망 색 (연한 회색)
lips_drawing_spec = mp_drawing.DrawingSpec(thickness=1, color=(0, 0, 255))     # 입술 강조 색 (빨간색, 얇은 선)

# 이미지 경로
image_path = r"C:\\img.jpg"

def plt_imshow(title='image', img=None, figsize=(8, 5)):
    """Matplotlib을 사용하여 이미지 출력"""
    plt.figure(figsize=figsize)
    rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(rgbImg)
    plt.title(title)
    plt.xticks([]), plt.yticks([])
    plt.show()

# 이미지 읽기
image = cv2.imread(image_path)

if image is None:
    print("이미지를 불러올 수 없습니다. 경로를 확인하세요.")
else:
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        
        # 얼굴 메쉬 추출
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        if not results.multi_face_landmarks:
            print("이미지에서 얼굴을 찾을 수 없습니다.")
        else:
            print(f"얼굴 {len(results.multi_face_landmarks)}개를 찾았습니다.")
            
            annotated_image = image.copy()
            for face_landmarks in results.multi_face_landmarks:
                # 전체 메쉬망 그리기
                mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,  # 전체 메쉬망
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mesh_drawing_spec
                )
                
                # 입술 부분만 다른 색과 얇은 선으로 강조
                mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_LIPS,  # 입술 부분만
                    landmark_drawing_spec=None,
                    connection_drawing_spec=lips_drawing_spec  # 얇은 빨간색 선
                )
                
            # 결과 출력
            plt_imshow("Face Mesh with Highlighted Lips", annotated_image, figsize=(10, 10))
