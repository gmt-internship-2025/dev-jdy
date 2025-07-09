### 예제코드
 
- 필요한 라이브러리 추가 설치

```
pip3 install scikit-image
```

- 테스트할 얼굴 이미지 파일 다운로드    

![image](https://github.com/user-attachments/assets/c034b8aa-43d4-458f-b0a2-48b89403a0c8)


- 예제 코드 실행

```
python3 test_face.py
```

- 얼굴 detect 결과

```
Face 0: Left: 154, Top: 114, Right: 333, Bottom: 293
```

![image](https://github.com/user-attachments/assets/34635e6b-5619-43a2-a317-d5919b258ad8)


- 눈동자 detect 코드 추가
- Dlib의 shape_predictor_68_face_landmarks.dat.bz2 모델 파일 다운로드

```
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
```

- 다운로드 받은 파일 압축해제

```
bunzip2 shape_predictor_68_face_landmarks.dat.bz2
```

- 예제 코드 실행

```
python3 test_face_eyes.py
```

- 눈동자 detect 결과

```
Face 0: Left: 154, Top: 114, Right: 333, Bottom: 293
Left eye center: [204.         165.66666667]
Right eye center: [297.33333333 169.83333333]
```

![image](https://github.com/user-attachments/assets/5f01178f-d25f-41f6-9392-6ff5e6005b4c)

```python
# 왼쪽 눈과 오른쪽 눈의 좌표 추출
left_eye = landmarks.parts()[36:42]  # 왼쪽 눈 (6개 점)
right_eye = landmarks.parts()[42:48]  # 오른쪽 눈 (6개 점)
    
# 왼쪽 눈의 중심 계산
left_eye_center = np.mean([(point.x, point.y) for point in left_eye], axis=0)
right_eye_center = np.mean([(point.x, point.y) for point in right_eye], axis=0)
```
- 얼굴 랜드마크 인덱스 코드 실행

```
python3 index_test.py
```

![image](https://github.com/user-attachments/assets/6f810e1c-21fb-4608-8d5d-9228f09d6988)


- 눈동자(중심) 계산법

```python
# 왼쪽 눈과 오른쪽 눈의 좌표 추출
left_eye = landmarks.parts()[36:42]  # 왼쪽 눈 (6개 점)
right_eye = landmarks.parts()[42:48]  # 오른쪽 눈 (6개 점)
    
# 왼쪽 눈의 중심 계산
left_eye_center = np.mean([(point.x, point.y) for point in left_eye], axis=0)
right_eye_center = np.mean([(point.x, point.y) for point in right_eye], axis=0)
```
