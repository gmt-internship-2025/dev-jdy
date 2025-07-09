### 예제코드

- 예제 코드 실행

```
python3 test_face_eyes.py
```

- 얼굴, 눈동자 detect 결과

```
Face Bounding Box: x: 143, y: 202, w: 251, h: 251
Left Eye: (0.3661453127861023, 0.3285500705242157)
Right Eye: (0.6384799480438232, 0.3410367965698242)
```

![image](https://github.com/user-attachments/assets/f6cb3c57-80a4-46fc-a995-5122db7b0c04)


- 얼굴 랜드마크 인덱스

![image](https://github.com/user-attachments/assets/b9eddffa-6d34-4185-83e6-fd4cbabb2327)


![image](https://github.com/user-attachments/assets/08a9aac8-890e-4b65-ac1e-881ef26b585b)


https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png

- 인덱스 번호가 순차적이지 않아 눈중심(눈동자)의 위치 좌표를 알아내기에 한계가 있음
- 코드에서 33, 263 인덱스만 적었으나 총 8개의 좌표가 표시되는 현상이 있음
