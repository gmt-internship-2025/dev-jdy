### **Framework for the Complete Gaze Tracking Pipeline**

https://github.com/pperle/gaze-tracking-pipeline.git

- 저장소 클론하기

```bash
git clone https://github.com/pperle/gaze-tracking-pipeline.git
```

- 클론한 폴더로 이동

```bash
cd gaze-tracking-pipeline
```

- 가상환경에서 필요한 라이브러리 설치
- 왜 가상환경을 쓰면 좋은가?
    - 라이브러리 충돌 방지
    - 깨끗한 환경 유지
    - 쉽게 초기화 가능

- 가상환경 python3-venv 패키지 설치

```bash
sudo apt update
sudo apt install python3-venv
```

- 가상환경 만들기

```bash
# 현재 폴더 안에 venv라는 폴더가 만들어지고 거기에 가상환경이 생성됨
python -m venv venv 
```

- 가상환경 활성화

```bash
source venv/bin/activate
```

```bash
(venv) jetson@jetson-desktop:~/Desktop$
```

- 필요한 라이브러리 설치 (PyTorch, torchvision)
    - 가상환경은 완전히 깨끗한 환경으로 시작하기 때문에 필요한 것들을 새로 설치해야 함.

```bash
pip install --upgrade pip
```

```bash
# JetPack 6.2용 PyTorch 설치
wget https://developer.download.nvidia.com/compute/redist/jp/v60dp/pytorch/torch-2.3.0a0+6ddf5cf85e.nv24.04.14026654-cp310-cp310-linux_aarch64.whl
pip install torch-2.3.0a0+6ddf5cf85e.nv24.04.14026654-cp310-cp310-linux_aarch64.whl
```

---
```
## requirements.txt
torchvision==0.10.0 # 버전 문제 발생
numpy==1.18.5
opencv_python==4.5.1.48
torch==1.9.0 # 설치완료
albumentations==1.1.0
matplotlib==3.4.3
mediapipe==0.8.7
pgi==0.0.11.2
pytorch_lightning==1.4.9
PyYAML==6.0
```
