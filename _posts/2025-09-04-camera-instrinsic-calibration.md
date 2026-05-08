---
layout: post
title: "Camera instrinsic calibration"
date: 2025-09-04T06:16:00.000Z
math: true
image:
  path: "/assets/img/posts/264cbb7d-7937-802f-9a3c-e6a4e823a971.webp"
categories:
  - "Team Vision"
---



## **Camera Calibratin** 

## calibration 이란?

<div class="notion-callout" style="display:flex;gap:0.75rem;align-items:flex-start;padding:1rem 1.1rem;margin:1.25rem 0;border-radius:14px;background:rgba(127, 127, 127, 0.12);border:1px solid rgba(127, 127, 127, 0.28);" markdown="1">
<div class="notion-callout__icon" style="font-size:1.25rem;line-height:1.4;flex:0 0 auto;">💡</div>
<div class="notion-callout__content" style="flex:1 1 auto;min-width:0;" markdown="1">

- 카메라의 **초점거리, 왜곡 계수, 가로세로 길이**등을 알아내기 위한 방법으로 보통 **체커보드나 april grid**등의 미리 사이즈가 어떻게 되어있는지를 알고있는 보드를 사용하여 수행한다.
![](/assets/img/posts/264cbb7d-7937-802f-9a3c-e6a4e823a971.webp)



![](/assets/img/posts/264cbb7d-7937-80c5-9000-efa90a9a053f.webp)



나의 경우에는 10 x 10에 한칸의 길이가 7cm인 체커보드를 사용하였다.

</div>
</div>









- 나는 ros2 에서 제공해주는 calibration 패키지를 통해 카메라의 캘리브레이션을 수행했다.

```bash
sudo apt install ros-humble-camera-calibration-parsers

sudo apt install ros-humble-camera-info-manager

sudo apt install ros-humble-launch-testing-ament-cmake

# 이 전에 내 ssh키를 github에 등록하도록 하자 
ssh-keygen -t ed25519 -C "you@example.com"

cd ~/colcon_ws/src

git clone -b <ros2-distro> git@github.com:ros-perception/image_pipeline.git

cd ..

colcon build

source install/setup.bash
```

- 먼저 필요한 패키지를 다운 받는다. 이후 빌드를 진행한다. 
  

```bash
ros2 topic list

ros2 topic hz /camera/image_raw
```

- 이런식으로 내 카메라 토픽이 ros2에서 잘 나오고 있는지 확인한다. 

```bash
ros2 run camera_calibration cameracalibrator --size 9x9 --square 0.07 --ros-args -r image:=/usb_cam/image_raw -p camera:=/my_camera

```

<details markdown="1">
<summary>파라미터들</summary>



## 📌 카메라 이름 관련

- `**c**`**,** `**-camera_name**`
  → 보정(calibration) 결과 YAML 파일 안에 저장될 **카메라 이름**을 지정합니다.
  
  (예: `--camera_name=my_camera` → 저장되는 파일의 루트 키가 `my_camera`로 들어감)
---

## 📌 체스보드(패턴) 관련 옵션

- `**p**`**,** `**-pattern**`
  → 사용할 보정 패턴 종류 선택
  
  - `'chessboard'` : 일반 체스보드
  - `'circles'` : 원 패턴
  - `'acircles'` : 비대칭 원 배열(asymmetric circles)
  - `'charuco'` : ChArUco 패턴
- `**s SIZE**`**,** `**-size=SIZE**`
  → 체스보드의 **내부 코너 개수** (예: 7x9 체스보드면 `--size 7x9`)
  
  > 여기서 NxM은 “사각형 개수”가 아니라 “내부 교차점 개수”임에 주의!
- `**q SQUARE**`**,** `**-square=SQUARE**`
  → 한 칸의 실제 **크기(단위: m)**.
  
  예: 20mm 정사각형이면 `--square 0.02`
---

## 📌 ROS 통신 관련 옵션

- `**-approximate=SECS**`
  → 스테레오 카메라처럼 두 이미지가 약간 시간 차이 있을 때, 몇 초까지 허용할지 지정.
- `**-no-service-check**`
  → 실행 시 `set_camera_info` 서비스가 켜져 있는지 확인하는 과정을 생략.
---

## 📌 최적화 관련 옵션

- `**-fix-principal-point**`
  → 주점(principal point)을 이미지 중앙으로 고정. (cx, cy를 변수가 아니라 고정값으로 둠)
- `**-fix-aspect-ratio**`
  → 초점거리 fx, fy를 같게 맞춤 (픽셀 비율 1:1로 강제).
- `**-zero-tangent-dist**`
  → 왜곡 보정에서 **접선 왜곡(p1, p2)**을 0으로 고정.
- `**k NUM_COEFFS**`**,** `**-k-coefficients=NUM_COEFFS**`
  → 반径 왜곡 계수(k1~k6) 몇 개 쓸지 선택 (기본은 2개 k1, k2만).
- `**-disable_calib_cb_fast_check**`
  → OpenCV 체스보드 코너 찾기에서 **빠른 실패 검사(Fast check)** 비활성화.
  
  (체스보드가 제대로 안 잡힐 때 이 옵션 켜면 더 정확히 찾을 수 있음, 대신 느려짐)


</details>





![](/assets/img/posts/264cbb7d-7937-8046-9ab5-f74d27adfd39.gif)

- 체커보드가 꽤 무거워서 힘들었다. 약 40장 정도의 캘리브레이션 이미지가 찍히고 나니 캘리브레이션이 버튼이 떳다!
  - 이후 save → commit 한 후에 종료 (터미널에 카메라의 계수가 뜬다)
  - 이후 저장하고 싶은 디렉토리로 가서 카메라 파라미터를 저장한다.
  
  ```bash
  tar -xzvf /tmp/calibrationdata.tar.gz
  ```
  
  ![](/assets/img/posts/264cbb7d-7937-8078-b12e-f9b8f68e3fd6.webp)
  
  ![](/assets/img/posts/264cbb7d-7937-8098-a9d7-e8148bea9634.webp)
  
  ![](/assets/img/posts/264cbb7d-7937-80f6-8136-c592ed2a2c9f.webp)


