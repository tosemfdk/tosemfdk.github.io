---
layout: post
title: "[Unist 복원 프로젝트] Part 6. Fast-Livo SLAM"
date: 2026-05-11T08:09:00.000Z
math: true
image:
  path: "/assets/img/posts/35dcbb7d-7937-808b-95db-f2a1fd163a6e.webp"
categories:
  - "UNIST"
---



## Fast-livo란?

![](/assets/img/posts/35dcbb7d-7937-808b-95db-f2a1fd163a6e.webp)

- Lidar-Inertial-Visual slam으로써 매우 강건하게 작동하는 slam이다. 
FAST-LIVO2는 LiDAR, 카메라, IMU 데이터를 단일 통합 복셀 맵에 실시간으로 결합하여 드론과 같은 로봇이 극한 환경에서도 정확하게 위치를 잡고 3D 지도를 그리게 해주는 고속 SLAM 시스템이다. **특징 추출 없이 원시 데이터를 직접 사용하는 방식과 순차적 업데이트 전략으로 연산 효율을 극대화**했으며, 평면 정보 활용, 조도 변화 보정, 센서 사각지대 보완 기술을 더해 기존 모델보다 훨씬 강력한 정밀도를 자랑한다. 이를 통해 저사양 임베디드 보드에서도 끊김 없이 작동하며, **자율 비행이나 고정밀 3D 렌더링(3DGS) 등 다양한 실전 분야에서 압도적인 성능**을 보여주는 혁신적인 기술이다.





## rig for mapping

이런식으로 우체통 처럼 생긴 센서 rig를 제작하여 안에서 arduino를 통해서 동기화하고 jetson을 통해서 센서데이터를 수집할 수 있도록 하였다.

![](/assets/img/posts/35dcbb7d-7937-8075-a0e9-d4664dea3040.webp)

![](/assets/img/posts/35dcbb7d-7937-8088-bffe-ede7eedd0488.webp)

![](/assets/img/posts/35dcbb7d-7937-803e-8be2-fa38f6e82d60.webp)

![](/assets/img/posts/35dcbb7d-7937-8098-a9fc-fd6a19f15f2d.webp)

## camera lidar calibration

![](/assets/img/posts/35dcbb7d-7937-8023-816a-e6b582e50b2c.webp)

- livox lidar가 calibration board테두리를 기준으로 실제로 존재하지 않는 검정색 point를 찍는것을 확인하였다. 이는 캘리브레이션을 진행할 때에 매우 큰 정확도 하락으로 이어졌다..


![](/assets/img/posts/35dcbb7d-7937-80d6-8dfa-d8ba1b588b3c.webp)

![](/assets/img/posts/35dcbb7d-7937-80ec-9770-c91ca6ded044.webp)

- 실제로 projection한 포인트가 보드와 일치하지 않는 것을 확인하였고 maual 한 캘리브레이션 방식을 사용하여 맞추었다.
![](/assets/img/posts/35dcbb7d-7937-80be-8982-cb5c2f46f16b.webp)





## Fast-Livo2 Ros2 porting

기존의 fast-livo코드는 ros1을 기준으로 구성되어있어 ros2로 간단하게 변경하였다.

![](/assets/img/posts/35dcbb7d-7937-80cb-8130-e6b9c05e0159.webp)

![](/assets/img/posts/35dcbb7d-7937-80a5-bec2-cf5585e397a0.gif)



- 지금은 가장 단순하게 카메라와 lidar의 fov가 겹치는 부분에 대해서만 colored point를 생성하고 이를 map frame상에서 누적하여 visualization하는 방식이다. 또한 지금은 voxelization한 후의 포인트를 back projection하고 있기 떄문에 포인트가 dense하게 나오고 있지 않지만 이러한 부분은 수정가능하다.


