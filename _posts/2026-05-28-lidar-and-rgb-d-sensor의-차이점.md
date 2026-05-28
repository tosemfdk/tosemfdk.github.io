---
layout: post
title: "Lidar and RGB-D sensor의 차이점"
date: 2026-05-28T11:06:00.000Z
math: true
image:
  path: "/assets/img/posts/36ecbb7d-7937-81d8-b845-ec60ba87d1fe.webp"
categories:
  - "study"
---



# **RGB-D depth camera와 LiDAR는 악조건에서 왜 다르게 무너지는가**

<div class="notion-callout" style="display:flex;gap:0.75rem;align-items:flex-start;padding:1rem 1.1rem;margin:1.25rem 0;border-radius:14px;background:rgba(37, 99, 235, 0.12);border:1px solid rgba(37, 99, 235, 0.28);" markdown="1">
<div class="notion-callout__icon" style="font-size:1.25rem;line-height:1.4;flex:0 0 auto;">🔦</div>
<div class="notion-callout__content" style="flex:1 1 auto;min-width:0;" markdown="1">

핵심은 둘 다 빛을 쓰지만 depth를 계산하는 방식이 다르다는 점이다. IR 패턴 기반 RGB-D는 “패턴이 보여야” 하고, ToF depth camera는 “빛의 시간/위상이 읽혀야” 하며, LiDAR는 “각 레이저 ray의 return이 잡혀야” 한다.

</div>
</div>

![Depth 계산 원리 비교: IR 패턴 기반 RGB-D, ToF depth camera, LiDAR](/assets/img/posts/36ecbb7d-7937-81d8-b845-ec60ba87d1fe.webp)

## **1. RGB-D depth camera도 한 종류가 아니다**

RGB-D는 RGB 영상과 depth map을 함께 제공하는 센서군을 말한다. 하지만 depth를 만드는 방식은 크게 **Structured Light, Active Stereo, ToF**로 나뉜다.

1. **Structured Light:** 예전 Kinect v1처럼 IR 점무늬/패턴을 물체에 쏘고, 그 패턴이 표면에서 어떻게 변형되어 보이는지 분석해 depth를 계산한다.
![](/assets/img/posts/36ecbb7d-7937-808a-82c4-db010e417149.webp)

1. **Active Stereo**: RealSense D400 계열처럼 두 IR 카메라와 IR projector를 사용한다. 왼쪽/오른쪽 IR 이미지에서 같은 패턴 조각의 위치 차이, 즉 disparity/correspondence를 찾아 depth를 계산한다.
   ![](/assets/img/posts/36ecbb7d-7937-80a6-b51b-d531ac52e1ae.webp)
$$
Z = \frac{fB}{d}
$$

여기서 Z는 깊이, f는 초점거리, B는 두 카메라 사이 baseline, d는 disparity다. 중요한 점은 “IR 두 개의 밝기 차이”가 아니라 두 IR 이미지에서 같은 점 또는 패턴 조각을 매칭한다는 것이다.

1. **ToF depth camera:** Kinect v2, Azure Kinect 계열처럼 IR/NIR 빛을 쏘고 돌아오는 시간 또는 위상 차이를 픽셀별로 계산한다. 패턴 모양 자체를 분석하는 방식은 아니다.
## **2. IR 패턴 기반 RGB-D가 악조건에서 약한 이유**

Structured Light와 Active Stereo는 표면에 투사된 IR 패턴이 IR 카메라에서 안정적으로 보여야 한다. 검은색·정반사·투명체·강한 외부광에서는 이 전제가 깨진다.

- **검은색/어두운 표면**: 검은 가죽, 어두운 플라스틱, 어두운 천은 IR을 많이 흡수할 수 있다. return 신호가 약해지면 패턴이 보이지 않고, correspondence를 찾지 못해 depth hole이 생긴다.
- **반짝이는 표면/금속/가죽**: diffuse하게 흩뿌려져야 할 패턴이 특정 방향으로 정반사되거나 번진다. 표면에 패턴이 “붙어” 보이지 않으므로 매칭이 실패한다.
- **유리/투명체**: IR이 통과하거나 다른 물체에서 반사되어 들어오면 표면의 실제 거리가 아니라 잘못된 거리 또는 hole이 나온다.
- **강한 햇빛**: active IR 패턴이 외부 IR 성분에 묻혀 contrast가 낮아진다.
## **3. ToF depth camera는 패턴 분석보다 낫지만 만능은 아니다**

ToF는 패턴 매칭이 필요 없으므로 텍스처가 없는 흰 벽 같은 표면에서는 오히려 잘 작동할 수 있다. 하지만 결국 빛이 센서로 돌아와야 하므로 return이 약하거나 여러 경로로 섞이면 depth 품질이 떨어진다.

- **검은색 표면**: 반사 신호가 약해져 depth noise가 증가하거나 hole이 생긴다.
- **유리/거울**: 빛이 통과하거나 다른 방향으로 튀어 실제 표면 위치와 다른 depth가 계산될 수 있다.
- **금속/광택 표면**: multipath, saturation, wrong depth가 발생하기 쉽다.
- **여러 ToF 센서 동시 사용**: 서로의 active IR 신호가 interference를 만들 수 있다.
> **ToF의 대표적인 문제는 multipath interference다. 빛이 테이블 → 벽 → 센서처럼 여러 경로로 돌아오면 센서는 어느 경로의 거리를 읽어야 하는지 혼동한다.**

## **4. LiDAR는 무엇이 다른가**

LiDAR도 넓게 보면 ToF 계열이 많다. 하지만 일반 RGB-D ToF 카메라처럼 넓은 장면 전체를 dense pixel로 한 번에 읽기보다, 좁은 레이저 beam을 ray별로 쏘고 그 return time을 직접 측정하는 구현이 많다.

- 패턴을 표면에 맺히게 해서 분석하는 것이 아니라, 각 ray가 돌아왔는지와 언제 돌아왔는지를 본다.
- 텍스처가 없어도 상관없다. 하얀 벽, 무늬 없는 바닥, 특징점 없는 표면도 return이 잡히면 3D point가 된다.
- beam에 에너지가 집중되고 수신기가 장거리 return을 받도록 설계된 경우가 많아, 일반 IR 패턴 기반 RGB-D보다 악조건에서 일부 point를 더 잘 건질 수 있다.
- 하지만 검은 표면, 유리, 거울, 매우 비스듬한 입사각에서는 LiDAR도 미검출, ghost point, wrong return, multi-return 문제를 겪는다.
## **5. 핵심 비교표**

| **구분** | **IR 패턴 기반 RGB-D** | **ToF depth camera** | **LiDAR** |
| --- | --- | --- | --- |
| **대표 원리** | IR 패턴/스테레오 매칭 | 빛의 시간/위상 측정 | 레이저 return 시간 측정 |
| **Depth 계산** | 패턴 위치 차이, disparity | per-pixel phase/time | per-ray time-of-flight |
| **텍스처 없는 벽** | projector 패턴이 보이면 가능 | 비교적 강함 | 강함 |
| **검은색 표면** | 매우 약함 | 약함 | 약하지만 상대적으로 더 버팀 |
| **정반사 표면** | 패턴 깨짐 | multipath/wrong depth | ghost/미검출 가능 |
| **유리** | 매우 약함 | 매우 약함 | 매우 약함: 통과/반사 혼재 |
| **햇빛** | IR 패턴이 묻힘 | active IR 신호가 묻힘 | 실외용 LiDAR는 상대적으로 강함 |
| **출력 형태** | dense depth image | dense depth image | sparse/semi-dense point cloud |
| **Normal 계산** | depth map 주변 픽셀로 계산 | depth map 주변 픽셀로 계산 | point cloud local plane fitting |

![악조건별 depth 실패 모드와 normal 계산 영향](/assets/img/posts/36ecbb7d-7937-8136-bc2d-c82571300996.webp)

## **6. Normal 계산 관점에서의 차이**

Normal 계산 자체는 RGB-D와 LiDAR가 수학적으로 비슷해 보일 수 있다. 둘 다 주변 3D 점을 모아 local plane을 맞추고, 그 평면의 수직 방향을 normal로 삼는다.

- RGB-D: depth image의 주변 픽셀을 3D로 unproject한 뒤 local plane 또는 cross product로 normal을 계산한다.
- LiDAR: point cloud에서 k-nearest neighbor 또는 radius neighbor를 모아 local plane fitting으로 normal을 계산한다.
차이는 입력 3D 점의 신뢰도와 분포다. RGB-D는 dense하지만 악조건에서 hole과 noisy depth가 생긴다. LiDAR는 sparse하지만 살아남은 point가 있으면 더 넓은 범위에서 기하 정보를 유지할 수 있다.

예를 들어 검은 가죽 소파에서는 RGB-D depth map 자체가 뚫려 normal을 계산하지 못할 수 있다. 반면 LiDAR는 일부 return이라도 살아 있으면 sparse point cloud 기반으로 local plane normal을 추정할 수 있다.

## **결론**

> **RGB-D 패턴 방식은 “무늬가 보여야 depth를 계산”하고, ToF는 “빛이 돌아오는 시간/위상을 봐야 depth를 계산”하고, LiDAR는 “각 레이저 ray의 return을 직접 재서 sparse하지만 강한 기하 정보를 얻는다.”**

따라서 “빛을 쏜다”는 점은 RGB-D와 LiDAR가 비슷하지만, depth 계산 방식이 다르기 때문에 악조건에서 무너지는 방식도 다르다. LiDAR도 검은색·유리·정반사에는 약하지만, 일반적인 IR 패턴 기반 RGB-D보다 텍스처 부족에 훨씬 둔감하고 일부 return만 살아도 3D 기하 정보를 남길 가능성이 높다.



