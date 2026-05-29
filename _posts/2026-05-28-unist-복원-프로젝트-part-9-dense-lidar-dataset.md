---
layout: post
title: "[Unist 복원 프로젝트] Part 9. dense lidar dataset"
date: 2026-05-28T12:54:00.000Z
math: true
image:
  path: "/assets/img/posts/36ecbb7d-7937-80c0-a8e1-c2c0f922a016.webp"
categories:
  - "UNIST"
---

## 현재 상황

- 현재 상황은 lidar slam (fast-livo2)가 겉보이게는 정확해 보이지만 pose를 colmap의 pose와 비교해볼때 distance에는 2~6cm의, rotation에서는 2~3 도의 평균적인 오차가 있는 것을 확인하였다. 이러한 얼핏 보기에는 작아보이는 오차만으로도 3DGS 결과물에는 차이가 많이 발생하는 것을 확인하였다. 
  - **colmap 결과물 PSNR** : 약 28 db
  - **online slam 결과물의 PSNR** : 약 21 db
- 이러한 문제의 원인에는 첫번째로 lidar - camera extrinsic calibration에서 문제가 있거나 fast-livo2의 slam결과물에서 이상이 있는 것으로 짐작할 수 있다. 
  - global 최적화를 제대로 구현한다면 이러한 문제가 해결될 수도 있겠지만 병목이 생기고 있으니 일단 다음 단계로 먼저 넘어가기로 한다.
## dense lidar point 중첩

- livox mid360 lidar는 128채널 같은 라이다에 비해서 이동하면서 online slam할 경우에 생기는 **lidar point가 mesh를 만들기에 dense하지 못하다**. 따라서 삼각대에 고정해놓고 라이다 토픽을 특정 시간동안 중첩시켜서 dense한 라이다 포인트를 얻을 수 있도록 하려고 한다. 
- 이후에 취득한 dense PCD들을 icp와 같은 매칭기법을 통해서 병합하고 작은 scene의 dens lidar point map을 취득 할 수 있다. 


### dataset 취득

![](/assets/img/posts/36ecbb7d-7937-80c0-a8e1-c2c0f922a016.webp)

![](/assets/img/posts/36ecbb7d-7937-8078-ba88-c89d9ca97f89.webp)

- 새로운 rig는 카메라의 1/4 inch hall과 딱 맞아 떨어진다. 
**데이터셋 취득 전략**

1. 한곳에서 120초 동안 pcd (약 2천3000만 포인트)를 중첩한다.
1. 그곳에서의 이미지를 저장한다.
   1. 이름을 sequence로 저장하여 나중에 icp매칭에서 순차적으로 병합한다.
1. 삼각대 방향과 위치를  옮긴후 1,2를 반복한다. 


**dataset**

![](/assets/img/posts/36ecbb7d-7937-80bc-b022-e7cefd7bf3a4.webp)

![](/assets/img/posts/36ecbb7d-7937-80a5-8cdf-c13766e32677.gif)



- pcd와 동시에 취득한 이미지가 중복되어있어서 1시간동안 촬영한 pcd file도 중복되어있지 않을까 두려웠지만,, 다행히 pcd data는 잘 취득된 것 같다. 
- 1~2, 3~4, 5~6 번같이 2개씩 단위로 각도를 180도 회전해가면서 데이터를 수집하였다. 각 포인트가 0.5gb에 point개수도 같으니 ICP를 돌리는 것은 voxelize한 후에 진행하면 것 같다. 




## ICP **strategy**

먼저 순서대로 취득한 PCD point는 인접한 번호끼리 물리적으로 가까운 거리에 위치하기 때문에 먼저 이웃끼리 1~2, 3~4, 5~6과 같이 병합한 후에 병합한 결과물을 다시 이웃끼리 병합하는 식으로 진행하였다.



**failure case**

![](/assets/img/posts/36fcbb7d-7937-804b-91da-eeb8a52b71bf.webp)

![](/assets/img/posts/36fcbb7d-7937-801a-92e1-edfa0dc39bf3.webp)

- 완전히 좌표계가 틀어져 버린 case 이다. 인접한 번호 사이에서 중복되는 feature가 별로 없을때에 이러한 현상이 나타날 수 있다.
![](/assets/img/posts/36fcbb7d-7937-80f4-9e78-e866a462cf10.webp)

- 잘 정합된 것처럼 보이지만 직사각형인 방의 특성상 서로 정반대에 있는 면에만 중복되는 feature가 있다면 (ex 계단, & TV) 그러한 feature를 기준으로  두 **PCD의 관계가 180도 틀어져 버리는 경우**가 있다. 자세히 보면 한쪽 면에만 존재하는 작은 칠판이 양쪽에 존재하는 것을 확인할 수 있다.


![](/assets/img/posts/36fcbb7d-7937-808f-8fff-d9cb2d9a6011.webp)

![](/assets/img/posts/36fcbb7d-7937-80c1-8eb1-d8533cc16f36.webp)

- lidar의 한계로 인해서 어쩔수 없이 ghosting point가 맺히는 현상이 있다.
- lidar 센서의 outlier로 실제로 존재하지 않는 먼거리에 점이 찍히는 것을 확인할 수 있다.




### Result

먼저 인접한 포인트 번호끼리 merge한 후에 육안으로 보기에 포인트끼리 매끄럽게 합쳐진 결과물을 가지고 **1 ~ 18번 pcd 까지의 융합 결과를 한번 생성**하였다.

**과정**

1. 원본 PCD 준비
   - data_0001~0030.pcd
   - 각 파일 약 2,400만 points
1. ICP용 voxel 생성
   - dense 원본을 바로 ICP하지 않고, 각 PCD를 **5cm voxel**로 downsample 하였다.
   - 이후 outlier 방지를 위해 각 voxel cloud에서:
     abs(x)<=10, abs(y)<=10, abs(z)<=10
     범위만 남김 **(이 작업이 poincloud상의 원점에서 멀리 떨어진 이상한 outlier를 날려주어 포인트간의 merging quality를 높여주었다.)**
1. 초기 trusted anchor를 구성
   - **시각 검수로 괜찮다고 판단한 조합**들을 먼저 합침
   - 최종적으로 0001~0018까지 accepted root 생성
   - 이후 0019~0030은 하나씩 붙이는 방식으로 진행
1. interactive sequential ICP
   - 시작 root: 0001~0018
   - 0019부터 0030까지 하나씩:
     1. 현재 accepted root에 새 PCD 하나를 ICP
     1. Open3D 창으로 확인
   - 결과:
     accepted: 0001~0019, 0022~0030
     rejected: 0020, 0021 (이 reject된 포인트들은 애초에 point가 방해물에 의해서 넓은 커버리지를 가지지 못하는 경향이 있었다.)
1. pose tightening
   - accepted된 각 PCD pose를 더 미세 조정
   - 방식:
     - 각 scan을 하나씩 잡고
     - 그 scan을 제외한 나머지 accepted scan들로 global map 생성
     - 해당 scan을 그 map에 point-to-plane ICP로 다시 정렬
   - 0001은 anchor로 고정해서 전체 좌표계가 흔들리지 않게 함
   - 2 iteration 수행
   - 변화량은 대부분 mm 단위라 안정적으로 수렴
1. 최종 dense 병합
   - ICP는 cropped voxel cloud로 했지만,
   - 최종 dense PCD는 crop하지 않음
   - accepted raw PCD 전체 points에 refined pose를 적용해서 병합
   - 최종:
     ICP/refined_abs10_poses/dense_refined_no_crop.pcd
   - 포함:
     0001~0019, 0022~0030
   - 제외:
     0020, 0021
   - 총 points:
     671,806,848 (6억 7000만개)


![](/assets/img/posts/36fcbb7d-7937-80ab-a663-c18d3676d75e.webp)

![](/assets/img/posts/36fcbb7d-7937-8024-964a-c60bac59ef54.gif)

