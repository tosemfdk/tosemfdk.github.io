---
layout: post
title: "8월 30일 - image retrieval 성능 개선"
date: 2025-09-01T04:07:00.000Z
math: true
notion_source_id: "261cbb7d-7937-8044-8652-cf70f31a68ec"
image:
  path: "/assets/img/posts/261cbb7d-7937-8023-922c-cf77a3939381.webp"
categories:
  - "서울대학교 여름방학 인턴"
---

### Idea

- Dinov2의 이미지  **순수 이미지 feature**로 top k 검색
- Top K image의 sift descriptor 생성 후 feature 추출  
- 2d - 2d correspondence를 이용해서 essential matrix 를 추출 한 후 epipolar 기하학을 만족시키는 점을을 inlier, 임계값 보다 만족하지 못하는 점을 outlier로 분류 → lnlier의 비율, inlier의 최소 개수 등으로 filtering
- top k 개의 이미지를 새롭게 선정


### image feature 기반 방식 vs feautre + geometry

> 왼쪽이미지는 기존방식, 오른쪽 이미지는 새로운 방식의 결과물이다.



- 이미지 기반으로 찾을때에는 비슷한 이미지의 반대쪽 복도 이미지를 retrieval 하는 문제점이 있었다.
- geometrical 기반으로 할때에는 동일한 복도에서의 이미지를 retrieval하는 모습을 띈다.
![](/assets/img/posts/261cbb7d-7937-8023-922c-cf77a3939381.webp)



![](/assets/img/posts/261cbb7d-7937-80c4-ae2f-eb870b3f8302.webp)





- 사람과 같은 동적 물체를 좀 더 엄밀하게 걸러낼 수 있다. 
![](/assets/img/posts/261cbb7d-7937-801b-b7e1-e847305d6d80.webp)



![](/assets/img/posts/261cbb7d-7937-8020-b874-cf98775963b2.webp)



- 아직 threshold, inlier rate 등의 파라미터 최적화가 되지 않아서 오히려 이상한 이미지의 priority가 상승하는 경우도 있음
![](/assets/img/posts/261cbb7d-7937-8080-95ee-f22ffe050ba4.webp)



![](/assets/img/posts/261cbb7d-7937-807d-b277-c18785593ef9.webp)





- 움직이는 물체까지 feature를 잡아버리기 때문에 똑같은 동적물체가 있는 이미지라면 retrieval된다.
  ![](/assets/img/posts/261cbb7d-7937-80ee-98fd-ff41d258a326.webp)




### 다음주까지 개선해야 할 사항



- depth prior를 이용한 2d - 3d correspondence를 이용한 geometrycal verification
  - rosbag상에 있는 depth 이미지를 수집하여 map 상에 있는 depth prior와 비교하여 더욱 면밀한 geometrical 검증을 진행한다.
- VGGT를 사용하는 것은 오버헤드가 크므로 wildgs slam의 3d gaussian map을 로딩한 후 diff-gaussian-rasterization 를 활용하여 우리가 map상에서 구한 (가벼운) 초기 포즈를 ssim과 같은 loss로 $R,T$ 를 최적화 하는 과정을 진행한다. 
  - [https://github.com/rmurai0610/diff-gaussian-rasterization-w-pose/tree/43e21bff91cd24986ee3dd52fe0bb06952e50ec7](https://github.com/rmurai0610/diff-gaussian-rasterization-w-pose/tree/43e21bff91cd24986ee3dd52fe0bb06952e50ec7)
- 이미지 retrieval 성능을 통계적으로 평가 
  - rosbag상에 있는 pose를 map 포즈로 맞춘 후 비교하면 모든 쿼리이미지로 성능을 평가할 수 잇을 것이다.
























