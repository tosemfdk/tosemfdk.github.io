---
layout: post
title: "9월 13일 - vggt제거, gaussian based refine, 2d pose 비교"
date: 2025-09-14T02:12:00.000Z
math: true
notion_source_id: "26ecbb7d-7937-80ef-818b-d9d350c88519"
image:
  path: "/assets/img/posts/26ecbb7d-7937-80fb-afa3-f145c0269c2e.webp"
categories:
  - "서울대학교 여름방학 인턴"
---



# VGGT 제거 & gaussian map based refine

> **VGGT로 init pose를 만드는 것은 카메라 pose의 성능을 높이는데 매우 도움**이 될 수 있지만 매우 **무겁고** 일반화용으로 **학습된 모델에 의존**하게 될 수 있다. 반면 g**aussian map 을 base로 카메라 pose R,t를 refine**하게 되면 **현재 맵에 딱 맞는 미세 정합으로 더 가볍고 일관된 로컬 정확도**를 얻을 수 있을 것이라고 기대할 수 있다.



- calibrated camera이기 때문에 $x_2Ex_1​=0$ 이고 쿼리 프레임으로부터 ref 프레임까지의 상대변환 $R,t$ 를 구할 수 있게 된다. 또한 ref image들의 world scale로 맞춰주는 과정을 진행한다.
  - 이를 모든 ref 이미지에 대해서 반복 하고 top1의 retrieval image에서의 rotation값을 차용하고 모든 ref 이미지의 translation의 median값을 차용해서 최종 initail query pose를 구성하게 된다.
# gt pose compare

![](/assets/img/posts/26ecbb7d-7937-80fb-afa3-f145c0269c2e.webp)

- 여태까지 3차원이라고 전제하고 계산했던 gt pose vs query pose 비교가 알고보니 2d - 3d compare였다. x,y,yaw에 해당하는 값 밖에 없어서 결국 3d pose를 Z+축으로 정사영시키고 distance, yaw를 비교해야 했다.


## 실내

![](/assets/img/posts/26ecbb7d-7937-8008-8bc6-fc88e76ec78d.webp)

### best retrieval

![](/assets/img/posts/26ecbb7d-7937-802a-99f7-c238e594928d.webp)

### worst retrieval

![](/assets/img/posts/26ecbb7d-7937-800c-ab34-cd9960d67906.webp)



### Rotation error

![](/assets/img/posts/26ecbb7d-7937-8023-944e-eb24d5fcd8eb.webp)

![](/assets/img/posts/26ecbb7d-7937-80de-9ee1-dc5c4d9e9300.webp)

### translation error

![](/assets/img/posts/26ecbb7d-7937-80cc-9b8c-e6170be77d39.webp)

![](/assets/img/posts/26ecbb7d-7937-8063-9b8e-ec552114cdde.webp)



## 실외

![](/assets/img/posts/26ecbb7d-7937-80a6-aa25-de46524d69dc.webp)

### best retrieval

![](/assets/img/posts/26ecbb7d-7937-80c0-9f49-dd2915a98477.webp)

### Worst retrieval

![](/assets/img/posts/26ecbb7d-7937-802b-9d43-f062a18086f3.webp)



### Rotation error

![](/assets/img/posts/26ecbb7d-7937-8004-bfc6-c9912adbb733.webp)

![](/assets/img/posts/26ecbb7d-7937-80de-bf9c-cad3dc2445b1.webp)

### Translation error

![](/assets/img/posts/26ecbb7d-7937-80dc-b41b-d671e33d0278.webp)

![](/assets/img/posts/26ecbb7d-7937-800f-8bbe-e64b02f3e4a3.webp)





