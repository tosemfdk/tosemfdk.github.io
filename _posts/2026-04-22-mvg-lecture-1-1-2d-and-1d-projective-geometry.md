---
layout: post
title: "[MVG] Lecture 1-1: 2D and 1D projective geometry"
date: 2026-04-22T08:28:00.000Z
math: true
image:
  path: "/assets/img/posts/568cbb7d-7937-8286-99d8-81e44317c20d.webp"
categories:
  - "study"
---



## Introductions 

### **projective transformation이란**

![](/assets/img/posts/568cbb7d-7937-8286-99d8-81e44317c20d.webp)

- 평행한 line도 projective transformation 이후에는 평행하지 않게 된다.
- 직사각형도 더이상 직사각형이 아니게 된다.
- circle도 마찬가지이다. 
- projective transformation 이후에 보존되지 않는 것을 보았다.
  - 각도, 거리, 거리의 비 모두 보존되지 않음


### **projective geometry 이란**

![](/assets/img/posts/f4fcbb7d-7937-8258-bc18-811fb68285b9.webp)

- straightness: projective geometry에서도 보존되는 특성
  - 예) **line은 projective transformation 이후에도 line이다.**
  - “projective transformation에 대해 invariant하다”
- 이것을 통해 projective transformation 을 “직선을 보존시키면서 변형하는 어떠한 mapping”이라고 정의할 수 있음.


## Euclidean V.S. Projective

![](/assets/img/posts/02ccbb7d-7937-82c8-8a7f-818178ad3e12.webp)



### Cartesian coordinates and Homogeneous coordinates

![](/assets/img/posts/505cbb7d-7937-8319-9a90-81af2f60ce32.webp)

- projective geometry에서는 euclidean geometry에선 exceptional case인 “**infinite**” 를 다룰 수 있다.
- `euclidean space` → `cartesian coordinates` ($\in \mathbb R$)로 표현
- `projective space` ($\mathbb{P}^2$) → `homogeneous coordinates` 로 표현
  - $(kx, ky, k) = k(x, y, 1)$ → **모든** $k$**에 대해서 equivalent**하다.
- Homogeneous coordinates → Cartesian coordinates
  1. $k$를 나눠 $(x, y, 1)$로 만들고, 
  1. 맨 뒤 좌표를 없애면,
  projective space에서 euclidean space로 만들 수 있다.


### **point at infinity**

- 3개의 숫자로 표현하게 되면서, 마지막 숫자 $k=0$ 을 통해 “**point at infinity**”를 표현할 수 있다. 
- $(x, y)$를 $(kx, ky, k)$로 만듦으로써 Euclidean space를 Projective space로 확장할 수 있다.
![](/assets/img/posts/688cbb7d-7937-8349-a3d0-814a19ae1a57.webp)

- 위 “모든 k에 대해서 equivalent하다” 라는 문장을 시각화하여 설명하고 있다.
- point at infinity의 위치도 위 그림에서 확인할 수 있다.
- $(0, 0, 0)$**: 이건 homogeneous coordinate에서도 정의되지 않는다.**


## The 2D Projective Plane

![](/assets/img/posts/aadcbb7d-7937-83a2-bc51-812779ae0b89.webp)

- homogeneous coordinate에서 point와 line의 표현은 **interchangeable (dual)**
- projective space에서의 point: ray로 표현
- ~projective space에서의 line: plane으로 표현~
  - ~따라서 line은 이 plane의 normal vector로 표현할 수 있다.~


## Lines and Points

![](/assets/img/posts/932cbb7d-7937-8333-9d1b-01dfb66ea807.webp)

- In Cartesian coordinates
  - $ax+by+c = 0$:  이런 다항식으로 **line**을 표현할 수 있음.
  - cartesian coordinate 에서 x-y plane 상의 line을 표현
![](/assets/img/posts/c0ecbb7d-7937-8312-ac32-817bd254683e.webp)

- $ax+by+c = 0$ $\iff$ $(ka)x + (kb)y + (kc) = 0$
  - 벡터 $(a, b, c)$와 line간에는 **one-to-one mapping이 아니다. 하나의 line을 나타내는 벡터는 1개가 아니다!**
- 즉 $(a, b, c)$와 $k(a, b, c)$ 는 non-zero k에 대해 **equivalent class**이다. 
- $(0, 0, 0)$은 어떠한 correspond line이 없다.






## Incidence relations

![](/assets/img/posts/250cbb7d-7937-838b-a945-8198c7a33b87.webp)

- $ax+by+c=0$
  - $(x,y)$는 계수$(a, b, c)$가 표현하는 line 상에 위치할 때 성립한다.
- 위 다항식을 vector들의 내적으로 다시 쓸 수 있다.
  - $(x, y, 1)$: cartesian coordinate에서의 $(x, y)$를 homogeneous coordinate로 표현한 좌표


### **Point** $x$ **lies on the line** $l$

![](/assets/img/posts/30bcbb7d-7937-8347-8ccc-81d9e67f46ac.webp)

- $x^Tl = 0$ (ax+by+c=0에서 기인한 것을 위에서 봤었음)
- 이렇게 point와 line간의 incidence 관계를 내적을 이용해 표현할 수 있다.
- 내적 → 교환법칙이 성립 (이건 duality relationship에서 다룰 때 더 자세히 볼 것)
- `projective space에서의 DoF: independent ratio의 개수`
  - point와 line 모두 2dof를 갖는다.
  - {a : b : c} → 두개의 독립적인 비를 찾을 수 있음


### **Intersection of lines**

![](/assets/img/posts/5a4cbb7d-7937-828a-9e4c-01fc97654d50.webp)

- 모든 line들간의 intersection은 하나의 점으로 표현할 수 있음.
  - 평행한 line들도 point at infinity에서 만나기 때문 (in homogenous coordinates)
- intersection point x: 두 line의 cross product로 구할 수 있음
  - cross product를 수행 → 하나의 벡터가 나오는데, homogeneous coordinate에서는 point가 ray와 같기 때문에 의미가 맞다.
- proof
  - cross product하면 각 벡터에 수직인 벡터가 나옴
  - 따라서 line 각각과 cross product로 나온 벡터를 내적하면 0이 나옴
  - 이것은 line 상에 위치한 point와 해당 line간의 관계를 표현하는 식임
  - 따라서 두 line의 외적은 point를 나타냄


### **Line joining points**

![](/assets/img/posts/063cbb7d-7937-837c-8a76-812ebe4d4915.webp)

- 두 point간의 외적은 line을 의미함
  - 두 point는 각각 원점에서 출발하는 ray로 볼 수 있음.
  - 그 두 ray의 외적은 두 ray에 수직인 벡터를 의미함
  - 이 벡터는 어떤 평면에 normal vector로 볼 수 있음
  - 이 normal vector를 가지는 평면은 homogeneous coordinate 상에서 line을 의미함
  - 따라서 두 point간의 외적은 어느 하나의 line을 나타낼 수 있음 .




## Ideal points and line at infinity

![[https://pointatinfinityblog.wordpress.com/2016/04/11/points-at-infinity-i-projective-geometry/](https://pointatinfinityblog.wordpress.com/2016/04/11/points-at-infinity-i-projective-geometry/)](/assets/img/posts/e87cbb7d-7937-823d-afef-018575beac66.webp)

![](/assets/img/posts/5f6cbb7d-7937-823d-a624-8106102f03f9.webp)

- $l=(a, b, c)$라면 평행한 다른 $l^\prime$은 $l^\prime=(a, b, c^\prime)$으로 표현할 수 있음.
  - $a$와 $b$는 기울기를 의미 → 둘이 같아야 함
  - 따라서 $c$와 $c^\prime$만 다른 형태
- $l\times l^{\prime}$:
  - line간의 intersection 을 구하는 식임 (평행한 직선들의 교점을 구하기 위해)
  - 앞의 $(c^\prime - c)$는 $(b, -a, 0)$ 벡터에 대한 scale factor로 볼 수 있음
  - 하지만 $(b, -a, 0)$은 scale 에 영향을 받지 않음 (세번째 숫자가 0이므로)
  - 세번째 숫자가 0인 것을 point at infinity라고 배웠음
  - 따라서 두 평행한 line은 point at infinity에서 만난다.


![[https://www.mauriciopoppe.com/notes/mathematics/geometry/projective-space/](https://www.mauriciopoppe.com/notes/mathematics/geometry/projective-space/)](/assets/img/posts/7f6cbb7d-7937-8264-880d-01621b65d27f.webp)

![](/assets/img/posts/2c6cbb7d-7937-832f-b023-81054320bca4.webp)

- point at infinity는 어느 line at infinity 상에 위치할 것.
  - 두 평행한 line은 point at infinity에서 만난다.
  - 즉 두 평행한 line은 line at infinity와 point at infinity에서 만난다.
- point at infinity(ideal point)인 $(b, -a, 0)$은 방향 벡터로 생각해볼 수 있는데,
  - 두 평행한 직선들의 방향이다.
  - 두 직선들의 normal 방향이다.
  - 즉 $(b,-a)$가 특정되면, 해당 방향을 가지는 모든 평행한 선들이 특정된다는 뜻이고, 이는 곧 그 선들이 $(b,-a)$에서 모두 만난다는 의미를 가진다.
  - 이런 의미에서, $(b,-a)$는 line at infinity 위에 존재하고, 이 line at infinity는 두 평행한 선들의 교차점이 되므로 line at infinity를 2D projective space에서의 line들의 **방향들의 집합**으로 생각할 수 있음.
> **remark**

- line과 line간의 교차점이 하나의 point가 된다는 것과, 두개의 point가 line 상에 존재한다는 내용을 서술할 때
- Euclidean space에서는 위에 대해 “평행한 선”들끼리에 대해서는 예외를 둘 수밖에 없다.
- 즉 projective space에서는 이러한 내용을 더욱 간단하게 설명할 수 있게 된다. (평행한 선들을 따로 예외로 처리하지 않고 이들마저 모두 line at infinity 상에 위치한 ideal point에서 만난다는 것으로 다른 경우와 동일하게 처리할 수 있기 때문)
- 하지만 책에선 이 ideal point와 line at infinity를 특별한 것으로 취급할 것이라 함


## Duality principle

![](/assets/img/posts/b08cbb7d-7937-826d-b675-812c7cbe5e2a.webp)

- point와 line은 서로 dual이다.
  - homogeneous coordinate의 표현 상에서 dual
  - projective space 상에서의 연산에 대해서 dual
- line과 point가 interchangeable 하다는 특성으로 duality principle을 이끌어냄.


## 참고자료

- [https://www.youtube.com/watch?v=LAHQ_qIzNGU&list=PLxg0CGqViygP47ERvqHw_v7FVnUovJeaz&index=1](https://www.youtube.com/watch?v=LAHQ_qIzNGU&list=PLxg0CGqViygP47ERvqHw_v7FVnUovJeaz&index=1)


