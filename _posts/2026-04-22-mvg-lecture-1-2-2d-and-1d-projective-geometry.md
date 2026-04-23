---
layout: post
title: "[MVG] Lecture 1-2: 2D and 1D projective geometry"
date: 2026-04-22T08:28:00.000Z
math: true
image:
  path: "/assets/img/posts/bb2cbb7d-7937-82aa-b1a1-81c37df83c76.webp"
categories:
  - "study"
---



# Conics

<div class="notion-callout" style="display:flex;gap:0.75rem;align-items:flex-start;padding:1rem 1.1rem;margin:1.25rem 0;border-radius:14px;background:rgba(127, 127, 127, 0.12);border:1px solid rgba(127, 127, 127, 0.28);" markdown="1">
<div class="notion-callout__icon" style="font-size:1.25rem;line-height:1.4;flex:0 0 auto;">💡</div>
<div class="notion-callout__content" style="flex:1 1 auto;min-width:0;" markdown="1">

### Conics and Dual Conics in 2d projective space

![](/assets/img/posts/bb2cbb7d-7937-82aa-b1a1-81c37df83c76.webp)

- conic: plane에서 2차식으로 표현되는 curve들을 일컬음
- Euclidean space상에서의 conic
  - 3개의 main types: hyperbola(쌍곡선), ellipse(타원), parabola(포물선)


---

### Conic section

![](/assets/img/posts/4afcbb7d-7937-8226-92d0-8133248f080d.webp)

- 위와 같이 두개의 inverted cone과 plane간의 intersection을 이용해서 conic section(=set of conics)를 정의한다.


---

### Mathematical formulation of conics

![](/assets/img/posts/c55cbb7d-7937-830c-9ebb-81802ae1cf33.webp)

- cartesian coordinates: 첫번째 식과 같이 2차식으로 정의됨.
- homogeneous coordinates:
  - homogenizing을 통해 homogeneous coordinate에서의 식을 유도할 수 있음
  - $x = (x_1 / x_3)$, $y = (x_2 / x_3)$
  - $x_1$, $x_2$, $x_3$로 parameterized & $(a, b, c, d, e, f)$의 coefficient를 가지는 식으로 유도 가능
### Mathematical formulation of conics - matrix form

![](/assets/img/posts/819cbb7d-7937-8326-b1dc-01548ad6cfcd.webp)

![](/assets/img/posts/12acbb7d-7937-82ef-afe0-81220bda879b.webp)

- $\bold{x}^TC\bold{x} = 0$
- homogeneous representation으로는 ratio가 곧 DoF임
  - ${a:b:c:d:e:f}$ 에서의 독립적인 ratio는 총 5개.
  - 따라서 projective space 상에서 conic은 **5DoF**를 가짐

</div>
</div>





<div class="notion-callout" style="display:flex;gap:0.75rem;align-items:flex-start;padding:1rem 1.1rem;margin:1.25rem 0;border-radius:14px;background:rgba(127, 127, 127, 0.12);border:1px solid rgba(127, 127, 127, 0.28);" markdown="1">
<div class="notion-callout__icon" style="font-size:1.25rem;line-height:1.4;flex:0 0 auto;">💡</div>
<div class="notion-callout__content" style="flex:1 1 auto;min-width:0;" markdown="1">

## Points on the conics

![](/assets/img/posts/5f4cbb7d-7937-8384-98f8-0114181be782.webp)

- conic 위의 한 점 $x_i$는 하나의 constraint를 가짐
  - 위 2차식을 만족해야 한다는 constraint
- 두번째 식과 같이 첫번째 식을 다시 쓸 수 있음
  - polynomial을 계수와 변수의 dot product로의 표현으로 바꿈
  - conic C를 위와 같이 6-dim vector로 표현할 수 있다.


![](/assets/img/posts/59ccbb7d-7937-82df-8ec2-01074f8d8016.webp)

- c는 5DoF이므로, 5개의 constraints(5개의 conic 상의 점들)만 있으면 conic의 coefficient를 특정할 수 있게 된다.
  - 즉 5개의 점들로 conic을 특정할 수 있다는 말.
  - 이것을 5개의 점들에 대한 constraints들로 이루어진 5x6 matrix의 null vector라고 말하고 있다.

</div>
</div>





<div class="notion-callout" style="display:flex;gap:0.75rem;align-items:flex-start;padding:1rem 1.1rem;margin:1.25rem 0;border-radius:14px;background:rgba(127, 127, 127, 0.12);border:1px solid rgba(127, 127, 127, 0.28);" markdown="1">
<div class="notion-callout__icon" style="font-size:1.25rem;line-height:1.4;flex:0 0 auto;">💡</div>
<div class="notion-callout__content" style="flex:1 1 auto;min-width:0;" markdown="1">

### Dual Conics

### Dual conics - Tangent lines

![](/assets/img/posts/782cbb7d-7937-828f-8b60-017f70cce3d1.webp)

- conic에 접하는 case를 살펴볼 것
- degenerate case로 두 점에서 만나거나 아예 만나지 않은 case는 추후에 살펴볼 것
- $l = Cx$
  - $l^Tx$: line과 그 위의 점간의 관계식임을 위에서 살펴봤었음
  - $x^TCx$: conic equation


### Dual conics - point conics and dual conics

![](/assets/img/posts/c9ccbb7d-7937-826b-b173-81b7e5e6d2af.webp)

- point conic: 5개의 point로 conic을 결정할 수 있다고 했음. 이렇게 point로 정의하는 conic을 point conic이라고 함
- dual(line) conic: line과 C*로 정의됨
  - C*: dual conics
  - $l^TC*l$: line lies on the conic 에 대한 식
- dual conic은 마찬가지로 5DoF를 가짐

</div>
</div>



<div class="notion-callout" style="display:flex;gap:0.75rem;align-items:flex-start;padding:1rem 1.1rem;margin:1.25rem 0;border-radius:14px;background:rgba(127, 127, 127, 0.12);border:1px solid rgba(127, 127, 127, 0.28);" markdown="1">
<div class="notion-callout__icon" style="font-size:1.25rem;line-height:1.4;flex:0 0 auto;">💡</div>
<div class="notion-callout__content" style="flex:1 1 auto;min-width:0;" markdown="1">

## Dual conic and the inverse of the conic

**For non-singular case**

![](/assets/img/posts/d78cbb7d-7937-83a7-9297-818825d588f8.webp)

- non-singular symmetric matrix C*의 경우 $C^*=C^{-1}$ 이 성립
  - 즉 degenerate case가 아니라면(line이 tangent line이라면) 위 식이 성립한다.
- proof)
  - $l = Cx$ 와 dual conic을 이용해서 유도할 수 있음
- 맨 아래 식 $(C^{-1})^{T} = C^{-1}$ 은 Conic이 symmetric matrix여서 성립하는 것
  - 대칭행렬의 특성으로 역행렬또한 대칭행렬이다.
- 따라서 $l^TC^{-1}l = 0$ 이라는 식을 유도할 수 있게 된다. → $C^{-1} = C^*$


### Geometrically

![](/assets/img/posts/019cbb7d-7937-8340-abe3-814c4e57ab3f.webp)

</div>
</div>



<div class="notion-callout" style="display:flex;gap:0.75rem;align-items:flex-start;padding:1rem 1.1rem;margin:1.25rem 0;border-radius:14px;background:rgba(127, 127, 127, 0.12);border:1px solid rgba(127, 127, 127, 0.28);" markdown="1">
<div class="notion-callout__icon" style="font-size:1.25rem;line-height:1.4;flex:0 0 auto;">💡</div>
<div class="notion-callout__content" style="flex:1 1 auto;min-width:0;" markdown="1">

### Degenerate conics

![](/assets/img/posts/a8bcbb7d-7937-825d-8dd8-012df77e353d.webp)

![](/assets/img/posts/cdacbb7d-7937-8210-a303-81633c05f272.webp)

- tangent line $l$ 이 만약 $x$ 뿐만 아니라 conic 위의 점 $y$에서도 만난다면?
  - $y^TCy = 0$: conic 상의 점 $y$이므로 이 식이 성립
  - $l^Ty = 0$: $l$ 위의 점 $y$이므로 이 식이 성립
  - $l^Ty = 0$에 $l = Cx$를 대입 → $x^TCy = 0$이 성립
- $x + \alpha y$: line 그 자체를 의미
  - $x,y$와 만나는 tangent line $l$ 상의 임의의 point들을 의미
  - 즉 tangent line이 $x$외의 다른 conic 상의 한점에서 만난다면, 그것은 두 점에서 만나는 것이 아니라 tangent line 상의 모든 점에서 만난다는 뜻
- 이 degenerate conic의 경우, conic은 더이상 ellipse나 parabola, 등의 모양으로 생각할 수 없다.
### Line conics (rank-2)

![](/assets/img/posts/59ccbb7d-7937-8344-88a9-81143c865bb7.webp)

- not of full rank == singular case → degenerate case
  - C: 3x3 matrix이므로, full rank는 3일 것.
  - 즉 rank가 2 이하라면 degenerate case임
- degenerate conics 종류:
  - two line(rank 2)
  - repeated line(rank 1)
- **Line conics (rank-2) 정의**
  - line l과 m으로 3x3 matrix를 구성했음
  - $lm^T$: outer product
  - $ml^T$: conic C가 symmetric matrix가 되도록 만들어줌
  - 두개의 line$(l, m)$으로 이루어진 conic 식을 그대로 conic equation에 넣어보면 위와 같이 두개의 항으로 나눠진다.
  - 위 example의 식이 두 선으로 만들어진 점 코닉의 정의임. 유도 과정이 아니라 그냥 식을 외우면 됨.


### Repeated line일 때

![](/assets/img/posts/76fcbb7d-7937-8212-82c0-81b2340c3fb6.webp)

- 위 식의 $m$을 $l$로 바꾸어 repeated line을 적용해본다.


### Line conic C*

![](/assets/img/posts/07bcbb7d-7937-82b1-bd86-8158bbb2b064.webp)

- $C^* = xy^T + yx^T$
  - 위 식은 3x3 symmetric matrix를 만들기 위한 일반적인 식임.
- C가 full rank라면 $C^* = C^{-1}$ 임
  - 따라서 현재 degenerate case의 경우에는 이것이 성립하지 않음.

</div>
</div>





<div class="notion-callout" style="display:flex;gap:0.75rem;align-items:flex-start;padding:1rem 1.1rem;margin:1.25rem 0;border-radius:14px;background:rgba(127, 127, 127, 0.12);border:1px solid rgba(127, 127, 127, 0.28);" markdown="1">
<div class="notion-callout__icon" style="font-size:1.25rem;line-height:1.4;flex:0 0 auto;">💡</div>
<div class="notion-callout__content" style="flex:1 1 auto;min-width:0;" markdown="1">

### Planar projective transformation (H, Homography)

![](/assets/img/posts/449cbb7d-7937-8235-8de5-815e98781fc8.webp)

- projectivity:
  - invertible mapping이다.
  - $P^2$ ⇒ $P^2$
  - $x = hx’$ ⇒ $x’ = h^{-1}x$
- projectivity $\iff$ collineation $\iff$ projective transformation $\iff$ homography
- 3x3 non-singular matrix H로 표현할 수 있음
- collinearity에 대한 증명
  - $l^Tx_i = 0$ ⇒ 중간에 Identity를 의미하는 $H^{-1}H$를 넣어도 식은 유지된다.
  - 따라서 이것을 point에 대한 것과 line에 대한 식으로 나눠보면 $H$로 매핑된 $x$는 여전히 $H^{-T}$로 매핑된 line $l$ 위에 있게 된다.
  - 이것은 line $l$ 상에 있던 모든 $x_i$ ($\forall i = 1, 2, 3$)에 대해 모두 성립하기 때문에, 이들은 변형 후에도 같은 line 위에 있게 된다.
  - 이것이 collinearity 성질이다.
<div class="notion-callout" style="display:flex;gap:0.75rem;align-items:flex-start;padding:1rem 1.1rem;margin:1.25rem 0;border-radius:14px;background:rgba(127, 127, 127, 0.12);border:1px solid rgba(127, 127, 127, 0.28);" markdown="1">
<div class="notion-callout__icon" style="font-size:1.25rem;line-height:1.4;flex:0 0 auto;">💡</div>
<div class="notion-callout__content" style="flex:1 1 auto;min-width:0;" markdown="1">

duality로 다시 생각해보면,

- point를 $H$로 변환하는 것은,
- line을 $H^{-T}$로 변환하는 것으로 생각할 수 있다.
- 2D-GS 라는 논문에서 이 duality를 이용한 식이 나옴


![2D Gaussian Splatting for Geometrically Accurate Radiance Fields, Huang et al. SIGGRAPH24](/assets/img/posts/7e4cbb7d-7937-8340-bbbb-01f905d8aa23.webp)

- $(WH)^{-1}$ 대신 $(WH)^T$ 를 line에 대해 적용하는 것을 볼 수 있음
- 위와 같이 matrix의 inverse는 수치 상 불안정할 수 있으므로 point의 변환을 line의 변환으로 생각하여 이를 해결한다.

</div>
</div>

</div>
</div>





<div class="notion-callout" style="display:flex;gap:0.75rem;align-items:flex-start;padding:1rem 1.1rem;margin:1.25rem 0;border-radius:14px;background:rgba(127, 127, 127, 0.12);border:1px solid rgba(127, 127, 127, 0.28);" markdown="1">
<div class="notion-callout__icon" style="font-size:1.25rem;line-height:1.4;flex:0 0 auto;">💡</div>
<div class="notion-callout__content" style="flex:1 1 auto;min-width:0;" markdown="1">

## Properties of H

![](/assets/img/posts/506cbb7d-7937-82f1-9998-01b2199d963d.webp)

- non-singular 3x3 matrix
  - invertible임
- homogeneous matrix
  - independent ratio의 개수가 중요 ⇒ 8DoF를 갖게 됨


---

### Homography의 의미

![](/assets/img/posts/4b4cbb7d-7937-83f5-80c1-016fbbdc2703.webp)

- origin으로부터 출발하는 어느 ray가 있고, 그것에 intersect되는 plane이 두개가 있을 때, 그 두개의 intersection point간의 관계를 알고 싶을 때
  - 이것을 H라는 관계로 생각해볼 수 있다.
  - $x^\prime = Hx$
---

### Examples

![](/assets/img/posts/4b8cbb7d-7937-8336-a4aa-0114056e606a.webp)

- 첫번째 그림
  - 어느 공간의 planar 상의 점 3개와 그 점 3개를 각 image1과 image2에 projection 시킨 그림
  - 동일한 점에서 projection 된 것이기 때문에 image1의 점에서 image2의 대응되는 점으로 변환하는 Homography를 구할 수 있음
- 두번째 그림
  - 마찬가지로 X라는 공간상의 동일한 점으로 image1과 image2에 projection 하고 있음
- 세번째 그림도 마찬가지


※ 여기서는 이 homography를 line과 conic에 적용하는 것을 살펴볼 것.  (이후(lecture 4)에는 point로 homography를 구하는 방법을 살펴볼 것)

</div>
</div>





<div class="notion-callout" style="display:flex;gap:0.75rem;align-items:flex-start;padding:1rem 1.1rem;margin:1.25rem 0;border-radius:14px;background:rgba(127, 127, 127, 0.12);border:1px solid rgba(127, 127, 127, 0.28);" markdown="1">
<div class="notion-callout__icon" style="font-size:1.25rem;line-height:1.4;flex:0 0 auto;">💡</div>
<div class="notion-callout__content" style="flex:1 1 auto;min-width:0;" markdown="1">

## Transformation on the lines and conics

![](/assets/img/posts/a33cbb7d-7937-832f-ac51-81a8b0b7aa1e.webp)

![](/assets/img/posts/89fcbb7d-7937-83dc-a341-81e3ebb05e44.webp)

- $x^\prime=Hx$ 에서의 $H$를 conic에 적용해보면
  - $C^{\prime} = H^{-T}CH^{-1}$ 가 된다.
  - dual conic은 위 식에서 전체에 inverse를 취한 꼴 (C* = C^-1임을 상기하자)
- proof
  - $x$대신 homography로 변환된 $x^\prime$에 대한 식으로 바꿔보면 위와 같은 변환된 conic 식을 유도할 수 있다.

</div>
</div>







### Summary of hierarchy of transformations

![](/assets/img/posts/220cbb7d-7937-839b-8d1f-011dbba91a17.webp)

![](/assets/img/posts/f79cbb7d-7937-836e-b20d-81774f6080df.webp)

- $A = sRK + tv^T$
  - $s$: scale matrix
  - $R$: rotation matrix
  - $K$: upper triangular matrix
  - $t$: translation vector
  - $v$: 3by1 vector


### projective geometry of 1D ($\mathbb{P}^1$)

![](/assets/img/posts/554cbb7d-7937-835d-ad88-019ed30f9f7d.webp)

- 여전이 line(ray)위의 수많은 점들을 생각해볼 수 있는데,
  - 2D projective space에서는 intersection을 생각해볼 때 plane을 다뤘음
  - 1D projective space에서는 line임
- 2x2 matrix 
  - homogeneous coord에선 3DoF임


### Cross ratio

![](/assets/img/posts/ce9cbb7d-7937-827d-98dc-01224ccedd83.webp)

- 2D에서는 직선이라는 성질은 보존되었음
- 1D에서는 projective transformation을 수행하면 cross ratio가 보존된다.
- cross ratio: 1D projective space에서의 기본적인 invariant임
  - 정의: line 위에 4개의 점이 있다고 할 때,
  - 각 점들로 이루어진 2x2 matrix의 determinant를 이용하여 계산된다.
  - **네 점의 상대적 위치 관계를 하나의 값으로 나타낸다.**


![](/assets/img/posts/8d1cbb7d-7937-8200-b650-813c5bda742e.webp)

- ray가 있고, line이 있음
  - 2D 였으면 ray와 plane이었음


### concurrent lines

- 1D projective space에도 point와 line간의 dual이 있음
- line에 대해 collinear 관계에 있는 points들에 대한 dual을 의미
- 정의:
  - 공간 상에서 같은 점을 지나는 직선들을 의미한다.
- 부가설명
  - 어느 plane에서 line들($l_1, l_2, l_3, l_4$)이 어느 하나의 점(위 그림에선 c)에서 만난다(concurrent)는 것은,
  - 그것의 dual plane에서는 하나의 다른 line과 위 line들($l_1, l_2, l_3, l_4$)이 교차하는 점들이 그 다른 하나의 line 위에 있게 된다는 것을 나타낸다.
  - 이것이 concurrent line과 collinear point간의 dual 관계이다.


- coplanar points:
  - 어느 2D projective space 상의 plane위에 있는 $x_i$들을 생각해보자
  - 이 $x_i$들에 $C$를 origin으로 하는 projection을 수행함으로써 line에 각 점들이 맺힐 때,
  - 이 각 점들은 cross ratio를 보존하게 된다?








## 참고자료

- [https://www.youtube.com/watch?v=gQ7IUS8NKCI&list=PLxg0CGqViygP47ERvqHw_v7FVnUovJeaz&index=2](https://www.youtube.com/watch?v=gQ7IUS8NKCI&list=PLxg0CGqViygP47ERvqHw_v7FVnUovJeaz&index=2)


