---
layout: post
title: "[MVG] Lecture 2-2: Rigid body motion and 3D projective geometry"
date: 2026-04-22T08:28:00.000Z
math: true
image:
  path: "/assets/img/posts/40bcbb7d-7937-8208-a0a9-01576bf90c54.webp"
categories:
  - "study"
---

# Projective Geometry and Transformations of 3D

3D space, 즉 $\mathbb{P^2}$에서의 homogeneous coordinate는 $\mathbb{P^2}$에서의 homogeneous coordinate의 확장으로 볼 수 있다. 따라서 많은 특성들 역시 $\mathbb{P^2}$의 확장으로 볼 수 있다. 하지만, Dimension의 추가로 인해 다음과 같은 특성을 가지게 된다.

<div class="notion-callout" style="display:flex;gap:0.75rem;align-items:flex-start;padding:1rem 1.1rem;margin:1.25rem 0;border-radius:14px;background:rgba(127, 127, 127, 0.12);border:1px solid rgba(127, 127, 127, 0.28);" markdown="1">
<div class="notion-callout__icon" style="font-size:1.25rem;line-height:1.4;flex:0 0 auto;">💡</div>
<div class="notion-callout__content" style="flex:1 1 auto;min-width:0;" markdown="1">

1. 두 직선은 projective plane에서 항상 만난다. 평행한 두 직선도 무한대 점에서 만난다. 하지만 3d space에서 만나지 않을수도 있다.

</div>
</div>

## Points in $\mathbb{P^3}$

3D space에서의 point X는 다음과 같이 표현된다.

$X = (X_1,X_2,X_3,X_4)^T$ 단, $X_4\neq0$

이를 inhomogeneous coordinate로 변환시 $(X,Y,Z)^T=(X_1/X_4,X_2/X_4,X_3/X_4)$와 같다.
이때 $X_4=0$인 경우는 points at infinity를 나타낸다. 이는 벡터로도 볼 수 있다.

## Projective Translation of Points in $\mathbb{P^3}$

Projective Translation은 non-singular한 4$\times$4 행렬 $H$에 의해 정의되는 선형 변환이다. 이때 점 $X$는 $X'$으로 변환된다. 수식으로는 다음과 같이 표현된다.

$$
X'=HX
$$

### $H$의 특징

1. $H$는 4$\times$4 homogeneous matrix이다
   1. 1개의 scaling factor를 포함해 15dof이다.
   1. non-singular이다. (일대일 대응이기 때문에 inverse가 존재하는 non-singular이어야 한다.)
1. Collineation (lines are mapped to lines)
   1. $\mathbb{P^2}$에서와 동일하게 직선을 다른 직선으로 mapping함. 즉 Projective Translation은 Collineation이다.
1. incidence relation이 보존됨.
   1. 직선과 평면의 교차점, collinearation, 평면 위의 점의 위치등이 유지
   <div class="notion-callout" style="display:flex;gap:0.75rem;align-items:flex-start;padding:1rem 1.1rem;margin:1.25rem 0;border-radius:14px;background:rgba(127, 127, 127, 0.12);border:1px solid rgba(127, 127, 127, 0.28);" markdown="1">
   <div class="notion-callout__icon" style="font-size:1.25rem;line-height:1.4;flex:0 0 auto;">💡</div>
   <div class="notion-callout__content" style="flex:1 1 auto;min-width:0;" markdown="1">
   
   원점에 대한 내용 찾아보기
   
   </div>
   </div>
## Planes in $\mathbb{P^3}$

$\mathbb{R^3}$에서의 평면은 다음과 같은 선형방정식으로 나타난다.

$$
\pi_1X+\pi_2Y+\pi_3Z+\pi_4=0
$$

하지만 homogeneous coordinate에서는 $X = \frac{X_1}{X_4}, Y = \frac{X_2}{X_4}, Z = \frac{X_3}{X_4}$이기 때문에 다음과 같이 표현할 수 있다.

$$
\pi_1 X_1 + \pi_2 X_2 + \pi_3 X_3 + \pi_4 X_4 = 0
$$

이는 평면 $\pi$ 위의 Point X와 평면 $\pi$의 내적으로도 표현할 수 있다.

$$
\pi^TX=0, \ \pi=\begin{bmatrix}\pi_1\\ \pi_2 \\ \pi_3 \\ \pi_4 \end{bmatrix}
$$

이때 평면에서 scaling factor는 유의미한 정보를 가지고 있지 않기 때문에 3dof를 가진다. 따라서 평면의 방정식은 3개의 비율로만 결정되기 때문에 다음과 같이 나타낼 수 있다.

$$
\pi_1:\pi_2:\pi_3:\pi_4
$$

평면의 앞의 3개의 계수는 법선 벡터(normal vector)를 나타낸다.

$$
n=(\pi_1,\pi_2,\pi_3)^T
$$

따라서 inhomogeneous coordinate에서 다음과 같이 나타낼 수 있다.

$$
\mathbf{n} \cdot \tilde{\mathbf{X}} + d = 0,\ \tilde{X} = (X,Y,Z),\ X_4 = 1,\ d = \pi_4
$$

또한, 원점으로부터 평면의 거리는 $\frac{d}{\|\mathbf{n}\|}$로, 원점에서부터 평면까지의 직교 거리(perpendicular Distance)를 의미한다.

---



점 $X'=HX$와 같이 나타낼 수 있기 때문에 plane transformation은 다음과 같이 나타낼 수 있다.

$$
\pi'=H^{-T}\pi
$$

이는 다음에 의해 유도될 수 있다.

Incidence Relation에 의해  $X'$ 역시 $\pi'$ 위에 존재해야 하기 때문에 다음을 만족하여야 한다

$$
\boldsymbol{\pi}^T \mathbf{X} = 0  \\  \boldsymbol{\pi'}^T H \mathbf{X} = 0 \\(\boldsymbol{\pi'}^T H) \mathbf{X} = 0 \\H^T \boldsymbol{\pi'} = \boldsymbol{\pi} \\ \boldsymbol{\pi'} = H^{-T} \boldsymbol{\pi}
$$

## Three Points Define a Plane

$X_i$가 평면 $\pi$ 위에 있는 점인 경우 $\pi^TX_i=0$을 만족한다. 따라서 다음 식 역시 만족한다.

$$
\begin{bmatrix}X_1^T\\X_2^T\\X_3^T\end{bmatrix}\pi=0
$$

이때 $[X_1, X_2,X_3]^T$는 일반적인 경우, 즉 한 직선 위에 있지 않은 경우 linearly independent하고 rank가 3이다. 따라서 1차원의 null space를 가진다. 만약 한 직선 위에 있는 경우, 2차원의 null space를 가진다. 이는 collinear points의 직선을 축으로 여러개의 치원들을 null space으로 가진다.

![](/assets/img/posts/40bcbb7d-7937-8208-a0a9-01576bf90c54.webp)

## Three planes Define a Point

세 평면이 만나는 경우도 위와 같이 rank와 null space의 차원을 이용해 설명 가능하다.

$$
\begin{bmatrix}\pi_1^T\\\pi_2^T\\\pi_3^T\end{bmatrix}X=0, \begin{bmatrix}\pi_1^T\\\pi_2^T\\\pi_3^T\end{bmatrix} = A
$$

이때 $\begin{bmatrix}\pi_1^T\\\pi_2^T\\\pi_3^T\end{bmatrix}$는 $3\times4$이고, $A$의 rank에 의해 교점이 결정된다. 이는 point-plane duality를 보여준다.

![](/assets/img/posts/24ccbb7d-7937-839f-bcab-0198c1c5b990.webp)

## Parametrized Points on a Plane

$\mathbb{P^3}$에서 평면 $\pi$위의 점 $X$는 다음과 같이 나타낼 수 있다.

$$
X=Mx
$$

이때 M은 4$\times$3 행렬이고, $x$는 3차원 벡터이며, 평면을 따라 변하는 매개변수 벡터이다. 또한 $M=[M_1,M_2,M_3]$이고, $X=M_1x_1+M_2x_2+M_3x_3$으로 나타낼 수 있다. 따라서 $X$가 형성하는 linear subspace는 $\pi$의 subspace이다. 

## M matrix와 Null-space

$X=Mx$이기 때문에 $\pi^TM=0$이다.  이는 $\pi_1M_{i1}+\pi_2M_{i2}+\pi_3M_{i3}+\pi_4M_{i4}=0$이 성립한다고 정리 할 수 있다. 따라서 4개의 변수에 대해 1개의 제약조건이 결정되었기 때문에 3dof이다. M은 rank 3이기 때문에, 각각의 $M_i$들은 평면의 basis가 되고, $x$는 parameterization variable이다.

<div class="notion-callout" style="display:flex;gap:0.75rem;align-items:flex-start;padding:1rem 1.1rem;margin:1.25rem 0;border-radius:14px;background:rgba(127, 127, 127, 0.12);border:1px solid rgba(127, 127, 127, 0.28);" markdown="1">
<div class="notion-callout__icon" style="font-size:1.25rem;line-height:1.4;flex:0 0 auto;">💡</div>
<div class="notion-callout__content" style="flex:1 1 auto;min-width:0;" markdown="1">

즉 $M_i$들은 모두 $\pi$의 basis (평면위의 점)가 되고, $x$는 coefficient가 된다.

</div>
</div>

$M$은 유일하지 않다. $\pi=(a,b,c,d)^T, a \neq0$일 경우, $M^T$는 다음과 같이 정리할 수 있다.

$$
M^T=[P|I_{3\times3}], where \quad P=(-\frac{b}{a}, -\frac{c}{a}, -\frac{d}{a})^T
$$

## Lines in $\mathbb{P^3}$

두 점을 연결하는 선 또는 두 평면의 교차선으로 나타낼 수 있다.

이때 직선의 자유도는 4이다. 이는 다음을 통해 증명 가능하다.

<div class="notion-callout" style="display:flex;gap:0.75rem;align-items:flex-start;padding:1rem 1.1rem;margin:1.25rem 0;border-radius:14px;background:rgba(127, 127, 127, 0.12);border:1px solid rgba(127, 127, 127, 0.28);" markdown="1">
<div class="notion-callout__icon" style="font-size:1.25rem;line-height:1.4;flex:0 0 auto;">💡</div>
<div class="notion-callout__content" style="flex:1 1 auto;min-width:0;" markdown="1">

1. 직선은 두개의 서로 수직한 평면의 교차점으로 볼 수 있다.
1. 각각의 평면은 2개의 자유도를 가진다.
1. 따라서 직선은 2+2의, 자유도를 가진다.

</div>
</div>

직선은 4자유도를 가지기 때문에 표현하기 위해서는 5dim vector로 표현되어야 한다. 하지만 이를 대체하기 위해 행렬 표현과  Plücker 좌표 두가지 표현을 사용한다.

## 직선의 행렬표현

3d space에서 두 점 A,B가 주어졌을때, 직선은 이를 연결하는 점들의 span으로 표현할 수 있다.

$$
X=\lambda A+\mu B
$$

따라서 직선은 $W^T$의 span으로 표현되고,  $W=\begin{bmatrix}A^T\\B^T\end{bmatrix}$이다. 즉 $W^T$의 column space가 직선이다. 또한, 두 평면 P와 Q에 대해서도  W의 null-space의 원소이다. 따라서 $W_P$ = 0, $W_Q$ = 0이다. 그러므로 $A^TW=0, \ B^TW = 0$이다. 이를 정리해보면 P와 Q는 두 점 A, B를 포함하는 평면이다. 그 이유는 내적값이 0이기 때문이다.
직선은 두 평면의 intersection으로 정의되기 때문에 직선은 두 평면을 동시에 만족하는 점들의 집합이다. 따라서 이 직선을 포함하는 모든 평면들은 P와 Q의 선형조합으로 생성될 수 있다.

$$
\lambda P+\mu Q
$$

이렇게 생성되는 평면들을 Pencil of planes라고 한다.

### dual representation of line

line을 두 평면의 intersection이라고 할 때 line은 $2\times4$ matrix $W^*$의 row space의 span으로 나타내진다.

$W^*=\begin{bmatrix}P^T\\Q^T\end{bmatrix}$

이때 아래와 같은 특징을 가진다.

1. $W^{*T}=\lambda P+ \mu Q$는 직선을 axis로 하는 pencil of planes을 의미한다.
1. $W^*$의 2 dim null-space는 pencil of points로 직선을 따라 존재하는 점들의 span이다. 즉 $W^*W^T=W^TW^*=0_{2\times2}$이다.
## Null-Space and Span Representation

평면은 직선과 한 점으로 표현될 수 있다. 이는 null-space와 span을 이용해 설명된다.

$$
M = \begin{bmatrix}W\\X^T\end{bmatrix}=\begin{bmatrix}A^T\\B^T\\X^T\end{bmatrix}
$$

이때 M의 null-space가 2차원인 경우 X는 W 위에 존재하고, 그렇지 않은 경우 $M\pi=0$이다. 즉 null-space가 1차원이다. 이는 평면이 결정되었음을 의미한다.

이때 M은 $W^*$를 통해서도 표현될 수 있다. 이때 X는 $W^*$와 $\pi$의 교점이다.

$$
M = \begin{bmatrix}W^*\\X^T\end{bmatrix}=\begin{bmatrix}P^T\\Q^T\\X^T\end{bmatrix}
$$

만일 M의 null-space가 2dim이면 X가 $W^*$위에 존재하는 것이다. 그렇지 않은 경우 $M\pi=0$을 만족한다.

## Plücker Line Coordinates

![](/assets/img/posts/d57cbb7d-7937-8282-9a00-01442e7c6013.webp)

Plücker Line Coordinate는 6개의 non zero element로 이루어짐. 직선은 두 점 A와 B로 정의될 수 있으며 이를 이용해 Plücker Line Coordinate를 구성할수 있다.

$$
\mathcal{L} = \{ l_{12}, l_{13}, l_{14}, l_{23}, l_{42}, l_{34} \}
$$

$\mathcal{L}$의 각 요소는 다음을 의미한다.

첫 번째 3개 요소 d: 직선의 방향 벡터(direction vector)

마지막 3개 요소 m: 직선의 모멘트 벡터(moment vector)

$d=B-A$

$m=A\times B$

Plücker Line Coordinate를 이용하면 두 직선의 교차 여부 역시 알 수 있다. 이는 두 직선을 정의하는 네 점이 한 평면 위에 존재(coplanar)할 경우 가능하다.

<div class="notion-callout" style="display:flex;gap:0.75rem;align-items:flex-start;padding:1rem 1.1rem;margin:1.25rem 0;border-radius:14px;background:rgba(127, 127, 127, 0.12);border:1px solid rgba(127, 127, 127, 0.28);" markdown="1">
<div class="notion-callout__icon" style="font-size:1.25rem;line-height:1.4;flex:0 0 auto;">💡</div>
<div class="notion-callout__content" style="flex:1 1 auto;min-width:0;" markdown="1">

두 직선을 $\mathcal{L}, \mathcal{\hat{L}}$이라 했을때

- 직선 $\mathcal{L}$ 은 점 **A,B** 를 지나는 직선이다.
- 직선 $\mathcal{\hat{L}}$ 은 점 $\hat{A},\hat{B}$ 를 지나는 직선이다.
이때 두 직선이 교차할 조건은 $det[A,B,\hat{A},\hat{B}]=0$이다. 이는 Plücker Line Coordinate를 이용하면 다음과 같이 나타낼 수 있다.

$$
l_{12} \hat{l}_{34} + \hat{l}_{12} l_{34} + l_{13} \hat{l}_{42} + \hat{l}_{13} l_{42} + l_{14} \hat{l}_{23} + \hat{l}_{14} l_{23} = (\mathcal{L} | \hat{\mathcal{L}}) = 0
$$

이를 증명하자면, $d$와 $\hat{m}$, $m$과 $\hat{d}$는 수직이어야 한다. 즉 다음이 성립한다.

$$
 d^T\hat{m}+m^T\hat{d}=0
$$

이는 $det[A,B,\hat{A},\hat{B}]$과 동일하다.

</div>
</div>

---

이는 duality에 의해 다음과 같이 쓰일 수 있다.

$(\mathcal{L} | \hat{\mathcal{L}}) = det[P,Q,\hat{P},\hat{Q}]=0$

<div class="notion-callout" style="display:flex;gap:0.75rem;align-items:flex-start;padding:1rem 1.1rem;margin:1.25rem 0;border-radius:14px;background:rgba(127, 127, 127, 0.12);border:1px solid rgba(127, 127, 127, 0.28);" markdown="1">
<div class="notion-callout__icon" style="font-size:1.25rem;line-height:1.4;flex:0 0 auto;">💡</div>
<div class="notion-callout__content" style="flex:1 1 auto;min-width:0;" markdown="1">

두 직선을 $\mathcal{L}, \mathcal{\hat{L}}$이라 했을때

- 직선 $\mathcal{L}$ 은 평면 $P,Q$ 의 교차선이다.
- 직선 $\mathcal{\hat{L}}$ 은 평면 $\hat{P},\hat{Q}$ 의 교차선이다.
이때 두 직선이 교차할 조건은 $det[P,Q,\hat{P},\hat{Q}]=0$이다. 이는 다음과 같이 나타낼 수 있다.

$$
(\mathcal{L} | \hat{\mathcal{L}}) = (P^T A)(Q^T B) - (Q^T A)(P^T B) = 0
$$

</div>
</div>

## Quadrics and Dual Quadric

Quadirc은 $\mathbb{P^3}$에 존재하는 곡면의 표면을 정의하는 식이다

$$
X^TQX=0
$$

- Q는 symmetric한 $4\times4$ matrix이다.
- $Q^T=Q$
- $det(Q)$에 의해 정해진다.
- symmetric하기 때문에 10dof지만 scale factor를 제거하면 9dof이다.
- Q가 singular하다면, quadric이 퇴화하여 몇개의 점으로 정의된다.
- quadric과 $\pi$의 접선은 Conic C이다.
![](/assets/img/posts/a2ccbb7d-7937-83f1-a2ea-013ecf38c5d9.webp)

$$
  \begin{align}X &= Mx \\    X^T Q X &= 0 \\    x^T M^T Q M x &= 0 \\    C &= M^T Q M \\    x^T C x &= 0\end{align}
$$

- $Q'=H^{-T}QH^{-1}$
  <div class="notion-callout" style="display:flex;gap:0.75rem;align-items:flex-start;padding:1rem 1.1rem;margin:1.25rem 0;border-radius:14px;background:rgba(127, 127, 127, 0.12);border:1px solid rgba(127, 127, 127, 0.28);" markdown="1">
  <div class="notion-callout__icon" style="font-size:1.25rem;line-height:1.4;flex:0 0 auto;">💡</div>
  <div class="notion-callout__content" style="flex:1 1 auto;min-width:0;" markdown="1">
  
  $$
  \begin{align}
  X'^TQ'X'=0 \tag{1}\\
  (X^TH^T)Q'(HX)=0 \tag{2}\\
  X^TQX=0\tag{3} \\
  Q=H^TQ'H \tag{4}\\
  Q'=H^{-T}QH^{-1}\tag{5}
  \end{align}
  $$
  
  </div>
  </div>
### Dual Quadrics

Point를 이용한 방법이 아닌 평면을 이용한 방법

$$
\pi^TQ^*\pi=0
$$

$Q^*=$adjoint Q 또는 $Q^{-1}$

$$
{Q^*}'=HQ^*H^T
$$

![](/assets/img/posts/ecccbb7d-7937-8375-978a-81485fc8cb44.webp)

![](/assets/img/posts/c4ecbb7d-7937-82a0-9b1a-01e26d28711c.webp)

![](/assets/img/posts/b63cbb7d-7937-8398-9e0f-818ad5b8ee04.webp)

## Hierarchy of Transformations

2D에서 3D로 확장할 때 사용

![](/assets/img/posts/0b6cbb7d-7937-82d1-a3ff-8130b0ae7919.webp)

### **Projective 변환 (15 dof)**

- **행렬 구조:** $\begin{bmatrix} A & t \\ v^T & v \end{bmatrix}$ 
- **특징:**
  - 가장 일반적인 변환으로, 직선이 직선으로 유지되는 모든 변환을 포함한다.
  - 원근 투영(perspective projection)을 설명할 수 있음.
  - **불변 속성:** 교차 및 접촉하는 표면의 관계 보존.
---

### **2. Affine 변환 (12 dof)**

- **행렬 구조:**  $\begin{bmatrix} A & t \\ 0^T & 1 \end{bmatrix}$ 
- **특징:**
  - 평행성(parallelism) 유지.
  - 선형 변환과 병진(translation)이 포함됨.
  - **불변 속성:** 평면의 평행성, 부피 비율(volume ratio), 무게 중심(centroid), 무한 평면(π_∞).
---

### **3. Similarity 변환 (7 dof)**

- **행렬 구조:** $\begin{bmatrix} sR & t \\ 0^T & 1 \end{bmatrix}$ 
- **특징:**
  - 크기(scale)를 유지한 채 회전 및 이동이 가능.
  - **불변 속성:** 절대 원뿔(absolute conic, Ω_∞).
---

### **4. Euclidean 변환 (6 dof)**

- **행렬 구조:** $\begin{bmatrix} R & t \\ 0^T & 1 \end{bmatrix}$
- **특징:**
  - 순수한 회전(rotation)과 병진(translation) 변환.
  - 부피(volume) 보존.
  - rigid body motion
