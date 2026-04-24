---
layout: post
title: "[MVG] Lecture 2-1: Rigid body motion and 3D projective geometry"
date: 2026-04-22T08:28:00.000Z
math: true
image:
  path: "/assets/img/posts/bc1cbb7d-7937-8269-b4a1-01ea7019a1a5.webp"
categories:
  - "study"
---

# Three-Dimensional Euclidean Space

$\mathbb{E^3}$을 3차원 유클리드 공간을 나타내는 기호로 사용한다.  이때 $\mathbb{E^3}$에 속하는 모든 점 p는 아래와 같이 3개의cartesian coordinate를 가지는 $\mathbb{R^3}$의 한 점으로 표현될 수 있음. 따라서 $\mathbb{E^3}$과 $\mathbb{R^3}$ 사이에 일대일 대응이 존재함을 확인할 수 있다.

$\mathbf{X} \doteq \begin{bmatrix} X_1, X_2, X_3 \end{bmatrix}^{T} = \begin{bmatrix} X_1 \\ X_2 \\ X_3 \end{bmatrix} \in \mathbb{R}^3$

> $\mathbb{E^3}$(Euclidean space)와 $\mathbb{R^3}$(cartesian coordinates) 사이에 일대일 대응이 존재

# Vector

![](/assets/img/posts/bc1cbb7d-7937-8269-b4a1-01ea7019a1a5.webp)

rigid body motion을 설명하기 위해 vector에 대한 설명 필요.
vector $\mathbb{v}$는 두 점 $p,q \in \mathbb{E^3}$에 의해 $p$에서 $q$까지 연결하는 방향이 있는 화살표로 정의됨.
이때 $p$가 $X$에 대응되고, $q$가 $Y$에 대응되는 경우 $v$는 아래와 같이 쓸 수 있다.
$v=Y-X \in \mathbb{R^3}$
다음과 같이 base point, 위의 상황에서는 $p,q$에 해당되는 기준점에 의해 정의되는 벡터를 Bound Vector 라고 한다.

![](/assets/img/posts/3cfcbb7d-7937-8233-ad6b-815bfe3d1981.webp)

이와 다르게 base point에 의존하지 않는 벡터를 Free Vector 라고 하며,  $Y-X = Y'-X'$을 만족하는 두개의 점 $(p,q)$와 $(p',q')$은 동일한 Free Vector를 정의한다.

![](/assets/img/posts/a11cbb7d-7937-82e6-8ad0-017beb0c7e88.webp)

base point가 origin인 경우, $X$ = 0, $Y=v$이다.
모든 free vector의 집합은 linear vector space를 형성하고, 이는 두개의 벡터 $v,u\in \mathbb{E^3}$의 선형 결합으로 정의된다. 즉 $\mathbb{R^3}$의 모든 point는 linear combination으로 나타낼 수 있고, $\mathbb{E^3}$ 자체가 선형 벡터 공간임을 나타낸다.
$\alpha v + \beta u = \begin{bmatrix} \alpha v_1 + \beta u_1, \alpha v_2 + \beta u_2, \alpha v_3 + \beta u_3 \end{bmatrix}^{T} \in \mathbb{R}^3, \quad \forall \alpha, \beta \in \mathbb{R}$

> base point에 의해 정의되는 벡터는 Bound Vector라고 한다.
base point에 의존하지 않고, $Y-X = Y'-X'$를 만족하는 벡터를 Free Vector라 한다.
Free Vector는 평행이동 된 경우 동일하다.
free vector의 집합인 linear vector space는 두 벡터의 선형 결합으로 정의된다.

# Inner and Outer Products in $\mathbb{E^3}$

![](/assets/img/posts/b57cbb7d-7937-8391-ad13-81684e56b6ce.webp)

$u,v$의 내적은 아래와 같이 정의된다.
$\langle u, v \rangle \doteq u^T v = u_1 v_1 + u_2 v_2 + u_3 v_3, \quad \forall u, v \in \mathbb{R}^3$
이는 길이를 정의할때도 사용된다. 두 벡터 사이의 각도는 다음과 같이 정의된다. $\cos \theta = \frac{\langle u, v \rangle}{\|u\| \|v\|}$
$\langle u, v \rangle = 0 \Rightarrow \text{orthogonal vectors}$

![](/assets/img/posts/82bcbb7d-7937-82f4-964a-016f6875346a.webp)

두 벡터의 외적은 새로운 벡터를 생성한다. 이때 방향은 오른손 법칙을 따르고, 새롭게 생성된 벡터는 두 벡터에 수직하다.

$u \times v \doteq
\begin{bmatrix} 
u_2 v_3 - u_3 v_2 \\ 
u_3 v_1 - u_1 v_3 \\ 
u_1 v_2 - u_2 v_1 
\end{bmatrix} 
\in \mathbb{R}^3$
두 벡터의 linear combination을 외적한 것은 각각을 외적한 후 linear combination한 것과 동일하다. 

$u \times (\alpha v + \beta w) = \alpha (u \times v) + \beta (u \times w), \quad \forall \alpha, \beta \in \mathbb{R}$

외적의 순서는 방향을 결정한다. 따라서 외적의 순서가 바뀌면 부호가 바뀐다.
$u \times v = -v \times u$

![](/assets/img/posts/93dcbb7d-7937-8386-8d08-01343e55bce8.webp)

$u,v$의 외적은 $v$에서 $u\times v$로의 map으로 표현 가능하다. →  $\mathbb{R^3}$ to $\mathbb{R^3}$ : $v \mapsto u \times v$ 반대 역시 가능하다.
이때 map을 skew-symmetric matrix라 하고, $\hat{u}$로 표현한다. 이때 $\hat{u}$는 $v$에 linear하다.
$\hat{u} \doteq
\begin{bmatrix} 
0 & -u_3 & u_2 \\ 
u_3 & 0 & -u_1 \\ 
- u_2 & u_1 & 0
\end{bmatrix} 
\in \mathbb{R}^{3 \times 3},\ u \times v = \hat{u} v,\ \hat{u}^{T} = -\hat{u}$

# Coordinate Frames

![](/assets/img/posts/cffcbb7d-7937-8214-b7a4-014b4cb80452.webp)

rigid object는 오른손 법칙을 만족하는 right-handed orthonormal frame을 가지며 이를 object coordinate frame 또는 body frame이라고 부른다. 또한, rigid-body motion은 reference frame과의 상대적인 motion으로 정의될 수 있으며, 이를 world frame이라 한다.

![](/assets/img/posts/1f6cbb7d-7937-8306-b461-01da8f94c0a4.webp)

다음과 같이 body frame $F_C$와 world frame $F_W$가 있을 때 motion은 translation vector T와 Rotation vector R로 나타낼 수 있음. 

# Rigid-body Motion

![](/assets/img/posts/be2cbb7d-7937-8235-a586-016132b0a1a3.webp)

이때 rigid-body motion은 motion에 관계 없이 distance가 유지된다.
$\| \mathbf{X}(t) - \mathbf{Y}(t) \| \equiv \text{constant}, \quad \forall t \in \mathbb{R}$

![](/assets/img/posts/e2ccbb7d-7937-8352-acae-014a842ecd14.webp)

또한, rigid-body motion은 다음과 같은 family of maps, 즉 함수 집합으로 표현된다.
$g(t) : \mathbb{R}^3 \to \mathbb{R}^3; \quad \mathbf{X} \mapsto g(t)(\mathbf{X})$
$g(t)$는 거리가 유지된다는 조건 하에 강체상의 모든 점의 좌표가 어떻게 변하는지를 나타낸다. 또한 시간을 생략해 다음과 같이 쓰기도 한다. 
$\mathbb{R}^3 \to \mathbb{R}^3; \quad \mathbf{X} \mapsto g(\mathbf{X})$ 
벡터 역시 이를 만족한다. $v = \mathbf{Y} - \mathbf{X}$일때 g에 의해 아래와 같이 새로운 벡터를 얻을 수 있다.
$u = g_*(v) \doteq g(\mathbf{Y}) - g(\mathbf{X})$
rigid-body motion은 길이가 보존되기 때문에 $| g_*(v) \| = \| v \|, \quad \forall v \in \mathbb{R}^3$이다.

# Euclidean Transformation

![](/assets/img/posts/855cbb7d-7937-827f-9a8a-0115502a9fa5.webp)

길이를 보존하는 map을 Euclidean Transformation이라 한다. 3차원 공간에서 이러한 Euclidean translation은 E(3)이라 나타낸다. 하지만, 이것이 rigid-body motion이라는 뜻은 아니다. $f : \begin{bmatrix} X_1, X_2, X_3 \end{bmatrix}^{T} \ \text{오른손 좌표계} \mapsto \begin{bmatrix} X_1, X_2, -X_3 \end{bmatrix}^{T} \ \text{왼손 좌표계}$와 같이 길이는 보존되지만 방향은 보존되지 않는 경우가 존재하기 때문에 rigid-body motion은 거리와 방향이 모두 보존되는 motion이다.

> rigid-body motion

내적은 polarization identity로 나타낼 수 있다.
$\langle u, v \rangle = \frac{1}{4} \left( \| u + v \|^2 - \| u - v \|^2 \right)$

$\| u + v \| = \| g_*(u) + g_*(v) \|$이기 때문에, 아래의 식이 성립한다.
$\langle u, v \rangle = \langle g_*(u), g_*(v) \rangle, \quad \forall u, v \in \mathbb{R}^3$

<details markdown="1">
<summary>증명</summary>

$\langle u, v \rangle = \| u \| \| v \| \cos \theta$



</details>

![](/assets/img/posts/f00cbb7d-7937-8321-9c61-815e1031df1e.webp)

세 벡터 사이의 triple product 역시 보존된다. 이는 rigid-body의 부피 역시 보존됨을 의미한다.
$\langle g_*(u), g_*(v) \times g_*(w) \rangle = \langle u, v \times w \rangle$
 $\text{Volume} = \text{area of base} \times \text{height}

= \| a \times b \| \| c \| |\cos \phi|

= |(a \times b) \cdot c|.$

# Orthogonal Matrix Representation of Rotation

![](/assets/img/posts/77ecbb7d-7937-83ca-b9e4-8190396fb9e8.webp)

원점을 중심으로 회전하는 강체를 가정하였을 경우, $F_C$와 $F_W$의 관계는 세개의 orthonormal vectors의 좌표로 결정된다. 이는 다음과 같다. $r_1 = g_*(e_1), \quad r_2 = g_*(e_2), \quad r_3 = g_*(e_3) \in \mathbb{R}^3$ 이때 $r_1, r_2, r_3$은 body frame의 unit vector이며, 강체의 회전은 $3\times3$ matrix로 결정된다.
$R_{wc} \doteq [r_1, r_2, r_3] \quad p \in \mathbb{R}^{3 \times 3}$
이때 $R_{wc}$의 column vector들이 각각 $r_1,r_2,r_3$이다. 따라서 Orthogonal matrix이며, 아래와 같은 조건을 만족한다.
$r_i^T r_j = \delta_{ij} \doteq
\begin{cases} 
1, & \text{for } i = j, \\
0, & \text{for } i \neq j
\end{cases}, \quad \forall i,j \in \{1,2,3\}
\\R_{wc}^T R_{wc} = R_{wc} R_{wc}^T = I
\\ R_{wc}^{-1} = R_{wc}^{T}
\\ \det(R_{wc}) = +1 \ by\ right-handed\ frame$

이때 $R_{wc}$는 special Orthogonal matrix라고 하며 이는 방향을 유지한다는 뜻을 포함하고 있다.
위와 같은 특징들로 인해 모든 특수 직교 행렬은 다음과 같이 표기된다.
$SO(3) \doteq \{ R \in \mathbb{R}^{3 \times 3} \mid R^T R = I, \det(R) = +1 \}$
또한, 이는 회전 행렬이라 불린다.


# Euler Angles to Rotation Matrix

![](/assets/img/posts/40ccbb7d-7937-8219-9adc-81be3b765cc3.webp)

yaw angle은 z축을 기준으로 회전시킨것

$R_z(\gamma) =
\begin{bmatrix}
\cos(\gamma) & -\sin(\gamma) & 0 \\
\sin(\gamma) & \cos(\gamma) & 0 \\
0 & 0 & 1
\end{bmatrix}$

![](/assets/img/posts/249cbb7d-7937-832d-9e85-0189df6b47e3.webp)

pitch angle은 y축을 기준으로 회전시킨것

$R_y(\beta) =
\begin{bmatrix}
\cos(\beta) & 0 & \sin(\beta) \\
0 & 1 & 0 \\
-\sin(\beta) & 0 & \cos(\beta)
\end{bmatrix}$

![](/assets/img/posts/b9acbb7d-7937-82f7-883f-0195df2adb81.webp)

roll angle은 x축을 기준으로 회전시킨것

$R_x(\alpha) =
\begin{bmatrix}
1 & 0 & 0 \\
0 & \cos(\alpha) & -\sin(\alpha) \\
0 & \sin(\alpha) & \cos(\alpha)
\end{bmatrix}$

## Tait-Bryan angle

yaw-pitch-roll 순으로 회전시킨 것으로, 최종적으로는 아래와 같다.

$R_3^0 = R_z(\gamma) R_y(\beta) R_x(\alpha)=
\begin{bmatrix}
\cos\gamma & -\sin\gamma & 0 \\
\sin\gamma & \cos\gamma & 0 \\
0 & 0 & 1
\end{bmatrix}
\times
\begin{bmatrix}
\cos\beta & 0 & \sin\beta \\
0 & 1 & 0 \\
-\sin\beta & 0 & \cos\beta
\end{bmatrix}
\times
\begin{bmatrix}
1 & 0 & 0 \\
0 & \cos\alpha & -\sin\alpha \\
0 & \sin\alpha & \cos\alpha
\end{bmatrix}$  

# Rigid-body motion and its Representations

![](/assets/img/posts/934cbb7d-7937-8368-8d9d-81f297d89e5b.webp)

world frame에서 rigid-body 위의 한 점 p를 $X_W$라 하자. 이때 $X_W$는 $T_{WC}$와 $X_C$의 합으로 표현되며, 다음과 같다. 
$X_W=T_{WC}+X_C$ 
이때,  $T_{WC}$는 $F_C$에 속하고, $X_C$는 $F_W$에 속한다.
$X_C$는 $F_C$ 위의 점이기 때문에 $R_{WC} X_C$이다. 따라서 $X_W$는 다음과 같고, homogeneous coordiantes를 사용시 linear form으로 나타낼수 있다.
$X_W = R_{WC} X_C + T_{WC}$ as linear form, $\bar{X}_w = \begin{bmatrix} X_w \\ 1 \end{bmatrix} = \begin{bmatrix} R_{WC} & T_{WC} \\ 0 & 1 \end{bmatrix}\begin{bmatrix} X_c \\ 1 \end{bmatrix} \doteq g_{WC} \bar{X}_c$

$g = \begin{bmatrix} R & T \\ 0 & 1 \end{bmatrix} \in \mathbb{R}^{4 \times 4}$

따라서 위의 rigid-body translation, 즉 Special Euclidean Transformation는 다음과 같다.

$SE(3) \doteq \left\{ g = \begin{bmatrix} R & T \\ 0 & 1 \end{bmatrix} \ \Big| \ R \in SO(3), T \in \mathbb{R}^3 \right\} \subset \mathbb{R}^{4 \times 4}$

여러 translation의 composition은 다음과 같이 이루어진다. 좌측에서 우측으로 이루어진다.
$g_1 g_2 =
\begin{bmatrix} R_1 & T_1 \\ 0 & 1 \end{bmatrix}
\begin{bmatrix} R_2 & T_2 \\ 0 & 1 \end{bmatrix}
=
\begin{bmatrix} R_1 R_2 & R_1 T_2 + T_1 \\ 0 & 1 \end{bmatrix}
\in SE(3)$

또한, inverse는 다음과 같다.

$g^{-1} =
\begin{bmatrix} R & T \\ 0 & 1 \end{bmatrix}^{-1}
=
\begin{bmatrix} R^T & -R^T T \\ 0 & 1 \end{bmatrix}
\in SE(3)$

![](/assets/img/posts/553cbb7d-7937-831a-9d81-81b5274a0bd6.webp)

다음 그림에서 도출할 수 있듯,$X_2 = g_{21} X_1, \quad X_3 = g_{32} X_2, \quad X_3 = g_{31} X_1$를 만족한다. 이를 통해 다음과 같은 composition rule을 얻을 수 있다.

$g_{32} g_{21} = g_{31}$
또한, 이를 통해 rule of inverse 역시 얻을 수 있다.
$g_{21}^{-1} = g_{12}$
이를 종합해보면 다음과 같은 composition rules를 얻을 수 있다.

$X_i = g_{ij} X_j, \quad g_{ik} = g_{ij} g_{jk}, \quad g_{ij}^{-1} = g_{ji}$


### 참고자료

<div class="notion-video-embed" style="margin:1.25rem 0;">
<div style="position:relative;padding-bottom:56.25%;height:0;overflow:hidden;border-radius:12px;">
<iframe src="https://www.youtube.com/embed/p05akAOSrUA" title="Embedded video" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen style="position:absolute;top:0;left:0;width:100%;height:100%;border:0;"></iframe>
</div>
</div>

