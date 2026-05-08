---
layout: post
title: "[MVG] Lecture 5-1 ~ 5-2: Camera models and calibration"
date: 2026-04-22T08:28:00.000Z
math: true
image:
  path: "/assets/img/posts/2fccbb7d-7937-83e1-87da-016e3fb561d5.webp"
categories:
  - "study"
---

# **Camera models and calibration**



### Learning Outcomes

<div class="notion-callout" style="display:flex;gap:0.75rem;align-items:flex-start;padding:1rem 1.1rem;margin:1.25rem 0;border-radius:14px;background:rgba(127, 127, 127, 0.12);border:1px solid rgba(127, 127, 127, 0.28);" markdown="1">
<div class="notion-callout__icon" style="font-size:1.25rem;line-height:1.4;flex:0 0 auto;">❗</div>
<div class="notion-callout__content" style="flex:1 1 auto;min-width:0;" markdown="1">

![](/assets/img/posts/2fccbb7d-7937-83e1-87da-016e3fb561d5.webp)

- pinhole모델의 카메라 projection
- projection matrix에서의 카메라 중심, principal planes, principal point, principal axis등의 개념
- forward & backward projection
- 캘리브레이션을 통한 instrinsic & extrinsic 값 찾기

</div>
</div>



## What is a Camera?

<div class="notion-callout" style="display:flex;gap:0.75rem;align-items:flex-start;padding:1rem 1.1rem;margin:1.25rem 0;border-radius:14px;background:rgba(127, 127, 127, 0.12);border:1px solid rgba(127, 127, 127, 0.28);" markdown="1">
<div class="notion-callout__icon" style="font-size:1.25rem;line-height:1.4;flex:0 0 auto;">❗</div>
<div class="notion-callout__content" style="flex:1 1 auto;min-width:0;" markdown="1">

![](/assets/img/posts/a85cbb7d-7937-83bb-bfc5-01e47d25ffb1.webp)

- 3차원의 물체를 $p^2$차원으로 매핑하는것

</div>
</div>

### 카메라 모델

<div class="notion-callout" style="display:flex;gap:0.75rem;align-items:flex-start;padding:1rem 1.1rem;margin:1.25rem 0;border-radius:14px;background:rgba(127, 127, 127, 0.12);border:1px solid rgba(127, 127, 127, 0.28);" markdown="1">
<div class="notion-callout__icon" style="font-size:1.25rem;line-height:1.4;flex:0 0 auto;">❗</div>
<div class="notion-callout__content" style="flex:1 1 auto;min-width:0;" markdown="1">

![](/assets/img/posts/86acbb7d-7937-82d7-9f04-01487abcc8c2.webp)

**카메라 모델**

- 중심 사영(Central Projection) : 한 점에서 모든 광선이 나와서 평면에 맺히는 방식입니다.
- Finite Centre(유한한 중심) : 카메라의 중심이 유한한 위치에 존재하며 일반적인 projective 카메라가 이에 해당합니다.
- Centre at infinity (무한대에 있는 중심) : 카메라의 중심이 무한대에 위치한 경우이며 affine 카메라가 이에 해당합니다. 원근투영이 적용되지 않습니다.
### 기본적인 pinhole모델

![](/assets/img/posts/8eccbb7d-7937-8245-8e2f-8156906571b9.webp)

- Image plane은 x,y평면에 평행합니다.
- projection의 중심 C를 유클리드 좌표계의 원점이라고 합니다. 또한 C와 p사이의 실제 거리는 초점거리 f가 됩니다.


![](/assets/img/posts/27ecbb7d-7937-83e9-8598-81258be1a6d5.webp)

- 오른쪽 그림의 두 삼각형의 닮음을 이용하면 이미지 평면에 사영된 포인트의 좌표는 $({fX\over z}, {fY\over z}, f)$가 됩니다. 하지만 3D에서 2D로의 사영이므로 세번째 좌표는 무시할 수 있게 됩니다.
![](/assets/img/posts/7f2cbb7d-7937-83fb-a416-01bf72741f52.webp)

- 용어들
  - Camera centre : 이미지 평면을 지나는 모든 선이 지나는 점..
  - Principal axis, point :  유클리드 좌표계의 z축과 그 축이 이미지 평면과 만나는 점
  - principal plane : 유클리드 좌표계의 xy평면
![](/assets/img/posts/f30cbb7d-7937-8296-94be-817cc8111ca5.webp)

기존에 유클리드 좌표계를 homogeneous 좌표계로 변환한다면

- 3d point $\vec X$ : $(X, Y, Z) \to (X, Y, Z, 1)$
- 2d point  $\vec x$ : $({fx\over z}, {fx\over z}) \to (fx, fy, z)$
- 이러한 3d point에서 2d point로의 변환을 matrix 곱의 형태로 나타낸다면 $\vec x = P\vec X$입니다.
  - 이때 $P$는 $\begin{bmatrix} {f} & 0 & 0&0 \\ 0&f&0&0\\0&0&1&0 \end{bmatrix} = \begin{bmatrix} {f} & 0 & 0 \\ 0&f&0\\0&0&1 \end{bmatrix}\begin{bmatrix} {1} & 0 & 0&0 \\ 0&1&0&0\\0&0&1&0 \end{bmatrix}$의 형태로 나타낼 수 있습니다.


![](/assets/img/posts/8f0cbb7d-7937-82c1-a230-011332b33f59.webp)

- 실제 3d point가 사영되었을때 원점은 principal point가 아닐 수도 있고 그것은 정의하기 나름입니다. 이러한 점을 보상하기 위해 2d 이미지의 원점과 principal plane의 차이로 $p_x, p_y$를 정의하고 $P$ matrix에 추가합니다.


![](/assets/img/posts/dd1cbb7d-7937-83b9-95c2-0145b14cda7c.webp)

- camera to image의 선형변환을 나타내는 matrix를 calibration matrix 혹은 intrinsic matrix라고 부릅니다.


![](/assets/img/posts/7afcbb7d-7937-83a0-aa74-813d8da3cf40.webp)

- 지금 까지는 $X_{cam}$에서 이미지 평면으로의 사영을 알아보았습니다. 하지만 일반적으로 3d point는 world 좌표계라는 카메라 좌표계와는 다른 유클리드 좌표계로 표현됩니다.
- world 좌표계에서 카메라 좌표계로의 변환은 $[R|t]$라는 회전과 평행이동의 특성을 가지는 강체 변환입니다.


![](/assets/img/posts/320cbb7d-7937-832c-9ec6-81db07cab0d8.webp)

![](/assets/img/posts/37acbb7d-7937-82fe-8860-81b906fbc998.webp)

카메라 좌표계의 중심을 world 좌표계를 기준으로 $C$로 나타낼 수 있습니다.

- $X_{cam} = \begin{bmatrix} R&-RC\\0&1\end{bmatrix}X_{world}$로 표현할 수 있습니다. 이때 모든 좌표는 homogeneous 좌표계로 표현되므로 $R$은 회전변환을 나타내고 $-RC$는 평행이동을 나타낸다고 볼 수 있습니다.
- $x_{image} = K\begin{bmatrix}I|0 \end{bmatrix}\begin{bmatrix}R&-RC\\0&1 \end{bmatrix}X_{world}$입니다. 이떄 $x_{image} = K\begin{bmatrix}R&-RC \end{bmatrix}X_{world}$ 로 나타낼 수 있습니다.
- $P = KR\begin{bmatrix}I|-C \end{bmatrix}$로 표현할때 $\begin{bmatrix} R|-RC \end{bmatrix}$를 extrinsic matrix라고 합니다. 이때 $P$는 9의 자유도를 가지게 됩니다.


![](/assets/img/posts/c91cbb7d-7937-8349-97ac-814444b4ded4.webp)

![](/assets/img/posts/080cbb7d-7937-8361-8342-01c1990fcf18.webp)

- Rotation matrix $R$은 yaw, pitch, roll (YPR)의 Z,Y,X축 각도변환 회전행렬을 순서대로 적용한 행렬입니다.
- 2D space에서의 회전행렬은 yaw 변환밖에 없기 떄문에 1의 자유도를 가지고 3D space에서의 회전행렬은 3의 자유도를 가집니다.
- Orthonormal한 회전행렬의 특성 덕분에 다음과 같은 특성을 가지게 됩니다.
  - 오른손 좌표축에 대한 회전행렬의 경우 +1, 외손 좌표축에 대한 회전행렬의 경우 -1의 determinat값을 가지게 됩니다. (일반적으로 3d회전은 오른손 좌표계를 기준으로 합니다.)
    <details markdown="1">
    <summary>오른손 좌표계에 왼손 좌표계 회전행렬을 가하면?</summary>
    
    ![](/assets/img/posts/cd7cbb7d-7937-8259-8782-81dedf739083.webp)
    
    
    
    </details>
    
    ![](/assets/img/posts/19fcbb7d-7937-8265-8867-0129ced3b9e6.webp)
  - $R$의 Transpose는 inverse입니다.
    <details markdown="1">
    <summary>why?</summary>
    
    ![](/assets/img/posts/b84cbb7d-7937-82d2-b0c7-816dcb5e2d4c.webp)
    
    
    
    </details>
  - 행렬의 각 colum벡터가 서로 orthonormal basis이기 떄문에 두 column벡터의 외적이 다른 colum 벡터를 나타내게 됩니다.
  - 두 colum 벡터의 내적은 0입니다.
  - 모든 column벡터의 크기는 1입니다.


![](/assets/img/posts/601cbb7d-7937-83a3-ad78-01b80ac1977a.webp)

- Intrinsic matrix에 대해 더 자세하게 살펴본다면 x축의 초점거리과 y축의 초점거리는 다를 수 있습니다. 따라서 이미지 pixel은 정사각형이 아닐 수 있고 심지어 skew 될 수 도 있습니다. 이러한 x축에 대한 skew값을 파라미터 s로 나타낼 수 있습니다.
  <details markdown="1">
  <summary>x축으로만 skew가 일어나는 것인가?</summary>
  
  ![](/assets/img/posts/f96cbb7d-7937-83fe-9425-81f83d222d7b.webp)
  
  ![](/assets/img/posts/1d5cbb7d-7937-832b-aa64-013b60b0cbeb.webp)
  
  
  
  </details>


![](/assets/img/posts/657cbb7d-7937-83fe-946f-014d021636cc.webp)

![](/assets/img/posts/c3bcbb7d-7937-823d-aefc-01bed7cc0bf7.webp)

- 결론적으로 projection matrix $P$는 11의 자유도를 가지게 됩니다. 
- 또한  pinhole 카메라 모델은 finite projective camera model이고 $P = KR[I|-C] = M[I|M^{-1}P_4] =$$\begin{bmatrix} P_1&P_2&P_3&P_4 \end{bmatrix}$ 로 나타낼 수 있습니다. 이떄 $C$값이 존재해야 하므로 3 by 3 matrix인 $M(KR)$  은 non singular합니다.
![](/assets/img/posts/a04cbb7d-7937-822e-acfe-018c8b74b3a0.webp)

![](/assets/img/posts/9bbcbb7d-7937-83af-989d-811f3dd517db.webp)

- projection matrix $P$는 rank 3이고 Null space에 속하는 4차원 벡터는 카메라의 중심을 의미합니다. 이 카메라 중심은 사영공간에서는 존재하지만 이미지 평면에서의 대응점이 존재하지 않습니다. $PC = 0$
- 증명
  - 사영공간에서의 점 $A$와 $C$사이의 점 $X$를 이미지 평면에 사영시킬때 이 점 $X$를 사영시킨 점과 $A$를 사영시킨 점은 동일합니다. 따라서 $PC = 0$


![](/assets/img/posts/a95cbb7d-7937-83b0-bd17-01e640b29083.webp)

카메라 projection matrix $P = [\vec p_1,\vec p_2, \vec p_3,\vec p_4]$라고 할때 각 $\vec p_1, \vec p_2, \vec p_3$는 world 좌표계에서 x, y, z축의 무한대의 점이 이미지 plane에 사영된 값과 같습니다.

EX) : $[\vec p_1,\vec p_2, \vec p_3,\vec p_4]\begin{bmatrix} 0\\0\\1\\0\end{bmatrix} = p_3$

$\vec p_4$는 $\begin{bmatrix} 0\\0\\0\\1 \end{bmatrix}$인 world 좌표계의 원점이 이미지 평면으로 사영된 값입니다.



![](/assets/img/posts/c9ccbb7d-7937-83da-84b1-01b2e312c105.webp)

- $P$ matrix의 row vector를 볼때 각 row는 특정 Plane을 나타낼 수 있습니다.
- Principal plane
  ![](/assets/img/posts/c8acbb7d-7937-831b-83b8-016902530bf4.webp)
  
  - $X$를 principal plane 위의 점이라고 생각할때 카메라 센터 $C$와 $X$사이에 선을 그으면 이미지 평면에서의 무한대의 점에서 만나게 됩니다. 따라서$\begin{bmatrix} P_1^T\\P_2^T\\P_3^T \end{bmatrix}X = \begin{bmatrix} X\\Y\\0\end{bmatrix}$입니다. 이때 항상 $P_3^TX = 0$이므로 $P_3$는 principal plane을 의미하게 됩니다.
- Axis plane
  ![](/assets/img/posts/0a9cbb7d-7937-83ba-94bf-8156ffe740bd.webp)
  
  - $PX = (x, 0, w)^T$일때 $(x, 0, w)^T$는 이미지 평면에서 $y=0$인 선이고 $PC = 0$이므로 그 선과 카메라 원점을 지나는 평면 $p_2$는  $P_2^TX = 0$의 식으로 나타낼 수 있습니다. 
- Principal axis, point
  ![](/assets/img/posts/7b8cbb7d-7937-8311-b601-016ba093f8e8.webp)
  
  ![](/assets/img/posts/014cbb7d-7937-83db-8ad1-813187abe9ef.webp)
  
  - Principal axis는 principal plane($P_3)$에 normal vecotor이므로  $m^3$vector입니다. 하지만 이 $m^3$벡터의 방향이 카메라의 방향인지 카메라 반대방향인지 모르기 때문에 $det(M)$을 구하여 방향을 지정해 줍니다.
  - $det(M)$이 양수이면 오른손 좌표계, 음수이면 왼손 좌표계입니다. 이때$det(M)\vec m^3$는 항상 카메라의 방향을 띄게 됩니다.

</div>
</div>

  

## Summary 

<div class="notion-callout" style="display:flex;gap:0.75rem;align-items:flex-start;padding:1rem 1.1rem;margin:1.25rem 0;border-radius:14px;background:rgba(127, 127, 127, 0.12);border:1px solid rgba(127, 127, 127, 0.28);" markdown="1">
<div class="notion-callout__icon" style="font-size:1.25rem;line-height:1.4;flex:0 0 auto;">❗</div>
<div class="notion-callout__content" style="flex:1 1 auto;min-width:0;" markdown="1">

![](/assets/img/posts/66ecbb7d-7937-8232-a992-8189ab258e91.webp)



![](/assets/img/posts/6eccbb7d-7937-83ce-ac37-01c8c5623285.webp)

</div>
</div>





## Action of a Projective Camera on Points

<div class="notion-callout" style="display:flex;gap:0.75rem;align-items:flex-start;padding:1rem 1.1rem;margin:1.25rem 0;border-radius:14px;background:rgba(127, 127, 127, 0.12);border:1px solid rgba(127, 127, 127, 0.28);" markdown="1">
<div class="notion-callout__icon" style="font-size:1.25rem;line-height:1.4;flex:0 0 auto;">❗</div>
<div class="notion-callout__content" style="flex:1 1 auto;min-width:0;" markdown="1">

![](/assets/img/posts/2eecbb7d-7937-823b-96a6-810ca7c0931e.webp)

- forward projection : 앞서 본 일반적인 projective camera는 3D에서 2D로의 매핑을 나타냈습니다. 3차원에서의 무한대의 점은 2차원 상에서 vanishing point로 표현되며 이는 오직 3 by 3 matrix인 $M$의 영향을 받습니다.


![](/assets/img/posts/f0ccbb7d-7937-8307-b536-810433999bc3.webp)

![](/assets/img/posts/55bcbb7d-7937-83bd-a790-81e18e7e491a.webp)

- backward projection : projection matrix $P$는 3 by 4 matrix이므로 non invertible하고 따라서 관측된 이미지 상의 $x$를 통해 원본 $X$의 정확한 위치를 복원할 수 는 없습니다. 하지만$X$는 $C$와 $x$를 이은 선 위에 존재하므로 $X(\lambda) = P^+x + \lambda C$로 나타낼 수 있습니다. ($P^+$ =  pesudo inverse)
- finite camera center일때 $M$은 Invertible하고 $\begin{bmatrix} M^{-1}x\\0 \end{bmatrix}$은 무한대 방향의 ideal한 점(방향)을 나타내게 됩니다. 또한 Camera center는 $\begin{bmatrix} -M^{-1}p_4\\1 \end{bmatrix}$이므로 카메라 원점 $C$로부터 $M^{-1}x$방향에 있는 점을 표현 할 수 있습니다.


![](/assets/img/posts/b07cbb7d-7937-8272-861b-813863ee4d6d.webp)

![](/assets/img/posts/913cbb7d-7937-820a-a3d3-01cd3d7adb4d.webp)

- $X$ : 3d point
- $C$ : camera center
- $P\vec X = \vec x = w(x, y, 1)^T$일때 $P^{3T}\vec X=w$입니다.
이때 $det(M)$은 $m3$가 카메라의 방향을 향하도록 해주고 $m^3$와 $X-C$의 내적은 $w$이므로 depth를 내적으로 표현할 수 있게 됩니다.

</div>
</div>



## Decomposition of the camera matrix



<div class="notion-callout" style="display:flex;gap:0.75rem;align-items:flex-start;padding:1rem 1.1rem;margin:1.25rem 0;border-radius:14px;background:rgba(127, 127, 127, 0.12);border:1px solid rgba(127, 127, 127, 0.28);" markdown="1">
<div class="notion-callout__icon" style="font-size:1.25rem;line-height:1.4;flex:0 0 auto;">❗</div>
<div class="notion-callout__content" style="flex:1 1 auto;min-width:0;" markdown="1">

![](/assets/img/posts/377cbb7d-7937-8382-b5ae-015a7138a60c.webp)

이제부터 카메라 matrix $P$를 $KR[I|-RC]$로 변환하는 법을 알아보겠습니다.



![](/assets/img/posts/638cbb7d-7937-82f3-ad8f-81c462b8a89b.webp)

camera centre $C$ :  $P$ matrix의 각 row vector는 두개의 axis plane과 하나의 principal plane을 나타내고 이 세 평면의 접점이 카메라 센터 이므로 $P$를 SVD를 통해 나누고 $\sum$의 singular value를 구하므로써 $C$를 구할 수 있습니다.

<details markdown="1">
<summary>why?</summary>

$P$의 null space는 1차원 벡터 C로 주어집니다.

![](/assets/img/posts/961cbb7d-7937-82a9-8d53-8190989a0363.webp)

![](/assets/img/posts/bd0cbb7d-7937-8361-953b-01dfcf458413.webp)

![](/assets/img/posts/adfcbb7d-7937-826a-9de0-01a5668a542f.webp)



</details>

![](/assets/img/posts/b2dcbb7d-7937-831d-ad04-81286fa3ee88.webp)

![](/assets/img/posts/02bcbb7d-7937-83e9-a5ba-01238bccd861.webp)

$M = KR$이고 $K$가 upper triangle matrix이기 때문에 RQ decompostion을 통해 분해하여 $M = RQ = KR$로 나타낼 수 있습니다.

이때 $MR^T=K$이고 $K = \begin{bmatrix}*&*&*\\0&*&*\\0&0&1 \end{bmatrix}$이기 때문에 4개의 equation이 나오게 되고 Rotation matrix는 3개의 자유도를 가지기 때문에 해를 확정할 수 있게 됩니다.

<details markdown="1">
<summary>why?</summary>

![](/assets/img/posts/e4bcbb7d-7937-8310-a844-818cc04c409b.webp)

![](/assets/img/posts/43dcbb7d-7937-833f-8753-81796fc6f857.webp)



</details>

![](/assets/img/posts/db4cbb7d-7937-83d3-a54a-010b0d849785.webp)

camera 는 필연적으로 projective하며 3차원에서 2차원으로 매핑시킨다.

</div>
</div>



## Affine camera : cameras at infinity

<div class="notion-callout" style="display:flex;gap:0.75rem;align-items:flex-start;padding:1rem 1.1rem;margin:1.25rem 0;border-radius:14px;background:rgba(127, 127, 127, 0.12);border:1px solid rgba(127, 127, 127, 0.28);" markdown="1">
<div class="notion-callout__icon" style="font-size:1.25rem;line-height:1.4;flex:0 0 auto;">❗</div>
<div class="notion-callout__content" style="flex:1 1 auto;min-width:0;" markdown="1">

![](/assets/img/posts/68bcbb7d-7937-8242-a4a8-01098109db37.webp)

카메라 중심이 무한대의 평면 위에 있다는 점을 확인할 수 있는 2가지 방법

1. 카메라 센터는 $P$의 null space이므로 $(\vec d, 0)$으로 표현할 수 있고$M\vec d = 0$을 만족하므로 $d$는 $M$의 null vector이다.
1. $P$의 세번째 row는 카메라 센터가 존재하는 principal plane인데 (0,0,0,1)이므로 무한대의 평면을 의미한다.
![](/assets/img/posts/4f1cbb7d-7937-8325-a306-01543a8c8d57.webp)

![](/assets/img/posts/f47cbb7d-7937-8283-a35f-01f3b37febae.webp)

![](/assets/img/posts/1ddcbb7d-7937-8390-90ad-0136e32f11e6.webp)

Affine camera의 특이한 점은 3차원에 2차원으로 투영해주는 3 by 4 matrix에서 세번째 column이 0이고 orthographic projection 이 이루어 진다는 점입니다. 

affine camera에서는 왜곡이 없기 때문에 $p_x, p_y$를 관습적으로 0으로 두는 경향이 있습니다.

![](/assets/img/posts/3b1cbb7d-7937-8363-9f73-81c66b79008f.webp)

- 따라서 affine camera matrix는 8의 자유도를 가집니다. 
  - instrinsic에서 3, rotation에서3, traslation에서 
    <details markdown="1">
    <summary>왜 rotation에서 3인가?</summary>
    
    $r^{1T},r^{2T},r^{3T}$가 외적관계로 dependent하기 때문에 $r^{1T},r^{2T}$가 결정되기 위해 $r^{3T}$가 결정되어야 함
    
    
    
    </details>
- $P_A$는 3의 rank를 가지는 $P_p$와 달리 2의 rank를 가지게 됩니다.


![](/assets/img/posts/bb3cbb7d-7937-82cc-8207-01f439e93203.webp)

Affine camera의 특징

- 3차원 상의 무한대점이 2차원의 무한대점에 매핑됩니다. 3rd row가 $(0,0,0,1)$이기 때문입니다.
- 3차원에서 평행한 두선이 사영 후에도 평행을 유지합니다. affine 특성을 유지합니다. 원근 왜곡이 없습니다.


### Orthographic projection

![](/assets/img/posts/a6acbb7d-7937-8227-a916-81ad725ee0be.webp)

![](/assets/img/posts/583cbb7d-7937-838e-a4a8-012a6ccfba11.webp)

- 카메라 캘리브레이션 matrix가 identity matrix입니다. (스케일링 효과가 없습니다)
- depth정보를 없앱니다.
- 카메라센터가 infinity에 있습니다.
- instrinsic parameter가 없으므로 5의 자유도를 가집니다.
- $P$ matrix의 마지막 row는 항상 $(0,0,0,1)$입니다.


![](/assets/img/posts/40acbb7d-7937-83e3-a06e-8151da12400a.webp)

![](/assets/img/posts/227cbb7d-7937-82e2-af58-01409e65b6aa.webp)

![](/assets/img/posts/a40cbb7d-7937-82d9-940f-81e66fb817a3.webp)

</div>
</div>



