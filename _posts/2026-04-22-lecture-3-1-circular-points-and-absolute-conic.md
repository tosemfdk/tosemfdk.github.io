---
layout: post
title: "Lecture 3-1: Circular points and Absolute conic"
date: 2026-04-22T08:28:00.000Z
math: true
image:
  path: "/assets/img/posts/358cbb7d-7937-805b-ad4f-d011ff105944.webp"
categories:
  - "study"
---

<div class="notion-callout" style="display:flex;gap:0.75rem;align-items:flex-start;padding:1rem 1.1rem;margin:1.25rem 0;border-radius:14px;background:rgba(127, 127, 127, 0.12);border:1px solid rgba(127, 127, 127, 0.28);" markdown="1">
<div class="notion-callout__icon" style="font-size:1.25rem;line-height:1.4;flex:0 0 auto;">💡</div>
<div class="notion-callout__content" style="flex:1 1 auto;min-width:0;" markdown="1">

1. Affine rectification과 metric rectification task에 대해서 설명한다.
1. Affine rectification task를 위해 line at infinity의 정의와 이것을 이용하는 방법을 배운다.
1. Metric rectification task를 위해 circular points의 정의와 dual absolute conic과 이것을 이용한 방법을 배운다.

</div>
</div>

# Affine Rectification and Metric rectification

**Affine rectification**

- 원근 투영으로 인해, 평행한 직선이 평행하지 않게 된다. 
- 따라서 ideal point들과 vanishing line 또한 finite 범위로 들어온다.
- 이러한 이미지를 평행한 선들은 평행하도록 이러한 투영 왜곡(projective distortion)을 제거하는 과정을 Affine rectification 이라고 한다.
- line at infinity 이용
**Metric rectification**

- 위 Affine rectification을 수행해도 평행한 직선의 경우 평행하도록 만들어주었지만, 각도가 90도 였던 두 직선 사이의 관계는 여전히 복구하지 못했다.
- 이를 각도까지 복원하여, 실제 세계의 것과 scale 만 다른 이미지를 만들어내는 작업을 metric rectification이라고 한다.
- 단, 여기서 scale은 복원하지 못한다.
- Circular points와 Absolute dual conic을 이용
# The Line at Infinity

Line at infinity의 성질을 알아보자.

- $\mathbf{l}_\infty$ = [0 0 1]$^\top$
- $\mathbf{l}_\infty$ 는 projective transformation 중 affinity 변환에 대해서 invariant하다.
- ※ point에 대한 projective transformation $\mathbf{H}$를 dual 관계인 line에 적용할 경우엔 $\mathbf{H}^{-\top}$ 으로 적용해야 한다.
- 이 $\mathbf{l}_\infty$를 찾는 것이 ***affine property**들을 찾는데에 도움을 준다는 것을 이후에 봐볼 것
- ※ affine property: parallelism(평행성), 직선성~, ratio of lengths(길이의 비)~
<details markdown="1">
<summary>**Recap) Hierarchical Projective Transformation**</summary>



</details>



[slide]

- 위 affinity 외의 다른 projective transformation을 봐보면, ideal point와 $\mathbf{l}_\infty$가 더 이상 infinity에 있지 않게 되는 것을 볼 수 있음
- 첫번째 식은 ideal point를 projective transform 할 때이다.
- 두번째 식은 line at infinity를 projective transform 할 때이다.


[slides]

- 위에서 $\mathbf{l}_\infty$는 affine transformation에 대해선 invariant 하다고 했다. 
- 하지만 pointwise(point level)에선 그렇지 않다.
- → $\mathbf{l}_\infty$ 위의 한 점(ideal point)를 affine transformation을 하면 여전히 $\mathbf{l}_\infty$ 위에 위치하지만, 동일한 점은 아니라는 뜻
- 만약 A가 scale 변환만 수행하는 diagonal matrix라면 point 또한 동일하게 유지된다.


# Recovery of Affine Properties from Images

이제 affine rectification 에 대해서 살펴볼 것



![](/assets/img/posts/358cbb7d-7937-805b-ad4f-d011ff105944.webp)

- **Affine rectification**: image 상의 line at infinity를 이용하면 ***projective distortion**을 제거할 수 있다.
- ※ projective distortion: 평행한 선이 투영 변환으로 인해 평행하지 않게 되는 경우 이를 projective distortion이라고 한다.
- projective distortion이 있는 이미지의 경우, line at infinity는 더 이상 infinity에 있지 않게 된다. (ideal point 또한 마찬가지)
- projective transformation($\mathbf{H}_{\mathbf{p}}$) on $\mathbf{l}_\infty$: 위에서 봤듯이 더 이상 infinity에 위치하지 않는다.
- 우리의 목표는 원래 $\mathbf{l}_\infty$이었던 finite line을 다시 $\mathbf{l}_\infty$으로 mapping하는 projective transformation $\mathbf{H}_{\mathbf{p}}^{\prime}$ 을 찾는 것이다.
- 이 때, 위 그림처럼 순서를 나타내면 다음과 같다.
  1. 1) affine 성질을 보존하는 이미지 ⇒ $\mathbf{H}_{\mathbf{p}}$ ⇒ 2) affine 성질을 잃어버린 이미지
  1. 2) affine 성질을 잃어버린 이미지 ⇒ $\mathbf{H}_{\mathbf{p}}^{\prime}$ ⇒ 3) affine 성질을 보존하는 이미지
  1. 1) affine 성질을 보존하는 이미지 ⇒ $\mathbf{H}_{\mathbf{A}}$ ⇒ 3) affine 성질을 보존하는 이미지
- 우리는 이 세번째 case가 왜 affine transformation($\mathbf{H}_{\mathbf{A}}$) 인지를 위에서 살펴봤다. (요약하자면 affine 성질을 보존하는 projective transformation은 affine transformation 밖에 없어서)
- ※ 위 첫번째와 세번째 그림을 보면, 평행한 선은 유지가 되지만 ideal point의 위치는 달라졌다. 하지만 여전히 ideal point이고 이는 line at infinity 상에 있다.


[slides]

- **Affine rectification 문제:** 
  - Given) 이미지 상에 투영된 line at infinity $\mathbf{l}$ = $(l_1, l_2, l_3)$$^\top$($l_3$ ≠ 0) 가 주어질 때 (이미지에서 $\mathbf{l}$을 계산할 수 있을 때)
  - Find) $\mathbf{H}_{\mathbf{p}}^{\prime}$를 찾는 것 ($\mathbf{H}_{\mathbf{p}}^{\prime}$는 투영된 $\mathbf{l}_\infty$(line at finite)을 다시 $\mathbf{l}_\infty$(line at infinite)로 mapping하는 변환임을 상기)
- **Solution**:
$$

$$



<div class="notion-callout" style="display:flex;gap:0.75rem;align-items:flex-start;padding:1rem 1.1rem;margin:1.25rem 0;border-radius:14px;background:rgba(127, 127, 127, 0.12);border:1px solid rgba(127, 127, 127, 0.28);" markdown="1">
<div class="notion-callout__icon" style="font-size:1.25rem;line-height:1.4;flex:0 0 auto;">💡</div>
<div class="notion-callout__content" style="flex:1 1 auto;min-width:0;" markdown="1">

**Affine Rectification 알고리즘 정리**

[slides]

1. 투영된 plane의 vanishing line $\mathbf{l}$을 투영된 평행선들 두 쌍의 교차점으로 구할 수 있다.
1. 임의의 $\mathbf{H}_{\mathbf{A}}$를 선택해서 $\mathbf{H}_{\mathbf{p}}^{\prime}$ = $\mathbf{H}_{\mathbf{A}}$$\mathbf{H}_{\mathbf{p}}^{-1}$를 계산
   이 때, $\mathbf{H}_{\mathbf{A}}$는 [ 행렬 ] 꼴이다. 
1. $\mathbf{H}_{\mathbf{p}}^{\prime}$를 계산하여 구했으면, 이것을 주어진 이미지에 적용하여 affinely rectified 이미지를 만듦 (평행성 복구)
1. Affine property들은 affinely rectified image로부터 복구할 수 있음 (평행선, ratio of lengths(?))
1. Note: 각도는 여전히 복구하지 못한다. (projective distortion만 rectify 했고, 여전히 affine distortion은 남아있음)


※ projective distortion(평행선 왜곡) → affine distortion(각도 왜곡) → similar distortion (scale 왜곡)

</div>
</div>

<div class="notion-callout" style="display:flex;gap:0.75rem;align-items:flex-start;padding:1rem 1.1rem;margin:1.25rem 0;border-radius:14px;background:rgba(127, 127, 127, 0.12);border:1px solid rgba(127, 127, 127, 0.28);" markdown="1">
<div class="notion-callout__icon" style="font-size:1.25rem;line-height:1.4;flex:0 0 auto;">💡</div>
<div class="notion-callout__content" style="flex:1 1 auto;min-width:0;" markdown="1">

**projective distortion은 실제 사진을 찍는 과정에서 언제 발생할까?**

1. World의 3D point $\mathbf{X}$를 [R|t]로 camera coordinate의 point $\mathbf{X_C}$로 변환
1. $\mathbf{X_C}$를 normal image plane(focal length가 1인 image plane)에 투영 → $\mathbf{X_{normal}}$ 
(➡️ 바로 여기서 projective distortion이 발생. 이 과정에서 Z축 값으로 나누게 되는데, 이러한 비선형 변환으로 linearity가 왜곡됨)
1. normal image plane에 투영된 점 $\mathbf{X_{normal}}$을 camera의 intrinsic $\mathbf{K}$를 통해 실제 pixel coordinate로 변환

</div>
</div>



# Computing a Vanishing Point from a Length Ratio



[slides]

- 이전 예시: ideal point와 line at infinity를 알면 affine property들을 알 수 있다는 것을 배웠음
- 반대로, affine property들을 알면, ideal point와 line at infinity를 구할 수 있다.
  - 이미지에서 하나의 line 상에 있는 a’, b’, c’을 확인할 수 있는 경우
  - world 상에 corresponding collinear points a, b, c를 갖는 line이 있다고 가정해보자.
  - d(a, b) : d(b, c) = a : b 를 우리가 안다고 하자. ( d(x,y)는 x와 y의 Eulidean distance를 의미 )
- Solution
  - 1D projective space를 다시 복습…


# Circular Points and Their Dual

circular points or absolute points

[slides] circular points의 정의



[slides] $\mathbf{H_S}$ (similar transformation)에 대해 invariant한 circular points

- circular points $\mathbf{I}$, $\mathbf{J}$의 경우 projective transformation $\mathbf{H}$가 similarity인 경우 invariant하다.
- converse is also true) $\mathbf{I}$와 $\mathbf{J}$가 변환 이후 invariant 하다면, 이 변환은 similarity이다.
- proof)
$$

$$



[slides] Circular points 이름의 유래

- 2D projective space에서의 모든 circle에 대해 $\mathbf{l}_\infty$와의 교점이 두개가 나온다.
- 이 때 모든 circle 모양의 conic은 $\mathbf{x}_1^{2} + \mathbf{x}_2^{2} = 0$  으로 나온다. ($\mathbf{l}_\infty$와 교점을 갖는 conic은 homogeneous coordinate에서 equivalent한 것들을 모두 빼면 이 하나의 형태밖에 없음.
- 그래서 이 두 점을 circular points라고 한다.
<div class="notion-callout" style="display:flex;gap:0.75rem;align-items:flex-start;padding:1rem 1.1rem;margin:1.25rem 0;border-radius:14px;background:rgba(127, 127, 127, 0.12);border:1px solid rgba(127, 127, 127, 0.28);" markdown="1">
<div class="notion-callout__icon" style="font-size:1.25rem;line-height:1.4;flex:0 0 auto;">💡</div>
<div class="notion-callout__content" style="flex:1 1 auto;min-width:0;" markdown="1">

**circle conic**

$$
\mathbf{x}_1^{2} + \mathbf{x}_2^{2} = 0
$$

위 conic을 matrix form으로 쓰면 다음과 같다.

$$
\begin{pmatrix}1 & 0 & 0 \\0 & 1 & 0 \\0 & 0 & 0\end{pmatrix}
$$

</div>
</div>



- 




