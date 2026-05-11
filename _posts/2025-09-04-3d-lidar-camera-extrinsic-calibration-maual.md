---
layout: post
title: "3D LIDAR-Camera extrinsic calibration (maual)"
date: 2025-09-04T06:34:00.000Z
math: true
image:
  path: "/assets/img/posts/265cbb7d-7937-8074-9a24-ff25931e0b85.gif"
categories:
  - "Team Vision"
---

# extrinsic calibration이란?

<div class="notion-callout" style="display:flex;gap:0.75rem;align-items:flex-start;padding:1rem 1.1rem;margin:1.25rem 0;border-radius:14px;background:rgba(127, 127, 127, 0.12);border:1px solid rgba(127, 127, 127, 0.28);" markdown="1">
<div class="notion-callout__icon" style="font-size:1.25rem;line-height:1.4;flex:0 0 auto;">💡</div>
<div class="notion-callout__content" style="flex:1 1 auto;min-width:0;" markdown="1">

[https://github.com/koide3/direct_visual_lidar_calibration?tab=readme-ov-file](https://github.com/koide3/direct_visual_lidar_calibration?tab=readme-ov-file)

카메라와 라이다의 상대적인 거리를 찾기 위한 변환 처리 과정이다. 

- input : 카메라 image와 info, 3d lidar point가 담긴 rosbag file
- output : 카메라와 라이다의 변환관계가 담긴 yaml파일

</div>
</div>





## 진행방법

<div class="notion-callout" style="display:flex;gap:0.75rem;align-items:flex-start;padding:1rem 1.1rem;margin:1.25rem 0;border-radius:14px;background:rgba(127, 127, 127, 0.12);border:1px solid rgba(127, 127, 127, 0.28);" markdown="1">
<div class="notion-callout__icon" style="font-size:1.25rem;line-height:1.4;flex:0 0 auto;">💡</div>
<div class="notion-callout__content" style="flex:1 1 auto;min-width:0;" markdown="1">

### installation

- 나는 도커를 통해 이미지를 다운받았다.

```bash
docker pull koide3/direct_visual_lidar_calibration:humble


# bag파일을 전처리 하기 위한 명령어 
docker run   --rm   --net host   --runtime=nvidia   --gpus all   --env DISPLAY=$DISPLAY   --env XAUTHORITY=${XAUTHORITY:-$HOME/.Xauthority}   -v /tmp/.X11-unix:/tmp/.X11-unix   -v /home/loe/workspace/autoware_carla/data/rosbag_extrinsic_calib_left:/tmp/input_bags   -v /home/loe/workspace/autoware_carla/data/result:/tmp/preprocessed   koide3/direct_visual_lidar_calibration:humble   ros2 run direct_visual_lidar_calibration preprocess -adv /tmp/input_bags /tmp/preprocessed

# 수작업으로 라이다와 카메라의 correspondence를 지정해주기 위한 명령어
docker run   --rm   --net host   --runtime=nvidia   --gpus all   --env DISPLAY=$DISPLAY   --env XAUTHORITY=${XAUTHORITY:-$HOME/.Xauthority}   -v /tmp/.X11-unix:/tmp/.X11-unix   -v /home/loe/workspace/autoware_carla/data/rosbag_extrinsic_calib_right:/tmp/input_bags   -v /home/loe/workspace/autoware_carla/data/result:/tmp/preprocessed   koide3/direct_visual_lidar_calibration:humble   ros2 run direct_visual_lidar_calibration initial_guess_manual /tmp/preprocessed

```




### data collection

<div class="notion-video-embed" style="margin:1.25rem 0;">
<div style="position:relative;padding-bottom:56.25%;height:0;overflow:hidden;border-radius:12px;">
<iframe src="https://www.youtube.com/embed/Urs36qdSQm0" title="Embedded video" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen style="position:absolute;top:0;left:0;width:100%;height:100%;border:0;"></iframe>
</div>
</div>

![](/assets/img/posts/265cbb7d-7937-80a6-af55-eeaedaaa6ff7.webp)

![](/assets/img/posts/265cbb7d-7937-8007-a04a-cf084d979d30.webp)

- bag record를 통해 약 40초간 bag파일을 수집해줍니다. (16채널이므로 많이 따야한다)
- 이후 위의 preprocess command를 입력한다.


### preprocess

```bash
docker run \
  --rm \
  -v /home/loe/workspace/autoware_carla/data/calibration_center:/tmp/input_bags \
  -v /home/loe/workspace/autoware_carla/data/lidar_calibration:/tmp/preprocessed \
  koide3/direct_visual_lidar_calibration:humble \
  ros2 run direct_visual_lidar_calibration preprocess -a /tmp/input_bags /tmp/preprocessed
```

![](/assets/img/posts/265cbb7d-7937-8074-9a24-ff25931e0b85.gif)

- 2~3분정도의 처리시간이 소요된다.

![](/assets/img/posts/265cbb7d-7937-8097-87e3-ec3b1ddd550d.webp)



![](/assets/img/posts/265cbb7d-7937-80fa-b0e6-cb38343fc75d.webp)

- 이후 내가 지정한 result파일에 ply파일과 png파일이 잘 저장되었는지 확인한다.


### manual pointing

```bash
docker run   --rm   --net host   --runtime=nvidia   --gpus all   --env DISPLAY=$DISPLAY   --env XAUTHORITY=${XAUTHORITY:-$HOME/.Xauthority}   -v /tmp/.X11-unix:/tmp/.X11-unix   -v /home/loe/workspace/autoware_carla/data/rosbag_extrinsic_calib_right:/tmp/input_bags   -v /home/loe/workspace/autoware_carla/data/result:/tmp/preprocessed   koide3/direct_visual_lidar_calibration:humble   ros2 run direct_visual_lidar_calibration initial_guess_manual /tmp/preprocessed

```



![](/assets/img/posts/265cbb7d-7937-802e-b546-fb108d26f214.gif)

![](/assets/img/posts/265cbb7d-7937-8073-8473-db4f740fb071.webp)

왼쪽 카메라가 잘 calibration된 모습이다.

</div>
</div>

