---
layout: post
title: "[Unist 복원 프로젝트] Part 2. slam을 사용해보자"
date: 2026-04-22T02:27:00.000Z
math: true
image:
  path: "/assets/img/posts/34acbb7d-7937-80ff-81ef-c3463e2c478a.gif"
categories:
  - "UNIST"
---

# 현재 상황

유니스트를 복원하기로 하였는데 **실내 / 실외를 나누어서 복원**하게 되었다. 실외는 **전용 모바일로봇이 돌아다니며 전용 데이터셋**을 얻을 수 있게 될 것 같고 **실내에서는 일단 내가 지금 속해잇는 인공지능 대학원 106동 전체**를 복원해보는 것으로 진행하기로 했다..



# DiskChunGS

[https://rffr.leggedrobotics.com/works/diskchungs/](https://rffr.leggedrobotics.com/works/diskchungs/)

- 교수님이 추천해주신 논문으로 google과 ETH의 RSL 연구실에서 나온 논문이다. 대표적은 특징은 아래와 같고 large scale의 scene을 메모리 효율적으로 mapping할 수 있도록 다양한 방법을 제시한다.
![](/assets/img/posts/34acbb7d-7937-8007-bac7-d8c232843203.webp)



## 실험 결과

[https://github.com/leggedrobotics/DiskChunGS](https://github.com/leggedrobotics/DiskChunGS)

![](/assets/img/posts/34acbb7d-7937-803e-bfdd-dfe4c3d37b08.gif)

kitty dataset 같은 장거리 데이터셋에서 높은 성능을 보였다는게 이 논문의 주요 contribution인데.. 내가 실험해보았을때에는 이렇게 high quality의 결과물은 나오지 않았다,,



### **tum RGB-D dataset**

![](/assets/img/posts/34acbb7d-7937-80ff-81ef-c3463e2c478a.gif)

- RGB-D dataset에서 rgb만을 이용하여 base pipeline을 돌려보았다,, 하지만 psnr은 22.08 정도로 퀄리티가 그렇게 좋지는 않다. 또한 논문에서 주장하는 direct sampling을 통한 세밀한 특징표현도 잘 이루어지지 않는 모습이다.. → 추후 원인 파악




### **konkuk mono dataset**

![](/assets/img/posts/34acbb7d-7937-80b9-8a95-cc5a91f4732b.gif)

- 내가 직접 iphone 4k video로 취득한 long sequence의 데이터셋이다.. 이 데이터셋은 orb-slam3에 필요한 캘리브레이션 파라미터가 없어서 중간에 계속 tracking이 끊겨 일부 데이터 셋만으로 진행하였다. 
- 퀄리티가 초반에는 괜찮다가 tracking을 잃어버린 이후부터는 완전히 이상해지는것을 확인할 수 있다. 
- 또한 동적물체(차량)의 영향을 받아 아티펙트가 남는 것 또한 확인할 수 있다.


### **zed camera (stereo) dataset**

![](/assets/img/posts/34acbb7d-7937-806b-a201-f15d85ad0d51.gif)

![](/assets/img/posts/34acbb7d-7937-8098-8e10-d7332f1c9301.gif)

- 캘리브레이션 된 데이터셋이고 야외 풀숲에서 찍은 데이터 셋 이다. 
- 하늘에 대한 영역을 잘 일반화 하지 못하는 것 같고 풀 숲이나 풀 밭에 대한 표현력도 많이 떨어진다.. → 원이파악 필요함. frame당 학습량이 문제일 수도 있고.. 최적의 학습 세팅을 찾아야할지도,,


# FVDB & FVDB reality-capture

![](/assets/img/posts/34acbb7d-7937-8045-95be-eec55ab1a9f3.webp)

- FVDB는 sparse한 공간정보를 gpu에 효율적으로 표현하기 위한 프레임 워크이고 fvdb reality capture는 fvdb를 활용하여 3DGS, mesh generation등의 application을 할 수 있도록 모아놓은 프레임워크이다. 
- 특징으로는 colmap 데이터 대신에 드론, 로봇, 차량 등에서 받아온 센서데이터(pointcloud, images, camera pose)를 받아 3dgs나 mesh generation등을 할 수 있다. → fvdb가 large scale에서 좋다고 하는 이유가 이것 때문인 듯 싶다. 다량의 데이터에서 한번에 위와같은 작업을 할 수 있도록 다양한 세팅을 조절할 수 잇을 것 같다.


## 실험 세팅

FVDB를 활용하기 위해서 odometry가 정확하게 나오는 모바일 로봇이 필요한데.. 내가 가지고 있는 당장 사용할 수 있는 로봇이 unitree go2밖에는 없다. 따라서 unitree go2의 센서데이터를 가지고 fvdb를 한번 사용해보기로 했다. 

```bash
ros2 topic list

/api/assistant_recorder/request
/api/assistant_recorder/response
/api/audiohub/request
/api/audiohub/response
/api/bashrunner/request
/api/bashrunner/response
/api/config/request
/api/config/response
/api/fourg_agent/request
/api/fourg_agent/response
/api/gas_sensor/request
/api/gas_sensor/response
/api/gesture/request
/api/gpt/request
/api/gpt/response
/api/motion_switcher/request
/api/motion_switcher/response
/api/obstacles_avoid/request
/api/obstacles_avoid/response
/api/pet/request
/api/pet/response
/api/programming_actuator/request
/api/programming_actuator/response
/api/robot_state/request
/api/robot_state/response
/api/sport/request
/api/sport/response
/api/sport_lease/request
/api/sport_lease/response
/api/uwbswitch/request
/api/uwbswitch/response
/api/videohub/request
/api/videohub/response
/api/vui/request
/api/vui/response
/arm_Command
/arm_Feedback
/audiohub/player/state
/audiosender
~/camera/camera_info
/camera/image_raw~
~/cmd_vel
/cmd_vel_joy
/cmd_vel_out~
/config_change_status
/diagnostics
/frontvideostream
/gas_sensor
/gesture/result
/gnss
/go2_states
/gpt_cmd
/gptflowfeedback
/imu
/joint_states
/joy
/joy/set_feedback
/lf/lowstate
/lf/sportmodestate
/lio_sam_ros2/mapping/odometry
/lowcmd
/lowstate
/multiplestate
~/odom~ /parameter_events
/pctoimage_local
/pet/flowfeedback
/point_cloud2
/pointcloud/aggregated
/pointcloud/downsampled
/pointcloud/filtered
/programming_actuator/command
/programming_actuator/feedback
/public_network_status
/qt_add_edge
/qt_add_node
/qt_command
/qt_notice
/query_result_edge
/query_result_node
/robot_description
/rosout
/rtc/state
/rtc_status
/scan
/selftest
/servicestate
/servicestateactivate
/sportmodestate
/tf
/tf_static
/uslam/client_command
/uslam/cloud_map
/uslam/frontend/cloud_world_ds
/uslam/frontend/odom
/uslam/localization/cloud_world
/uslam/localization/odom
/uslam/navigation/global_path
/uslam/server_log
/utlidar/client_cmd
/utlidar/cloud
/utlidar/cloud_base
/utlidar/cloud_deskewed
/utlidar/grid_map
/utlidar/height_map
/utlidar/height_map_array
/utlidar/imu
/utlidar/lidar_state
/utlidar/mapping_cmd
/utlidar/range_info
/utlidar/range_map
/utlidar/robot_odom
/utlidar/robot_pose
/utlidar/server_log
/utlidar/switch
/utlidar/voxel_map
/utlidar/voxel_map_compressed
/uwbstate
/uwbswitch
/videohub/inner
/webrtc_req
/webrtcreq
/webrtcres
/wirelesscontroller
/wirelesscontroller_unprocessed
/xfk_webrtcreq
/xfk_webrtcres
```

- 사실 위에는 기본적으로 나오지 않는 ros topic까지도 섞여있다. 나중에 제대로 된 토픽 list를 올리도록 하겠다. 내가 수집한 토픽은 아래와 같다.
  - /utlidar/robot_odom : unitree go2의 내장 lidar, imu, 다리 움직임을 통합하여 계산한 오도메트리이다. 아마 로봇 본체의 흔들림을 최소화 하였을 것 같다.
  - /frontvideostream : unitree ros2 msg 형태로 비디오 스트림이 인코딩되어서 들어온다. 720p, 360p, 180p의 영상이 stream으로 들어온다. (Annex-B 형태의 H.264 인코딩 비디오 payload)
  - /utlidar/cloud_deskewed : go2내부에서 돌아가는 slam map의 완성본을 저장하는 cmd를 찾지 못해서..이런 deskew 되고 있는 라이다 포인트를 받아와서 Odom에 맞추어 중첩해주었다..이렇게 임시방편으로 진행해서 그런지 결과가 썩 좋지는 않다..


### 실험 결과 

**lidar camera calibration**

![](/assets/img/posts/34acbb7d-7937-8081-8224-f83ee4127f9f.webp)

- 먼저 데이터셋에서 lidar와 카메라의 extrinsic을 한번 맞추어보았다. 보이는대로 조금씩 틀어져있는것을 확인할 수 있다.. 만약 제대로 fvdb를 활용하려면 아마 cailbration을 다시 맞추어야 할 것 같다.
```xml
## camera urdf
<joint name="front_camera_joint" type="fixed">
  <origin rpy="0 0 0" xyz="0.32715 -0.00003 0.04297" />
  <parent link="base" />
  <child link="front_camera" />
</joint>
## lidar urdf
<joint name="radar_joint" type="fixed">
  <origin xyz="0.28945 0 -0.046825" rpy="0 2.8782 0" />
  <parent link="base" />
  <child link="radar" />
</joint>
```

따라서 라이다에서 카메라 로의 수식은 아래와 같다. 

**T_camera_radar = inverse(T_base_camera) × T_base_radar**

$$
T_{CR}=T_{CB}^{-1}\times T_{BR}
$$



![](/assets/img/posts/34acbb7d-7937-80e3-a58e-c0888abff412.gif)

- 아티펙트도 많고.. fvdb에 사용하기 위해서는 undistort과정을 거치게 되는데.. 카메라의 intrinsic을 내가 남의 go2것으로 사용해서 그런 것 같다.. 




## 추후 시도 및 생각

- 아무래도 fvdb는 robot system의 정교함에 camera pose를 기대는 만큼 정교한 calibration과 계획적인 실험이 필요할 것 같다.
  - go2 robot의 camera & lidar calibration필요
  - fvdb 의 최적의 학습 spot 찾기
  - 계획적인 데이터셋 취득
- DiskChunGS와 Loger와 같은 large scale에서의 카메라 only (mono, stereo)방법을 계속 시도해볼지 아니면 fvdb를 신뢰할지 계획 수정필요
- 실내 8층 건물을 복원하는걸 milestone으로 잡는 것.








