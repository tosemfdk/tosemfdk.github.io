---
layout: post
title: "Intel realsense & mid360s ROS2 installation"
date: 2026-04-24T04:56:00.000Z
math: true
image:
  path: "/assets/img/posts/34ccbb7d-7937-801d-a0b5-e0bddbc5ac7a.gif"
categories:
  - "development tips"
---

## Livox SDK installation

[https://github.com/Livox-SDK/Livox-SDK2](https://github.com/Livox-SDK/Livox-SDK2)

- livox sdk2는 Mid-360s까지 지원한다. 


### 설치

```bash
git clone https://github.com/Livox-SDK/Livox-SDK2.git
cd ./Livox-SDK2/
mkdir build
cd build
cmake .. && make -j4 ## jetson에서는 빌드를 원활히 하기 위해서 core의 수를 제한한다.  
sudo make install
```



### 예제 실행

- 예제의 config file의 lidar_ip를 나의 lidar의 ip로 수정합니다. 이때 기본적으로 lidar의 시리얼 넘버의 맨 마지막 두자리수가 lidar의 ip 주소가 됩니다. 제꺼는 44로 끝났기 떄문에 192.168.1.144가 됩니다.
- (이미지는 다른 블로그에서 긁어왔습니다.)

```bash
# lidar ip 수정
nano /path/to/Livox-SDK2/samples/livox_lidar_quick_start/mid360s_config.json

cd /path/to/livox-SDK2

# 패킷이 들어오는지 확인합니다.
./build/samples/livox_lidar_quick_start/livox_lidar_quick_start  samples/livox_lidar_quick_start/mid360s_config.json
```

![](/assets/img/posts/34ccbb7d-7937-803c-94ee-e9c677a36d94.webp)

![](/assets/img/posts/34ccbb7d-7937-8067-8916-f4c82ade248b.webp)



- 먼저 위에 것을 진행하기 전에 내 host pc의 ip에 192.168.1.x와 소통할 수 있는 프로파일을 생성합니다.

```bash
sudo ip addr add 192.168.1.5/24 dev eno1 #저는 기존에 있던 프로파일인 eno1에 추가하였습니다.
```



## livox ros2 driver installation

- 저는 jetson에 humble을 미리 설치한 후에 진행하였습니다.
[https://github.com/Livox-SDK/livox_ros_driver2.git](https://github.com/Livox-SDK/livox_ros_driver2.git)

```bash
git clone https://github.com/Livox-SDK/livox_ros_driver2.git ws_livox/src/livox_ros_driver2

./build.sh humble

cd /path/to/ws_livox
source install/setup.bash

# 여기서 ip setting을 아래와 같이 변경
vim install/livox_ros_driver2/share/livox_ros_driver2/config/MID360s_config.json

```

```json
{
  "lidar_summary_info" : {
    "lidar_type": 8
  },
  "Mid360s": {
    "lidar_net_info" : {
      "cmd_data_port"  : 56100,
      "push_msg_port"  : 56200,
      "point_data_port": 56300,
      "imu_data_port"  : 56400,
      "log_data_port"  : 56500
    },
    "host_net_info" : [
      {
        "host_ip"        : "192.168.1.5",
        "cmd_data_port"  : 56101,
        "push_msg_port"  : 56201,
        "point_data_port": 56301,
        "imu_data_port"  : 56401,
        "log_data_port"  : 56501
      }
    ]
  },
  "lidar_configs" : [
    {
      "ip" : "192.168.1.144", // 내 lidar ip
      "pcl_data_type" : 1,
      "pattern_mode" : 0,
      "extrinsic_parameter" : {
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": 0.0,
        "x": 0,
        "y": 0,
        "z": 0
      }
    }
  ]
}

```



이렇게하고 launch 파일을 실행하면 ros2 topic이 잘 나오는 것을 확인할 수 있습니다.

```bash
ros2 launch livox_ros_driver2 rviz_MID360s_launch.py

ros2 topic list
```

![](/assets/img/posts/34ccbb7d-7937-8082-8cd9-ffdd425fb5be.webp)

![](/assets/img/posts/34ccbb7d-7937-801d-a0b5-e0bddbc5ac7a.gif)





## intelrealsense 설치

<div class="notion-video-embed" style="margin:1.25rem 0;">
<div style="position:relative;padding-bottom:56.25%;height:0;overflow:hidden;border-radius:12px;">
<iframe src="https://www.youtube.com/embed/FpMCJsg_KmE" title="Embedded video" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen style="position:absolute;top:0;left:0;width:100%;height:100%;border:0;"></iframe>
</div>
</div>

- 이분이 jetson관련된 컨텐츠를 상당히 많이 찍으시는데 나는 그냥 믿고 본다 5:18 초부터의 설치내용을 따랐다.


### realsense에 맞는 커널 빌드 (jetpack 6.2 미포함)

```bash
git clone https://github.com/jetsonhacks/jetson-orin-librealsense.git

cd jetson-orin-librealsense

sha256sum -c install-modules.tar.gz.sha256
tar -xzf install-modules.tar.gz
cd install-modules

sudo ./install-realsense-modules.sh


```

- 커널 모듈을 깔면 이러한 장점이 있다고 한다.
  - **심층 프레임 형식 지원(Depth-related streaming formats)**: RealSense 카메라의 고유한 깊이(Depth) 데이터를 원활하게 스트리밍할 수 있도록 커널 수준에서 호환성을 확보합니다.
  - **프레임 메타데이터 접근(Per-frame metadata attributes)**: 프레임마다 포함된 상세 정보를 추출할 수 있게 되어, 더 정밀한 AI 비전 처리가 가능해집니다 
  - **IMU 센서 지원**: 가속도계와 자이로스코프(Gyroscope) 같은 관성 측정 장치(IMU)를 시스템이 인식하고 사용할 수 있도록 합니다. 이 모듈들이 없으면 카메라의 모션 데이터를 활용할 수 없습니다.
  - **네이티브 성능 최적화**: 커널 수준에서 드라이버를 직접 패치하고 로드하기 때문에, 프로덕션 환경에서 가장 **안정적이고 높은 성능**을 기대할 수 있습니다.


```bash
# gpg key add
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys FB0B24895113F120

sudo add-apt-repository "deb https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main" -u
# intelrealsense 설치
sudo apt-get install librealsense2-utils
sudo apt-get install librealsense2-dev

# viewer로 firmware update & 장치 인식
realsense-viewer
```



![](/assets/img/posts/34ccbb7d-7937-8074-b047-d53d1e35f180.webp)

