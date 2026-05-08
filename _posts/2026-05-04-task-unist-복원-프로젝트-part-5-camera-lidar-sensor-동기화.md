---
layout: post
title: "Task[Unist 복원 프로젝트] Part 5. Camera Lidar Sensor 동기화"
date: 2026-05-04T05:31:00.000Z
math: true
image:
  path: "/assets/img/posts/356cbb7d-7937-8074-86c6-d3f19f3a7d7c.webp"
categories:
  - "UNIST"
---

# 데이터 동기화

- 데이터 동기화란 여러가지 센서 값을 한번에 처리할때 외부에서 harware trigger를 지원하는 센서같은 경우에는 하나의 **MCU로 외부 입력을 주어 동일한 시간에 센서가 값을 내보내도록 할 수 있다**. 나같은 경우에 사용하는 **mid360s나 d455, hikrobot같은 머신비전 카메라는 외부 트리거를 지원**한다.


### line configure

![](/assets/img/posts/356cbb7d-7937-8074-86c6-d3f19f3a7d7c.webp)

![](/assets/img/posts/356cbb7d-7937-809f-89cf-f6c42f3de51f.webp)

![](/assets/img/posts/356cbb7d-7937-8073-a650-c78ad706117a.webp)





## lidar

- mid360s 는 PTP 기술을 지원한다.  Master와 slave의 시계를 같은 timeline으로 맞추는 기술이라고 생각하면 될 것 같다. RJ45 케이블이 연결되어있으면 하드웨어에 있는 시계가 PTP 패킷이 오가는 순간에 timestamp를 찍어 매우 강건하게 시간 동기화를 할 수 있다. (orin nano는 지원을 안하기 때문에 software PTP로 진행하다. 소프트웨어적 오차가 반영된다.)
  ![](/assets/img/posts/356cbb7d-7937-802c-83a8-cd794f25ee4f.webp)
  
  - 따라서 mid360s에서 나오는 timestamp는 호스트 clock과 어느정도 시간이 맞는다고 볼 수 있다.
- 8번 pin에 PPS신호를 쏴주어 매 초의 경계를 라이다에게 알려주어 시간축이 틀어지는 현상을 방지한다.
  - 카메라와 다르게 PPS로 신호가 들어갈때 데이터를 찍어내는것이 아니므로 주의해야한다.
## Camera

- 카메라도 PTP를 지원하는 기종이 있지만 내 카메라는 지원하지 않기 떄문에 external hardware trigger를 통해 10hz의 pwm신호를 쏴주어 맞추게 된다. 이때 jetson orin으로 interupt가 발생했을때의 시간을 패킷으로 보내어 현재 ros time과 아두이노 time의 차이를 어느정도 유추할 수 있다.


## Arduino

<details markdown="1">
<summary>code</summary>

```json
// Arduino GIGA R1
// USB-C Serial: Jetson/Mini PC와 통신
// D2: Camera trigger
// D3: Optional LiDAR PPS 1Hz

const int CAM_TRIG_PIN = 2;
const int LIDAR_PPS_PIN = 3;

bool running = false;
bool use_lidar_pps = false;

uint32_t cam_period_us = 100000;  // 10 Hz
uint32_t last_cam_us = 0;
uint32_t last_pps_us = 0;

uint32_t cam_seq = 0;
uint32_t pps_seq = 0;

void pulsePin(int pin, uint32_t high_us) {
  digitalWrite(pin, HIGH);
  delayMicroseconds(high_us);
  digitalWrite(pin, LOW);
}

void setup() {
  pinMode(CAM_TRIG_PIN, OUTPUT);
  pinMode(LIDAR_PPS_PIN, OUTPUT);
  digitalWrite(CAM_TRIG_PIN, LOW);
  digitalWrite(LIDAR_PPS_PIN, LOW);

  Serial.begin(115200);
  while (!Serial) {}

  Serial.println("GIGA_TRIGGER_READY");
}

void handleCommand(String cmd) {
  cmd.trim();

  if (cmd == "START") {
    running = true;
    Serial.println("OK START");
  } else if (cmd == "STOP") {
    running = false;
    Serial.println("OK STOP");
  } else if (cmd.startsWith("FPS ")) {
    float fps = cmd.substring(4).toFloat();
    if (fps > 0.1 && fps <= 60.0) {
      cam_period_us = (uint32_t)(1000000.0 / fps);
      Serial.print("OK FPS ");
      Serial.println(fps);
    }
  } else if (cmd == "PPS_ON") {
    use_lidar_pps = true;
    Serial.println("OK PPS_ON");
  } else if (cmd == "PPS_OFF") {
    use_lidar_pps = false;
    Serial.println("OK PPS_OFF");
  }
}

void loop() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    handleCommand(cmd);
  }

  if (!running) return;

  uint32_t now = micros();

  // Camera trigger: 10~15Hz 등 원하는 fps
  if ((uint32_t)(now - last_cam_us) >= cam_period_us) {
    last_cam_us = now;

    // 이 줄을 Jetson이 읽고 host time으로 timestamp를 찍음
    Serial.print("CAM,");
    Serial.print(cam_seq++);
    Serial.print(",");
    Serial.println(now);

    pulsePin(CAM_TRIG_PIN, 100);  // 100 us pulse
  }

  // LiDAR PPS: 선택사항. 쓸 거면 반드시 1Hz
  if (use_lidar_pps && (uint32_t)(now - last_pps_us) >= 1000000UL) {
    last_pps_us = now;

    Serial.print("PPS,");
    Serial.print(pps_seq++);
    Serial.print(",");
    Serial.println(now);

    pulsePin(LIDAR_PPS_PIN, 100);  // PPS high > 1us 조건 만족
  }
}
```



</details>

- 아두이노의 GND와 라이다, 카메라의 GND를 묶음어야지만 트리거 신호가 잘 전달된다.


## Tips

- 아래와 같이 livox의 **custom message를 통해 라이다 토픽의 timestamp가 결국 첫번째 포인트의 ros 시간대**에 불과하고 이후에 **각 포인트마다 몇 초의 시간 offset이 흘렀는지를 확인**할 수 있게 된다. 이는 매우 중요한데 livox lidar는 imu또한 캘리브레이션 되어있기때문에 **imu topic의 값을 통해 lidar point의 각 offset 마다의 움직임을 보간**하여 **한 시점에서의 descew된 pointcloud로 복원**할 수 있게 된다. 이러한 포인트를 **특정한 시점의 이미지에 투영**하면 되기 때문에 카메라와 Lidar HZ를 맞추는 것은 매우 중요하다.
- 또한 라이다 토픽의 주기를 10hz로 하였을때 한 Topic이 담고있는 포인트들의 Offset을 전부 더하면 100ms가 된다.

```json
jetson@loe:~$ ros2 topic echo /livox/lidar --once
header:
  stamp:
    sec: 1777892189
    nanosec: 400149246
  frame_id: livox_frame
timebase: 1777892189400149246
point_num: 20064
lidar_id: 192
rsvd:
- 0
- 0
- 0
points:
- offset_time: 0
  x: -2.1410000324249268
  y: -1.5369999408721924
  z: 2.4860000610351562
  reflectivity: 88
  tag: 0
  line: 0
- offset_time: 4947
  x: -2.0759999752044678
  y: -1.399999976158142
  z: 2.5390000343322754
  reflectivity: 77
  tag: 0
  line: 1
- offset_time: 9894
  x: -2.006999969482422
  y: -1.2589999437332153
  z: 2.5799999237060547
  ...
```

![](/assets/img/posts/356cbb7d-7937-80bc-b5f9-ed6104a30b78.webp)

