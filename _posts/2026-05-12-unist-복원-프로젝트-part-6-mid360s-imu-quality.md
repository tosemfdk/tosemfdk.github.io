---
layout: post
title: "[Unist 복원 프로젝트] Part 6. Mid360s IMU quality"
date: 2026-05-12T03:29:00.000Z
math: true
image:
  path: "/assets/img/posts/35ecbb7d-7937-8043-ab58-e45d4ab3bdad.webp"
categories:
  - "UNIST"
---



# IMU quality 분석

IMU의 퀄리티를 분석하는 것으로는 크게 정적 분석과 동적인 분석으로 나눌 수 있게 된다.

정적인 분석으로는 아래와 같은 분석이 있다. 

## 정적분석



 **노이즈 레벨(White Noise) 확인:**

움직임이 전혀 없는데도 가속도나 자이로 값이 계속 변한다면 노이즈가 심한 것이다.. 보통 raw 데이터를 그래프(Plot)로 띄워놓고 진폭이 얼마나 되는지 확인한다.

**바이어스(Bias) 측정:**
수평이 완벽한 곳에 뒀을 때 가속도 $z$축이 $9.81m/s^2$ (혹은 $1g$)에 가깝게 나오는지, 그리고 자이로스코프의 3축 값이 모두 0에 수렴하는지 확인해본다. 0에서 벗어난 정도가 **'Bias'**이며, 이게 크면 나중에 로봇이 가만히 있어도 한쪽으로 흐르는 **Drift** 현상이 발생하게된다.

- IMU의 예제 데이터
  - IMU topic
    ```bash
    ros2 topic echo /livox/imu --once
    header:
      stamp:
        sec: 1778499862
        nanosec: 765184222
      frame_id: livox_frame
    orientation:
      x: 0.0
      y: 0.0
      z: 0.0
      w: 1.0
    orientation_covariance:
    - 0.0
    - 0.0
    - 0.0
    - 0.0
    - 0.0
    - 0.0
    - 0.0
    - 0.0
    - 0.0
    angular_velocity:
      x: -0.03331834077835083
      y: 0.015504890121519566
      z: 0.004326723515987396
    angular_velocity_covariance:
    - 0.0
    - 0.0
    - 0.0
    - 0.0
    - 0.0
    - 0.0
    - 0.0
    - 0.0
    - 0.0
    linear_acceleration:
      x: -0.01059205736964941
      y: -0.0040344963781535625
      z: 0.9981179237365723
    linear_acceleration_covariance:
    - 0.0
    - 0.0
    - 0.0
    - 0.0
    - 0.0
    - 0.0
    - 0.0
    - 0.0
    - 0.0
    ---
    ```
  - 수평으로 가만히 놓았을때 가속도의 z값이 1g에 가깝게 나오는 것을 확인할 수 있다.
  - 움직임이 없기 때문에 각속도 성분이 모두 0에 가깝게 나오는 것을 확인할 수 있다.


**Allan Variance (알란 분산) 분석:**
약 1시간 이상 데이터를 가만히 받아 Allan Variance 그래프를 그려보는 것이 좋다. 이를 통해 해당 센서의 **Random Walk**와 **Bias Instability** 같은 핵심 파라미터를 정밀하게 추출할 수 있다.

- 2시간 정도의 정적 데이터
  - bag info 
    ```bash
    ros2 bag info imu_2h/
    
    Files:             imu_2h_0.db3
    Bag size:          532.6 MiB
    Storage id:        sqlite3
    Duration:          7109.139410828s
    Start:             May 11 2026 07:15:31.464983152 (1778498131.464983152)
    End:               May 11 2026 09:14:00.604393980 (1778505240.604393980)
    Messages:          1421805
    Topic information: Topic: /livox/imu | Type: sensor_msgs/msg/Imu | Count: 1421805 | Serialization Format: cdr
    
    ```
    
    - 1421805개의 메세지로 초당 토픽의 개수가 약 199.998hz에 달하는 것을 확인할 수 있다.


- Allan Variance
  ![](/assets/img/posts/35ecbb7d-7937-8043-ab58-e45d4ab3bdad.webp)
  
  - 각 축마다 편차가 있지만 (z축의 시간 당 noise가 가장 작다) 11.5deg/hour 의 틀어짐이 있다고 볼 수 있다. 
  - 시간이 지남에 따라서 각도 오차가 심각하게 벌어지게 되는것을 확인할 수 있다.
  - $\text{RMS} = {\sqrt{\sigma_x^2+\sigma_y^2+\sigma_z^2\over3 }}$ 이다.




- 다른 IMU의 Allan 분석
![](/assets/img/posts/35ecbb7d-7937-8040-8880-c18c5e0d5f5f.webp)

![](/assets/img/posts/35ecbb7d-7937-8050-b65e-d29cc7ab2e21.webp)

- blue : Vi-Sensor, ADIS16448, `200Hz`
- red : 3dm-Gx4, `500Hz`
- green : DJI-A3, `400Hz`
- black : DJI-N3, `400Hz`
- circle : xsens-MTI-100, `100Hz`
![](/assets/img/posts/35ecbb7d-7937-80fb-9b35-c33c4852a0e4.webp)



## 동적분석

- 동적 분석에서는 특정 축만 회전 시켜서 다른 축에 변화가 있는지 확인하는 실험
  - 테스트시에 특정 축만 흔들어주는것이 거의 불가능 하므로 일단은 포기한다.
- imu를 원점에서 시작해서 공중에서 8자로 마구 흔든 후에 다시 원점에 복귀했을때 0값에서 얼마나 틀어졌는지 확인하는 원점 회귀 분석이 있다. 
  - 기준점을 두고 공중에서 livox lidar를 8자로 흔들어보았다.
  ![](/assets/img/posts/35ecbb7d-7937-8008-b1bc-f88ce57c213e.gif)
  
  - 이후 plot juggler를 통해서 적분 필터를 씌워 실제 각도의 drift가 얼마나 발생하였는지를 확인하였다.
    ![](/assets/img/posts/35ecbb7d-7937-8054-ad4f-f4bdaf0c17fc.webp)
    
    - **x축으로 3도, y축으로 0.5도, z축으로 9도 정도 틀어짐**을 확인할 수 있었다. 20초 동안 흔들었음에도 최대 9도까지밖에 틀어지지 않은것은 보정이 없을때의 **적분오차가 발생함을 감안하면 상당히 괜찮은 결과**라고 생각한다.


