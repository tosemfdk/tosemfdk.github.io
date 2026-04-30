---
layout: post
title: "[Unist 복원 프로젝트] Part 4 Sensor Rig calibration"
date: 2026-04-28T05:28:00.000Z
math: true
image:
  path: "/assets/img/posts/352cbb7d-7937-8027-9f04-da9c71071dca.gif"
categories:
  - "Unist"
---

# Sensor & Rig

![](/assets/img/posts/351cbb7d-7937-806c-bb86-db07c5dd2d4e.webp)

### mid 360s

- 고정형 라이다로 회전형라이다와 다르게 비 **반복적인 pattern**이 나타나는 것이 특징이다.
- livox sdk2, ros2 패키지를 설치하여 토픽을 확인할 수 있으며 **PTP와 외부 line을 통해 데이터 동기화**를 지원한다.


## D435f

- intel realsense의 depth camera이며 mono rgb와 양쪽 적외선 센서의 disparity를 활용한 depth 출력이 있다.
- 카메라 위의 몰렉스 단자를 통해서 1번 pin을 통해 외부트리거 데이터 동기화를 지원하지만 하지만 이는 **depth 출력에대해서만 지원한다…** 조금 더 상위 버전인 d455계열에서는 depth & rgb 싱크를 지원하기때문에 카메라를 교체할 필요가 있다.
## RIG

- 센서들의 마운트로 현재는 jetson이나 선들이 들어갈 자리가 없기 때문에 해당 사항을 반영하고 프로파일로 두손으로 잡을 수 있게 하여 추후 다시 제작하려고 한다. 
  




# Calibration

## Intrinsic calibration

[https://github.com/rameau-fr/MC-Calib](https://github.com/rameau-fr/MC-Calib)

- 해당 패키지를 사용하여 realsense viewer에서 이미지를 약 30장 정도 수집하여 calibration 하였다.
### 보드 generation

![](/assets/img/posts/351cbb7d-7937-8085-8f57-c2d8c522e976.webp)

- docker 안에서 mc-calib 패키지를 빌드하고 board generation코드를 통해서  5 by 7 보드를 작성하고 프린트하였다.
<details markdown="1">
<summary>config</summary>

- 측정결과 박스의 길이는 29cm, 아르코마커의 크기는 22cm였다.

```yaml

######################################## Boards Parameters ###################################################
# [중요] 출력할 보드의 칸 수입니다. (보통 가로가 더 길게 만듭니다)
number_x_square: 7  
number_y_square: 5  
resolution_x: 1000  # 출력용이므로 해상도를 높였습니다.
resolution_y: 1000  
length_square: 0.029  
length_marker: 0.022 
number_board: 1     # 카메라 1대이므로 보드 1개면 충분합니다.
boards_index: []    # 비워두면 0번 인덱스 보드를 사용합니다.

# [중요] 출력 후 자로 잰 실제 사각형의 한 변 길이 (예: 4.0cm면 4.0 입력)
square_size: 0.029  

############# Boards Parameters for different board size #################
number_x_square_per_board: []
number_y_square_per_board: []
square_size_per_board: []

######################################## Camera Parameters ###################################################
distortion_model: 0  # 리얼센스 RGB는 Brown(0) 모델을 씁니다.
distortion_per_camera : [] 
number_camera: 1     # ### 1로 수정 (카메라 1대)
refine_corner: 1  
min_perc_pts: 0.3    # ### 0.3으로 하향 (보드가 살짝 가려져도 인식되게 함)

cam_params_path: "None" 
fix_intrinsic: 0     # 내측 파라미터를 새로 구할 것이므로 0 유지

######################################## Images Parameters ###################################################
# [중요] 실제 사진이 들어갈 경로 (미리 폴더를 만들어두세요)
root_path: "../data/my_realsense/images" 
cam_prefix: "Cam_"
# 처음엔 "None"으로 두세요. (이미 특징점을 뽑아둔 게 아니기 때문)
keypoints_path: "None"  

######################################## Optimization Parameters ###################################################
quaternion_averaging: 1  
ransac_threshold: 5      # 너무 낮으면 인식이 까다로우니 5 정도로 완화
number_iterations: 1000  

######################################## Hand-eye method #############################################
he_approach: 0 

######################################## Output Parameters ###################################################
# [중요] 결과 파일이 저장될 위치
save_path: "../data/my_realsense/output"
save_detection: 1     # 인식 결과를 눈으로 확인하기 위해 1로 설정
save_reprojection: 1  # 보정 결과를 확인하기 위해 1로 설정
camera_params_file_name: "realsense_intrinsic.yml"
```



</details>



### 결과

![](/assets/img/posts/351cbb7d-7937-80a2-9560-d67c6fc545c0.webp)

```yaml
nb_camera: 1
camera_0:
   camera_matrix: !!opencv-matrix
      rows: 3
      cols: 3
      dt: d
      data: [ 602.67991775128519, 0., 321.40607074071954, 0.,
          603.24811474916351, 255.87074469949764, 0., 0., 1. ]
   distortion_vector: !!opencv-matrix
      rows: 1
      cols: 5
      dt: d
      data: [ 0.030677480758436598, 0.50569847109540544,
          0.0023319303176375468, -0.002259486639755007,
          -1.78567003855249 ]
   distortion_type: 0
   camera_group: 0
   img_width: 640
   img_height: 480
   camera_pose_matrix: !!opencv-matrix
      rows: 4
      cols: 4
      dt: d
      data: [ 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
          1. ]
```

- 결과적으로 3d 로 추정한 point의 projection error도 매우 낮게 나왔고 카메라의 해상도, 주점, 초점거리, 왜곡 계수등을 알수있게 되었다.




## lidar-camera calibration

- lidar based calibration은 colored pointcloud를 만들거나 camera pose를 구할때 매우 정밀할 필요가 있다.
- 또한 lidar가 비반복적인 패턴을 가지는 livox mid 360s lidar이기 때문에 general한 lidar에서 전부 통용되는 calibration 방식이 필요했다.


### Automatic Extrinsic Calibration of a Camera and a 3D LiDAR using Line and Plane Correspondences

[https://publications.ri.cmu.edu/storage/publications/2018/09/Zhou18iros.pdf](https://publications.ri.cmu.edu/storage/publications/2018/09/Zhou18iros.pdf)

- lidar와 카메라의 강체변환(6dof)을 체커 캘리브레이션 보드의 plane과 가장 자리의 4개 line을 구하고 이를 이미지의 line과 맞추는 식으로 제약 조건을 건다. 




### FAST-Calib: LiDAR-Camera Extrinsic Calibration in One Second

[https://arxiv.org/pdf/2507.17210](https://arxiv.org/pdf/2507.17210)

- 내부의 4개의 구멍이 뚫린 캘리브레이션 보드에 동그란 구멍을 뚫어 라이다에서 해당 원을 인식하게 하여 중심의 좌표와 카메라의 픽셀을 매칭하는 2d-3d correspondence를 4개 구할 수 있게 된다. 


위의 두 논문을 참조하여 일반 체커보드에서도 모든 lidar에 사용가능한 캘리브레이션 tool을 공부도 할 겸 제작해보겠다.



### Repository



### 데이터 수집

![](/assets/img/posts/351cbb7d-7937-80ca-a189-ce331d030878.webp)

![](/assets/img/posts/351cbb7d-7937-802f-b57d-c55dd1a02675.webp)

- 이렇게 이미지에 꽉 차는 한도에서 캘리브레이션 보드를 RIG와 최대한 가깝게 하여 총 9set의 pcd, png format의 데이터셋을 취득하였습니다.
- 데이터 set별로 plane의 방향을 변경하기 위헤 좌우로 기울이거나 바닥에 받침을 넣어 위아래로 기울이면서 수집한 데이터셋의 다양성을 확보하였습니다.


**촬영된 png file**

![](/assets/img/posts/351cbb7d-7937-80b5-bc42-d0b6e53d5413.webp)

![](/assets/img/posts/351cbb7d-7937-8028-ae8b-efc493045002.webp)



**100개의 topic을 중첩한 lidar PCD file**

![](/assets/img/posts/351cbb7d-7937-8034-8213-f3be5ab5e829.webp)



lidar 데이터 실시간 compression..

## image processing



Undistorting

먼저 line의 직선성을 확보하기 위해서 왜곡이 되어있는 이미지를 undistort합니다.

![](/assets/img/posts/351cbb7d-7937-8006-8f52-dd0e97eb1d54.webp)

![](/assets/img/posts/351cbb7d-7937-80b9-84e9-ebba3239c896.webp)



- 먼저 체커보드는 사각형의 size가 10cm, 흰색 테두리의 길이가 7.5cm인 7x10 보드이다.
- 먼저 이미지의 체커보드상의 inner coner를 찾고 이 점들은 간격과 line이라는 점을 알고 있기 때문에 PnP 알고리즘을 통해 점들의 3d 좌표를 알아낼 수 있게 되고 체커보드의 Camera coordinate상의 평면을 구할 수 있게 된다.
  ![](/assets/img/posts/351cbb7d-7937-80a7-92a6-cd26bffe009e.webp)
- 이후 알고있는 카메라 보드의 prior를 통해 실제 보드의 outerline을 다시 projection하여 추측할 수 있게 됩니다. 
  ![](/assets/img/posts/351cbb7d-7937-80cb-be28-df3ab8b7f789.webp)


### lidar processing

![](/assets/img/posts/351cbb7d-7937-800f-9def-d4f2db7d03f9.webp)

100개의 lidar packet이 중첩된 라이다를 살펴볼때 intensity를 기준으로 물체를 파악할 수 있으므로 굉장히 dense하게 장면이 표현된 것을 확인할 수 있다. 



**Passthrough filter**

먼저 좌표축 기준으로 아래와 같은 기준으로 포인트를 필터링 해보았다.

- **x축 : [0, 3], y축 : [-1, 1] , z축 : [-2, 2]**
![](/assets/img/posts/352cbb7d-7937-803a-a8ca-ce84604dde22.webp)

![](/assets/img/posts/351cbb7d-7937-80e0-970d-e3796224466d.webp)



**RANSAC + line filtering**

- 이후 ransac 알고리즘을 통해 평면을 추출할 수 있는데 **ransac에 평면이 두개라는 prior를 주고 찾아낸 두 평면**중에서 inlier와 원점과의 거리를 비교하여 가장 먼 포인트를 가지는 평면을 바닥이라고 휴리스틱하게 생각**하여 제거**한다.
![](/assets/img/posts/351cbb7d-7937-806e-b1dd-d2e8ca8cc97b.webp)



- 이후에 남겨진 체커보드의 노멀방향으로 다시 passthrough filter를 적용한다.
  - 여기서 threshold를 정하는 기준은 평면의 median값을 기준으로 inlier들의 편차를 구하고 그 편차들의 median값을 기준으로하여 3배의 거리값으로 잡는다 이렇게 medain값을 활용하면 일부의 outlier에 기준 threshold가 끌려가는 일을 방지할 수 있게 된다.
![](/assets/img/posts/351cbb7d-7937-807a-a403-ee7a3d6f94e7.webp)

![](/assets/img/posts/351cbb7d-7937-8045-b66c-f99a0b567200.webp)



- 이후에 캘리브레이션 보드의 line에 해당하는 포인트를 구해야 한다. 이를 위해 fast-calib에서 사용한 **world Z축 기준 projecting 후에 어떤 점의 3cm이내에서 서로 다른 두 포인트**를 찾았을때 그 포인트들의 **최소각도가 100도** 이상을 넘어가면 그 포인트는 line이라고 판단하게 된다. 이후 선4개를 2d ransac
  ![](/assets/img/posts/352cbb7d-7937-8076-b7c7-e59b12079852.webp)
![](/assets/img/posts/352cbb7d-7937-80d0-b0e5-ef0e0e4534f4.webp)

![](/assets/img/posts/352cbb7d-7937-80bd-a90e-cab2ea84be78.webp)





### constraint

이미지 상의 체커보드는 PnP알고리즘을 통해 평면으로 복원이 가능한 상태이므로 해당 평면 위의 테두리 line 4개 또한 3차원 구조로 복원이 가능하다.  따라서 lidar coordinate의 line에 해당하는 점을 강체변환 $[R,t]$ 하면 line의 방정식에 겹쳐져야 한다. 또한 lidar 좌표계에서의 평면 법선벡터 또한 Rotation후 카메라좌표계의 평면의 법선벡터와 일치해야한다.

$$
 R_L^C n_i^L = n_i^C
$$

- 회전후에 평면의 법선벡터가 일치해야 한다. 
$$
R_L^C d_{ij}^L = d_{ij}^C
$$

- 회전후에 선의 방향벡터가 일치해야한다.
$$
(n_i^C)^T \left( R_L^C P_{im}^L + t_L^C \right) + d_i^C = 0
$$

- 강체변환 후에 lidar point가 평면 위에 위치한다.
$$
\left( I - d_{ij}^C (d_{ij}^C)^T \right) \left( R_L^C Q_{ijk}^L + t_L^C - P_{ij}^C \right) = \mathbf{0}_{3 \times 1}
$$

- 평면의 가장자리에 있는 edge lidar point가 카메라 프레임의 해당 line에 일치해야 한다.





이러한 제약조건을 통해서 비선형 최적화를 통해서 6DOF의 강체변환을 찾아낼 수 있다. 이때 다른 데이터셋에서 결과를 더 찾아낸 뒤 더욱 정밀하게 최적화 할 수도 있다. 

## 최종결과

<details markdown="1">
<summary>extrinsic calibration matrix</summary>

```json
{
  "datasets": [
    "lidar_calib_1",
    "lidar_calib_3",
    "lidar_calib_6",
    "lidar_calib_9"
  ],
  "convention": "X_camera = R_camera_lidar @ X_lidar + t_camera_lidar",
  "initialization": {
    "source": "mean of per-dataset single-pose extrinsics",
    "R_initial": [
      [
        -0.0048061490359144745,
        -0.9999882059490377,
        -0.0006992097469291688
      ],
      [
        0.499783244488621,
        -0.0017964343685613479,
        -0.8661486485309476
      ],
      [
        0.8661371770452253,
        -0.0045122928079937286,
        0.4997859839508738
      ]
    ],
    "t_initial_m": [
      0.040013180656062124,
      -0.05601998581840048,
      -0.02236459929256138
    ]
  },
  "extrinsic_camera_from_lidar": {
    "R": [
      [
        -0.008362180201372804,
        -0.999953702093284,
        0.004761051587161502
      ],
      [
        0.4993563392372271,
        -0.008300862075517113,
        -0.8663569369217278
      ],
      [
        0.8663563472416389,
        -0.0048671715337643695,
        0.49940263338851726
      ]
    ],
    "t_m": [
      0.04072979539322377,
      -0.05562777172574635,
      -0.02256087729880189
    ],
    "rvec_rodrigues": [
      0.8170976140347784,
      -0.8171977063776494,
      1.4220513193419562
    ],
    "T_4x4": [
      [
        -0.008362180201372804,
        -0.999953702093284,
        0.004761051587161502,
        0.04072979539322377
      ],
      [
        0.4993563392372271,
        -0.008300862075517113,
        -0.8663569369217278,
        -0.05562777172574635
      ],
      [
        0.8663563472416389,
        -0.0048671715337643695,
        0.49940263338851726,
        -0.02256087729880189
      ],
      [
        0.0,
        0.0,
        0.0,
        1.0
      ]
    ]
  },
  "extrinsic_lidar_from_camera": {
    "R": [
      [
        -0.008362180201372804,
        0.4993563392372271,
        0.8663563472416389
      ],
      [
        -0.999953702093284,
        -0.008300862075517113,
        -0.0048671715337643695
      ],
      [
        0.004761051587161502,
        -0.8663569369217278,
        0.49940263338851726
      ]
    ],
    "t_m": [
      0.04766442958469282,
      0.04015634356852685,
      -0.03712046104252349
    ],
    "T_4x4": [
      [
        -0.008362180201372804,
        0.4993563392372271,
        0.8663563472416389,
        0.04766442958469282
      ],
      [
        -0.999953702093284,
        -0.008300862075517113,
        -0.0048671715337643695,
        0.04015634356852685
      ],
      [
        0.004761051587161502,
        -0.8663569369217278,
        0.49940263338851726,
        -0.03712046104252349
      ],
      [
        0.0,
        0.0,
        0.0,
        1.0
      ]
    ]
  },
  "optimization": {
    "method": "joint numeric Gauss-Newton/LM over all dataset point-to-plane, point-to-line, and vector-alignment residuals",
    "iterations": 5,
    "history": [
      {
        "iteration": 0,
        "cost": 0.5049365414815065,
        "rms": 0.005717370378268873
      },
      {
        "iteration": 1,
        "cost": 0.4940143522746105,
        "rms": 0.005655196630836511,
        "accepted": true,
        "step_norm": 0.006750044827729621,
        "damping": 4e-05
      },
      {
        "iteration": 2,
        "cost": 0.49400873007730856,
        "rms": 0.005655164450879034,
        "accepted": true,
        "step_norm": 0.00016437561794220391,
        "damping": 1.6000000000000003e-05
      },
      {
        "iteration": 3,
        "cost": 0.494008726948166,
        "rms": 0.005655164432968605,
        "accepted": true,
        "step_norm": 3.904814330779087e-06,
        "damping": 6.400000000000001e-06
      },
      {
        "iteration": 4,
        "cost": 0.4940087269468828,
        "rms": 0.00565516443296126,
        "accepted": true,
        "step_norm": 9.3220536902031e-08,
        "damping": 2.560000000000001e-06
      },
      {
        "iteration": 5,
        "cost": 0.4940087269468816,
        "rms": 0.005655164432961256,
        "accepted": true,
        "step_norm": 3.4848571745603715e-11,
        "damping": 0.01600000000000001
      }
    ],
    "weights": {
      "plane_weight": 1.0,
      "line_weight": 1.0,
      "vector_weight": 2.0
    },
    "sample_counts": {
      "lidar_calib_1": {
        "board_points": 6000,
        "line_points": {
          "edge_line_1": 148,
          "edge_line_2": 147,
          "edge_line_3": 143,
          "edge_line_4": 101
        }
      },
      "lidar_calib_3": {
        "board_points": 6000,
        "line_points": {
          "edge_line_1": 178,
          "edge_line_2": 179,
          "edge_line_3": 146,
          "edge_line_4": 112
        }
      },
      "lidar_calib_6": {
        "board_points": 6000,
        "line_points": {
          "edge_line_1": 188,
          "edge_line_2": 156,
          "edge_line_3": 152,
          "edge_line_4": 103
        }
      },
      "lidar_calib_9": {
        "board_points": 6000,
        "line_points": {
          "edge_line_1": 167,
          "edge_line_2": 146,
          "edge_line_3": 129,
          "edge_line_4": 83
        }
      }
    }
  },
  "per_dataset": {
    "lidar_calib_1": {
      "dataset_dir": "lidar_calib/lidar_calib_1",
      "line_assignment_camera_to_lidar": {
        "top": "edge_line_2",
        "right": "edge_line_3",
        "bottom": "edge_line_4",
        "left": "edge_line_1"
      },
      "line_correspondence": {
        "top": {
          "camera_target_line": "top",
          "lidar_fitted_line": "edge_line_2",
          "camera_length_m": 0.8500000089406966,
          "lidar_length_m": 0.8581813133995057,
          "length_error_m": 0.008181304458809047,
          "point_to_line_rms_m": 0.011655763201932283,
          "point_to_line_p95_m": 0.01650459571172292,
          "point_count": 147
        },
        "right": {
          "camera_target_line": "right",
          "lidar_fitted_line": "edge_line_3",
          "camera_length_m": 1.1500000208616257,
          "lidar_length_m": 1.1597767474630047,
          "length_error_m": 0.009776726601379071,
          "point_to_line_rms_m": 0.01683714210884253,
          "point_to_line_p95_m": 0.02391144129994102,
          "point_count": 143
        },
        "bottom": {
          "camera_target_line": "bottom",
          "lidar_fitted_line": "edge_line_4",
          "camera_length_m": 0.8500000089406966,
          "lidar_length_m": 0.8146160048711587,
          "length_error_m": -0.0353840040695379,
          "point_to_line_rms_m": 0.012038227595929199,
          "point_to_line_p95_m": 0.01849210626875715,
          "point_count": 101
        },
        "left": {
          "camera_target_line": "left",
          "lidar_fitted_line": "edge_line_1",
          "camera_length_m": 1.1500000208616257,
          "lidar_length_m": 1.1019400929379914,
          "length_error_m": -0.04805992792363423,
          "point_to_line_rms_m": 0.018071326913684727,
          "point_to_line_p95_m": 0.022505247105706987,
          "point_count": 148
        }
      },
      "residuals": {
        "point_to_plane": {
          "count": 6000,
          "rms_m": 0.0038897587765183645,
          "median_abs_m": 0.002600133012758299,
          "p95_abs_m": 0.007707370397515303,
          "max_abs_m": 0.01324938651585672
        },
        "point_to_line": {
          "top": {
            "lidar_line": "edge_line_2",
            "count": 147,
            "rms_m": 0.011655763201932283,
            "median_m": 0.011218018321713047,
            "p95_m": 0.01650459571172292,
            "max_m": 0.02095323330035858
          },
          "right": {
            "lidar_line": "edge_line_3",
            "count": 143,
            "rms_m": 0.01683714210884253,
            "median_m": 0.016156975379911297,
            "p95_m": 0.02391144129994102,
            "max_m": 0.027916564803959897
          },
          "bottom": {
            "lidar_line": "edge_line_4",
            "count": 101,
            "rms_m": 0.012038227595929199,
            "median_m": 0.009928277017858521,
            "p95_m": 0.01849210626875715,
            "max_m": 0.03222412138398979
          },
          "left": {
            "lidar_line": "edge_line_1",
            "count": 148,
            "rms_m": 0.018071326913684727,
            "median_m": 0.018118986038606948,
            "p95_m": 0.022505247105706987,
            "max_m": 0.025229883578374752
          }
        }
      },
      "visualization": {
        "visualization_dir": "lidar_calib/joint_extrinsic_debug_1_3_6_9/lidar_calib_1",
        "outputs": [
          "01_joint_line_correspondence_camera_frame.png",
          "02_joint_transformed_lidar_features_camera_frame.png",
          "03_joint_image_overlay_transformed_lidar_edges.png",
          "04_joint_side_colored_lidar_features_lidar_frame.pcd",
          "05_joint_side_colored_lidar_features_camera_frame.pcd",
          "06_rgb_colored_lidar_points_lidar_frame.pcd",
          "07_rgb_colored_lidar_points_camera_frame.pcd",
          "08_rgb_colored_lidar_projection_on_image.png"
        ],
        "rgb_colorized_lidar_points": {
          "source_pcd": "lidar_calib/lidar_calib_1/accumulated_cloud.pcd",
          "image_source": "lidar_calib/lidar_calib_1/image_processing_debug/01_undistorted.png",
          "coordinate_convention": "Colors are sampled from the undistorted image after projecting X_camera = R_camera_lidar @ X_lidar + t_camera_lidar.",
          "input_points": 1999200,
          "valid_colored_points": 210357,
          "valid_ratio": 0.10522058823529412,
          "files": [
            "06_rgb_colored_lidar_points_lidar_frame.pcd",
            "07_rgb_colored_lidar_points_camera_frame.pcd",
            "08_rgb_colored_lidar_projection_on_image.png"
          ]
        }
      }
    },
    "lidar_calib_3": {
      "dataset_dir": "lidar_calib/lidar_calib_3",
      "line_assignment_camera_to_lidar": {
        "top": "edge_line_3",
        "right": "edge_line_2",
        "bottom": "edge_line_4",
        "left": "edge_line_1"
      },
      "line_correspondence": {
        "top": {
          "camera_target_line": "top",
          "lidar_fitted_line": "edge_line_3",
          "camera_length_m": 0.8500000089406965,
          "lidar_length_m": 0.8466210033330098,
          "length_error_m": -0.0033790056076866692,
          "point_to_line_rms_m": 0.00949517760457148,
          "point_to_line_p95_m": 0.015201354450476748,
          "point_count": 146
        },
        "right": {
          "camera_target_line": "right",
          "lidar_fitted_line": "edge_line_2",
          "camera_length_m": 1.1500000208616257,
          "lidar_length_m": 1.193538892131577,
          "length_error_m": 0.043538871269951374,
          "point_to_line_rms_m": 0.015833040375684438,
          "point_to_line_p95_m": 0.02210280816905716,
          "point_count": 179
        },
        "bottom": {
          "camera_target_line": "bottom",
          "lidar_fitted_line": "edge_line_4",
          "camera_length_m": 0.8500000089406966,
          "lidar_length_m": 0.8030916966719228,
          "length_error_m": -0.04690831226877379,
          "point_to_line_rms_m": 0.01192815558935444,
          "point_to_line_p95_m": 0.01947363963603024,
          "point_count": 112
        },
        "left": {
          "camera_target_line": "left",
          "lidar_fitted_line": "edge_line_1",
          "camera_length_m": 1.1500000208616257,
          "lidar_length_m": 1.220914230971693,
          "length_error_m": 0.07091421011006727,
          "point_to_line_rms_m": 0.014102135625487759,
          "point_to_line_p95_m": 0.018733861682275404,
          "point_count": 178
        }
      },
      "residuals": {
        "point_to_plane": {
          "count": 6000,
          "rms_m": 0.0037038346165021046,
          "median_abs_m": 0.002458099966917482,
          "p95_abs_m": 0.007290763874200301,
          "max_abs_m": 0.012850078582897106
        },
        "point_to_line": {
          "top": {
            "lidar_line": "edge_line_3",
            "count": 146,
            "rms_m": 0.00949517760457148,
            "median_m": 0.00840862825606237,
            "p95_m": 0.015201354450476748,
            "max_m": 0.01903749747115249
          },
          "right": {
            "lidar_line": "edge_line_2",
            "count": 179,
            "rms_m": 0.015833040375684438,
            "median_m": 0.015499734284581467,
            "p95_m": 0.02210280816905716,
            "max_m": 0.02603904406336013
          },
          "bottom": {
            "lidar_line": "edge_line_4",
            "count": 112,
            "rms_m": 0.01192815558935444,
            "median_m": 0.010498968260455536,
            "p95_m": 0.01947363963603024,
            "max_m": 0.024259815646898924
          },
          "left": {
            "lidar_line": "edge_line_1",
            "count": 178,
            "rms_m": 0.014102135625487759,
            "median_m": 0.013662246788741015,
            "p95_m": 0.018733861682275404,
            "max_m": 0.02259823600206271
          }
        }
      },
      "visualization": {
        "visualization_dir": "lidar_calib/joint_extrinsic_debug_1_3_6_9/lidar_calib_3",
        "outputs": [
          "01_joint_line_correspondence_camera_frame.png",
          "02_joint_transformed_lidar_features_camera_frame.png",
          "03_joint_image_overlay_transformed_lidar_edges.png",
          "04_joint_side_colored_lidar_features_lidar_frame.pcd",
          "05_joint_side_colored_lidar_features_camera_frame.pcd",
          "06_rgb_colored_lidar_points_lidar_frame.pcd",
          "07_rgb_colored_lidar_points_camera_frame.pcd",
          "08_rgb_colored_lidar_projection_on_image.png"
        ],
        "rgb_colorized_lidar_points": {
          "source_pcd": "lidar_calib/lidar_calib_3/accumulated_cloud.pcd",
          "image_source": "lidar_calib/lidar_calib_3/image_processing_debug/01_undistorted.png",
          "coordinate_convention": "Colors are sampled from the undistorted image after projecting X_camera = R_camera_lidar @ X_lidar + t_camera_lidar.",
          "input_points": 1999200,
          "valid_colored_points": 210532,
          "valid_ratio": 0.10530812324929972,
          "files": [
            "06_rgb_colored_lidar_points_lidar_frame.pcd",
            "07_rgb_colored_lidar_points_camera_frame.pcd",
            "08_rgb_colored_lidar_projection_on_image.png"
          ]
        }
      }
    },
    "lidar_calib_6": {
      "dataset_dir": "lidar_calib/lidar_calib_6",
      "line_assignment_camera_to_lidar": {
        "top": "edge_line_2",
        "right": "edge_line_3",
        "bottom": "edge_line_4",
        "left": "edge_line_1"
      },
      "line_correspondence": {
        "top": {
          "camera_target_line": "top",
          "lidar_fitted_line": "edge_line_2",
          "camera_length_m": 0.8500000089406966,
          "lidar_length_m": 0.8608281871354383,
          "length_error_m": 0.010828178194741689,
          "point_to_line_rms_m": 0.007116770683866854,
          "point_to_line_p95_m": 0.011341491589701522,
          "point_count": 156
        },
        "right": {
          "camera_target_line": "right",
          "lidar_fitted_line": "edge_line_3",
          "camera_length_m": 1.1500000208616257,
          "lidar_length_m": 1.1566708559519951,
          "length_error_m": 0.006670835090369476,
          "point_to_line_rms_m": 0.017626615687712486,
          "point_to_line_p95_m": 0.025931328571858574,
          "point_count": 152
        },
        "bottom": {
          "camera_target_line": "bottom",
          "lidar_fitted_line": "edge_line_4",
          "camera_length_m": 0.8500000089406966,
          "lidar_length_m": 0.8044747175313265,
          "length_error_m": -0.04552529140937012,
          "point_to_line_rms_m": 0.00716433977857711,
          "point_to_line_p95_m": 0.0123988571872727,
          "point_count": 103
        },
        "left": {
          "camera_target_line": "left",
          "lidar_fitted_line": "edge_line_1",
          "camera_length_m": 1.1500000208616257,
          "lidar_length_m": 1.1879566232786893,
          "length_error_m": 0.03795660241706367,
          "point_to_line_rms_m": 0.014894419594905565,
          "point_to_line_p95_m": 0.020475130110097235,
          "point_count": 188
        }
      },
      "residuals": {
        "point_to_plane": {
          "count": 6000,
          "rms_m": 0.003806214242344193,
          "median_abs_m": 0.0024925394300157055,
          "p95_abs_m": 0.007717753139604167,
          "max_abs_m": 0.012373104225932519
        },
        "point_to_line": {
          "top": {
            "lidar_line": "edge_line_2",
            "count": 156,
            "rms_m": 0.007116770683866854,
            "median_m": 0.006083115877611233,
            "p95_m": 0.011341491589701522,
            "max_m": 0.014578114727400194
          },
          "right": {
            "lidar_line": "edge_line_3",
            "count": 152,
            "rms_m": 0.017626615687712486,
            "median_m": 0.017755624668868757,
            "p95_m": 0.025931328571858574,
            "max_m": 0.029924017723918735
          },
          "bottom": {
            "lidar_line": "edge_line_4",
            "count": 103,
            "rms_m": 0.00716433977857711,
            "median_m": 0.006633118768363527,
            "p95_m": 0.0123988571872727,
            "max_m": 0.016778938240924076
          },
          "left": {
            "lidar_line": "edge_line_1",
            "count": 188,
            "rms_m": 0.014894419594905565,
            "median_m": 0.014481873410810905,
            "p95_m": 0.020475130110097235,
            "max_m": 0.022837889757130393
          }
        }
      },
      "visualization": {
        "visualization_dir": "lidar_calib/joint_extrinsic_debug_1_3_6_9/lidar_calib_6",
        "outputs": [
          "01_joint_line_correspondence_camera_frame.png",
          "02_joint_transformed_lidar_features_camera_frame.png",
          "03_joint_image_overlay_transformed_lidar_edges.png",
          "04_joint_side_colored_lidar_features_lidar_frame.pcd",
          "05_joint_side_colored_lidar_features_camera_frame.pcd",
          "06_rgb_colored_lidar_points_lidar_frame.pcd",
          "07_rgb_colored_lidar_points_camera_frame.pcd",
          "08_rgb_colored_lidar_projection_on_image.png"
        ],
        "rgb_colorized_lidar_points": {
          "source_pcd": "lidar_calib/lidar_calib_6/accumulated_cloud.pcd",
          "image_source": "lidar_calib/lidar_calib_6/image_processing_debug/01_undistorted.png",
          "coordinate_convention": "Colors are sampled from the undistorted image after projecting X_camera = R_camera_lidar @ X_lidar + t_camera_lidar.",
          "input_points": 1999200,
          "valid_colored_points": 212525,
          "valid_ratio": 0.10630502200880353,
          "files": [
            "06_rgb_colored_lidar_points_lidar_frame.pcd",
            "07_rgb_colored_lidar_points_camera_frame.pcd",
            "08_rgb_colored_lidar_projection_on_image.png"
          ]
        }
      }
    },
    "lidar_calib_9": {
      "dataset_dir": "lidar_calib/lidar_calib_9",
      "line_assignment_camera_to_lidar": {
        "top": "edge_line_3",
        "right": "edge_line_2",
        "bottom": "edge_line_4",
        "left": "edge_line_1"
      },
      "line_correspondence": {
        "top": {
          "camera_target_line": "top",
          "lidar_fitted_line": "edge_line_3",
          "camera_length_m": 0.8500000089406968,
          "lidar_length_m": 0.8595549696487086,
          "length_error_m": 0.009554960708011762,
          "point_to_line_rms_m": 0.011302021482822007,
          "point_to_line_p95_m": 0.016835102030136523,
          "point_count": 129
        },
        "right": {
          "camera_target_line": "right",
          "lidar_fitted_line": "edge_line_2",
          "camera_length_m": 1.150000020861626,
          "lidar_length_m": 1.1690999245601308,
          "length_error_m": 0.019099903698504894,
          "point_to_line_rms_m": 0.021566283026917484,
          "point_to_line_p95_m": 0.02984474589955417,
          "point_count": 146
        },
        "bottom": {
          "camera_target_line": "bottom",
          "lidar_fitted_line": "edge_line_4",
          "camera_length_m": 0.8500000089406968,
          "lidar_length_m": 0.779842476908071,
          "length_error_m": -0.07015753203262587,
          "point_to_line_rms_m": 0.020896959847425477,
          "point_to_line_p95_m": 0.029920197610877455,
          "point_count": 83
        },
        "left": {
          "camera_target_line": "left",
          "lidar_fitted_line": "edge_line_1",
          "camera_length_m": 1.150000020861626,
          "lidar_length_m": 1.222917260066337,
          "length_error_m": 0.07291723920471105,
          "point_to_line_rms_m": 0.017134301305617758,
          "point_to_line_p95_m": 0.02221115733747587,
          "point_count": 167
        }
      },
      "residuals": {
        "point_to_plane": {
          "count": 6000,
          "rms_m": 0.005913515801455583,
          "median_abs_m": 0.004677155919281528,
          "p95_abs_m": 0.010637062688794942,
          "max_abs_m": 0.020705514714475903
        },
        "point_to_line": {
          "top": {
            "lidar_line": "edge_line_3",
            "count": 129,
            "rms_m": 0.011302021482822007,
            "median_m": 0.010588412505704548,
            "p95_m": 0.016835102030136523,
            "max_m": 0.020747091603027377
          },
          "right": {
            "lidar_line": "edge_line_2",
            "count": 146,
            "rms_m": 0.021566283026917484,
            "median_m": 0.02046749785982417,
            "p95_m": 0.02984474589955417,
            "max_m": 0.03202346700408522
          },
          "bottom": {
            "lidar_line": "edge_line_4",
            "count": 83,
            "rms_m": 0.020896959847425477,
            "median_m": 0.019745773845317564,
            "p95_m": 0.029920197610877455,
            "max_m": 0.03234480403137506
          },
          "left": {
            "lidar_line": "edge_line_1",
            "count": 167,
            "rms_m": 0.017134301305617758,
            "median_m": 0.016950727895672663,
            "p95_m": 0.02221115733747587,
            "max_m": 0.02369302746215979
          }
        }
      },
      "visualization": {
        "visualization_dir": "lidar_calib/joint_extrinsic_debug_1_3_6_9/lidar_calib_9",
        "outputs": [
          "01_joint_line_correspondence_camera_frame.png",
          "02_joint_transformed_lidar_features_camera_frame.png",
          "03_joint_image_overlay_transformed_lidar_edges.png",
          "04_joint_side_colored_lidar_features_lidar_frame.pcd",
          "05_joint_side_colored_lidar_features_camera_frame.pcd",
          "06_rgb_colored_lidar_points_lidar_frame.pcd",
          "07_rgb_colored_lidar_points_camera_frame.pcd",
          "08_rgb_colored_lidar_projection_on_image.png"
        ],
        "rgb_colorized_lidar_points": {
          "source_pcd": "lidar_calib/lidar_calib_9/accumulated_cloud.pcd",
          "image_source": "lidar_calib/lidar_calib_9/image_processing_debug/01_undistorted.png",
          "coordinate_convention": "Colors are sampled from the undistorted image after projecting X_camera = R_camera_lidar @ X_lidar + t_camera_lidar.",
          "input_points": 1999104,
          "valid_colored_points": 203636,
          "valid_ratio": 0.10186363490843899,
          "files": [
            "06_rgb_colored_lidar_points_lidar_frame.pcd",
            "07_rgb_colored_lidar_points_camera_frame.pcd",
            "08_rgb_colored_lidar_projection_on_image.png"
          ]
        }
      }
    }
  },
  "output_files": [
    "00_joint_line_correspondence_montage.png",
    "00_joint_rgb_projection_montage.png",
    "joint_extrinsic_result.json",
    "joint_extrinsic_report.md"
  ]
}
```



</details>

![](/assets/img/posts/352cbb7d-7937-8027-9f04-da9c71071dca.gif)

