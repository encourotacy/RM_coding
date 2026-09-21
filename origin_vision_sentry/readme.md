# origin_vision_sentry

哨兵机器人视觉程序，集成 **常规自瞄、打前哨站、全向感知、打符**。比赛主程序 `ovsentry_mpc` 按电控/导航信号自动切换模式；调试时也可单独运行 `ovsentry_auto_aim`、`ovsentry_buff`、`ovsentry_omni`。
基于 [sp_vision_25](https://github.com/TongjiSuperPower/sp_vision_25) 框架精简而来，仅保留哨兵所需模块。

---

## 1. 功能概述

主程序 `src/ovsentry_mpc.cpp` 根据电控/导航信号及目标类型自动切换工作模式。打前哨站与打车上装甲板虽共用 YOLO 识别与 Tracker 框架，但目标特性、估计、瞄准、开火逻辑均不同，作为两个独立功能处理：

### 1.1 常规自瞄（打车上装甲板）

| 项 | 说明 |
|----|------|
| 目标 | 移动的整车（平移 / 低速小陀螺 / 高速小陀螺） |
| 识别 | 主相机 YOLO 检测装甲板四点 |
| 估计 | 整车 EKF 估计（旋转中心位置、速度、yaw 角、yaw 角速度、装甲半径、高度差） |
| 瞄准 | Aimer 跟随旋转装甲板，Planner MPC 轨迹规划，装甲板切换时提前减速 |
| 开火 | 按位置误差 + 高速强制开火（`high_spin_force_fire_enabled`） |
| 触发 | 主相机 tracker 处于 `tracking`/`detecting`，无打符请求 |

### 1.2 打前哨站（outpost）

| 项 | 说明 |
|----|------|
| 目标 | 固定位置、低速 / 静止旋转的三层装甲板 |
| 识别 | 主相机 YOLO 检测，Tracker 走前哨站三层关联分支（`outpost_target`） |
| 估计 | 三层装甲关联预测 + layer correction + 锁定判定（`outpost_layer_locked`） |
| 瞄准 | 相位窗口瞄准（`outpost_aim_coming_angle` / `outpost_aim_leaving_angle`） |
| 开火 | 两种子模式：<br>① 相位窗口开火（`outpost_phase_fire_enabled`，锁定后按相位打）<br>② 静止开火（`outpost_static_fire_enabled`，静止时直接瞄当前装甲） |
| 收敛保护 | EKF 未收敛或发散时停发指令（`outpost_convergence`），避免乱打 |
| 触发 | 主相机识别到前哨站装甲板（`ArmorName::outpost`），无打符请求 |

### 1.3 全向感知（OMNI）

| 项 | 说明 |
|----|------|
| 目标 | 主相机视野外的敌方装甲板 |
| 识别 | 左前 / 右前 / 正后三路 USB 相机并行 YOLO 检测 |
| 决策 | Decider 选最优目标，控制大 yaw 转向目标方位 |
| 重定向策略 | 冷却（`omni_retarget_cooldown_s`）+ 最小角度阈值（`omni_retarget_min_delta_deg`）+ 命令超时（`omni_command_timeout_s`） |
| 触发 | 主相机 tracker 进入 `lost`，无打符请求 |

### 1.4 打符（SMALL_BUFF）

| 项 | 说明 |
|----|------|
| 目标 | 能量机关（小符） |
| 识别 | Buff_Detector 检测扇叶与中心 R 标 |
| 估计 | EKF 预测符叶旋转角度与角速度 |
| 瞄准 | Buff_Aimer MPC 瞄准（`buff_aim_radius_m` 控制瞄点半径） |
| 开火 | 相位窗口开火（`fire_gap_time`） |
| 失效保护 | 检测短暂丢失时保持上一条指令（`buff_lost_cmd_hold_s`） |
| 触发 | 订阅 `/request_buff` 为 true |

### 模式优先级

打符 > 常规自瞄 / 打前哨站 > 全向感知

主相机 tracker 状态决定走自瞄还是全向：`tracking`/`detecting` 走自瞄（含前哨站分支），`lost` 走全向。打符由外部 `/request_buff` 请求抢占最高优先级。

---

## 2. 环境依赖

- Ubuntu 22.04
- OpenVINO 2024.6.0（`/opt/intel/openvino_2024.6.0`）
- ROS2 Humble + `rm_interfaces`（哨兵主程序依赖 ROS2 云台桥）
- Ceres Solver（打符求解用）
- 其余 apt 包：

```bash
sudo apt install -y git g++ cmake can-utils \
    libopencv-dev libfmt-dev libeigen3-dev libspdlog-dev \
    libyaml-cpp-dev libusb-1.0-0-dev nlohmann-json3-dev \
    libfastcdr-dev openssh-server screen
```

相机 SDK（按实际相机选其一）：
- 海康 MVS（`camera_name: hikrobot`）
- 迈德威 SDK（`camera_name: mindvision`）

---

## 3. 编译

```bash
cd /home/star/RM_coding/origin_vision_sentry
cmake -B build
make -C build -j$(nproc)
```

> 若 ROS2 环境未配置好，`io_ros2_gimbal` 目标会跳过，哨兵主程序不会编译。请先 `source /opt/ros/humble/setup.bash` 并确保 `rm_interfaces` 可被找到。

---

## 4. 运行主程序

### 4.1 直接运行

```bash
./build/ovsentry_mpc configs/sentry.yaml
```

带画面显示（默认开启；比赛主程序为四路相机拼接预览）：

```bash
./build/ovsentry_mpc configs/sentry.yaml
```

关闭显示（部署/自启用）：

```bash
./build/ovsentry_mpc configs/sentry.yaml --no-display
```

### 4.2 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `@config-path` | `configs/sentry.yaml` | yaml 配置路径 |
| `left` | 读 yaml `omni_left_path` | 左前 USB 相机设备名 |
| `right` | 读 yaml `omni_right_path` | 右前 USB 相机设备名 |
| `back` | 读 yaml `omni_back_path` | 正后 USB 相机设备名 |
| `left_yaw` | 60 | 左前相机中心 yaw 角 (deg) |
| `right_yaw` | -60 | 右前相机中心 yaw 角 (deg) |
| `back_yaw` | 180 | 正后相机中心 yaw 角 (deg) |
| `fov_h` | 120 | USB 相机水平视场角 (deg) |
| `fov_v` | 67 | USB 相机垂直视场角 (deg) |
| `no-display` | — | 关闭画面显示 |

示例（临时指定相机路径）：

```bash
./build/ovsentry_mpc configs/sentry.yaml left=video0 right=video2 back=video4
```

### 4.3 开机自启

```bash
chmod +x autostart.sh
# 将 autostart.sh 绝对路径写入 ~/.config/autostart/sp_vision.desktop
```

`autostart.sh` 会在工程目录下用 screen 后台启动：

```bash
./build/ovsentry_mpc configs/sentry.yaml --no-display
```

日志写入 `logs/` 目录。

### 4.4 单模式程序

四个入口各自有真实主循环。自瞄 / 打符 / 全向写在独立任务类里，比赛主程序 `ovsentry_mpc.cpp` 直接调用这三个任务，所以改 `auto_aim_task` / `buff_task` / `omni_task` 会立刻作用到主程序。单模式程序不会构造用不到的任务，避免抢 USB 设备。

| 程序 | 源文件 | 行为 |
|------|--------|------|
| `ovsentry_mpc` | `src/ovsentry_mpc.cpp` | 比赛用：打符请求 → tracker lost 走全向 → 否则自瞄 |
| `ovsentry_auto_aim` | `src/ovsentry_auto_aim.cpp` | 只跑自瞄 + MPC，lost 也不进全向 |
| `ovsentry_buff` | `src/ovsentry_buff.cpp` | 只打符，不看 `/request_buff` |
| `ovsentry_omni` | `src/ovsentry_omni.cpp` | 只跑三路 USB 全向，不要求主相机 lost |

```bash
./build/ovsentry_auto_aim configs/sentry.yaml
./build/ovsentry_buff configs/sentry.yaml
./build/ovsentry_omni configs/sentry.yaml
```

---

## 5. 配置说明

主配置文件 `configs/sentry.yaml`，关键段：

| 段落 | 说明 |
|------|------|
| `enemy_color` | 敌方颜色（red/blue） |
| `yolo_name` / `*_model_path` | 识别模型选择与权重路径 |
| `auto_aim_device` / `omni_device` | 推理设备（GPU/NPU/CPU） |
| `omni_*_path` | 三路 USB 相机 v4l 设备路径 |
| `omni_*_yaw_deg` / `omni_fov_*_deg` | 全向相机安装方位与视场角 |
| `camera_name` / `exposure_ms` | 主工业相机选择与曝光 |
| `ros2_gimbal` | ROS2 云台桥话题配置 |
| `R_camera2gimbal` / `camera_matrix` | 手眼标定与内参结果（由标定程序填入） |
| `omni_retarget_*` / `omni_command_timeout_s` | 全向感知重定向策略参数 |
| `buff_*` | 打符检测与瞄准参数 |

---

## 6. 标定

自瞄依赖相机内参与手眼标定，部署前需执行。标定程序位于 `calibration/`，配置在 `configs/calibration.yaml` 与 `configs/calibration_chessboard.yaml`。

### 6.1 采集标定数据

普通标定板：

```bash
./build/capture configs/calibration.yaml
# 采集图像+四元数到 assets/img_with_q/
```

棋盘格标定板：

```bash
./build/capture_chessboard configs/calibration_chessboard.yaml
# 采集到 assets/chessboard_calib/
```

### 6.2 标定相机内参

```bash
./build/calibrate_camera configs/calibration.yaml
# 或棋盘格版
./build/calibrate_camera_chessboard configs/calibration_chessboard.yaml
```

结果打印到终端，需手动复制 `camera_matrix` / `distort_coeffs` 到 `configs/sentry.yaml`。

### 6.3 手眼标定

```bash
./build/calibrate_handeye configs/calibration.yaml
# 或棋盘格版
./build/calibrate_chessboard_handeye configs/calibration_chessboard.yaml
```

结果 `R_camera2gimbal` / `t_camera2gimbal` 复制到 `configs/sentry.yaml`。

### 6.4 标定板位置（可选）

```bash
./build/calibrate_robotworld_handeye configs/calibration.yaml
```

### 6.5 视频拆分

将录像拆成图像+时间戳用于离线标定：

```bash
./build/split_video <视频路径>
```

---

## 7. 测试

测试程序位于 `tests/`，编译后位于 `build/`。常用测试：

### 7.1 自瞄离线回放（内录视频）

```bash
./build/auto_aim_test configs/demo.yaml assets/demo/demo
# 回放 assets/demo/demo.avi + demo.txt 测试自瞄整链路
```

### 7.2 打符离线回放

```bash
./build/auto_buff_test configs/sentry.yaml <视频路径>
```

### 7.3 实时识别测试

```bash
./build/camera_detect_test configs/sentry.yaml     # 主相机识别
./build/usbcamera_detect_test configs/sentry.yaml  # USB 相机识别
```

### 7.4 规划器测试

```bash
./build/planner_test configs/sentry.yaml        # 实车
./build/planner_test_offline configs/sentry.yaml # 离线
```

### 7.5 逻辑回归测试（无需硬件）

```bash
./build/ovsentry_omni_logic_test        # 全向感知逻辑
./build/outpost_auto_aim_logic_test      # 前哨站自瞄逻辑
./build/normal_auto_aim_v2_logic_test    # 常规自瞄逻辑
./build/ros2_gimbal_big_yaw_lookup_test  # 大 yaw 查表
./build/armor_type_logic_test            # 装甲类型逻辑
./build/pattern_geometry_test            # 几何模式
```

### 7.6 相机/通信测试

```bash
./build/camera_test configs/camera.yaml
./build/camera_thread_test configs/sentry.yaml   # 多线程检测（默认读 ascento.yaml，传 sentry.yaml）
./build/multi_usbcamera_test
./build/gimbal_test configs/sentry.yaml
./build/cboard_test
./build/dm_test
./build/fire_test
./build/handeye_test
./build/minimum_vision_system
```

> `camera_thread_test` 默认读 `configs/ascento.yaml`，运行时请传 `configs/sentry.yaml`。

---

## 8. 调试曲线（PlotJuggler）

主程序通过 `tools::Plotter` 以 UDP（`127.0.0.1:9870`）发送 JSON 调试数据。

1. 打开 PlotJuggler
2. 数据源选 **UDP Server**，端口 `9870`
3. 加载布局：
   - `mpc_layout.xml` — 自瞄/MPC 曲线（`gimbal_yaw`、`mpc_yaw`、`ref_yaw` 等）
   - `buff_layout.xml` — 打符曲线（`buff_yaw`、`cmd_yaw`、`R_yaw` 等）

代码中发送数据的字段见 `src/ovsentry/runtime.cpp` 的 `publish_telemetry()`。新增曲线只需往 `nlohmann::json data` 里加字段，再 `plotter.plot(data)` 即可。

---

## 9. 目录结构

```
origin_vision_sentry
├── src/ovsentry_mpc.cpp        # 比赛主程序：编排自瞄 / 全向 / 打符
├── src/ovsentry_auto_aim.cpp   # 只跑自瞄
├── src/ovsentry_buff.cpp       # 只打符
├── src/ovsentry_omni.cpp       # 只跑全向感知
├── src/ovsentry/
│   ├── runtime.*               # 相机、云台、帧状态、打点与显示
│   ├── auto_aim_task.*         # 自瞄任务（mpc 与 auto_aim 共用）
│   ├── buff_task.*             # 打符任务（mpc 与 buff 共用）
│   └── omni_task.*             # 全向任务（mpc 与 omni 共用）
├── configs/
│   ├── sentry.yaml             # 哨兵主配置
│   ├── calibration.yaml        # 标定配置
│   ├── calibration_chessboard.yaml
│   ├── camera.yaml             # 相机测试配置
│   ├── demo.yaml               # 离线回放测试配置
│   └── ascento.yaml            # camera_thread_test 默认配置
├── assets/                     # 模型权重、demo 素材
├── calibration/                # 标定程序
├── tests/                      # 测试程序
├── io/                         # 硬件抽象（相机/云台/C板/串口）
├── tasks/
│   ├── auto_aim/               # 自瞄算法
│   ├── auto_buff/              # 打符算法
│   └── omniperception/         # 全向感知
├── tools/                      # 工具库（日志/绘图/弹道/EKF 等）
├── autostart.sh                # 开机自启脚本
├── mpc_layout.xml              # PlotJuggler 自瞄布局
├── buff_layout.xml             # PlotJuggler 打符布局
└── CMakeLists.txt
```

---

## 10. 模式切换说明

比赛主程序 `ovsentry_mpc` 通过 ROS2 话题接收外部指令切换模式：

- `/request_buff`（`std_msgs/Bool`）：`true` 时进入打符模式，`false` 退出
- `/request_auto_aim_ignore`（`rm_interfaces/RequestAutoAimIgnore`）：忽略指定装甲 ID，用于导航追击时过滤目标

无外部打符请求时，主相机 tracker 处于 `tracking`/`detecting` 走自瞄（根据目标类型自动区分常规装甲板与前哨站分支），进入 `lost` 走全向感知。打符由 `/request_buff` 请求抢占最高优先级。

单模式程序 `ovsentry_auto_aim` / `ovsentry_buff` / `ovsentry_omni` 会忽略上述自动切换，始终停留在对应功能。
