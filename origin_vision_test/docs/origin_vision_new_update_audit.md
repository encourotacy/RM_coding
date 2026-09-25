# `origin_vision_new` 更新审计与 `origin_vision_test` 迁移边界

## 结论
- 这轮 `origin_vision_new` 相对 `sp_vision_25` 的增量，主要分成两块：
  - 通信/工程边界更新：`OVGimbal`、`ROS2Gimbal`、`CBoard` 图像时间补偿、CMake 中 ROS2 可选拆库。
  - auto_aim 算法更新：`Singer`、13/15 维状态、`Target/Tracker/Aimer` 改写、配置清理、实验链路。
- 本次移植到 `origin_vision_test` 只保留前者，算法侧仍保持 `sp_vision_25` 基线。

## 逐文件审计

### 1. 构建与目标边界
- `CMakeLists.txt`
  - 新增 `ovgimbal_auto_aim_test`、`ovgimbal_mpc` 入口，用于串口云台联调。
  - 原本顶层直接探测 ROS2 并让 `io` 库携带 ROS 依赖；新版本改为按 `io_ros2_nav`/`io_ros2_gimbal` 是否存在来决定是否编译对应入口。
  - `ovsentry_*` 入口单独依赖 `io_ros2_gimbal`，避免普通目标被 ROS2/FastDDS 依赖污染。
- `io/CMakeLists.txt`
  - `io` 主库新增 `ovgimbal.cpp`。
  - ROS2 相关源从 `io` 主库拆出，变为两个可选静态库：
    - `io_ros2_nav`：旧的 nav 发布/订阅桥。
    - `io_ros2_gimbal`：新的云台状态/控制桥。
  - 新增 `fastcdr` 检查，因为 `ROS2Gimbal` 走泛型序列化消息而不是编译期绑定消息类型。

### 2. 电控与姿态接口
- `io/cboard.hpp`
  - 新增 `imu_at_image()` 与 `offset_ms()`。
  - 新增 `data_prev_`、`has_prev_`、`imu_query_offset_`，用于图像时间戳查询时做稳定插值。
- `io/cboard.cpp`
  - 新增 `read_imu_query_offset()`，从 YAML `timing.offset` 读取固定补偿。
  - `imu_at()` 从“只看 ahead/behind”改成“三点窗口插值”，前向查询时不再立刻丢历史样本。
  - 新增 `imu_at_image()`，统一做 `image_timestamp + offset` 查询，替代各入口手写 `t - 1ms`。
  - `read_yaml()` 里新增 timing 日志输出。
- `io/ovgimbal.hpp/.cpp`
  - 新增 FYT 串口云台桥。
  - 接收侧：读 `0x14` 状态帧，解出四元数、yaw/pitch、角速度、弹速，并维护 IMU 队列。
  - 发送侧：把 `Command` 编成 `0x81` 控制帧，只发 `pitch/yaw/fire`。
  - 同样支持 `imu_at_image()` 固定补偿查询。
- `io/command.hpp`
  - 新增 `big_yaw`、`small_yaw`、`has_target_yaw`，只服务 `ROS2Gimbal` 的双 yaw 输出，不影响旧 `CBoard`/`OVGimbal`。

### 3. ROS2 通信
- `io/ros2/publish2nav.*`、`io/ros2/subscribe2nav.*`
  - 与 `sp_vision_25` 基本一致，没有核心协议改写。
- `io/ros2/ros2_gimbal.hpp/.cpp`
  - 新增基于 `GenericPublisher/GenericSubscription` 的云台桥。
  - 状态侧：从 YAML 读取 `status_topic/cmd_topic/msg_type`，反序列化 `Gimbal` 状态消息，恢复姿态、模式、角速度、弹速。
  - 兼容扩展：在现有状态字段后允许追加一个尾字段 `gimbal_big_yaw(float32, deg, 连续角)`；`ROS2Gimbal` 会按可选字段读取，缺失时保持旧协议兼容。
  - 控制侧：序列化 `GimbalCmd`，按 `pitch/big_yaw/small_yaw/fire/target_distance` 输出。
  - 若上层没填写 `big_yaw/small_yaw`，会退化为 `command.yaw` 双写，并打一次 warning。

### 4. 入口级改动
- `src/standard.cpp`
- `src/mt_standard.cpp`
- `src/mt_auto_aim_debug.cpp`
- `src/uav.cpp`
- `src/uav_debug.cpp`
- `src/sentry.cpp`
- `src/sentry_bp.cpp`
- `src/sentry_debug.cpp`
- `src/sentry_multithread.cpp`
- `src/auto_buff_debug.cpp`
  - 共同变化只有一项：`cboard.imu_at(t - 1ms)` 或 `cboard.imu_at(t)` 统一改为 `cboard.imu_at_image(t)`。
  - 这属于通信层时间戳口径统一，不改变算法流程。
- `src/ovgimbal_auto_aim_test.cpp`
- `src/ovgimbal_mpc.cpp`
  - 新增串口云台最小联调入口。
- `src/ovsentry_auto_aim_test.cpp`
- `src/ovsentry_mpc.cpp`
  - 新增 ROS2 云台桥最小联调入口。

### 5. 配置层改动
- `configs/standard3.yaml`
- `configs/standard4.yaml`
- `configs/uav.yaml`
  - 对通信移植真正有用的改动只有 `timing.offset`。
- `configs/sentry.yaml`
  - 对通信移植真正有用的改动有两项：
    - `timing.offset`
    - `ros2_gimbal.*`
  - 其余 `Singer/motion_model/top4_model` 均属于算法侧，本次不迁。

## 算法侧更新摘要
- `tasks/auto_aim/aimer.*`
  - 引入 `resistance_k`、`decision_speed_enter/exit`，并经历过高低速选板逻辑试验。
- `tasks/auto_aim/target.*`
  - 从 11 维扩到 13 维，引入 `Singer` 辅助状态与 `predict_with_singer()`。
- `tasks/auto_aim/tracker.*`
  - 新增 `SingerConfig`、实验链路配置与部分选板/观测逻辑变更。
- `tasks/auto_aim/top4_model.*`
  - `sentry_test` 分支新增 15 维实验模型。

这部分不进入 `origin_vision_test`。更详细的算法对比，直接参考：
- `origin_vision_new/docs/target_observation_prediction_vs_sp_vision_25.md`
- `origin_vision_new/docs/origin_vision_new_vs_sp_vision_25_baseline.md`

## 本次迁移落点
- 已迁入 `origin_vision_test`：
  - `CBoard imu_at_image`
  - `OVGimbal`
  - `ROS2Gimbal`
  - `Command` 双 yaw 字段
  - 顶层/IO CMake 的 ROS2 可选拆库
  - 最小联调入口与必要配置
- 未迁入 `origin_vision_test`：
  - `Singer`
  - `Target/Tracker/Aimer` 的建模和选板修改
  - `top4_experimental`
  - `origin_vision_new` 中针对 sentry/hero 的算法调参
