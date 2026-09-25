# 笔记

## 头文件

`ovsentry_omni_mpc.cpp` 开头这 15 个头文件按用途分成三组：字符串与二进制解码、标准容器，以及时间和多线程。

### 字符串和二进制数据

- `fmt/core.h` 提供 `fmt::format`，用来拼调试文字，比如相机名、耗时、yaw/pitch。
- `fastcdr/Cdr.h` 和 `fastcdr/FastBuffer.h` 是 CDR 解码。文件里用它们把 ROS 发来的原始消息拆成「要忽略的装甲板编号」列表。
- `cstdint` 提供 `uint8_t`、`uint32_t` 这种固定宽度整数，编号和个数都用它们。
- `cmath` 提供 `std::sqrt`，用来算目标在水平面上的距离。

### 容器

- `vector` 是可变长数组，这里用来存一串装甲板编号。
- `list` 是链表，检测结果 `std::list<auto_aim::Armor>` 用它装。
- `optional` 表示「可能没有」：没有最上方装甲板、没有全向候选目标时就是空的。
- `memory` 提供 `std::shared_ptr` 和 `std::unique_ptr`，ROS 节点、订阅和执行器都靠它们持有。
- `string` 是字符串。
- `algorithm` 提供查找、删除、排序这一类算法。文件里用 `std::sort` 给忽略编号排序，用 `std::find` / `std::find_if` 在列表里查找。

### 时间和线程

- `chrono` 提供时间点，每一帧的时间戳是 `steady_clock::time_point`。
- `thread` 用来单独开一条线程跑 ROS 的 `executor->spin()`。
- `mutex` 用来给忽略名单加锁，避免订阅回调和主循环同时改它。
- `atomic` 提供 `std::atomic_bool`，文件里的 `request_buff_` 用它在线程之间传递「要不要打符」这个开关。
