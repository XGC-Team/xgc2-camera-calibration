# 相机内参标定：仿真 / 物理实验代码审查

审查日期：2026-09-05。结论：**针孔 + 五参数畸变的求解路线合理，已读前后端 API 的名称和数据结构基本对应，但不能据此判定实验已端到端验收通过。** 本次提交修复一个已复现的启动阻塞问题，并保留尚未修复的性能、界面反馈问题和精度验收边界。

## 1. 审查基线与交付边界

| 层级 | 仓库 / 路径 | 固定 commit |
| --- | --- | --- |
| 入口 | `XGC-Team/xgc2-harness` | `d57136eeb6cc81bf4ad6e3fc3d14592851f9ef25` |
| 编排 | `devops` → `XGC-Team/xgc2-devops` | `bab01d4b8f84c1e82892fd283cf74dc431e0d441` |
| 实验工作流 | `devops/products/xgc2` → `XGC-Team/xgc2` | `097c9c1a6048b751d72b4100d240da772c6d9ef3` |
| 标定集合 | `devops/products/ros1/perception/calibration` → `XGC-Team/xgc2-calibration` | `91da17c57b94191f5d3e6a3131f4febbb26ff4af` |
| 标定实现 | 上述目录的 `camera-calibration` → 本仓库 | `99a5429ea6004d21b874fa4a7b7c7537e4574475` |

修复放在实际维护代码的本仓库，不在 harness 内复制实现，不直接推进上层 gitlink。合并后仍需按 camera-calibration → calibration → devops → harness 的依赖方向集成并验收。本次读取时本仓库 main 与上述 pin 一致。

范围：工作流生成脚本、ROS 入口、相机控制、板型定义、检测/采样关键路径、求解与保存关键路径、HTTP 路由及内嵌 React/legacy UI。不是全仓库审计；没有运行 ROS/Gazebo、USB 相机、NVENC、Media Edge 或浏览器整站，也没有跑完整仓库测试套件。外层 Core 编排执行器、代理重写、发布后的 process catalog 和实际 Gazebo SDF/渲染真值没有全部核验。

## 2. Findings（按优先级）

### F1 — P1：仿真时钟停止会卡住可选相机控制初始化【已修复】

位置：[原始入口 `maybe_camera_control`，L36–64](https://github.com/XGC-Team/xgc2-camera-calibration/blob/99a5429ea6004d21b874fa4a7b7c7537e4574475/xgc_camera_calibration/scripts/intrinsic_calibrator_web.py#L36-L64)。

原实现用 `rospy.Time.now() + rospy.Duration(timeout)` 作为截止时间，并调用 `rospy.Rate(10).sleep()`。启用 `/use_sim_time` 而 `/clock` 停止、目标模型又未出现时，这两者都依赖不再前进的时间，无法实现注释所承诺的超时回退。该函数在 HTTP server 创建前调用，因而界面/健康检查也可能迟迟不可用。

另一个问题是 `camera_control_timeout` 没有传入 `GazeboCameraControl(..., connection_timeout=...)`，构造器可以先按默认 10 秒等待，再另起一轮等待。

本提交改用 `time.monotonic()` 和有界 `time.sleep()`；连接发现与模型等待共享预算；传递配置的连接超时；拒绝非有限或非正超时。这里没有宣称整个进程启动具有硬实时超时：先前的板模型选择服务调用、线程调度和构造器内部的短暂稳定等待仍有各自边界。

验证：原入口完整文件的 Git blob SHA 在本地核对为 `8934d893e46892a7f7c9346781d438c310eb8bea`。新增测试执行真实入口模块，只替换外部 ROS 依赖和时钟。冻结 ROS 时钟的睡眠被设置为立即失败，避免回归测试本身挂住。修复前 9 个测试方法运行失败（unittest 报告 11 个 failure，包含 subtest）；修复后 9/9 通过。

### F2 — P2：物理采样池增长与持锁同步求解耦合，可能阻塞操作界面【未修复；静态确认，未做容量压测】

位置：[采样入池](https://github.com/XGC-Team/xgc2-camera-calibration/blob/99a5429ea6004d21b874fa4a7b7c7537e4574475/xgc_camera_calibration/src/xgc_camera_calibration/intrinsic_service.py#L1890-L2045)、[`calibrate` 持锁](https://github.com/XGC-Team/xgc2-camera-calibration/blob/99a5429ea6004d21b874fa4a7b7c7537e4574475/xgc_camera_calibration/src/xgc_camera_calibration/intrinsic_service.py#L2590-L2600)、[留一求解](https://github.com/XGC-Team/xgc2-camera-calibration/blob/99a5429ea6004d21b874fa4a7b7c7537e4574475/xgc_camera_calibration/src/xgc_camera_calibration/intrinsic_solver.py#L1495-L1620)。

物理模式没有相机控制时，每个严格合格的新 snapshot 都可入池；当前代码仅按 snapshot ID 排除完全重复的快照，不按姿态近重复限制入池数量。入口默认每 0.2 秒请求一次检测。对所选 N 个视图，求解除全量拟合外还会执行 N 次 N−1 视图拟合；剔除异常视图时还有额外拟合。

`calibrate()` 在会话 RLock 内同步完成这套计算，`state()`、`reset()`、采样和保存也需要同一把锁。线程化 HTTP server 并不能绕过这把锁。因此长时间采样后点击分析，状态请求和重置可能等待整个优化结束。没有测得具体卡顿时长，不应编造吞吐量或延迟数值。

建议：在锁内冻结带 revision 的观测快照，锁外执行有生命周期的求解任务，完成时以 revision/CAS 提交候选；提供计算状态和取消语义。把完整证据留存与有限求解子集分开，采用几何多样性抽样/近重复控制，并保留被选与未选索引。不要在此次小修复中擅自删除样本或添加未经产品确认的采样上限。

### F3 — P2：重置失败仍显示成功【未修复；隔离逻辑复现】

位置：[UI reset 回调](https://github.com/XGC-Team/xgc2-camera-calibration/blob/99a5429ea6004d21b874fa4a7b7c7537e4574475/web-src/src/intrinsic-legacy.ts#L220-L240)。React 页面实际动态导入该 legacy 模块，并非废弃代码。

`postJSON()` 把 HTTP/网络失败转为 `{ok:false,error:...}`，但 reset 回调忽略返回值，无条件显示 `Coverage reset.`。服务端确实可能因正在运行的 action 返回 409，或因网络失败根本没有完成重置。

隔离执行原回调、令请求返回 `{ok:false,error:'HTTP 409: action already running'}`，得到 `displayed_status: 'Coverage reset.'`。这验证的是回调逻辑，不是浏览器 E2E。建议与 candidate/save/continue 保持一致：检查 `r.ok === false`、显示错误，并避免失败后宣称清空完成。修改后还应构建并验证仓库发布的 Web 静态资源，因此本次没有只改 TypeScript 而漏更新产物。

## 3. 数学原理核对

### 3.1 模型及优化目标

平面标定点 `P_j=(X_j,Y_j,0)` 在第 i 视图中变换为 `R_i P_j+t_i=(Xc,Yc,Zc)`，归一化坐标 `x=Xc/Zc, y=Yc/Zc, r²=x²+y²`。本实现采用的五参数模型为：

```text
radial = 1 + k1*r² + k2*r⁴ + k3*r⁶
xd = x*radial + 2*p1*x*y + p2*(r² + 2*x²)
yd = y*radial + p1*(r² + 2*y²) + 2*p2*x*y
u = fx*xd + cx
v = fy*yd + cy
```

`calibrateCameraExtended(..., flags=0)` 联合估计 `fx, fy, cx, cy, k1, k2, p1, p2, k3` 和各视图位姿，以像素重投影平方误差为目标。未发现把 K 的像素单位和板的米制尺寸直接混用、把内参误当位姿、或把显示缩略图的尺寸直接作为源相机 K 尺度的证据。算法假设是单一固定内参的针孔 / plumb_bob 模型；不能据此保证任意广角、鱼眼、滚动快门或变焦相机都适用。

独立模型探针使用 8 个倾斜平面视图、每视图 144 个精确点、3840×2160 图像，分别验证零畸变和非零五参数畸变。OpenCV 4.13.0 / NumPy 2.3.5 下，两组 RMS 分别约 `5.7903e-5`、`5.8445e-5` 像素，最大相对焦距误差小于 `7.3e-8`。这是**同模型精确对应点的一致性检查**，没有执行仓库检测器、完整求解器的筛选/LOO、Gazebo 图像渲染或真实相机，不能当作实测标定精度。

### 3.2 AprilGrid、尺度和观测平面

`board_profiles.py` 定义 field 6×6：tag 0.088 m，gap 0.0264 m；A4 6×6：tag 0.024 m，gap 0.0072 m。这里 gap 是米制间隔，0.30 是 gap/tag 的比例，不能把 0.30 当成 0.30 m。对应整体边长分别为 0.660 m 和 0.180 m。

检测代码明确声明 Kalibr ID0 左下、图案旋转 180°，OpenCV 到对象点角点顺序为 `(1,0,3,2)`。严格求解使用经过完整 tag 质量筛选的角点和对象点；原始角点只用于标注。源 JPEG 可用时会重新在源分辨率定位，入池后冻结图像尺寸，尺寸改变的样本被拒绝。这些是正确的防混坐标措施。

但上述坐标约定是否与**实际安装的**板图案、SDF 材质、传感器光学中心和相机 profile 完全一致，仍需真值图像/标定板实物验证。本次未以只读 Python 约定替代这项验证。

### 3.3 可观测性、覆盖率和精度边界

求解器对每个视图投影雅可比拆分位姿列 `Jp` 与内参列 `Jk`，计算 `Jk - Jp*lstsq(Jp,Jk)`，再堆叠、列归一化并做 SVD。这是在排除位姿补偿后检查内参局部可观测性的合理做法。X/Y/Size/Skew 是采集指导，不是完整可观测性证明；AprilGrid Skew 使用单应局部各向异性，不再把纯平面内旋转当作倾斜信息。

**保存门槛是相对稳定性检查，不是绝对精度保证。** `_candidate_save_assessment_locked()` 的 held-out 上界为：

```text
median(training per-view RMS) + 3*hypot(detector_uncertainty, 1.4826*MAD)
```

例如训练 RMS 全为 20 px、MAD=0、AprilGrid uncertainty=0.75 px，则上界为 22.25 px；held-out RMS=20 px 不会被这一条不等式拒绝。这是可运行的代数反例，**不是已经构造出通过全部门槛并落盘的真实坏标定**。满秩、有限性、完整 LOO、ray 稳定性等其他检查仍然存在。建议按相机/用途定义独立的绝对像素、角度或应用误差预算，不擅自添加统一常数。

另外，`0.75 px` 来源于边拟合残差阈值，不是本次实测得到的统计标准差；checkerboard 分支把 `cornerSubPix` 的 0.01 停止准则作为 uncertainty，也不能解释成测量噪声。ray 指标取 9×7 网格的最大值，而阈值使用观测点总数的极值因子；这个统计解释还需要校准。全量异常视图筛选先于 LOO，留一结果不能等同于从未参与筛选的独立测试集。以上列为方法/验收风险，非本次已修复缺陷。

## 4. 前后端接线核对

工作流来源：[固定版本 provision 脚本](https://github.com/XGC-Team/xgc2/blob/097c9c1a6048b751d72b4100d240da772c6d9ef3/scripts/provision-camera-intrinsic-calibration-automation.sh)。

| 环节 | 已读代码中的对应关系 | 判断范围 |
| --- | --- | --- |
| 仿真 | wait-gazebo → movable camera → Media Edge → calibrator；`cameraStatic:false`，calibrator `cameraControl:true` | 定义层对应；Core fan-in/readiness 执行和 Gazebo runtime 未验收 |
| 物理 | V4L2 compressed topic → compressed RTP adapter → Media Edge → calibrator；`cameraControl:false` | 3840×2160/30 与 source ID、RTP port、socket 的定义和 binding 对应；硬件/NVENC 未验收 |
| 板型 | shared `boardProfile` → calibrator → resolver；仿真按同一 profile 选择 Gazebo board model | 代码路径对应；实际板模型、图案和渲染未验收 |
| 预览 / 求解 | WebRTC 归浏览器；标定使用低频 Media Edge snapshots 与源 JPEG 定位 | 未见把视频显示尺寸直接用于源分辨率 K 的问题；媒体时序/源身份仍需整体验证 |
| 内嵌 UI → API | GET `state` / `image.jpg`；POST `candidate` / `save` / `continue` / `reset` / `auto_run` 等 | 名称、方法与已读 HTTP route 对应；外层代理路径未全部核验 |
| 保存 | UI 带 `candidate_id`；HTTP 校验；service 在锁内检查当前 candidate ID、质量并调用原子 YAML 写入 | 不是只有前端禁用按钮；后端仍实施门槛 |
| 输出隔离 | root / `sim` 或 `phy` / camera_name；版本化 YAML | 仿真/物理目录分离，未见默认互相覆盖 |

不要把 provision 脚本中的额外单引号仅凭外观直接判定为 CSP 漏洞：这些参数可能经过 ROS CLI 的 YAML 解码，必须核对最终命令和响应头；本次不将此猜测列为 finding。对比去畸变图也只展示配置之间的映射变化，不能证明哪套 K/D 更接近真实相机。

## 5. 已执行与尚未执行

已执行：

```bash
python -m unittest discover -s xgc_camera_calibration/test -p test_intrinsic_startup_wall_clock.py -v
# 9 tests, OK（修复前失败）
python -m py_compile xgc_camera_calibration/scripts/intrinsic_calibrator_web.py xgc_camera_calibration/test/test_intrinsic_startup_wall_clock.py
python docs/reviews/intrinsic_math_probe.py
# 两组已知 K/D 模型一致性检查通过；输出上述相对门槛反例。
```

另执行了 Node 隔离回调逻辑复现 F3。没有运行整个 Python 测试目录、npm 构建、catkin 构建、Gazebo/ROS 端到端测试或真实相机测试；本地测试结果不代表远端 CI 已通过。

集成验收应覆盖：两种板型的仿真真值误差、暂停/恢复 `/clock` 与缺失相机、物理相机不同曝光/焦距与边缘采样、陈旧快照和源切换、候选期间并发采样与旧 candidate ID 拒绝、长时采样的 UI 响应、失败重置的真实错误显示，以及最终命令/响应头/代理路径和 YAML 下游加载。阈值由用途确定，不能把覆盖条变绿或 `save_ready` 作为全部验收。

## 外部数学 / API 依据

- [OpenCV 4.4 相机标定教程：针孔、径向/切向畸变与内参](https://docs.opencv.org/4.4.0/dc/dbb/tutorial_py_calibration.html)
- [ROS rospy 文档：sleep 使用 ROS 时间](https://docs.ros.org/en/kinetic/api/rospy/html/index.html)

这些文档支持模型和时钟语义；项目事实以本报告标注的固定源码为准。
