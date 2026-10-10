# 04 — DIC 观测与 FE 到 DIC 的映射

## 正式观测

DICObservation 保存 coordinates `[Q,2]`、displacement `[Q,2]`、valid_mask `[Q]`、
confidence `[Q]`、可选 covariance `[Q,2,2]`、cycle_count、loading_phase、
loading_state_ref、coordinate_system、coordinate_units 与 displacement_units。
covariance 的单位是 displacement_units 的平方，是每个点的二维位移协方差，
不是 Q 个点之间完整的空间协方差。
有效点的位移/协方差必须有限，协方差在相对浮点容差下对称且半正定。
质量分数属于 `[0,1]`；无效点可以保存 NaN 位移和 NaN covariance。

正式接口没有 damage/stress truth。metadata 只接受 source、specimen_id、camera_id、
calibration_id、timestamp、acquisition_id 的标量采集信息，拒绝任意标签张量。
loading_state_ref 只是载荷引用，没有可读的隐藏状态对象。
这防止常见的意外泄漏，并不构成针对恶意调用方的访问权限机制。
数据集与评价模块仍需管理 truth 通道隔离和训练数据审计。

## 形函数映射

对单元节点 \(x_0,x_1,x_2\)，解
\[
[x_1-x_0\ \ x_2-x_0]\begin{bmatrix}\xi\\\eta\end{bmatrix}=q-x_0.
\]
使用 \(N_0=1-\xi-\eta,\ N_1=\xi,\ N_2=\eta\)，得到
\[
u^{pred}_{DIC}(q)=N_0u_0+N_1u_1+N_2u_2.
\]
这是线性三角形 Lagrange 的参考形函数，见
[DefElement 的一次三角形定义](https://defelement.org/elements/examples/triangle-lagrange-equispaced-1.html)。
由分片线性多项式的再现性，本项目预期任意仿射位移在合法三角形内精确再现，
误差只来自浮点运算；该性质由独立解析测试验证。

实现按单元尺度归一化局部线性方程，兼容顺/逆时针。
每个有效查询点记录三节点索引与三权重，相当于稀疏 \(P_{F\to D}\) 的一行。
对二维位移两个分量使用同一权重，不构造 Q×V 的稠密矩阵。

## 有效性、边界与缓存

`ObservationOperator(mesh).prepare(points, coordinate_units=..., coordinate_system=...)`
返回不可变 ObservationMapping；`mapping.apply(nodal_displacement)` 返回 observation FieldData。
prepare 的坐标单位/坐标系必须与 mesh 完全相同，转换应在映射前完成。
apply 的 FE 位移必须引用该 mesh 的 node 空间，若自带坐标则也须与 mesh 一致。
输出位移保留输入 displacement units，不进行隐式单位转换。

三重心坐标均位于 `[0,1]` 的点有效。默认仅接受 64×机器 epsilon 的无量纲边界容差，
容差可显式减小，最大允许 1e-10；容差带内数值舍入的点吸附到边界。
不实施物理意义的域外外推。共享边界点确定性地选择 connectivity 中第一个单元。
真实网格外点：element/node 索引为 -1、权重为零、mask 为 False、预测位移为 NaN。
相关 FE 单元节点有无效位移时，观测预测也无效，包括零权重节点，采用保守规则。
后续 likelihood 的有效集合必须是 `DIC mask & mapping/prediction mask`，不能只用任一 mask。

每个 operator 绑定一个不可变 mesh，使用查询坐标内容的 SHA256 和有界 LRU 缓存。
相同坐标可复用映射；cache_size=0 可禁用，clear_cache 清空。
缓存不是持久化磁盘缓存，没有多线程共享保证；不同 mesh/坐标系应创建新 operator。
查询点允许为空；非有限坐标被拒绝，不能静默丢点。

## 范围限制

tri3 的线性形函数是本轮唯一插值 backend。
MeshData 支持 quad4/混合拓扑，但本算子会明确抛 UnsupportedElementError，
不能静默把四边形拆成未经声明的三角形。
每次未缓存的点定位复杂度 O(EQ)，适合验证与小网格。
大网格需要空间索引和批处理，仍须保留支撑、mask 和边界规则。
当前是 DIC 点采样模型，未建模实际 DIC subset/窗口平均、空间相关噪声或纹理误差。

## Smoke 的科学边界

使用四节点、两个不规则 tri3 与八个 DIC 点：六个域内/边界点，两个域外点。
节点位移直接来自解析仿射场；没有 FE 求解、没有疲劳实验。
误差与损伤更新结果记录在 [实施报告](PHASE0_IMPLEMENTATION_REPORT.md)。
