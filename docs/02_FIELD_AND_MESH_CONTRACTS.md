# 02 — 网格与场数据契约

## 独立空间基数

| 空间 | 记号 | 典型值形状 | 位置声明 |
|---|---|---|---|
| FE 节点 | V | 位移 `[V,2]` | node |
| FE 单元 | E | damage `[E]` | element |
| 积分点 | G | strain `[G,3]` | quadrature |
| DIC 查询点 | Q | 位移 `[Q,2]` | observation |

任何空间数量都可相同或不同；相同长度不表示同一个支撑。
规则图像可作为外部输入格式，但不得代替统一 mesh/field 接口。

## MeshData

输入节点 `[V,2]`、逐单元 ragged connectivity、逐单元类型、材料区域 `[E]`、
边界标签、长度单位、坐标系、mesh_id 与几何 metadata。
支持 tri3、按周边顺序给出的凸 quad4 及二者混合。
每种单元的 arity 由 ElementType.node_count 定义；高阶单元须新增类型、合法性验证及算子。
顺/逆时针均可；单元与节点顺序原样保存。

验证有限坐标、二维形状、整数且不越界的索引、单元节点唯一、单元非退化、
重复单元、材料区域非负、非流形边（共享超过两单元）与外边界标签。
几何判定按单元尺度归一化，极瘦退化单元会被拒绝。
四边形只接受凸、非交叉的低阶几何；暂不支持一般非凸四边形。

BoundaryTag 保存 node_ids `[B]` 和 facets `[F,2]`；标签不是约束本身。
DisplacementBC/TractionBC 在 SolverRequest 中指定约束与载荷。
不执行全域网格重叠、T junction、全域连通性、重复坐标、孤立节点等昂贵诊断；
backend 接入前必须增加相应质量检查，不能把 MeshData 验证等同于网格求解适定性。

`mesh.field_reference(node/element)` 返回规范空间引用。
重网格或改变节点/连接关系必须赋予新的 mesh_id，并显式映射旧场。
本轮没有重网格器、网格文件导入器或 GNN 图构建器。

## FieldReference 与 FieldData

FieldReference：`space_id, location, count, mesh_id`。
积分点必须单独声明，例如
`FieldReference('mesh-A:tri3-rule-v1', 'quadrature', G, 'mesh-A')`。
布局的积分阶数、权重与 element ownership 尚未实现；引用身份由调用方负责管理。

FieldData 输入：values `[count,...]`、location、units、reference 和/或 coordinates、
valid_mask `[count]` 与 metadata。带坐标时必须指定 coordinate_units 和 coordinate_system。
坐标必须有限；只有无效位置上的值可以为 NaN/Inf，算子输出选择 NaN。
mask 必须是真正的布尔数组，不能把整数隐式视为布尔值。
数值输入复制为不可写数组，外部数组变动不会改变场或缓存。

`require` 检查位置、单位、分量形状、完整有效性。
`assert_same_support` 比较位置、数量、引用、坐标/坐标系及掩膜；
`assert_compatible` 进一步检查单位和分量形状。
`with_values` 产生新字段，保留支撑契约，改单位必须显式传入。
没有场之间的隐式算术、单位换算、节点/单元投影或插值。

## MechanicsState

输入 mesh_id、role、damage、cycle_count、loading_state，及可选 displacement、
strain、stress、internal_variables。输出为验证后不可变状态。

damage：`[n]`，单位 `1`，node/element/quadrature，完全有效，范围 `[0,1]`。
displacement：`[V,2]`，node，单位 m/mm。
strain/stress：`[n,3]` 或 `[n,2,2]`，力学位置；strain 单位 `1`，stress 显式 Pa/kPa/MPa/GPa。
所有力学场必须引用 state.mesh_id；只有 mesh_id 的状态容器无法验证实际 V/E，
SolverRequest 和 SolverResult 通过具体 MeshData 执行这层检查。
internal_variables 是 `name -> FieldData`，每个变量也必须声明位置与单位。

`[n,3]` 的分量顺序/剪应变约定由具体本构定义；示例是工程应变
`[exx,eyy,gamma_xy]` 和应力 `[sxx,syy,sxy]`。跨模型适配必须显式转换，不能只检查 shape。
采用更一般张量、3D 或概率 belief 时应新增明确契约，不改变字段含义来绕过检查。

响应可以为 None，用于尚未求解的新状态。修改 damage 后，不得继续把旧响应当作平衡解。
DamageTransition 只返回新 damage FieldData，重求解/历史变量更新由 orchestration 负责。
内部变量是否保留、推进或作废取决于具体模型，不能统一清空塑性历史。

LoadingState 用 load_id/phase 引用外部载荷配置；CycleLoading 给出非负 delta_cycles，
以及可选的正频率 Hz。力/位移数值和材料参数在 solver request 中明确给出。
