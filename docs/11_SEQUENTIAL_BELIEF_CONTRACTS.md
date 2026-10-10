# Phase 2A sequential belief contracts

隐藏状态包含当前积分点损伤、材料参数与历史。首版只估计 Phase 1C 的低维初始损伤参数、
fatigue A 和显式 nuisance；后验定义为 `p(theta | observed prefix)`，不宣称任意 G 维场后验。

`MechanicsBelief`（别名 `BeliefState`）保存参数 MAP、数据信息秩、先验信息、局部协方差、
弱/零方向、当前 physics plug-in damage、循环数、拟合残差及 JSON provenance。
`parameter_mean` 是局部高斯近似的中心/MAP。非线性解码得到的 `current_damage` 不是严格损伤后验均值。

| Contract | Meaning |
|---|---|
| ObservationFrame / ObservationBatch | 仅实测 DIC、mask、协方差和公开载荷，带唯一 evidence ID |
| ObservationDesign | 公开未来坐标、mask、噪声设计与诊断载荷，没有实测位移/反力值 |
| PredictedObservation | 指定坐标上的预测位移，带 role 与 conditioning IDs |
| MechanicsBelief | prior 或已同化 prefix 的 filtered 参数 belief 和当前损伤 |
| StatePrediction | 新观测之前的 prediction，或固定 prefix 的 future forecast |
| Innovation | 分开标记 pre-assimilation、post-fit residual、held-out forecast innovation |
| ForecastResult | 固定 origin、conditioning IDs，严格晚于 origin 的开环结果 |

坐标是任意 `[Q,2]` mm；积分点 G、观测点 Q 和 FE 节点 V 独立。不转换成 H×W。
每帧 active=ROI AND validity；协方差按 point-major ux/uy 排列。mask 先选子矩阵，再 Cholesky 白化。
confidence 不当作方差。无噪声例使用显式 `1e-10 mm²` likelihood floor，它不是实验噪声校准。

Prefix MAP 每次从已观测前缀构建 Phase 1C `InverseObjective`，调用原有 bounded SciPy/TRF。
完整 prefix 和原始 Gaussian prior 各计一次。之前 posterior 只提供 warm start，不充当新先验。
cold starts 来源是固定归一化参数箱中心，不读取真值。各 start 的所有残差评价、失败和预算耗尽都保留。
warm-start 与 cold-start 可得到不同局部解，应比较目标与参数；没有全局最优或单峰保证。

`PhysicsPredictor` 对每个候选从它自己的初始 damage 建立新的 `MaterialHistoryState`，
实际调用 Phase 1B solver。canonical integration anchors 由 load segments 与 cycle_step 决定。
非网格时刻从前一 anchor 分支，查询不改变之后的轨迹。缓存只包含完整候选键及目标时刻对应的
不可变历史；失败 rollout 不缓存、不发布。诊断/未来位移总用目标时刻的新损伤重求解 FEM。
模型支持固定 Tri3 mesh；当前无跨 mesh history transfer。静态独立 load modes 不演化损伤。

公开诊断 pattern 必须是零 damage、N=0 的模板，包含该观测工况的实际载荷/BC。
load_history 明确提供疲劳循环；未来载荷是公开计划输入，不是未来 DIC。
初始 Gaussian 只参数化 N=0；演化场保留逐积分点 physics rollout，不投影为 Gaussian。
未知 E0 时模板中的 E0 被替换为参数箱中点，候选 solve 才替换为候选值。

数据 Fisher=`Jw.T @ Jw`，prior Fisher 独立保存，posterior Fisher 是两者之和。
SVD 在归一化参数空间的 square-root information 上进行；rank threshold 同时包含有限差分误差。
full covariance 只有 posterior rank=K 时可取得；否则只提供明确命名的 identifiable-subspace covariance，
保留 null directions，受 null space 影响的 Gaussian marginal 标准差为 infinity。
参数箱约束不算 Gaussian information，不解释为小方差；boundary/near-boundary、未收敛和导数未验证
使 Gaussian propagation 不可用。局部 Gauss-Newton 近似忽略 residual-weighted second derivatives，
不证明后验全局 Gaussian/单峰。优化器收敛不等于数据可辨识。

默认各帧噪声条件独立。存在共享配准偏差时声明 `independent_given_shared_registration`，
并提供共享 tx、ty、theta（未知或固定）；不是每帧重置/独立计数的偏差。
其他未建模跨帧噪声相关被拒绝；目前没有通用联合跨帧 covariance API。
预测 NIS 使用 `R + J C_parameter J.T` 的局部近似；post-fit residual 不当作独立 innovation。
高斯参数采样拒绝界外样本，不 clip，不采样未辨识方向；拒绝/物理失败全部计数。
有拒绝时的统计是条件 Gaussian pushforward，而不是完整非线性 posterior。

时间契约在 `TemporalSplit` 中固定：N=0/10/20 assimilation，N=30/40 forecast。
`load_observation_prefix` 根据公开 schedule 筛选后才打开 observation-only 文件；不接受 truth path。
estimator 的 assimilation_cutoff 在建立 objective 前拒绝未来帧。forecast 接口只接受 designs。
未来 DIC、damage truth 和误差只在 forecast 生成后由 evaluation 计算；不进入优化或初值选择。

`estimate_initial`、`predict_to_cycle`、`assimilate_observation`、`forecast_future` 构成可替换接口，
`BeliefEstimator` Protocol 为 Phase 2B 保留。新 estimator 可通过 Graph/GRU 输出参数、参数不确定性和
条件场，但须保留观测/evidence/time/role 契约并在同一个 evaluator 中比较。
本阶段不实现网络、训练、RL、RUL 或任意 latent damage correction。

验证入口：`python -W error scripts/verify_phase2a.py --all --output NEW_DIRECTORY`。
全部配置包括两个 100-seed 科学统计试验；完整 qualification 应运行 CLI matrix 和完整 pytest。
pytest 只在几个种子上检查统计流程，不能据此声称完成 100-seed qualification。
科学结果目录只创建新目录，不覆盖旧证据；每例保存配置、版本、源码 SHA256、seed、全部优化 history、
Jacobian/Fisher、诊断与 numerical results。
