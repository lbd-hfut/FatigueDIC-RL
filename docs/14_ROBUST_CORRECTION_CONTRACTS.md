# Robust correction contracts — Phase 2D

单位沿用 mm、N/mm²、cycle、hazard cycle⁻¹。所有参考 law 是 synthetic benchmarks，不是实验标定材料律。

主预测仍用 Phase1B solver 与原不可逆 DamageTransition。每个参数候选必须从自己的 N0 状态重演公开历史。
积分点、FE node、element 与 DIC point 不是同一空间；低维 c=[c0,c1] 是全轨迹固定材料修正系数。
所有步失败由原 solver 原子回滚；不会将网络输出作为 G 个独立 damage 增量。

`HistoricalCandidate` 只接受四个候选的历史验证 SSE、有效参数维数、观测分量数、收敛、边界及导数诊断。
它不接受路径、law-family 标签、SyntheticTruth 或未来误差。训练历史为 N0/10；N20 是预留历史验证。
最终模型确定后可在 N0/10/20 重标定；N30/40/70/120/220 只能 open-loop physics rollout。

`select_model` 使用 SSE+2×practical rank；收益必须超过 max(9.21,2√(2M))。
该规则是保守工程规则，不声称非嵌套模型的 SSE 差具有精确 χ² 分布或校准 Bayes evidence。
实践可观测性要求 data-only J 的全部 box-normalized singular values≥3、两尺度差分可靠、无活动边界。
数值秩和实践秩分开保存；ridge 不作为新增测量信息。

先优先选 Physics、A refit、A/m refit，再检查 K2 是否有独立历史收益。弱信息时保留 Physics 或已验证简单参数模型。
full-prefix refit 若失去实践可观测性、触及边界或未来公开物理 rollout 数值失败，可拒绝部署并回到 Physics。
这不允许读取未来真实观测进行拒绝。所有初选、部署拒绝、未收敛和硬失败必须保存。

`choose_shrinkage` 仅对独立历史验证通过的 K2 用固定 gate=[0,.25,.5,.75,1]；
score=SSE_validation+2 gate²||a||²，额外收益仍需越过相同门槛。
gate 冻结后，若在(0,1)，将 full-prefix c 固定并只重估 d0/共享 nuisance。
λ=λprior exp(gate α tanh Φa)，α=2；gate=0 精确保留 prior，zero hazard 仍为零。
这不是新的成核机制，也不是 RL policy。

`infer_models` 只读严格 public.json、N≤20 DIC 与显式授权的独立反力观测。
所有候选 forecast 完成后，独立 evaluation 模块才读真值、law-family 与未来 DIC 来评分。
shared tx/ty/theta 在全部 frames/specimens 中只声明一次；测量 covariance 为给定这些 nuisance 后的残余噪声，不能重复加入配准方差。
load_factor 只有有声明的独立载荷/反力信息或 nuisance 对照时开放；收敛不表示幅值与 damage 唯一可辨识。

输出复用 MechanicsBelief、PredictedObservation、StatePrediction、ForecastResult；
filtered 与 history-only failed-refit 的 prediction 角色不能混淆。所有位移指标必须匹配 physical/registered 坐标系。
Fisher 仅用于局部信息诊断，不继承 Phase2A/2B 未完全校准的 coverage 资格。
统计样本单位是独立物理 case；同一组五个加载 specimens、多个时刻、测点及配对网格不作额外独立样本。
