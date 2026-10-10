# Phase 2C frozen physics correction contracts

本阶段研究三个分离的问题：A/m parameter calibration、constitutive structural discrepancy、state estimation error。
M0同形同参；M1同形不同A/m；M2是合成参考 hazard `A*x^m*(1+beta*x/(1+x))`，不是实验疲劳律。

`HybridDamageLaw` 是独立 FatigueDamageLaw 组合器，不修改 Tri3、driver、DamageTransition 或原fatigue solver。
当前无量纲驱动x、损伤D、R、物理积分点坐标通过平滑basis `[G,K]`；一个全局 `[K]` 系数向量产生
`lambda_prior*exp(alpha*tanh(Phi*a))`，K=2/4/8，alpha=2，|a|<=4。不能输出G个独立损伤动作。
固定能量basis为 `(x/(1+x))^(j+1)`；无常数或log(x)项，避免与A/m的直接冗余。
空间basis使用明确mm参考原点和长度的cos函数；不依赖node/cell ID。
零prior hazard保持零；不包含新的损伤成核机制。该修正幅度范围也限制可表达的失配。

原Phase1B每步先当前D解min/max、算hazard、一次DamageTransition，再新D重新解FEM；失败原子rollback。
新CorrectionPredictor对每个参数候选从自身N0damage重演，使用public canonical time anchors和只读query分支。
DamageAndEvolutionParameters只声明参数归属：初态decoder仅消费d0，A/m/c用于law；初态不是自由G场。
旧PhysicsPredictor及全部原有数值代码保持原内容。

`CorrectionAcquisition`只读取严格public.json和cutoff内DICMeasurement；不导入training/evaluation/truth。
相同材料的多个load specimens共享一组d0/A/m/c，其DIC噪声条件独立。共享registration时全group只一个tx/ty/theta。
公开未来载荷计划不是未来观测。推断先完成所有forecast，再由evaluator打开truth和未来DIC计算误差。
train/validation/calibration/test按完整物理group划分，group内5load轨迹不跨split。
不同失配regime用相同case/noise seeds作配对对照；不能将这些配对复制当额外独立样本。

Mode A只作oracle state诊断；Mode B联合DIC前缀TRF估计d0及演化参数，复用原optimizer和whitened likelihood。
TRF完整prefix中每证据一次，coefficient正则一次；data Fisher与正则分开。
rank/full covariance只作局部信息诊断，模型失配下不授予interval coverage资格；不继承Phase2B未校准区间。
B3只N0/10，origin10；其N20是prediction，不能标成filtered20。其他主方法origin20。

T2只读synthetic train/validation **N<20** positive-hazard log-ratio标签；确切零hazard区域mask，无epsilon伪监督。
Torch梯度仅通过固定basis与K个有界系数；SciPy FEM不在autograd路径。checkpoint记录全部seed、split、hash与早停。
T3是真实FEM前缀多步目标的有限差分TRF，不声称端到端物理梯度或RL。
B4有额外oracle training信息；B0/B1/B2/B3/B5只DICprefix，必须分开解释。

冻结配置见configs/phase2c/study.toml。正式train/test分布、5training seeds、2OOD cases/scenario在查看test前固定。
未来载荷在N20后降低至每specimen原幅值的.25，以避免人为进入数值饱和，并研究驱动范围变化的长时预测。
这项载荷计划对所有方法一致，不根据结果修改。统计精度仅为CPU小规模原型。
