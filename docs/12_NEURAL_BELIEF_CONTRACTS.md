# Phase 2B 神经 belief 接口与实验契约

核心仍为可选 NumPy/SciPy 物理模块。torch 仅在 `neural` 路径导入；未增加 PyG/RL 依赖。
NeuralBeliefEstimator通过注入backend的observation_update调用图网络，自身不import torch/networks，
保留全部既有inference隔离测试；backend只接收ObservationFrame和公开物理模型。
网络与 Phase 2A 共用 ObservationFrame/Batch、MechanicsBelief、PredictedObservation、Innovation、
ForecastResult 和 TemporalSplit。额外 `initialize_belief` 是 `estimate_initial` 的兼容入口。

第一版未知固定参数[d0,fatigue A]，采用 Phase1C box 的无量纲z∈[-1,1]；初始damage参数与
fatigue-law参数不会成为随循环变化的控制动作。六参数Gaussian命名为
[d_base,amplitude,xc,yc,sx,sy]，其中amplitude≡A_D，与fatigue A不同。
所有damage通过物理坐标采样到G积分点，未来演化不强制保持Gaussian。

Q个active DIC节点由物理坐标kNN连接；几何等距离tie全部保留，Q=1使用self edge。
V个FE节点由Tri3 connectivity连接，element/积分点数据若要使用必须显式投影。
ROI AND validity先选择测量及协方差principal submatrix；无效NaN不进入网络。
空ROI、active域外点和重复坐标拒绝，要求调用方先建立采集层合并/联合协方差规则。
近邻点不因距离自动被视为独立证据；covariance相关矩阵提供有效分量数，池化和Pᵀ聚合均归一化。
采集confidence不作为inverse variance。

FE→DIC使用原Tri3形函数；DIC embedding→FE是按Pᵀ权重和逐node质量归一化，
无观测覆盖的FE节点保留独立FE特征。它是信息关联特征，不是posterior precision公式。
shape与global pooling要分别评价。DIC/FE重新编号只改变中间node张量顺序。

Cholesky innovation r=L⁻¹(o−prediction)用于原似然诊断。其分量随点排列依赖三角分解，
因此图node通道使用唯一对称白化Σ⁻¹/²e；范数与Cholesky相同，随点重排等变。
whitened node特征再用asinh(r/10)缩放，似然和NIS不使用该缩放。
统一长度尺度1mm、位移.01mm、σ通道1e−4mm、E通道2000N/mm²、traction10N/mm²、
时间40cycle；显式N和delta_N，不使用observation index充当疲劳循环。
global displacement-gradient/intercept特征由坐标上的最小二乘仿射回归计算，未解damage参数或读取truth。

Physics-guided条件每次从previous候选初始参数N=0重演Phase1B，重新FEM求目标damage响应。
参数更新后重新建立新的候选历史；新参数不套入旧material state继续推进。
网络给出有界中心及无量纲delta，记录before/after、pre-update预测和innovation。
GRU因果处理history；forecast固定N≤20 posterior，未来只有公开ObservationDesign。

训练是synthetic-supervised amortized inference。SciPy条件是停止梯度的预计算输入；
torch梯度仅穿过图encoder、融合、GRU、parameter/covariance heads。
训练条件混合nominal和训练label附近有界扰动；这叫teacher forcing，仅允许在train。
validation缓存使用声明nominal，不能用truth初始化；部署每步使用真实previous网络估计重演物理。
teacher/deployment差异必须通过真实部署validation测试检查，不能只报告teacher-forced准确性。
完整部署validation发现该差异后，uniform最终physics模型改为每步用自身previous mean，
停止梯度地更新条件，train/validation均不使用teacher label作为条件。
训练专用scalar homogeneous law与72次独立Phase1B/FEM对照，包括adaptive subdivision；
其假设仅适用于均匀损伤、均质矩形、自由泊松收缩、x方向10N/mm²载荷、R=-1、m=1、step2。
数据真值、部署推断及预测仍使用独立SciPy求解器。Gaussian原teacher-conditioned训练作为有限扩展保留。
额外observation-only控制使用训练acquisition的经典TRF pseudo-label，不读取内部damage标签，
fatigue A在N0不可识别，只能使用声明先验，不能把此控制称为无先验联合恢复。

四个split按独立物理case/轨迹seed固定分配；不同帧/重复noise不能跨split。
训练loader拒绝calibration/test；推断loader仅打开cutoff内observations，永不打开labels/future。
完整2×2及一般K×K Cholesky covariance是学习近似，PSD不表示校准或可辨识。
未知E0/损伤等严格退化输入应拒绝唯一damage输出，单独显示解析等价解或Eeff。
弱方向/单帧A信息不足用独立Fisher诊断展示，不能把training prior当作新增DIC信息。
参数Gaussian边际不clip到box后声称真实posterior；接近边界应标记，非线性damageplug-in不是后验均值。
