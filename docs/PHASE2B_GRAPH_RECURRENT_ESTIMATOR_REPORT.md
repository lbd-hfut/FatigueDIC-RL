# Phase 2B Graph + GRU sequential damage estimator

日期 2026-10-08；资格：**PHASE2B_GRAPH_ESTIMATOR_PARTIALLY_QUALIFIED**。
本阶段为 synthetic-supervised neural amortized sequential inference；完成后停止，无神经物理 correction/RL训练。
资格限于声明的已知 E0/BC、低维参数、合成采集分布；不等于任意G维damage posterior、真实材料校准或RUL。

主结论：五seed Uniform独立test的d0 RMSE .004333–.006085，A RMSE .0001292–.0001459；
受控点估计及时间留出gate通过。经典MAP仍更精确。隔离CPU基准示例，warm MAP 2.759s、
shape+cov .1491s，约18.5倍参数belief在线提速；FEM调用2535.7→129，NN本身约3.58ms。
这是三个固定validation案例的操作基准，不是所有geometry/hardware的加速保证；future uncertainty
传播仍约.711s，不能将NNforward时间冒充整个带不确定性forecast流程。
独立固定材料100noise复核中，A coverage最低.76、D40最低.55；registration bias下部分d0 coverage为0。
这些失败比总体test coverage更直接限制不确定性资格，未据此重新调校准系数。

## A. Git/source/environment audit

分支feature/phase0-architecture，HEAD dc239d1b7b6dd1e44d93e71bdcccccc0f12e8bc9仅Initial commit。
未提交工作树包含完整已验证阶段代码，HEAD不能代表科研基线；无commit/push/merge/reset/clean。
开发前169文件可恢复tar SHA256 `577b52dd46094a97d78f263228cd0db87f38910d02a08aaebc7ae3c7d678b849`，
manifest SHA256 `a2af2c71225546120baff04ca8ff846c4541b259b7223db9aaba0e7e1e02849a`；逐项恢复校验169/169。
历史10133结果文件独立登记哈希，源码快照排除results/cache/旧tar。原数值求解器及历史报告保留。
最终完整manifest、增量patch、恢复tar及历史资产复核在本run `delivery/`，其哈希由`delivery/manifest.json`登记。
正式源码冻结manifest SHA256 `aaeb418b9e4011800e107aab0abebcdbaf08146749b9bb9b9e11525a6eeb3e0e`、tar SHA256 `5066e2f1842bd74ac17f10e4075ed5f63541b5f56c1b88c3cbeb6e1902cdb886`。
完整199文件清单与可恢复tar在本run根目录；环境133个安装包的名称/版本见`environment_packages.json`。
Python3.13.12/numpy2.4.4/scipy1.17.1/torch2.13.0/pytest9.0.3/matplotlib3.10.9。
CUDA编译版本13.0，但nvidia-smi无法连接driver、torch CUDA unavailable、0devices，正式训练采用CPU。
16CPU约62GBRAM；每worker numerical/torch thread1；没有破坏原环境、安装PyG或大型RL依赖。
[完整baseline](../results/phase2b/20261008T114713Z/baseline/audit.json)，[Git基线建议](GIT_BASELINE_RECOMMENDATION.md)。

## B. Phase 2A uncertainty follow-up

原共享registration情景d0为89/100，Wilson95% [.81369,.93746]；A/tx/ty/θ为94/92/95/92%。
原D30/D40coverage95/95%，predictive normalizedNIS1.027/.959。原Phase2A部分资格不变。
新预冻结seeds22000–22099，d0/A/tx/ty/θ为97/96/95/94/96%，d0Wilson [.91548,.98975]；
D30/D40 .95/.94，NIS1.0280/1.0253。全Fisher交叉块非零，重建JᵀJ误差0，priorF=0，
同一个tx/ty/θ用于全部帧，未把共享nuisance再次加入R。原d0–A相关−.5615..−.5372，远离box边界。
首例observed Hessian与GN相对差3.175e−6；profileCI [.1191514,.1280894]，GN [.1191669,.1281216]。
16bootstrap仅形状诊断不作为coverage资格。新复核未重复显著欠覆盖，不撤销原89/100，未事后调整SD。
[完整专项报告](PHASE2A_UNCERTAINTY_FOLLOWUP.md)。

## C. Network architecture

第一版参数θ=[d0,fatigue A]，Phase1C物理box归一化z∈[-1,1]。64hidden、3层SiLU残差message passing，
DIC与FE双图encoder；1层64hidden GRUCell；tanh有界参数中心及显式before/after/Δz，完整Cholesky covariance。
d0/A为材料固定参数的估计，GRU更新belief，不成为随循环改变的控制action。
六参数扩展[d_base,amplitude(A_D),xc,yc,sx,sy]，fatigue A固定.0015；不输出G个自由damage值。
PyTorch可选extra `neural`；基础NumPy/SciPy导入不加载torch。NeuralBeliefEstimator注入graph backend，
自身不导入learning/truth，保留Phase2A原接口隔离测试。

## D. Graph and DIC–FE fusion

DIC图由真实mm坐标kNN(k6)建立，距离tie全保留；Q=1selfedge，支持16/37/64/256及变量Q。
ROI∧validity先选测量和协方差principal submatrix，无效NaN不进入网络；空ROI、域外、重复坐标明确拒绝。
FE图来自Tri3 connectivity；V node、element、G quadrature和Q DIC支持严格区分，材料/BC到node显式投影。
原FE→DIC形函数P定位并插值。fusionA global pooling，B normalized Pᵀembedding质量归一化；
无观测FE node保留FE分支。它不作为逆协方差，confidence不当precision；pooling归一化和fullcov有效点数
避免把高度相关/重复密集点数直接解释为新增独立证据。实际训练的点数范围16/64，其他点数为OOD诊断。
图通道采用对称Σ^−1/2e保证重排等变，其范数等于Cholesky L^−1e；真实innovation/NIS使用原Cholesky。
尺度：xy1mm，u.01mm，σ1e−4mm，E2000N/mm²，traction10N/mm²，N及ΔN除40cycle；
global仿射梯度/intercept是公开观测的最小二乘特征，未解damage/读取truth。
[接口契约](12_NEURAL_BELIEF_CONTRACTS.md)。

## E. Temporal physics-guided belief update

共有initialize_belief/estimate_initial、predict_to_cycle、assimilate_observation、forecast_future，复用
ObservationBatch/DICMeasurement/MechanicsBelief/PredictedObservation/Innovation/ForecastResult/TemporalSplit。
每次旧θ从N0建立候选一致history→Phase1B推进→新damage FEM→DIC预测→真实innovation→graph/GRU更新θ。
新θ重新从N0演化；禁止将其套入另一个候选的旧history。滤波前、后、future角色及conditioning IDs分别保存。
FE失败或backend异常整批belief/hidden rollback；未收敛MAP保留，不冒充成功收敛。
显式N与ΔN，N0/5/20与N0/10/20不是等价观测index；core拒绝未来、重复ID、非因果帧。
本K2 neural不估sharedregistration nuisance，显式shared-noise契约拒绝；偏置实验是unmodelled discrepancy。

## F. Parameter/uncertainty heads

完整2×2及6×6 Cholesky SPD，参数physical scaling显式。训练NLL用synthetic真实参数，不拟合Phase2A covariance。
N0对A的data Fisher rank1；learned有限A variance来自training prior而非新增DIC信息，显式warning。
N0 point A中心监督为声明training prior均值.0015；后续用真实参数监督，未用真实参数推断初始化。
Gaussian边际区间不clip后伪装posterior；box近边界/高斯mass跨box标记；数值可传播不等于统计校准。
当前damage是F(参数中心)plug-in，不等于E[D]；damage/u区间为checked FD局部参数pushforward。
unknown E0/非零imposed displacement等不支持的严格退化情景拒绝唯一damage输出；保留经典data-rank诊断。
fullcov输出与training-prior依赖不能当作data-only posterior唯一可识别性。

## G. Dataset/leakage audit

Uniform832独立物理轨迹，train512/val64/cal128/test128，seeds31000..31831；
Gaussian224，128/32/32/32，seeds61000..61223。按原始case分配，不按帧随机拆分。
每轨迹均直接Phase1B+Tri3生成truth、独立DIC坐标及完整noise covariance；public/observations与labelsNPZ分开。
Uniform训练E1800..2200、d0 .04..28、A .0006..0024、σ2..8e−5mm，80%独立/20%空间相关，
矩形width1.8..2.2/height.9..1.1、两类mesh nx2/4、少量missing与regular/random点集。
Gaussian q64、nx4/6、σ5e−6..2e−5，小规模训练范围见config/逐casepublic.json。
训练loader仅train/val，推断loader仅prefix公开观测；预测N30/40完成之前不打开未来DIC或labels。
训练仅N0/10/20；future truth不参与梯度、早停、超参数或校准。evaluation之后才读取未来用于指标。
所有case public/label/source/config hashes及noise seeds在manifest；future dropout预未知，forecast用prefix规划ROI。

## H. Training/cost and milestones

M0完成只读不确定性复核与完整357test baseline（292.03s）。M1数据契约/确定性physics数据；
M2五seed graph初估validation d0 RMSE .00759–.01014，单N0 A不可识别，未跳过基础gate。
M3原teacher-conditioned训练部署validation所有1920推断成功，但Gseed41002 A RMSE .0005342，
且通常弱于无physics GRU。保留失败/较差cohort；在cal/test之前诊断teacher→自身参数的条件偏移。
另一个Fseed41004过早停38epoch，val loss .0415；统一recurrent minimum80epoch，在validation-only独立目录
复跑该固定seed154epoch，val .002151，冻结选择；其他四seed原训练均≥80，不换seed。
最终Uniform G/H采用own previous mean的causal训练条件（不teacher输入），scalar均匀等价law在72次独立
Phase1B/FEM对照、adaptive大损伤候选验证。仅训练条件省成本；正式数据、推断和forecast均原SciPy。
条件全部stopgradient，GRU/graph/head仍torch梯度；没有穿越NumPy/SciPy autograd或伪造observation physics loss。
从随机初始化的cold-causal cohort仍有约.02 normalized validation loss；最终shape用已验证F的权重初始化，
仅将新增physics通道权重初置0（仍可训练），causal微调lr1e−4、最多80epoch、minimum40、patience25。
五seed选择epoch为1/14/23/1/13；不是用test选择最好seed。
global最终使用cold-causal cohort，shape使用F初始化；所以两者最终差异不能归因于fusion本身。
相同训练设置下的global/shape对称cold及teacher-cohort validation均保留；未证明形函数融合单独更优。
AdamW1e−3、weight decay1e−4、clip1、batch32、validation early stopping；所有五固定seeds保留。
point epochs上限160(uniform)/120(Gauss)，covariance60/40；cov阶段冻结point/encoder/GRU，仅训练uncertainty。
E observation-only控制用N0观测TRF pseudo-label，A声明先验，无内部damage labels；它无joint fatigue监督。
CPU实际offline生成/每seed训练、checkpoint/hash、loss histories、validation选epoch及最初失败日志均保留。
共71个checkpoint（含所有诊断/cohort），累计模型process-wall10587.96s（2.94h），不是测得的CPU core time；
primary dataset生成65.28s/25.94s，五seed与完整代价清单见`training_cost_audit.json`。
原始serial在seed41001四模型完成后由独立workers接管余四seed；预留目录触发no-overwrite exit1，
属于调度handoff，不是已保存模型训练失败。该原始exit1日志未删除。
早期训练label NPZ会materialize全轨迹数组，但loss只取固定参数，未来damage/u从未作为条件、梯度或选择指标；
最终训练/校准loader收紧为只materialize两个参数数组，并用NPZ访问追踪测试禁止未来field数组读取。
GPU不可用，未记录虚构GPU memory或训练加速。最终cohort选择由validation在校准/test前冻结。


## I. Controlled parameter/state recovery

全128 test cases的每seed指标（全部权重冻结后首次打开test评价；不挑最优seed）：

| 方法/seed | d0 RMSE | A RMSE | Current D RMSE | D40 RMSE | Online s | FEM solves |
|---|---:|---:|---:|---:|---:|---:|
| graph_gru_seed_41001 | 0.00613838 | 0.000139018 | 0.00488676 | 0.00636563 | 0.313347 | 123 |
| graph_gru_seed_41002 | 0.00529791 | 0.000136324 | 0.00475101 | 0.0063412 | 0.317057 | 123 |
| graph_gru_seed_41003 | 0.00620869 | 0.000139444 | 0.00462027 | 0.00560201 | 0.318207 | 123 |
| graph_gru_seed_41004 | 0.00721613 | 0.000142337 | 0.00537954 | 0.0062202 | 0.308421 | 123 |
| graph_gru_seed_41005 | 0.00554786 | 0.000139966 | 0.00489118 | 0.00659722 | 0.314385 | 123 |
| graph_initial_seed_41001 | 0.00825842 | 0.000520073 | 0.0122458 | 0.0222699 | 0.0153318 | 1 |
| graph_initial_seed_41002 | 0.00950794 | 0.000517777 | 0.0122699 | 0.0221508 | 0.0148118 | 1 |
| graph_initial_seed_41003 | 0.00773123 | 0.00051956 | 0.0120138 | 0.0222023 | 0.0154938 | 1 |
| graph_initial_seed_41004 | 0.0100508 | 0.000519868 | 0.0131371 | 0.0229187 | 0.0150835 | 1 |
| graph_initial_seed_41005 | 0.00846081 | 0.000518739 | 0.0123613 | 0.0223671 | 0.01498 | 1 |
| graph_observation_only_seed_41001 | 0.00280331 | 0.000524423 | 0.0105071 | 0.0216164 | 0.0153616 | 1 |
| graph_observation_only_seed_41002 | 0.00513029 | 0.00052406 | 0.0107169 | 0.0215015 | 0.015254 | 1 |
| graph_observation_only_seed_41003 | 0.00262112 | 0.000524327 | 0.0103878 | 0.0214001 | 0.0151274 | 1 |
| graph_observation_only_seed_41004 | 0.00471214 | 0.000524092 | 0.0110092 | 0.021842 | 0.0155086 | 1 |
| graph_observation_only_seed_41005 | 0.00345103 | 0.000524015 | 0.0105265 | 0.0215251 | 0.0150628 | 1 |
| independent_frame_inverse | N/A | N/A | 0.00148692 | N/A | 0.108758 | 25.9375 |
| physics_global_seed_41001 | 0.0139642 | 0.000405163 | 0.0110349 | 0.0143591 | 0.325557 | 129 |
| physics_global_seed_41002 | 0.0131414 | 0.000405857 | 0.00923601 | 0.0136986 | 0.326361 | 129 |
| physics_global_seed_41003 | 0.0122875 | 0.00042499 | 0.00687938 | 0.0112705 | 0.324339 | 129 |
| physics_global_seed_41004 | 0.0107122 | 0.000353203 | 0.00516147 | 0.00833451 | 0.319484 | 129 |
| physics_global_seed_41005 | 0.0143652 | 0.000413933 | 0.00833005 | 0.0100374 | 0.321497 | 129 |
| physics_prior | N/A | N/A | 0.0595461 | 0.0653365 | 0.0096735 | 0 |
| physics_shape_seed_41001 | 0.00574617 | 0.000145853 | 0.00409899 | 0.00529837 | 0.326561 | 129 |
| physics_shape_seed_41002 | 0.00608464 | 0.000133492 | 0.00483735 | 0.00591031 | 0.337734 | 129 |
| physics_shape_seed_41003 | 0.00433337 | 0.000129268 | 0.00349896 | 0.00510034 | 0.326897 | 129 |
| physics_shape_seed_41004 | 0.00495136 | 0.000133137 | 0.00408256 | 0.00546039 | 0.322863 | 129 |
| physics_shape_seed_41005 | 0.00510492 | 0.00012918 | 0.00410949 | 0.0053566 | 0.317355 | 129 |
| physics_shape_uncertainty_seed_41001 | 0.00574617 | 0.000145853 | 0.00409899 | 0.00529837 | 0.320425 | 129 |
| physics_shape_uncertainty_seed_41002 | 0.00608464 | 0.000133492 | 0.00483735 | 0.00591031 | 0.331931 | 129 |
| physics_shape_uncertainty_seed_41003 | 0.00433337 | 0.000129268 | 0.00349896 | 0.00510034 | 0.320687 | 129 |
| physics_shape_uncertainty_seed_41004 | 0.00495136 | 0.000133137 | 0.00408256 | 0.00546039 | 0.325733 | 129 |
| physics_shape_uncertainty_seed_41005 | 0.00510492 | 0.00012918 | 0.00410949 | 0.0053566 | 0.3241 | 129 |
| prefix_cold_map | 0.00154049 | 0.000126718 | 0.00127603 | 0.00358762 | 10.1192 | 3018.94 |
| sequential_warm_map | 0.00154049 | 0.000126718 | 0.00127603 | 0.00358762 | 9.13604 | 2699.62 |

## J. Multi-step forecasting

N≤20 assimilation与N30/40 open-loop严格分离。E/control只用N0，forecasts origin0；F/G/H/MAP使用全prefix，origin20。
independent current-frame inverse没有初始d0/A识别，forecast为N/A。所有current/displacement/heldout spatial/D30/D40原始每case结果
在test_final/records.json与raw/；forecast未持续读取未来DIC修正。

![open-loop](../results/phase2b/20261008T114713Z/figures/open_loop_forecasts.png)

## K. Noise/registration calibration

每seed独立cal128，以每prefix/参数标准化绝对error的预定95%分位数确定D尺度，Ccal=D Craw D；相关项保留。
具体q=ceil((128+1)*.95)/128、NumPy `higher`，实际取第124顺序统计量，含有限样本保守padding；
Gaussian cal32取最大值。没有声称这给出精确95%联合或逐条件覆盖保证。
它是learned marginal error calibration，不是精确联合posterior/全条件coverage保证；未在test上调SD。
full noise R用于innovation和predictive NIS，field点均值不冒充独立Bernoulli样本。Wilson间/width/standardizedRMSE
在summary，未来damage、mean displacement与predictive observation分开。N0弱A方向必须保持prior依赖。

| seed | d0 raw/cal coverage | A raw/cal coverage | Current D coverage | D30/D40 coverage |
|---|---|---|---:|---|
| physics_shape_uncertainty_seed_41001 | 1.0000/0.9844 | 1.0000/0.9375 | 0.9844 | [0.984375, 0.9921875] |
| physics_shape_uncertainty_seed_41002 | 1.0000/0.9609 | 1.0000/0.9688 | 0.9453 | [0.9453125, 0.9453125] |
| physics_shape_uncertainty_seed_41003 | 1.0000/0.9688 | 1.0000/0.9375 | 0.8906 | [0.8984375, 0.9140625] |
| physics_shape_uncertainty_seed_41004 | 1.0000/0.9609 | 0.9766/0.9766 | 0.9375 | [0.9609375, 0.9453125] |
| physics_shape_uncertainty_seed_41005 | 1.0000/0.9922 | 1.0000/0.9141 | 0.9141 | [0.9296875, 0.9453125] |

![calibration](../results/phase2b/20261008T114713Z/figures/parameter_calibration.png)

固定mechanics .12/.0015、seeds53000–53099，每场景100独立采集：

| Noise scenario/seed | d0/A coverage | D40 coverage | NIS30/40 |
|---|---|---:|---|
| independent/41001 | [1.0, 0.96] | 1.0000 | [1.020203124955055, 0.9890053372150112] |
| independent/41002 | [1.0, 0.89] | 1.0000 | [1.021910209962237, 0.9870962140925283] |
| independent/41003 | [1.0, 0.87] | 0.6400 | [1.1033068677071147, 1.076925323345993] |
| independent/41004 | [1.0, 0.93] | 0.5500 | [1.1373568433457186, 1.1051229260579756] |
| independent/41005 | [1.0, 0.76] | 0.8200 | [1.038944532346217, 1.0266668115628774] |
| registration/41001 | [1.0, 1.0] | 1.0000 | [15.216793515901193, 15.15039061312018] |
| registration/41002 | [0.0, 0.75] | 0.0800 | [15.21670695344767, 15.151048675131712] |
| registration/41003 | [0.0, 0.58] | 0.0100 | [15.255427212927168, 15.18859424642789] |
| registration/41004 | [0.0, 0.87] | 0.0000 | [15.282927404402903, 15.223759977878045] |
| registration/41005 | [0.0, 0.58] | 0.0000 | [15.221683207045276, 15.158616700289095] |

配准bias对K2模型和K2 MAP均未建nuisance；与Phase2A K5共享配准复核物理信息不同，不能声称公平同等nuisance识别。
固定matched mechanics的A coverage为.96/.89/.87/.93/.76，D40为1/1/.64/.55/.82；
parameter/calibration与field pushforward不是同一个资格。NIS约.99–1.14接近1，仍不能证明参数或damage区间校准。
registration的future NIS约15.15–15.28，暴露持续观测失配；bias/width/coverage每seed均保留。

## L. Cross-mesh/OOD/Gaussian

17预声明场景×12独立case，加12paired cross-mesh reference（不是额外独立试验）；包含Q16/64/256/37、
regular/clustered、independent/spatial、registration、unknown load1.1、nx8 mesh、E2600、3×.7 geometry、
m=2 model discrepancy、no noise、missing45%、irregular N0/5/20。ID calibration原样transfer，不用OOD拟合。
paired mesh保持同参数/点/噪声，truth displacement差<1e−12；原始paired每case gap见summary。
共3456条OOD结果，其中15条失败来自同一个clustered case的数值奇异full covariance；
每个F/G/H五seed拒绝更新，clustered每方法失败频率1/12。成功11例的RMSE不代表全部12例，
未添加未声明nugget或事后variance floor来消除失败。未知load amplitude是模型假设被故意违反，
provenance中的known_BC是estimator假设，不能读作该OOD真实载荷已知。

| OOD scenario | H D40 RMSE across5 seeds min/max | H failure total |
|---|---|---:|
| q16 | 0.00462863/0.00596562 | 0 |
| q64 | 0.00235212/0.0051704 | 0 |
| q256 | 0.00392741/0.00548642 | 0 |
| q37 | 0.00531159/0.00630424 | 0 |
| regular | 0.00262433/0.0055962 | 0 |
| clustered | 0.0689997/0.0881729 | 5 |
| independent | 0.00414943/0.00756952 | 0 |
| spatial | 0.00829667/0.00937426 | 0 |
| registration | 0.0207066/0.0268606 | 0 |
| unknown_loading | 0.0571551/0.078973 | 0 |
| cross_mesh | 0.00393288/0.00763354 | 0 |
| heldout_material | 0.00535423/0.024105 | 0 |
| cross_geometry | 0.0112121/0.0156532 | 0 |
| model_discrepancy | 0.0149815/0.0225634 | 0 |
| no_noise | 0.00293464/0.00639761 | 0 |
| missing | 0.00654359/0.0100433 | 0 |
| irregular_time | 0.00598698/0.0123959 | 0 |
| cross_mesh_reference | 0.00303332/0.00633926 | 0 |

![OOD](../results/phase2b/20261008T114713Z/figures/OOD_forecast_diagnostics.png)

Gaussian32独立test、cal32的小样本探索，完整6×6cov；较窄training priors及弱方向使参数恢复有限，
没有将小damage RMSE解释为六参数可靠恢复或高维posterior资格。每seed/经典4例结果保存：

| Gaussian seed | 6-parameter RMSE | Current D RMSE | D40 RMSE | coverage |
|---|---|---:|---:|---|
| physics_shape_uncertainty_seed_71001 | [0.014303627478226645, 0.06144903759978945, 0.15089469328087352, 0.04367160930617585, 0.07089339139784612, 0.05645146911080453] | 0.0403129 | 0.0411967 | [1.0, 0.96875, 0.90625, 0.8125, 1.0, 1.0] |
| physics_shape_uncertainty_seed_71002 | [0.014218965154808104, 0.060761002512847766, 0.1478162416068982, 0.04150220575569023, 0.07137216475645707, 0.05634960414882365] | 0.0382447 | 0.0390783 | [1.0, 1.0, 1.0, 0.90625, 0.96875, 1.0] |
| physics_shape_uncertainty_seed_71003 | [0.01437828124905091, 0.06400503944817176, 0.15164890875754625, 0.043694662601503995, 0.07486652298701339, 0.05584659417453657] | 0.0392378 | 0.0400725 | [1.0, 0.96875, 0.90625, 0.9375, 1.0, 1.0] |
| physics_shape_uncertainty_seed_71004 | [0.01454221014997249, 0.06467631174374644, 0.15156402052246976, 0.04412438226066215, 0.0756874204099559, 0.05673635066600468] | 0.0400897 | 0.0409371 | [1.0, 0.96875, 0.96875, 0.90625, 0.9375, 1.0] |
| physics_shape_uncertainty_seed_71005 | [0.014383012655493192, 0.06432250028113169, 0.1556722350864714, 0.043895994749758714, 0.07271488461437889, 0.057026009373335976] | 0.0404123 | 0.0413095 | [0.96875, 0.96875, 0.90625, 0.875, 1.0, 0.9375] |

## M. Classical MAP comparison

以下只比较预冻结前16个test cases，全部方法相同observations/noise/public inputs（E/control仅N0作为结构性baseline）：

| 方法/seed | d0 RMSE | A RMSE | Current D RMSE | D40 RMSE | Online s | FEM solves |
|---|---:|---:|---:|---:|---:|---:|
| graph_gru_seed_41001 | 0.00703529 | 0.000149654 | 0.00587922 | 0.00742012 | 0.440974 | 123 |
| graph_gru_seed_41002 | 0.0056883 | 0.000134571 | 0.00549768 | 0.00727721 | 0.449739 | 123 |
| graph_gru_seed_41003 | 0.00584868 | 0.000133537 | 0.00547291 | 0.00778517 | 0.51435 | 123 |
| graph_gru_seed_41004 | 0.00862582 | 0.000130031 | 0.00661226 | 0.00741442 | 0.448348 | 123 |
| graph_gru_seed_41005 | 0.00365359 | 0.000143831 | 0.00383297 | 0.00644031 | 0.479123 | 123 |
| graph_initial_seed_41001 | 0.0087322 | 0.000478448 | 0.0116102 | 0.0218879 | 0.0198031 | 1 |
| graph_initial_seed_41002 | 0.0114803 | 0.000451204 | 0.0155478 | 0.0251878 | 0.0181444 | 1 |
| graph_initial_seed_41003 | 0.00753563 | 0.000472985 | 0.0136525 | 0.0239127 | 0.0219128 | 1 |
| graph_initial_seed_41004 | 0.0110582 | 0.000475385 | 0.0133546 | 0.0236587 | 0.0188477 | 1 |
| graph_initial_seed_41005 | 0.00844048 | 0.000477794 | 0.0135085 | 0.0238925 | 0.0206106 | 1 |
| graph_observation_only_seed_41001 | 0.00283491 | 0.000471696 | 0.00996442 | 0.0198732 | 0.0234856 | 1 |
| graph_observation_only_seed_41002 | 0.00542809 | 0.00046956 | 0.0111815 | 0.0211051 | 0.0220672 | 1 |
| graph_observation_only_seed_41003 | 0.00297324 | 0.000471565 | 0.00946155 | 0.0193483 | 0.0191805 | 1 |
| graph_observation_only_seed_41004 | 0.00578338 | 0.000471055 | 0.0105557 | 0.0204839 | 0.0204903 | 1 |
| graph_observation_only_seed_41005 | 0.00393238 | 0.000470866 | 0.0105838 | 0.0205062 | 0.0191132 | 1 |
| independent_frame_inverse | N/A | N/A | 0.00148692 | N/A | 0.108758 | 25.9375 |
| physics_global_seed_41001 | 0.012687 | 0.000247323 | 0.00928373 | 0.0100751 | 0.46544 | 129 |
| physics_global_seed_41002 | 0.0125632 | 0.000300515 | 0.0117895 | 0.0151618 | 0.484807 | 129 |
| physics_global_seed_41003 | 0.010721 | 0.000355587 | 0.00647447 | 0.0106036 | 0.441078 | 129 |
| physics_global_seed_41004 | 0.00861748 | 0.000290895 | 0.00430232 | 0.00621101 | 0.482543 | 129 |
| physics_global_seed_41005 | 0.0147387 | 0.000343252 | 0.00712165 | 0.00705894 | 0.442876 | 129 |
| physics_prior | N/A | N/A | 0.0595461 | 0.0653365 | 0.0096735 | 0 |
| physics_shape_seed_41001 | 0.00499094 | 0.00015011 | 0.00450031 | 0.00631259 | 0.496955 | 129 |
| physics_shape_seed_41002 | 0.00712819 | 0.000137516 | 0.00666226 | 0.00836331 | 0.515132 | 129 |
| physics_shape_seed_41003 | 0.00392185 | 0.000121147 | 0.00370507 | 0.00578023 | 0.457063 | 129 |
| physics_shape_seed_41004 | 0.00590874 | 0.000134069 | 0.00503288 | 0.0070326 | 0.48352 | 129 |
| physics_shape_seed_41005 | 0.00483322 | 0.000131709 | 0.0050041 | 0.00721364 | 0.451955 | 129 |
| physics_shape_uncertainty_seed_41001 | 0.00499094 | 0.00015011 | 0.00450031 | 0.00631259 | 0.477375 | 129 |
| physics_shape_uncertainty_seed_41002 | 0.00712819 | 0.000137516 | 0.00666226 | 0.00836331 | 0.463826 | 129 |
| physics_shape_uncertainty_seed_41003 | 0.00392185 | 0.000121147 | 0.00370507 | 0.00578023 | 0.43845 | 129 |
| physics_shape_uncertainty_seed_41004 | 0.00590874 | 0.000134069 | 0.00503288 | 0.0070326 | 0.479788 | 129 |
| physics_shape_uncertainty_seed_41005 | 0.00483322 | 0.000131709 | 0.0050041 | 0.00721364 | 0.457756 | 129 |
| prefix_cold_map | 0.00154049 | 0.000126718 | 0.00127603 | 0.00358762 | 10.1192 | 3018.94 |
| sequential_warm_map | 0.00154049 | 0.000126718 | 0.00127603 | 0.00358762 | 9.13604 | 2699.62 |

TRF复用Phase1C，prefix冷/暖均只用当前prefix，保存Jacobian/Fisher/历次optimization；未收敛保留且单列。
只比较同case cost，不把离线生成/训练成本隐藏于在线latency。online包含observations加载/物理rollout与diagnostic写入，
network_seconds另列；forecast mean/uncertainty成本分开，FEMsolve统计包括全部疲劳endpoint求解。
SciPy replay仍限制physics-guided端到端速度；NNforward快不能等价宣称全流程相同比例加速。
另有`benchmark_final/`单进程、thread1、所有训练/evaluation worker退出后的三固定validation案例基准，
不在timed inference内写诊断文件。E/prior online只统计N0 belief构建，current20是后续预测，
不能将其较短时间理解为与全prefix MAP相同信息的状态估计。
Uniform B只识别当前D，未来N/A；Gaussian B在已知A下用N20单帧识别初始六参数并physics rollout，
不是假定当前damage仍为Gaussian。不同参数模型的N/A边界在原始record中明确标注。

![accuracy cost](../results/phase2b/20261008T114713Z/figures/fair_accuracy_cost.png)


位移指标补充：MAP为冻结16例，H为128例，以下表用于各模型指标审计，跨算法公平精度比较仍以上述同16例表为准。

| Method | current true u RMSE mm | fitted DIC RMSE mm | spatial-held-out true u RMSE mm | forecast true u30/40 RMSE mm |
|---|---:|---:|---:|---|
| prefix_cold_map | 7.37669e-06 | 6.6504e-05 | 7.20124e-06 | 1.42806e-05 / 2.30677e-05 |
| sequential_warm_map | 7.3767e-06 | 6.6504e-05 | 7.20124e-06 | 1.42806e-05 / 2.30677e-05 |
| physics_shape_uncertainty_seed_41001 | 2.69661e-05 | 6.52038e-05 | 2.6908e-05 | 3.12181e-05 / 3.88087e-05 |
| physics_shape_uncertainty_seed_41002 | 3.12817e-05 | 6.8227e-05 | 3.16043e-05 | 3.58059e-05 / 4.3201e-05 |
| physics_shape_uncertainty_seed_41003 | 2.31238e-05 | 6.23314e-05 | 2.29866e-05 | 2.88514e-05 / 3.71597e-05 |
| physics_shape_uncertainty_seed_41004 | 2.59361e-05 | 6.41878e-05 | 2.6092e-05 | 3.08092e-05 / 3.85585e-05 |
| physics_shape_uncertainty_seed_41005 | 2.75465e-05 | 6.50791e-05 | 2.74776e-05 | 3.20162e-05 / 3.98119e-05 |

完整每seed predictive coverage/width/NIS见`displacement_metric_audit.json`；未用fit RMSE代替damage或forecast指标。

## N. Ablation and failed experiments

A prior，B per-frame current inverse，C cold prefixMAP，D warmMAP，E graphN0+obs-onlypseudo监督控制，
F无physics GRU，G global/shape physics-GRU，H shape+fullcov；H point weights固定G，故H/G相同点误差，
不是额外独立算法点准确性提升。global/shape都训练五seed；不预设shape或G优于F/MAP。
初始teacher-cohort、deployment validation偏移、seed41004早停和orchestrationhandoff/测试失败日志均保存，
最终selection不换seed、不按test最好模型挑选。完整每seed loss和代价在training*/summary/history.json。
Gaussian训练规模小，teacher→deployment偏移仍为限定风险，没有看test后追加调参。

## O. Full tests/scientific verification

开发前357 passed，exit0；正式完整pytest exit0、wall796.40s，
实际计数见[完整日志](../results/phase2b/20261008T114713Z/pytest_final.txt)。历史全部37个CLI（含Phase0 smoke）实际重跑，失败0。
最终冻结199个源码/测试/配置/工程文件后再次完整复验：**389 passed，0失败，581.89s**，
子进程exit0，wall584.04s；日志为`pytest_final_source_frozen.txt`，源码逐项复核199/199未变化。
原Phase2A全部14类科学场景重新运行，并不改其资格。
正式Phase2B A–P全部执行/登记，trained五seed permutation/checkpoint因果检查、Q/V/G、fusion、no-leakage、
rollback、uncertainty/calibration、MAPcomparison、跨mesh/geometry、model discrepancy、成本、truth隔离均在
[科学验收](../results/phase2b/20261008T114713Z/scientific_final/verification.json)。长训练和统计CLI不放进普通pytest。
controlled point gate=True；neural coverage gate=False；regression gate=True。

## P. Failure modes/limits

本阶段不保证GNN更准确，经典低维MAP在matched model中通常更精确；physics innovation的额外收益需独立统计。
learned Gaussian与fullcov只给近似条件误差描述，prior支撑单N0 A，不同case条件和OOD不能由总体coverage替代。
局部线性field pushforward未证明global非高斯/多峰/边界coverage；Gaussian6参数样本32统计精度有限。
未知载荷、固定未建registration及fatigue函数形式误差能造成persistent innovation/forecast偏差及CI失效，
不能靠任意改damage吸收后声称解决物理失配；unknownE严格退化不输出唯一真实D。
物理guidance保留大量FEMreplay；CPU训练预算和小数据对泛化有限，GPU加速尚未实际验证。
qualification是部分资格；无任意G维损伤、真实实验、RUL、RL或physicscorrection实现。
Uniform场的global仿射回归特征已包含大量信息，本阶段未证明图message passing不可替代普通MLP/DeepSets。
固定matched-case欠覆盖的bias、条件covariance和相关项需要进一步诊断；不武断归因于某一种曲率机制。
原Phase2A的97/100独立复核也不作为神经Cov真值或替代这些失败的证据。

## Q. Recommended Phase 2C experiments (不执行)

先研究共享registration/load nuisance显式扩展与observability：同prefix下classical/neural具有同物理信息。
使用独立cal/test与预冻结更大noise/OOD cohorts检验covariance与conditional coverage，研究bounded/non-Gaussian表示。
对模型函数形式错误保留innovation与forecast偏差，比较可校准A、错误固定A、wrong exponent的可解释残差。
physics correction先从低维可辨识残差/参数校准实验开始，验证能量/单调damage/一致history，避免G维自由action。
若引入可微FEM，先建立adjoint/implicit derivative并与FD验证；不能假定SciPy autograd。
在明确cost/accuracy/物理证据后再判断神经correction或RL的必要性。本轮在Phase2B停止。

复现CLI：`generate_phase2b_dataset.py`，`train_phase2b.py`/`train_phase2b_parallel.py`，
`diagnose_phase2b_training_seed.py`，`verify_phase2b_uniform_condition.py`，`train_phase2b_causal.py`，
`train_phase2b_gaussian.py`/`train_phase2b_gaussian_uncertainty.py`，`evaluate_phase2b.py`（val/cal/test），
`verify_phase2b_generalization.py`，`verify_phase2b_fixed_noise.py`，`regress_phase0_to_2a.py`，`verify_phase2b.py`。
配置/随机seed/依赖/source hashes及每次command/execution记录都在本run；output须为新目录，拒绝覆盖。
