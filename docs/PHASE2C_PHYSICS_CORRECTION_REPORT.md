# Phase 2C — Physics-Guided Fatigue Damage Evolution Correction

日期：2026-10-09（Asia/Shanghai）。正式资格：**PHASE2C_PHYSICS_CORRECTION_PARTIALLY_QUALIFIED**。

在冻结的高信噪比、已知材料/边界、均匀初始损伤和合成结构失配分布中，低维 DIC-only correction
具有可重复的预测收益。它不替代正确模型的 A/m 重标定，不代表任意损伤场可辨识，
不代表真实材料、裂纹、网格客观局部化或概率不确定性已经验证。**Phase 3 RL：NO-GO。**

主证据：[最终审计](../results/phase2c/20261008T233840Z/final_audit_v4/audit.json)、[公平主实验逐行结果](../results/phase2c/20261008T233840Z/final_audit_v4/fair_records.json)、
[研究输出](../results/phase2c/20261008T233840Z/)、[位移坐标系评分复核](../results/phase2c/20261008T233840Z/metric_frames/records.json)。
所有原始 seed、优化过程、失败/未收敛诊断、checkpoint 和历史结果保留。

## A. Git、源码与环境审计

分支 `feature/phase0-architecture`；HEAD `dc239d1b7b6dd1e44d93e71bdcccccc0f12e8bc9` 是 initial commit，
**不是当前已验证完整源码版本**。开发前 README 已修改，主要源码、测试、配置、报告与结果尚未提交。
本阶段没有 commit/push/merge/reset/clean，也没有修改驱动、安装依赖或替换历史报告。

开发前源文件225项；其聚合 SHA256 为
`04f4391f031eb58fa79f6473489f11b7dc4a668ba3726395e5c03b53a3bd0c5e`。
[源码快照与恢复说明](../results/phase2c/20261008T233840Z/baseline/RECOVERY.md) 同时保存逐文件 SHA、Git read-only 状态、
完整 HEAD→工作区补丁（含未提交源码）和轻量 tar。M4另存源码快照。打包显式排除全部 results、旧tar、checkpoint与缓存。
最终完整源码（含本文档）的 SHA、逐文件 manifest、最终tar、开发前→最终增量patch及实际恢复检验见
[最终源码审计](../results/phase2c/20261008T233840Z/final_source_verified/audit.json)。本文不能以自身内容固定自己的哈希，完整哈希以外部manifest为准。

最终科学程序的数值源码指纹（src/scripts/configs/phase2c，不含结果/本文档）为 `e5991a3ddd1e73aaaedfefc8fa1e62b888ab1d7a3939eff050c5ba538225f84c`；
各生成/训练/验证阶段另存当时指纹。后续接口校验与评分修复没有改变训练数据或冻结权重；实际五seed重训重放逐位一致。
71个旧训练checkpoint SHA保持一致；77,837个历史资产 path/size/mtime均未变化。
原225项源文件中，只有README有本阶段说明更新；旧数值源码、测试和正式报告内容保持原SHA。

Python3.13.12，NumPy2.4.4，SciPy1.17.1，PyTorch2.13.0，pytest9.0.3，Matplotlib3.10.9。
CUDA编译版本13.0，但 nvidia-smi exit9、driver不可用、torch.cuda=False，设备数0。
因此执行CPU小规模原型；没有深度图网络大规模训练。OMP/BLAS/MKL线程设为1，科学CLI最多3个CPU进程。
新 correction 数值路径可在屏蔽torch导入时正常导入；torch只属于监督训练/checkpoint路径。
[环境和依赖审计](../results/phase2c/20261008T233840Z/baseline/environment.json)、[隔离复核](../results/phase2c/20261008T233840Z/dependency_isolation.json)。
Git冻结建议另见 [Phase2C Git建议](GIT_PHASE2C_BASELINE_RECOMMENDATION.md)，没有执行Git写操作。

## B. 冻结的物理模型失配问题

三类任务分别评价：parameter calibration、constitutive structural discrepancy、state estimation error。
prior为 λ=A xᵐ，x=max(Y_a/Yref−η,0)，Yref=.025 N/mm²、η=0；x、m无量纲，A与λ单位cycle⁻¹。
Y_a由原Phase1B实际 min/max FEM 能量振幅驱动计算，不以DIC噪声直接替代内部驱动力。

M0：A=8e−5、m=1且形式一致。M1：wrong_A取A=1.28e−4；wrong_m取m=1.18；wrong_Am同时改变。
M2：λtrue=A xᵐ(1+3x/(1+x))。这是受控synthetic oracle，**不是已经实验验证的疲劳律**。
它连续、非负，x=0时λ=0，beta无量纲。在公开多载荷驱动区间内不能由一个A/m在全部工况精确重现。
α=2的修正幅度限制与β=8 OOD另行保留；空间oracle为额外 exp(.3 cos(πX/2mm))，定义在物理坐标中。

所有主案例都是独立材料组的五个公开载荷specimens，共享同一初始d0和材料律。
载荷牵引幅值为10×[.15,.35,.7,1.4,2.8] N/mm²；R=−1；N0/10/20同化。
公开未来计划在N20后将幅值乘.25，评价N30/40/50/70/120/220。
该计划在查看测试前冻结，用来测试驱动区间变化并避免无意义的饱和；没有依据预测误差改动载荷或测试分布。

## C. Hybrid damage law 与原物理更新

λhybrid=λprior exp[α tanh(Φa)]，α=2，|a_j|≤4，K=2/4/8。
乘性因子位于[e⁻²,e²]；原hazard为零时仍为零，**不表示新损伤成核机制**。
非法系数、错误积分布局、非有限输入/输出显式拒绝。

继续调用原DamageTransition：Dnew=Dmax−(Dmax−Dold)exp(−ΔNλ)，Dmax=.95。
Tri3、EnergyAmplitudeDriver、DamageTransition和原FatigueEvolutionSolver均未改动。
每个子步用当前D求min/max、计算hazard、仅执行一次增量更新，再用新D重新求解FEM；
新平衡解失败时原子回滚，不提交半步损伤。各步保留MaterialHistoryState、循环数和布局归属。
G为积分点数，不等于FE node/element/DIC point数；新增G=3、element=1的真实耦合测试与passive history验证。

## D. 低维修正表示与实际学习架构

C1是原模型A/m经典重标定。C2采用固定光滑物理特征 t=x/(1+x)，φ_j=t^(j+1)，
只学习全局K个系数；不是隐藏G个自由参数。系数a在整个开环预测中固定，δ随当前物理x变化。
该basis在x趋零时factor趋1，不能精确表示所有纯A常数重标定；这也是优先采用C1的表达能力理由。
首版D/R/history没有独立学习项：D及R通过已验证FEM/driver影响x，不声称已经学习材料记忆或任意R机制。

固定空间basis为t或t²乘[1,cos(πX/2mm),cos(πY/1mm),cosX cosY]，参考原点和mm长度显式保存。
不依赖数组排序、单元ID或节点编号；同一物理坐标的basis跨网格相同。
每次推断保存 basis、a、factor、原/修正hazard、系数敏感性、当前damage及六个forecast damage/位移/应变/应力字段。
监督模块是PyTorch的**K参数固定basis非线性回归**；没有MLP、Graph encoder或新的神经状态估计器。

C3 graph-conditioned extension、B6和fixed-vs-graph消融：**NOT EXECUTED**。
K≥4已有弱/null方向，且首版没有独立空间训练矩阵；不以增大网络绕过可辨识性限制。
空间固定basis的真实FEM诊断和小规模DIC标定已执行，不能替代C3验收。

## E. 状态估计与损伤律修正的分离

Mode B采用原Phase1C TRF及Phase2A prefix MAP似然/ForwardAdapter，共同估计d0和允许的演化参数。
每个参数候选从与自身d0一致的N0状态重演完整public load history，不能套用其他候选历史。
统一输出MechanicsBelief、PredictedObservation和ForecastResult；新的predictor通过原estimator协议集成测试。
current20 filtered、prediction和open-loop forecast标签保持明确。B3仅同化0/10，N20是prediction，forecast origin为10。

Mode A仅在正式结果完成后，用synthetic真实N20状态进行memoryless-law重启诊断；绝不混入DIC-only成绩。
另有correct-physics oracle reference：仅真A/m/β等物理形式已知，d0仍由同一DIC prefix估计。
这是extra-information参考，其truth-parameter IO明确声明，不能算普通observation-only方法。

Mode C Phase2B neural estimator substitution：**NOT EXECUTED**。
冻结2B权重的fatigue A支持区间[1e−4,.004]，本阶段名义A=8e−5在其外；
重写解码bounds或临时重训会破坏同等物理信息的替代对照。本轮只建立可靠Mode B基线。

## F. 三类标定/训练及信息预算

T1：有界无量纲TRF，重新标定A或A,m。T3：真实FEM多步DIC prefix有限差分TRF，
目标为1/2∑‖L⁻¹(pred−DIC)‖²+.001/2∑a²。每个likelihood证据只出现一次，ridge也只加一次。
没有可微SciPy/NumPy FEM，也没有伪造端到端multi-step gradient或策略学习。

T2：仅synthetic train/validation的N<20 oracle positive-hazard log-ratio标签。
只有true与prior hazard都严格正才构造 log λtrue−log λprior；零hazard使用显式mask，未加epsilon制造标签。
torch梯度仅穿过α tanh(Φa)，训练features停止梯度，FEM用于数据生成及独立评价。
检查目标是否超过α可表达范围，不裁剪标签。
AdamW lr=.02、weight_decay=0、系数小ridge1e−7、clip1、最多1600epochs、patience150/min300、float64、validation选择epoch。
本原型只有K个全局系数，lr与预算在冻结配置中声明；不把该小规模参数学习包装为深度网络训练。

每个regime分别训练K=2/4/8和seed85000–85004，共75checkpoint。
**B4是按预声明实验regime配置的训练先验，不是根据测试DIC判断失配类型的统一分类器。**
其oracle hazard训练标签是额外信息；测试时系数固定，仅d0由DIC估计。
OOD始终使用原M2冻结权重，没有按OOD类型挑模型，没有测试集调参/校准。

A/m最初搜索盒过窄，独立no-noise pilot（seed80000）触及A上界2e−4。
随后统一扩大A至[1e−5,5e−4]、m至[.6,1.6]，相同80nfev预算、相同nominal初始化，重跑全部50条B1/B2。
主公平比较使用新结果；原305条、旧box结果、pilot和addendum均保留。
这项改动修复经典基线的信息/搜索限制，没有修改测试观测、噪声、训练权重或correction超参数。
新M2 A估计约2.24–2.28e−4，已远离新bounds；扩大搜索改善prefix拟合，却暴露更大的未来低驱动偏差。
[搜索范围附录](../configs/phase2c/calibration_search_addendum.toml)。

## G. Sensitivity、Fisher与可辨识性

使用原中心有限差分与步长复核、whitened Jacobian、SVD rank/weak/null诊断。
Fdata=JᵀJ；系数ridge的局部矩阵另外保存，不把它当DIC新增信息或已验证概率prior。
K2 energy：d0+c0+c1 rank3/3；K4 rank4/5；K8 rank5/9（固定点诊断）；
正式M2拟合分别保留相同rank限制。因此K4可准确预测某些轨迹，**不代表全部系数可唯一恢复**。

专用global diagnostic basis[1,log x]联合d0/A/m/c0/c1得到rank3/5：
常数修正与A在log-x系数零时具有精确等价；log-x方向与m在零修正处局部切线等价，
bounded tanh远离该点并非全局相同函数。生产energy basis去掉这两个直接冗余项。
共享tx/ty/θ的Jacobian和全部交叉Fisher块保留；不重复计入共享配准噪声，也不每帧重新声明nuisance。
未知load amplitude显示弱方向/边界解；满数值秩也不能解释为稳定恢复。
correction系数没有唯一oracle真值（参考律并非恰好位于固定basis族），系数RMSE标为N/A，评价函数/hazard及轨迹误差。
本阶段**不授予参数、损伤或预测区间coverage资格**；没有在calibration/test上拟合标准差缩放因子。

Phase2A原89/100覆盖和Wilson[81.369%,93.746%]原证据保留；独立新样本97/100不撤销原结果。
共享nuisance交叉块原审计未发现遗漏，不能据此把局部Gaussian当合格真实后验。
[原独立复核](PHASE2A_UNCERTAINTY_FOLLOWUP.md)和Phase2B未校准神经不确定性均保持原部分资格。

## H. 多步轨迹优化、数据拆分和时间泄漏

95个物理case组（regime内独立、regime间配对）：每regime train8/validation3/calibration3/test5，每组五条load trajectories；
seed分别81000起、82000起、83000起、84000起。regime之间使用配对参数/噪声seed，不能当25个额外独立重复样本。
train/validation/calibration只生成N0/10/20真实字段，未来真实字段不存在；calibration split本轮保留未使用。
主DIC irregular Q16，ROI/validity与完整component/spatial covariance复用原接口。
main σx=2e−7 mm（0.2nm），σy=1.25σx；这是**极高SNR受控实验**，比50nm OOD小250倍，不能作真实DIC精度声明。

test仅N≤20可进入拟合。未来设计只复制prefix坐标/mask/covariance并使用公开载荷计划；
所有load specimens的六个未来physics forecasts和诊断完成后，评价器才打开真实damage/参数/未来DIC。
实际IO拦截集成测试验证此顺序。普通方法和oracle reference的信息边界分别标记。
prediction/current fitted displacement/open-loop forecasting不互相冒充。

主公平矩阵305条、OOD208条、显式nuisance8条、correct-physics reference25条、空间basis2条，合计548行。
其中β8的两条B5保留UNCONVERGED_TENTATIVE；没有FAILED推断或不可逆损伤违例，不能把“有限结果”混同“全部收敛”。
另外保存旧box重复50条、串行cost3条、pilot、oracle-only状态诊断、replay和所有历史科学验证。

## I. 时间步收敛与长期预测

不同step使用同一冻结K4系数，无重训/重标定。
独立DOP853对相同空间FE离散、piecewise public loads积分，rtol1e−10/atol1e−12，
每RHS重新求min/max FEM；它是时间参考，不是连续空间真解。

| ΔN | Damage RMSE vs DOP853 | 加密阶数 |
|---:|---:|---:|
| 4.0 | 3.10476295e-05 | N/A |
| 2.0 | 1.55603724e-05 | 0.9966065403477505 |
| 1.0 | 7.78936475e-06 | 0.9982990018690382 |
| 0.5 | 3.89698195e-06 | 0.9991484295239307 |

保持约一阶冻结hazard时间误差；额外查询[0,7,20,33,220]与只查询220的最终damage逐位相同。
constant corrected hazard的解析semigroup在step10/5/2.5/1.25与N40验证到浮点误差。
非constant hazard有限step误差不强行宣称严格不变；材料函数不含observation index或ΔN参数。
未来horizon为1/2/3/5/10/20个10cycle区间，实际跨度10/20/30/50/100/200cycle（B3另从N10计算）。

## J. 空间basis与网格/几何泛化

固定空间basis在nx2/4/8网格，V6/15/45、G4/16/64上运行真实耦合。
N40 area-weighted mean damage约.1266576/.1266570/.1266575；max damage约.127839/.127991/.128028。
common64DIC displacement相对最细网格RMSE约1.68e−6/2.23e−7/0 mm。
FE node重新编号后damage差2.78e−17、DIC差2.08e−17；物理basis不变。
这些是有限场景离散敏感性检查，**不证明局部软化或裂纹网格客观**。

OOD cross_mesh为nx6/ny3，paired coarse reference用相同物理案例/点/noise seed；
均匀damage/轴向牵引位移接近仿射，因此跨网格差极小不能作为复杂局部场泛化证据。
cross_geometry为3×.7mm、heldout E=2600仍作为公开已知输入；R=0仍公开、driver重新计算。
没有声称未知E或材料机制已泛化。
真实空间失配的两个case中，fixed spatial K2的N220 RMSE=.0014465，global energy K2=.0078169；
这是新增空间信息的受控诊断，非训练过图basis或高维damage恢复。

## K. 公平基线比较

以下均为N220 damage RMSE：同一个五案例组、相同DIC/noise/public mechanics，B4另有privileged training信息。
B0只固定错误演化prior，d0由同样prefix估计；它没有偷用真实初态。

| 情景 | B0 prior | B1 refit A | B2 refit A,m | B5 DIC K2 | B4 supervised K4，全部五seed平均 |
|---|---:|---:|---:|---:|---:|
| matched | 6.97825e-07 | 1.93422e-06 | 1.2361e-05 | 8.42586e-06 | 2.75624e-06 |
| wrong_A | 0.00509787 | 1.92357e-06 | 1.2323e-05 | 0.000800111 | 0.000854725 |
| wrong_m | 0.00313974 | 0.00268522 | 9.4249e-06 | 0.000242649 | 0.000610219 |
| wrong_Am | 0.00979034 | 0.00436416 | 9.40254e-06 | 0.000651598 | 0.000166166 |
| structural | 0.0196014 | 0.00693257 | 0.00201021 | 0.000350085 | 0.000130534 |

M1 A/m refit优于correction，尤其wrong_m/wrong_Am必须允许m重新标定。
M2 K2相对B2的五个长期比值为 [0.1718576, 0.1700371, 0.1741413, 0.1819469, 0.1728404]；
中位比值0.1728404，约降低82.7%，5/5改善。
预声明gate为中位<.9、至少4/5改善、最差≤1.25，实际通过。
此结论仅对该高SNR多载荷结构失配分布成立，不表示所有未知疲劳律都能如此改善。

M0 physics prior的N220 RMSE约6.98e−7，DIC K2约8.43e−6：正确模型下自由修正过拟合噪声。
主supervised K4五seed的matched validation最大factor deviation约
[.000456,.000335,.000151,.000335,.001445]，四个seed未达到冻结2e−4 neutral阈值。
K2监督matched deviation均<2.83e−7；这不能事后替换预声明主K4并宣布全部合格。

| training seed | K4 N20 RMSE（五案例平均） | K4 N220 RMSE（五案例平均） |
|---|---:|---:|
| 85000 | 6.509471e-05 | 0.000134044 |
| 85001 | 6.033872e-05 | 0.0001238249 |
| 85002 | 6.043366e-05 | 0.0001239768 |
| 85003 | 6.340664e-05 | 0.0001303932 |
| 85004 | 6.806236e-05 | 0.0001404301 |

five training seed与five test case交叉结果是层次重复，不能伪称25独立test trajectories。
全部seed原始误差、模型权重、验证历史与系数可检查，不只汇报最优seed。

![基线、matched负对照和时间收敛](../results/phase2c/20261008T233840Z/final_audit_v4/comparison.png)

## L. 模型失配、噪声及OOD失败

下表N220 damage RMSE；B4使用同一M2冻结权重，不重新训练。每情景仅两个固定case，统计精度有限。

| 冻结 OOD 情景（每情景2案例） | B2 A,m | B5 DIC K2 | B4 K4 全五seed平均 |
|---|---:|---:|---:|
| cross_geometry | 0.00198117 | 0.000342339 | 0.000123841 |
| cross_mesh | 0.00203913 | 0.000349761 | 0.00013512 |
| different_d0 | 0.00241815 | 0.000357512 | 0.000205565 |
| heldout_E | 0.00134153 | 0.000260817 | 4.60757e-05 |
| independent_noise | 0.00348097 | 0.00161321 | 0.000467953 |
| registration_bias | 0.00558473 | 0.00260825 | 0.00175266 |
| spatial_discrepancy | 0.00897621 | 0.00781694 | 0.00777957 |
| spatial_noise | 0.0026527 | 0.00203165 | 0.000275991 |
| stress_ratio_zero | 7.26102e-05 | 4.27016e-05 | 3.51577e-05 |
| structural_beta8 | 0.00565969 | 0.00617579 | 0.0377043 |
| structural_form2 | 0.00170464 | 0.000198332 | 0.00411334 |
| unknown_loading | 0.0589266 | 0.0572558 | 0.0574855 |

independent/spatial noise σx=5e−5mm（50nm），空间covariance采用原.22mm核和component相关矩阵；
给定共享配准后独立noise，不把共享配准再计入每帧covariance。
β8 B5两个case耗尽80nfev，结果有限但未收敛；已完整保留。
unknown loading truth幅值乘1.07，固定错误BC导致damage误差约.057–.059。
放开load_factor并不能解决该信息不足：B2约.18185且load_factor到.85下界；B5约.06592且到1.15上界。
这些优化“收敛”与Fisher数值秩不表示物理解唯一，不能用correction吸收边界错误而宣布成功。

共享配准truth为tx5e−5mm/ty−3e−5mm/θ4e−5rad。
未建模B5长期RMSE=.0026083；一个共享tx/ty/θ建模后约.00035347，所有参数交叉块保留。
配准/载荷显式对照用相同case观测，没有更换seed。

最终评分审计发现，原未来registered predicted observation曾直接与unregistered displacement truth比较。
已在新评价代码区分physical FEM displacement与registered observation；新增无噪声集成测试二者均恢复。
对已冻结601条原记录仅重算指标，不重优化、不改预测、不覆盖原记录：
[matching-frame ledger](../results/phase2c/20261008T233840Z/metric_frames/records.json)。配准拟合4行的原mixed-frame位移指标不用于物理解读。
damage/参数/likelihood/cost及公平主结论不受该评分修复影响。
应力/应变振幅误差在此比例线性载荷下由max-field误差乘|1−R|/2得到，另存在该ledger；
uniform traction应力几乎不随damage变化，极小stress error本身没有校正识别意义。

## M. 关键消融与state/law分离结果

M2全部五case平均如下；B3的N20是prediction，不是filtered：

| 方法 | N20 damage RMSE | N220 damage RMSE | data rank / 参数维数 |
|---|---:|---:|---|
| B0_physics_prior | 0.0159209 | 0.0196014 | [1, 1, 1, 1, 1] / 1 |
| B1_refit_A | 0.000890744 | 0.00693257 | [2, 2, 2, 2, 2] / 2 |
| B2_refit_Am | 0.000201748 | 0.00201021 | [3, 3, 3, 3, 3] / 3 |
| B3_single_step_K2 | 3.49953e-05 | 0.00035889 | [3, 3, 3, 3, 3] / 3 |
| B5_multistep_K2 | 3.38983e-05 | 0.000350085 | [3, 3, 3, 3, 3] / 3 |
| B5_multistep_K4 | 3.40259e-06 | 2.69246e-05 | [4, 4, 4, 4, 4] / 5 |
| B5_multistep_K8 | 8.82126e-06 | 0.000106782 | [5, 5, 5, 5, 5] / 9 |
| ablation_no_physics_K2 | 0.0062754 | 0.0172295 | [3, 3, 3, 3, 3] / 3 |
| ablation_no_history_K2 | 4.32007e-05 | 0.000454777 | [3, 3, 3, 3, 3] / 3 |
| ablation_spatial_K2 | 0.00018569 | 0.00109013 | [3, 3, 3, 3, 3] / 3 |

no-physics ablation为λ=8e−5(Φa)²的非负候选，非经典物理律；结果不能将其“更差”解释为任意无物理模型都差。
no-history仅N20；single-step仅N0/10；multi-step仅N0/10/20。
K4/K8误差与成本、可辨识性共同报告，不能仅因更低RMSE就扩大K。
global-vs-spatial固定basis已执行；fixed-vs-graph、神经state替代对照为NOT EXECUTED，原因见D/E。

Mode A首个M2固定case：estimated-state+refit-Am N220 RMSE=.00211309；
oracle-state+同一refit-Am仍为.00208108，仅改善约1.5%；提高状态估计精度不足以消除演化律偏差。
oracle-state+错误nominal prior=.00569318；estimated-state+正确真实律=.000213401；oracle-state+正确律=0。
这分别隔离错误材料律与初始state误差，不能把oracle改善全部归功于可部署correction。

correct physics、人为固定d0比true少.01的负对照：DIC prefix诱导c0≈4、c1≈−3.97861边界附近，
未来N220从无修正RMSE=.0100688恶化到=.0168779；N0误差无法由演化修正消除。
这个oracle-only perturbation不是正式成绩，而是证明state错误会产生虚假的“材料修正”。
[独立状态/律诊断](../results/phase2c/20261008T233840Z/state_law_diagnostic/execution.json)。

## N. 计算预算、GPU与在线代价

数据生成95组wall37.88s；75个K参数监督训练wall8.13s（各模型wall总和19.07s，不等于CPU core-time）；
26组OOD生成48.39s。主评价2191.48s、OOD1437.90s、expanded calibration550.21s、nuisance322.06s，
这些任务部分并发，不能把wall直接相加为总耗时。所有依赖、seed、配置与checkpoint SHA保存。
未运行大规模Graph/MLP GPU训练，GPU memory为N/A。

| 首个冻结 M2 test case，串行单线程 BLAS | 在线秒数 | FEM solves |
|---|---:|---:|
| B2_refit_Am | 27.297 | 21855 |
| B5_multistep_K2 | 21.818 | 17125 |
| B4_supervised_K4_seed_85000 | 9.399 | 6805 |

串行同case成本包括prefix fit、data Fisher、五specimen六个长期forecasts与物理诊断，
**不是单次纯网络forward latency**；测量时可能有其他科研任务CPU竞争。
固定K系数deployment很便宜，但d0仍需经典MAP，FEM rollout仍必需；不将这个结果宣传为端到端神经加速。
主五case平均K4/K8 DIC修正约73.9/308.1s、40,087/173,559 FEM solves，显示高K明显增加成本。
疲劳内部时间步、Jacobian有限差分和候选完整重演都计入实际成本。

## O. 完整测试、科学CLI和可复现记录

开发前正确环境下389 passed；首次未设置PYTHONPATH的运行388 passed/1 failed（子进程导入），日志保留。
正式完整最终源码回归：**426 passed**，`python -m pytest -q -W error`，含全部389原测试与37新增测试。
早期425项及评分修复后的426项完整通过均保留；最后为torch可选路径加importorskip，重新运行全套。
另以阻断torch导入的隔离进程运行物理law测试，checkpoint测试仅在torch缺失时跳过，不改变有torch环境的测试要求。
所有37个Phase0–2A历史科学CLI实际运行exit0；首次漏PYTHONPATH的失败矩阵不删除。
Phase2B冻结全部权重/原观测/原calibration重新评价：原8888条记录参数、coverage相关数组和forecast误差逐位相同，
原FAILED状态也相同；另外执行1080条global-fusion OOD记录，合计9968条。
Gaussian follow-up使用正确旧seed71001–71005；首次辅助脚本误用41001的中断日志保留。
trained双图 permutation/causality/reload/ambiguity五checkpoint检查最大误差8.94e−8，checkpoint reload与causal prefix精确一致。

| # | 验收项目 | 实际状态 | 证据与范围 |
|---|---|---|---|
| 1 | Matched neutral correction | PARTIALLY_QUALIFIED | Exact zero coefficients are neutral; primary supervised K4 violates frozen 2e-4 factor tolerance in four of five seeds. |
| 2 | Constant/zero hazard | PASSED | scientific_primitives/constant and unit tests |
| 3 | Nonnegative damage rate | PASSED | bounded multiplicative factor and strict finite hazard contracts |
| 4 | Boundedness/irreversibility | PASSED | all formal forecasts have zero irreversible violations; unchanged Phase1B transition |
| 5 | Consistent material history | PASSED | candidate replay, exact layout ownership and G!=element-count/passive-history tests |
| 6 | FE rollback | PASSED | injected new-damage equilibrium failure preserves committed state; original rollback CLI |
| 7 | Frozen hazard/time consistency | PASSED | analytic semigroup; independent DOP853; query invariance |
| 8 | Parameter mismatch recovery | PASSED | expanded-budget-equivalent A/m refit on all five fixed cases per M1 regime |
| 9 | Structural discrepancy improvement | PASSED | frozen five-case paired K2 gate; high-SNR synthetic distribution only |
| 10 | Correction identifiability | PARTIALLY_QUALIFIED | K2 supported under known inputs; K4/K8 weak/null; A/constant and m/log-x tangent redundancy diagnosed |
| 11 | Future holdout | PASSED | all forecasts before truth/future DIC IO; N0/10/20 cutoff and explicit B3 origin10 |
| 12 | Independent/spatial DIC noise | PARTIALLY_QUALIFIED | frozen two-case OOD diagnostics, not statistical coverage qualification |
| 13 | Boundary/registration confounding | PARTIALLY_QUALIFIED | shared registration improves; unknown loading retains boundary and weak solutions |
| 14 | Cross-mesh sensitivity | PARTIALLY_QUALIFIED | paired global fields and genuine spatial FE refinement; no mesh-objective localization |
| 15 | Spatial basis validity | PASSED | coordinate-defined fixed basis, FE renumbering and coupled spatial rollout |
| 16 | Correct-physics negative control | PASSED_WITH_DEGRADATION | unnecessary flexible fits and biased-state correction worsen long forecasts; failures retained |
| 17 | Classical A/m comparison | PASSED | identical prefix and TRF80 budget, expanded A/m bounds; all old results retained |
| 18 | Five-seed reproducibility | PASSED | 75 frozen checkpoint reloads and five-seed actual retraining replay |
| 19 | Observation-only isolation | PASSED | strict public schema, guarded prefix loading and IO-boundary integration test |
| 20 | Complete Phase0-2B regression | PASSED | 426 pytest after metric-frame regression; all37 historic CLI; complete frozen Phase2B datasets/checkpoints/calibration |

实际五seed重新训练K4 M2，系数、最佳epoch、validation loss逐位一致；另重生成训练prefix FEM/DIC字段逐位一致。
所有75个checkpoint重新加载和checksum检验；推断不导入truth训练数据。见
[确定性重放](../results/phase2c/20261008T233840Z/replay/execution.json)、[最终pytest](../results/phase2c/20261008T233840Z/pytest_final_v3.txt)、
[历史CLI matrix](../results/phase2c/20261008T233840Z/historic_scientific_regression_v2/matrix.json)。
初次最终汇总将所有OOD强制要求收敛，识别出β8的两个budget-exhausted结果；
修正汇总为明确保留tentative状态，未放宽冻结M2主gate或修改旧测试。该日志也保留。

复现必须使用新目录；固定配置在 `configs/phase2c/`，脚本拒绝覆盖已有结果。
先设 `export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/fatigue-mpl`。
例：

```bash
python scripts/train_phase2c.py --stage dataset --output results/reproduce2c/dataset
python scripts/train_phase2c.py --stage supervised --dataset results/reproduce2c/dataset --output results/reproduce2c/training
python scripts/evaluate_phase2c.py --dataset results/reproduce2c/dataset --training results/reproduce2c/training --output results/reproduce2c/test
python scripts/supplement_phase2c_evaluation.py --suite fair --dataset results/reproduce2c/dataset --output results/reproduce2c/fair_calibration
python scripts/generate_phase2c_ood.py --output results/reproduce2c/ood_dataset
python scripts/evaluate_phase2c.py --partition ood --dataset results/reproduce2c/ood_dataset --training results/reproduce2c/training --output results/reproduce2c/ood
python scripts/supplement_phase2c_evaluation.py --suite nuisance --dataset results/reproduce2c/ood_dataset --output results/reproduce2c/nuisance
python scripts/supplement_phase2c_evaluation.py --suite spatial --dataset results/reproduce2c/ood_dataset --output results/reproduce2c/spatial_basis_followup
python scripts/supplement_phase2c_evaluation.py --suite reference --dataset results/reproduce2c/dataset --output results/reproduce2c/oracle_reference
python scripts/verify_phase2c.py --root results/reproduce2c --output results/reproduce2c/scientific_primitives_final
python scripts/diagnose_phase2c_state_law.py --root results/reproduce2c --output results/reproduce2c/state_law_diagnostic
python scripts/replay_phase2c.py --root results/reproduce2c --output results/reproduce2c/replay
python scripts/benchmark_phase2c.py --root results/reproduce2c --output results/reproduce2c/serial_cost
python scripts/audit_phase2c_metric_frames.py --root results/reproduce2c --output results/reproduce2c/metric_frames
python scripts/regress_phase0_to_2a.py --output results/reproduce2c/historic_scientific_regression_v2
python scripts/regress_phase2b_for_phase2c.py --historic results/phase2b/20261008T114713Z --output results/reproduce2c/frozen2b
python -m pytest -q -W error
```

`audit_phase2c.py`用于本次多次保留实验目录的最终审计，明确引用原run与follow-up路径；
全新一次性2B复跑的路径见frozen2b/execution.json，合并审计时需按suite对应读取，不能假定目录格式相同。
`report_phase2c.py`和`export_phase2c_source.py`生成新报告/源码归档；恢复patch已经在/tmp新目录实际执行，未改Git。

## P. 失败、限制与未解决问题

正式资格为PARTIALLY_QUALIFIED，因为主K4监督neutral四seed不达预设阈值，
高K弱可辨识、未知load/大失配/空间OOD存在退化，且每OOD只有两case，无概率校准资格。
没有把低残差、满数值秩或优化器收敛解释为唯一真实材料恢复。
没有用测试集coverage乘因子修正，也没有选择最优seed或删除失败。

训练是synthetic privileged监督；regime-specific prior额外信息、五specimens共享d0假设和0.2nm主噪声均限制实际应用。
首版材料律memoryless、二维小应变平面应力、比例载荷、uniform initial damage；
没有validated真实材料、多轴fatigue、空间模型独立训练、Graph-conditioned basis、有限应变、crack/RUL。
局部软化未经regularization的mesh objective验证；均匀场的cross-mesh接近等价不作高维泛化证据。
fixed low-dimensional correction提高预测不等于其系数有唯一物理解释。

## Q. Phase 3 RL Go/No-Go

**当前NO-GO，不自动进入RL。** M1应先采用A/m重标定；M2支持继续研究非RL低维物理修正，
但没有证明Actor-Critic或时间变化action具有必要性。correct-law negative control和错误state补偿说明，
把correction当任意控制动作可能破坏材料解释；这不能靠更复杂policy解决。

下一轮若获授权，应先扩大预声明noise/load/geometry/structural-family矩阵，增加独立load/反力信息，
检查state、boundary、registration与law误差，降低不可辨识K，建立真实材料/空间训练与独立验证。
然后再定义具有可观测决策信息、明确目标和额外收益的RL问题；现有证据不能支持直接训练SAC/TD3/PPO。
本阶段停在non-RL correction baseline。
