# Phase 2E — Observability-guided experimental design

资格：**`PHASE2E_OBSERVABILITY_PARTIALLY_QUALIFIED`**。RL：**NO-GO**。本轮是低维、非 RL 的观测设计和材料模型区分研究；阶段在此结束。

已有回归 **489 passed**（原469 + 新20）；新独立测试 **40 个物理世界**，五个主要设计臂，另有4个预算匹配反力对照。每个世界共享材料/初始参数的五个加载试件、所有帧和设计臂均作为配对数据，不增加独立样本数。硬失败世界 0，候选诊断失败事件 30，记录的损伤不可逆违例 0。候选边界与未收敛结果保留，ABSTAIN 保留在资格分母。

主要证据：[冻结方案](../results/phase2e/20261009T061530Z/independent_test_freeze.json)、[逐世界与逐设计结果](../results/phase2e/20261009T061530Z/independent_test_evaluation/results.json)、[冻结门槛统计](../results/phase2e/20261009T061530Z/independent_test_summary.json)、[可复现分析与描述统计](../results/phase2e/20261009T061530Z/analysis/)、[源码与资产保护](../results/phase2e/20261009T061530Z/preservation_audit.json)。下面的点估计和距离不构成校准模型概率。

## A. Phase 2D 失败机理与旧证据边界

Phase2D 160 世界由 test100、OOD24、diagnostic36组成；family总计 matched54、parameter30、S1 46、S2/S3/S4各6、S5 12。主 test 中 matched 每个噪声等级8例，parameter/S1 每级各6例，等级为2e-7、5e-5、2.5e-4、7.5e-4、1.25e-3 mm。[分组与完整失败诊断](../results/phase2e/20261009T061530Z/phase2d_failure_analysis.json)。

原8个历史可辨识 S1 世界的身份和结果如下。前6个 σ=2e-7 的世界选择 B5；0062/0065在5e-5选择B1。表中 nuisance 投影额外允许E0、load_factor、tx/ty/theta及d0变化；这是新的局部诊断条件，原8例的实际标定仍是已知E0/BC、独立测量噪声。投影参考点不同会产生不同剩余距离。

| 旧case | σ mm | B7 | 旧B7/B2 N220比 | 原始距离 | 投影+B2参考 | 投影+B5参考 |
|---|---|---|---|---|---|---|
| p2d-0042 | 2e-07 | B5 | 0.162089 | 105.717 | 71.9665 | 75.1743 |
| p2d-0043 | 2e-07 | B5 | 0.170013 | 51.5723 | 37.7763 | 39.3473 |
| p2d-0044 | 2e-07 | B5 | 0.171061 | 54.1577 | 39.5432 | 41.1652 |
| p2d-0045 | 2e-07 | B5 | 0.162019 | 106.263 | 72.0266 | 75.2227 |
| p2d-0046 | 2e-07 | B5 | 0.164214 | 63.3434 | 45.1241 | 47.2408 |
| p2d-0047 | 2e-07 | B5 | 0.180723 | 79.303 | 54.2284 | 56.5246 |
| p2d-0062 | 5e-05 | B1 | 4.53379 | 1.10423 | 0.213537 | 0.417893 |
| p2d-0065 | 5e-05 | B1 | 4.74302 | 0.513618 | 0.465355 | 0.454449 |

0062：B1/B2/B5 的 N20 白化验证 SSE 为130.434/150.605/148.840，B2历史最小奇异值2.84193低于旧门槛3；0065为183.082/185.620/184.382，复杂模型的分数收益未超过旧35.777门槛。全部历史候选均收敛且MAP位于内部。因此缺少支持复杂演化形状的历史观测，是比“优化器没有收敛”更充分的解释。B1在历史上看起来足够好，但在之后较低幅值载荷下缺乏外推精度。

两例原N220损伤RMSE，B7/B2/B5分别为0.0103073/0.00227345/0.00232027和0.00829676/0.00174926/0.00121586。0062的K2本身未优于A/m；0065虽有K2点估计收益，历史观测不足以可靠选择它。这些是失败诊断，不作为新资格测试。未改旧B7/B8代码、阈值、结果或阶段资格；B8原154个gate0、6个gate1，与B7同结果。[两例原始数值摘录](../results/phase2e/20261009T061530Z/old_two_failures_summary.json)。

## B. Model discriminability

对同一公开实验设计，分别以当前前缀拟合参数演化两个候选模型并重解新损伤FEM，计算 Δy=y1-y2。已有协方差分解 Σ=LLᵀ用于 d_w=L⁻¹Δy 与 J_n,w=L⁻¹J_n。所有参数灵敏度使用已验证的无量纲 box 坐标、中心有限差分两尺度检查。边界的虚拟设计灵敏度复用已有单边二阶检查并显式限制部署。

距离仅量化局部预测差异。两个分别拟合的复合模型有参数误差、非线性曲率和模型形式误差，距离大并不保证识别正确。用于旧8例的每假设2000次独立噪声MC固定预测均值和线性化点；它不重标定参数、不抽取材料后验，也不增加物理世界样本数。新的40世界逐一进行完整前缀拟合、设计、实际观测生成和选择，提供额外独立确认。

| 旧case / B2参考+d0 nuisance | 条件MC正确 | 条件MC错误 | 条件MC拒绝 | 条件测量总数 | 新增物理世界 |
|---|---|---|---|---|---|
| p2d-0042 | 4000 | 0 | 0 | 4000 | 0 |
| p2d-0043 | 4000 | 0 | 0 | 4000 | 0 |
| p2d-0044 | 4000 | 0 | 0 | 4000 | 0 |
| p2d-0045 | 4000 | 0 | 0 | 4000 | 0 |
| p2d-0046 | 4000 | 0 | 0 | 4000 | 0 |
| p2d-0047 | 4000 | 0 | 0 | 4000 | 0 |
| p2d-0062 | 51 | 25 | 3924 | 4000 | 0 |
| p2d-0065 | 58 | 18 | 3924 | 4000 | 0 |

上述正确/错误为固定模型均值条件下的分类计数，拒绝选择未被算作成功；不能当作真实材料模型识别率或复合模型参数不确定性校准。

## C. Nuisance projection 与精确零空间

稳定SVD给出 J_n,w=U S Vᵀ，只保留超过相对容差及有限差分分辨率的方向；d_perp=d_w-U_r(U_rᵀd_w)。保存原始、可吸收、剩余差异与奇异值；验证正交性和范数分解，不显式构造大矩阵P或普通求逆。生产设计仅投影公开声明未知的E0、loading或共享配准；d0及疲劳/修正参数在联合 Fisher 中保留完整交叉项。旧失败诊断另外展示投影d0的影响。投影是局部线性、未按box限制的切空间计算，不能单凭投影零声称两个非线性有界模型全局等价；下述FE零空间例另有界内精确等价参数。当前机制未积分所有候选参数的完整非线性后验。

解析退化 E_eff=E0(1-D) 已通过真实FE复核：(E0,D)=(2040,.12) 与 (2200,.184)的位移最大差为0，E_eff=1795.2 N/mm²。局部白化原始差266.319在E0 nuisance投影后仅2.56e-10；数值零空间按分辨率处理，固定均值MC全部ABSTAIN。普通满数值秩不能替代实际信息。[Fisher/Jacobian/零空间原始场](../results/phase2e/20261009T061530Z/information_probes_v3/static/)。

## D. 额外观测设计与预算

共同已观测前缀为N0/10，每世界五试件×16点×2分量×2帧=320标量。N10时仅用该前缀设计N10→20实验；实际完成10疲劳循环后获得N20观测。主要新增预算为160 DIC标量、每试件10循环、每世界50 specimen-cycles，预冻结候选幅值0.25/0.4/0.7/1和R=-1/0。D<0.6、名义应力≤35 N/mm²及原求解失败/单步损伤增量限制同时约束实验。未扩大范围。

五策略：fixed沿用原随机不规则16点与原schedule；uniform采用覆盖域的16点与原schedule；random在同一公开32个设计候选中预先种子抽样；Fisher最大化两候选模型条件演化最小奇异值的较小平方；discrimination最大化两个nuisance线性化参考的较小投影距离。候选由8个真实比例疲劳加载条件×4个物理坐标点集构成，连接/插值使用原FE形函数，无图像reshape。坐标池64公共网格点+原16不规则点只是本基准候选集，不改变底层任意DIC坐标契约。

预算相同指相同测量成本、循环上限、应力/损伤边界和TRF上限。不同幅值/R的实际能量暴露不相同，逐世界保存应力跨度平方×循环的公开proxy及实际damage increment。改变疲劳载荷会改变真实轨迹，所有比较误差均相对该设计自己的真实rollout；不把减小损伤路径等同于信息增加。该proxy不是耗散能或实际实验费用。反力1标量成本预定为2，15点×2+1反力每试件保持32单位，总计160成本、155实际标量；没有把额外反力藏在原160 DIC预算之外。

## E. 独立反力与静态诊断加载

固定同一损伤状态的真实FE结果：

| 静态方案 | 成本单位 | 实际标量 | data rank /3 | 奇异值 |
|---|---|---|---|---|
| original_16 | 160 | 160 | 1 | 581.096, 4.98359e-10, 1.75812e-10 |
| reaction_budget_matched_15 | 160 | 155 | 2 | 567.772, 310.518, 2.10205e-10 |
| static_two_modes_8_each | 160 | 160 | 1 | 994.186, 1.27623e-09, 1.75897e-10 |
| more_64 | 640 | 640 | 1 | 1117.62, 9.3808e-10, 3.81653e-10 |
| healthy_extra | 320 | 315 | 3 | 708.199, 313.149, 160.459 |

在载荷控制、均匀损伤、已知ν下，位移约束load_factor/[E0(1-D)]。独立净反力约束load_factor，使秩1→2；E0–D仍退化。额外健康试件使秩3，成本为320，属于额外信息对照。更多64点耗费640，沿已有方向增加信息，不移除零空间。固定损伤的轴向/剪切两个独立静态模式仍保持秩1；对空间非均匀参数的潜在作用未据此推断。

静态结果只满足 ∂u/∂A|D=∂u/∂m|D=0；如果施加了先验演化历史，静态数据可间接约束历史参数，但它本身不是hazard或演化速度测量。主要加载设计则真正推进FE–damage轨迹。反力正式确认仅为预冻结的两个load-unknown和两个material-unknown世界，精度受n2条件限制：

| 世界 | 观测臂 | B7部署 | E9部署 | E9 N220 RMSE | 成本 | 标量 |
|---|---|---|---|---|---|---|
| p2e-test-034 | discrimination | B0 | B0 | 0.064577 | 160 | 160 |
| p2e-test-034 | discrimination_reaction | B0 | B1 | 0.00134623 | 160 | 155 |
| p2e-test-035 | discrimination | B0 | B0 | 0.0575179 | 160 | 160 |
| p2e-test-035 | discrimination_reaction | B0 | B2 | 0.00190516 | 160 | 155 |
| p2e-test-036 | discrimination | B0 | B0 | 0.0222154 | 160 | 160 |
| p2e-test-036 | discrimination_reaction | B0 | B0 | 0.00418307 | 160 | 155 |
| p2e-test-037 | discrimination | B0 | B0 | 0.133482 | 160 | 160 |
| p2e-test-037 | discrimination_reaction | B0 | B0 | 0.133157 | 160 | 155 |

## F. 非RL设计与选择基线

所有策略使用同一候选模型B0、B1、B2、B5与80次TRF函数评价上限。B0仍估计必要d0及声明nuisance，其hazard形式/参数保持物理先验；B1估计A，B2估计A/m，B5为原DIC-only K2。prefix、full-prefix目标都从一致初始状态重演化，warm-start只提供参数初值，不重复累加旧posterior/likelihood。

冻结B7调用原score/rank/bounds规则，在同一实际N20观测上比较。独立命名 E9_PROJECTED_VALIDATION：B0预测残差若≤M+3√(2M)，保持物理模型或在弱信息下ABSTAIN；否则按收敛、边界、联合前缀+新帧box最小奇异值≥3筛选简单参数标定。只有投影模型差异≥3、独立N20条件正态方向分数>2.32635且验证残差改善，才支持K2；full-prefix再次做部署检查。ABSTAIN可部署B0或接受的简单标定，不宣称唯一材料律已识别。分数是预冻结条件plug-in screen，不是校准p值、模型posterior或复合假设检验。

## G. 噪声、nuisance与模型OOD

新世界：matched/parameter/S1各8；S2/S3/S4/S5各2；registration/load/material/spatial各2，合计40。development8和validation4使用不同种子；最终runtime另外在同一development load世界做重复接口/反力检查，不计新增独立试件。全部预冻结种子210000+/220000+/230000+与输入配置、单位、网格、公开未来载荷保留。

σ为5e-5/2.5e-4/7.5e-4 mm；沿用旧高SNR2e-7及1.25e-3资料用于诊断，不用于新资格。空间covariance和共享registration由既有接口处理；独立测量噪声条件于全局共享tx/ty/theta，未额外重复计入Σ。真实配准、load bias、E bias不进入设计器。主family三组的分层噪声描述如下；各噪声等级是不同物理世界，不能当作配对纯噪声因果效应。

| σ mm | 设计 | 主世界数 | 平均E9 N220 RMSE | ABSTAIN |
|---|---|---|---|---|
| 5e-05 | fixed | 12 | 0.00356 | 8 |
| 5e-05 | fisher | 12 | 0.00330647 | 8 |
| 5e-05 | discrimination | 12 | 0.00389017 | 8 |
| 0.00025 | fixed | 6 | 0.00559034 | 2 |
| 0.00025 | fisher | 6 | 0.00533515 | 2 |
| 0.00025 | discrimination | 6 | 0.00623321 | 2 |
| 0.00075 | fixed | 6 | 0.0144008 | 6 |
| 0.00075 | fisher | 6 | 0.0141573 | 6 |
| 0.00075 | discrimination | 6 | 0.0140961 | 6 |

令 x=Y_a/.025，Y_a与reference energy单位均为N/mm²，x、D、R和strength均无量纲，hazard单位cycle⁻¹。matched为A x^m，parameter保持形式、A变为先验的1.35–1.65倍且m=1.12–1.22；S1为A x^m[1+3x/(1+x)]，S2为A[max(x−.5,0)]^m，S3为A x^m exp[.55(R+1)]，S4为A x^m exp(3D)，S5为A x^m exp[.3 cos(πX/2mm)]。这些定义沿用未修改的 [reference law](../src/fatigue_dic_rl/correction/law_family_benchmarks.py)。先验A=8e−5 cycle⁻¹、m=1。K2仍为先验hazard乘exp[2 tanh(c0 t+c1 t²)]，t=x/(1+x)，c0/c1∈[−4,4]，不产生自由积分点damage action；既有hazard为零时仍为零。

S1饱和、S2阈值、S3额外R、S4损伤加速、S5连续空间变化沿用已验证synthetic reference定义；S2–S5始终结构OOD。S3只使用比例载荷。S2的严格零hazard、S3额外R及S5空间材料变化可能超出全局乘性K2族；不能归因于观测不足一个原因。这些law均是受控oracle，非实材料标定。已观测前缀的残差、优化收敛/边界、Fisher、实际加载差异、数值时间误差和未来误差分别保存。

| family（含uncertain子组） | n | fixed E9 N220 | Fisher E9 N220 | discrimination E9 N220 | discrimination拒绝 | K2部署 |
|---|---|---|---|---|---|---|
| matched | 12 | 0.0124737 | 0.0084914 | 0.00847386 | 4/12 | 0 |
| parameter | 8 | 0.00698029 | 0.00670002 | 0.00619994 | 6/8 | 0 |
| S1 | 12 | 0.0238787 | 0.0234487 | 0.0249792 | 12/12 | 0 |
| S2 | 2 | 0.00276476 | 0.00270552 | 0.00268152 | 1/2 | 0 |
| S3 | 2 | 0.000608874 | 0.000688268 | 0.000700864 | 1/2 | 0 |
| S4 | 2 | 0.00489783 | 0.00087343 | 0.00388325 | 1/2 | 0 |
| S5 | 2 | 0.00296296 | 0.00275733 | 0.00243157 | 0/2 | 0 |

## H. 新选择方法与冻结B7公平比较

所有未来N30/40/70/120/220在N≤20标定与选择完毕后仅作评价。所有世界采用2×1 mm几何、厚度.8 mm、六节点/四Tri3积分点、已知ν=.3；E/d0/load scale按各物理seed采样。没有宣称本轮跨网格或跨几何资格。以下每行B7和E9使用完全相同的实际观测、载荷、FE精度和时间split：

| 设计 | 世界 | 当前E9 damage RMSE | B7 N220 RMSE | E9 N220 RMSE | 空间holdout u RMSE mm | ABSTAIN | E9 K2 |
|---|---|---|---|---|---|---|---|
| fixed | 40 | 0.0107187 | 0.0127228 | 0.0128635 | 2.55035e-05 | 27 | 0 |
| uniform | 40 | 0.00930736 | 0.0116066 | 0.0115002 | 2.59822e-05 | 26 | 0 |
| random | 40 | 0.0127338 | 0.0153743 | 0.0151584 | 2.91332e-05 | 28 | 0 |
| fisher | 40 | 0.00932945 | 0.011427 | 0.0112733 | 2.32272e-05 | 27 | 0 |
| discrimination | 40 | 0.00973973 | 0.0116535 | 0.0117608 | 2.39379e-05 | 25 | 0 |

所有周期/方法/参数/场保留在原始records和npz。点估计改善可以来自更适当的A/m标定，而不是识别出了结构修正；必须与K2启用次数和ABSTAIN联合解读。一个明确的E9风险是adequacy仅看新N20帧：低幅值实验可能使N20信号不足，进而回退B0，尽管先前N0/10已呈现物理先验失配。开发S1 random就有这样的长期退化。这属于选择规则和观测设计相互作用的失败，不能全部解释为传感器噪声。正式test保留冻结规则；后续联合历史adequacy规则需要重新预注册、重新独立测试。

另一种实际独立test失败是p2e-test-017的discrimination设计（a=.7,R=0,outer点集）：E9在历史验证后接受B2，full-prefix拟合虽收敛且不触边，但实用秩2/3，因此部署检查强制退回B0。N220 B0/E9=0.00911974，冻结B7保留B1=0.00234323；B2点估计仅0.000504262。该退化来自重新线性化后的弱方向和回退规则相互作用，不是优化未收敛，也不是本例N20 adequacy分支。ABSTAIN或简单Physics fallback并不保证未来预测足够准确；不能用拒绝标签掩盖错误。后续更安全的简单模型回退必须重新冻结并独立测试。

以下为所有40世界的设计之间配对描述比较，负值表示前项误差更低；它没有参与选择器或资格门槛：

| E9设计对比 | 独立世界 | N220 RMSE差均值 | world-paired bootstrap95 |
|---|---|---|---|
| fisher − fixed | 40 | -0.00159025 | -0.00416266, -9.01721e-05 |
| fisher − random | 40 | -0.00388517 | -0.00986442, -0.000420039 |
| discrimination − fixed | 40 | -0.00110275 | -0.0037839, 0.000537557 |
| discrimination − random | 40 | -0.00339767 | -0.00938216, 2.81892e-05 |

Fisher相对fixed/random的总体平均误差降低，但这是异质场景、不同实际疲劳路径上的描述统计；多重比较未用作额外资格检验，也不证明结构识别可靠性或RL必要性。各设计仍以自己的真实trajectory计算误差，新增疲劳暴露不同，不能把误差降低全部称为信息收益。

![旧失败的局部区分度与独立预测比较](../results/phase2e/20261009T061530Z/analysis/comparison.png)

## I. 统计不确定性与冻结门槛

40世界是CPU方向筛查，其总二项Wilson最坏半宽约0.148；主family n8的同量约0.285。选择这一规模是声明的小规模CPU计算预算；冻结时已明确其不足以取得误激活≤10%的可靠资格，没有预先期待凭8个matched零激活通过该门槛。至少35个独立matched且零事件才可能使95% Wilson上界≤.10（仍不证明所有分布的可靠性）；本轮未根据结果事后扩大样本。主family n8、各OOD/uncertain条件n2，不能证明普遍可靠性。不能将200设计臂、五试件、帧或2000测量MC当成新世界。新matched n8零激活的95% Wilson上界0.324，达不到冻结0.10门槛；旧matched40不能与新8混用扩大资格样本。所有ABSTAIN计入structural分母，失败算无实际收益。

预定义门槛：matched E9−B0 N220平均excess bootstrap上界≤.001、max≤.01、false-K2 Wilson上界≤.10；S1相对同设计B7的实际收益≥.001至少80%，median ratio≤.90。配对均值bootstrap10000次、seed241001，计算单位world。没有按final test结果修改规则或最小收益。

| 设计 | matched误K2 | Wilson上界 | matched gate | S1实际收益 | median E9/B7 | S1 gate |
|---|---|---|---|---|---|---|
| fixed | 0/8 | 0.324408 | False | 0/8 | 1 | False |
| uniform | 0/8 | 0.324408 | False | 1/8 | 1 | False |
| random | 0/8 | 0.324408 | False | 1/8 | 1 | False |
| fisher | 0/8 | 0.324408 | False | 1/8 | 1 | False |
| discrimination | 0/8 | 0.324408 | False | 0/8 | 1 | False |

候选拟合总计832条，未收敛26、MAP触边161、实用秩不足421；这些不是独立物理样本，计数只是优化稳定性诊断。30个候选诊断失败事件全部为已触边参数无法执行prospective centered sensitivity，在6个世界×5设计重复记录；它们按规则被拒绝，没有FE hard failure、full-fit exception或forecast exception。完整拟合中26个未收敛和421个弱实用秩仍是真实方法限制，未筛除。[逐项频率](../results/phase2e/20261009T061530Z/analysis/optimization_frequency.json)。

不确定性资格：NOT QUALIFIED。设计的局部信息/距离未积分完整参数后验，未验证网络/模型posterior、damage field或forecast区间覆盖率。Phase2A原89%及Phase2B未校准协方差没有用作监督、真值或风险权重。数值满秩和小网络variance不被解释为科学校准。

## J. 物理一致性、时间推进与泄漏

底层Tri3、FatigueEvolutionSolver、DamageLaw、transition、ObservationOperator源文件均与开发前逐字节一致。每个参数候选从自己的d0及当前候选材料/疲劳参数重建历史；更新参数后重新演化至N20；每个新损伤重解FEM。hybrid hazard非负、损伤不可逆、物理单位及quad/node/DIC支撑显式分离。旧原子回滚测试全部保留。

真实材料状态随所选N10→20疲劳激励变化。当前观测前的prediction、N20 filtered状态与N>20 forecast用既有不同role标签。静态probe推进循环为0；主要试验每试件真实推进10循环，之后未来公开schedule完全开环。step2/1/.5在固定低维hybrid参数下，相对.5的damage RMSE为4.47951e-05/1.49931e-05，插入离网格查询时刻的后续状态差均为0。这是受控时间误差验证，没有声称所有空间law/网格收敛或局部损伤网格客观。

设计器只读共同N0/10；新N20设计先写入committed_design，测量随后产生。原producer兼容资产中有一组未使用的固定schedule N20帧，IO守卫拒绝设计阶段读取它。N>20真值在全部候选forecast_before_truth文件写入后才生成；未来实测读取数0。family/hazard/true damage只在独立producer和最后scorer使用。新规则只看已观测历史/公开加载/当前估计。旧8例只作诊断，资格用新的40世界。正式test运行前校验冻结config与172个runtime文件hash，最终再验一次。[freeze SHA256](../results/phase2e/20261009T061530Z/independent_test_freeze.json)：`4d1de98bd5fc734a0adeae6d357ab13357ed7bf9c3eb03505cb82eea34e0540b`。

## K. 计算成本

CPU SciPy/FEM，无新网络训练或RL依赖。GPU driver不可用，torch.cuda false；没有更换驱动、环境或基础求解器。Python3.13.12、NumPy2.4.4、SciPy1.17.1、PyTorch2.13.0 CUDA build13.0、pytest9.0.3，完整依赖export保留在baseline。正式独立评价6 workers，wall=5342.79s；这是一次配对全候选科研比较，非单模型的神经forward延迟。

| 设计 | prefix+library+arm median FEM | 同项median wall s | 同项max wall s | 比较中forecast median s | 选中投影距离median | 位移SNR median |
|---|---|---|---|---|---|---|
| fixed | 124292 | 180.483 | 640.239 | 8.93836 | 0.225089 | 108.132 |
| uniform | 126602 | 182.871 | 620.082 | 8.93892 | 0.263224 | 122.273 |
| random | 117165 | 175.125 | 591.224 | 8.85801 | 0.107789 | 47.5868 |
| fisher | 129530 | 192.231 | 624.759 | 8.97986 | 0.301454 | 134.178 |
| discrimination | 127767 | 188.816 | 586.292 | 8.94988 | 0.394239 | 116.769 |

公共library只算一次并在五设计间共享；上表按独立部署需承担的完整library成本报告。fixed/uniform实际可省设计library，原始组件成本可独立查看。arm含B0/B2/B5、需要的B1、B7/E9的完整拟合、Fisher和参考forecast，用于公平科学比较；不是孤立选择器的纯线上延迟。TRF nfev、FEM调用、历史/虚拟设计/选择/forecast成本均保留。inference FEM计数覆盖参数优化、虚拟设计、信息检查和candidate forecast；不包含私有oracle真值生产的FE次数。arm wall还包含合成测量/未来真值与文件写入，因此是整体科研workflow时间，不可视作精确纯在线时延；组件selection/forecast wall另有保存。计算预算按共同上限控制，实际收敛耗时不同，不宣称五策略恰有相同秒数或耗散能。

## L. Source、测试与恢复审计

branch `feature/phase0-architecture`；HEAD `dc239d1b7b6dd1e44d93e71bdcccccc0f12e8bc9`仍是initial commit，大部分完整科研代码未提交。开发前292文件与Phase2D最终manifest完全一致，canonical源码SHA为 `d37ba93feb411397cf28a41063b373b650ddcf13232764fb76d45e3d581ed9bf`；轻量快照SHA `fa155fa974be035772c51b0118888a3f735a513131b7724bd8dd0b3d6ef30a06`。[开发前恢复包](../results/phase2e/20261009T061530Z/baseline/RECOVERY.md)。未修改任何预存在源码/报告，保护156 checkpoint和191435历史资产的清单通过。没有commit/push/merge/reset/clean。

独立test的runtime、config、cases在开发/验证之后于2026-10-09 07:28:50 UTC冻结。开发过程中接口和数值分辨率检查有修复，早期运行日志保留；冻结前对最终runtime重新验证真实N20/反力/未来轨迹。测试阶段没有调整设计规则、加载候选或模型复杂度。

| 回归或科学项 | 实际执行与证据 | 状态 |
|---|---|---|
| Baseline / Final pytest | 469 / 489 passed；`-W error`；[日志](../results/phase2e/20261009T061530Z/pytest_final.txt) | PASSED |
| Phase0–2A CLI | 全部37项，包含全部coverage脚本；[matrix](../results/phase2e/20261009T061530Z/historic_regression/matrix.json) | PASSED |
| Phase2B frozen reevaluation | 全部声明uniform/OOD/fixed-noise/Gaussian/architecture；不重新校准；[逐数值对比](../results/phase2e/20261009T061530Z/phase2b_regression_audit/summary.json) | PASSED |
| Phase2C replay | 五训练seed系数/epoch/loss和原生成数据逐数值一致；[execution](../results/phase2e/20261009T061530Z/phase2c_replay/execution.json) | PASSED |
| Phase2D core prefix replay | 160世界、1280fit-stage，训练/验证SSE差0，不读truth | PASSED |
| 旧8例机理 / nuisance投影 | 8 used worlds，qualification count0 | PASSED（诊断） |
| 精确不可辨识 / 反力 / 静态独立加载 | 真实FE、数值零空间、成本1→2→3秩对照 | PASSED（声明范围） |
| 候选加载与因果实际rollout | 40世界×五主要臂及四反力臂，完整FE–damage | EXECUTED |
| 高噪声与ABSTAIN / 结构选择 | 冻结条件按上表评分，不移除困难情景 | PARTIALLY QUALIFIED |
| matched科学资格 | 点估计门槛与误激活Wilson置信上界均保留 | 见冻结gate；不能以小样本授满资格 |
| 时间步和query不变性 | step2/1/.5，固定参数真实solver | PASSED（单一受控hybrid） |
| 任意几何/高维场/完整非线性posterior | 本轮没有扩展 | NOT EXECUTED / NOT QUALIFIED |

旧Phase2C全305+OOD完整优化没有在本轮重新训练/优化全矩阵；本轮执行其确定性五seed和生成数据回放，并保留完整旧科研证据。全量pytest覆盖Phase0–2D。本轮没有将仅运行的子集说成全部旧科研脚本重跑。

[最终源码、增量patch与恢复验证](../results/phase2e/20261009T061530Z/final_source_verified/RECOVERY.md)；最终完整source SHA在外部manifest/audit中，避免报告自引用哈希。归档排除所有results/旧tar/checkpoint/cache；实际在空临时目录执行patch重建并逐文件核对。科研资产[delivery manifest](../results/phase2e/20261009T061530Z/delivery_assets.json)独立保留。[逐数组与预算/因果顺序独立审计](../results/phase2e/20261009T061530Z/scientific_audit.json)核对实际forecast文件、真值生成顺序、加权RMSE重算、不可逆性、匹配预算及冻结source，不替代完整的信息流形式证明。[Git baseline freeze proposal](GIT_BASELINE_FREEZE_PROPOSAL.md)仅供审阅，不执行Git写操作。

### 可复现执行

使用已保存的独立冻结记录，不重新运行freeze脚本来生成不同记录；恢复到新目录，在兼容环境使用 `PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`。所有输出目录必须新建：

```bash
python scripts/verify_phase2e.py --split test --freeze-record <restored-freeze-record> --output <new-independent-test-data> --workers 6
python scripts/evaluate_phase2e.py --dataset <restored-independent-test-data> --freeze-record <restored-freeze-record> --output <new-evaluation-directory> --workers 6
python scripts/summarize_phase2e.py --evaluation <new-evaluation-directory> --output <new-summary.json>
python scripts/probe_phase2e_information.py --case <restored-development-case-000> --output <new-information-probes>
```

正式已执行的配置、freeze、dataset/cases manifest、log和原始arrays位于本报告证据目录。分析程序保存为外部研究资产 `analysis/report_phase2e.py` 和 `analysis/audit_phase2e_artifacts.py`，仅在test结果封存后读取结果，不被designer或estimator导入。不是可微FEM或训练程序。

## M. 下一阶段 Go/No-Go 与科学限制

`PHASE2E_OBSERVABILITY_PARTIALLY_QUALIFIED` 仅涵盖所声明的有限合成、低维模型和信息诊断。结构family可靠性、完整不确定性及稀疏OOD统计均不足。自动ABSTAIN是一种诚实的输出，不会把缺少材料律信息转换成恢复真值的资格。位移拟合、数据满数值秩与长期演化模型正确性分别评价。整体位移SNR的分子是位移幅值，而材料律区分度的信号是候选模型位移差；前者高并不保证后者高，不能以整体SNR替代可辨识性。

RL维持 **NO-GO**，扩大K/Graph correction亦暂缓。独立载荷测量与健康材料诊断的小规模后续研究有条件值得继续；其改善的是信息条件，还没有证明需要学习policy。非RL候选设计、A/m标定和信息/残差拒绝机制已经是必要基线；实验设计优于某个随机设计不证明RL有必要。优先改善公开载荷/配准、独立健康材料信息及足够跨度的真实疲劳历史，重新设计独立试件/噪声标定与时间holdout。更长疲劳激励或更多点需要重新冻结成本和损伤安全范围，不能沿用本轮40测试调阈值后复用同样世界取得资格。

本轮没有证明真实材料、裂纹路径、局部软化网格客观性、任意高维损伤后验、神经区间校准或所有结构失配修正。没有进入下一阶段或RL。
