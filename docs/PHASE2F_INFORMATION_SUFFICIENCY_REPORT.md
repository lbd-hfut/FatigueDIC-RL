# Phase 2F — Source baseline readiness and fatigue-law information sufficiency

资格：**`PHASE2F_PILOT_PARTIALLY_QUALIFIED`**。Git 源码恢复预检通过；初始状态的健康/反力信息诊断通过；当前低维修正的结构材料律信息仍不足。**正式结构测试扩展与 RL：NO-GO。** 本轮停止。

原有489项回归、新增13项测试，共 **502 passed**。Development仅 **3个独立物理世界**、十个完整拟合协议、三个固定点信息协议；不是13个独立世界，没有执行正式独立测试。60份主forecast数组独立重算的最大指标差为0，损伤不可逆违例0；40份full-prefix候选中6份未收敛、2份触边、13份实用秩不足，全部保留。

可审阅结论是：健康基线＋反力解除声明的静态 E0–D–load 零空间，但不自动增加足够的结构演化律信息；更长/多幅值历史可改善选定模型内参数信息，拟合后的A/m与K2仍可几乎重合。若仅确定E_eff，不能称作恢复真实损伤。没有修改旧资格、B7/E9门槛或测试分母。

主要证据：[预先固定的Development方案](../results/phase2f/20261010T010645Z/development_v2/plan.json)、[逐协议原始结果](../results/phase2f/20261010T010645Z/development_v2/summary.json)、[信息/静态/时间诊断](../results/phase2f/20261010T010645Z/development_v2/diagnostics/summary.json)、[profile与逐方法描述指标](../results/phase2f/20261010T010645Z/development_v2/analysis/summary.json)、[独立数组与信息审计](../results/phase2f/20261010T010645Z/development_v2/scientific_audit.json)。没有计算可宣称的独立结构分类准确率。


## A. Git baseline readiness

[Phase 2E Git 冻结预检](PHASE2E_GIT_FREEZE_READINESS.md)给出精确源码清单、预期 staged diff、ignore/外部资产建议及推荐 commit/tag。branch 为 `feature/phase0-architecture`，HEAD `dc239d1b7b6dd1e44d93e71bdcccccc0f12e8bc9` 仍是 initial commit，不能代表当前完整源码。Phase 2E 的 314 文件 canonical source SHA256 为 `0f98b2593cf9f73c6b03eef462f736aeaab52cf9720f3cd986fca9224e9a2ae5`；完整包和 baseline+patch 两条路径都已在独立空目录恢复并逐文件验证。

旧 RECOVERY 的 `../baseline` 实际指向 Phase 2E，同文中 Phase 2C 只是旧临时目录前缀，具有误导性但不妨碍独立恢复。保留原文，并在新预检说明澄清。开发前源包、manifest、HEAD-worktree patch、环境、161 个 checkpoint 哈希和历史资产保护清单均保留在[本轮 baseline](../results/phase2f/20261010T010645Z/baseline/)。没有 Git 写操作、旧报告改写或科研结果覆盖。本轮源码作为后续增量单独保存；[最终 source/patch/独立恢复](../results/phase2f/20261010T010645Z/final_source_verified/RECOVERY.md)的哈希在外部 audit/manifest，避免报告自引用。

## B. Phase 2E evidence audit

读取正式报告、原始逐世界结果、冻结门槛与 delivery ledger。实际复核 40 个独立物理世界、204 个配对观测臂；Phase 2E 49916 项资产 SHA 全量核验通过。[只读证据复核](../results/phase2f/20261010T010645Z/phase2e_evidence_review.json)。原 Fisher S1 收益 1/8，discrimination 0/8，原 matched n8 的零激活 Wilson 上界约 .324；全部原资格与 RL NO-GO 保留。

旧世界仅用于机理审阅，不纳入新资格样本。p2e-test-017 的退化来自 full-prefix 实用秩不足后的冻结回退，不能归结为优化未收敛；新帧-only adequacy 的失败是另一种机制。原 Phase2D 0062/0065 的观测也不足以可靠支持复杂演化形状。不能把这些失败统称为测点太少：模型族不包含真机制、局部弱信息、选择规则回退及数值精度须分别检查。本轮没有修改 B7/E9 函数或阈值。

## C. Initial state、constitutive parameter、structural discrimination

初始均匀损伤的静态刚度为 E_eff=E0(1−d0)。载荷控制下，DIC 主要约束 load_factor/E_eff；增加相同模式测点或静态幅值不能解除精确退化。净反力约束载荷尺度，健康参照提供独立 E0 信息。静态诊断中 A/m/hazard 不参与 operator，∂u/∂A|D=∂u/∂m|D=0；不能将状态可观测性改善当作疲劳律识别成功。

疲劳参数在指定模型内的局部可辨识性来自真实 FE–damage 历史。每个参数候选由自己的 d0、E0、A/m 或 c0/c1 重建完整历史；每次新损伤都重新解 FEM。d0 与演化参数的联合 Fisher 保留交叉项。当前状态误差由隔离的 synthetic scorer 评价，不传给 estimator。

未知模型形式要求比较两个均经合法标定的复合模型。参数模型的满数值秩不证明结构模型能被识别；一个 A/m 模型仍可在有限驱动范围内近似结构真律。实际 d0 误差、两个非真值 cold start 和 profile 中 d0 的补偿均记录。cold start 研究优化初值，不能冒充“硬固定错误损伤先验”实验；后者本轮未进行。

## D. Independent reaction and healthy baseline

| N0信息 | rank /3 | 奇异值（box坐标） | d0条件灵敏度范数 | d0估计 | load_factor |
|---|---|---|---|---|---|
| DIC | 1 | 508.203, 1.05185e-09, 1.72165e-10 | 1.17262e-09 | 0.115924 | 0.993882 |
| DIC_force | 2 | 523.924, 292.193, 1.6506e-09 | 1.93341e-09 | 0.101179 | 1.0501 |
| DIC_healthy | 2 | 643.569, 147.064, 7.20693e-10 | 152.585 | 0.0845112 | 1.00644 |
| DIC_force_healthy | 3 | 644.998, 295.985, 144.247 | 152.585 | 0.0845112 | 1.0501 |

仅DIC与force组的条件d0信息约1e−9，不能恢复唯一damage。healthy-only组d0可识别但E/load仍有零方向。三组有界等价参数 E0=2100/2350/2550、d0=.07817/.17624/.24085、同一load_factor在DIC+force下最大位移差≤4.36e−17 mm；增加healthy后白化差异约34/111/206。这个实际有界FE例验证了非线性补偿，而不只是局部零空间。


健康参照是额外、独立 noisy displacement 测量，已知 D=0；它与损伤试件共享真实 E0 和载荷尺度，这是本受控合成假设。它不是从 oracle E0 直接建立的先验。未知 E0 的 public metadata 固定 nominal 2050 N/mm²，候选自己替换 E0。实际载荷偏差不进入估计初值，只能由观测识别。

反力仅在 N0 每试件测量一次，σ=.01 N，没有逐帧复制独立证据。force-only 仍不能由静态数据解除 E0–D；healthy-only 可以增加损伤对比信息，但 E0/load_factor 仍等价。两个独立来源合用才关闭本静态三参数零空间。健康试件的材料散布、载荷传感器系统误差、ν/厚度误差未校准，不能将本结果称为真实材料标定。

## E. Fatigue loading excitation and history

| S1配对协议 | cycle跨度 | 成本 | prefix平均ΔD | 幅值范围²×cycle proxy/试件 | Am s_min | K2 s_min |
|---|---|---|---|---|---|---|
| fixed_short | 20.0 | 480 | 0.00997276 | 80 | 2.74802 | 3.4235 |
| fixed_long | 60.0 | 480 | 0.0313083 | 240 | 9.41027 | 11.465 |
| multi_long | 60.0 | 480 | 0.0119758 | 102.6 | 3.42042 | 3.68088 |
| with_reaction_healthy | 60.0 | 650 | 0.0119758 | 102.6 | 3.41983 | 3.67931 |

proxy=ΣΔN[a(1−R)]²，不是耗散能或真实试验费用。fixed60相对short20增加暴露，局部s_min增强；multi60相对fixed60降低损伤增长，也改变驱动区间。dense history的额外640标量在固定线性化点产生PSD Fisher增量，Am/K2最小特征值分别28.78/41.87；这证明更多数据增加参数信息，不证明结构family可分。


候选预先固定于既有安全 envelope：名义应力≤35 N/mm²、D<.6、原单步损伤增量≤.03、步长2 cycles。multi 历史先 20 cycles 幅值1/R=−1，再20 cycles 幅值.4/R=−1，再20 cycles 幅值.7/R=0；实际求解 min/max FEM 和疲劳推进，没有用静态场替代演化。未来统一为幅值.4/R=−1。

三个世界各有五个共材料加载试件；所有对照共享同一物理世界及可配对的噪声种子。短历史 N0/10/20，长历史 N0/30/60，二者都是480 DIC标量，但完整暴露分别100/300 specimen-cycles。dense N0/10/…/60 只作固定估计点的信息诊断，1120 标量，不能声称等成本。增加反力+10成本单位，增加健康 DIC+160；比较负担逐案例保存。

未来仅评价 cutoff+[10,60,160]，短历史对应N30/80/180，长历史对应N70/120/220。不同当前 cycle、实际驱动和损伤路径必须一起报告，跨协议较低RMSE不等于纯信息增益。不同R也与幅值和时段配对改变，本轮没有隔离识别任意 mean-stress law。

## F. Nuisance-projected discriminability and nonlinear compensation

由已验证 covariance.factor 白化参数灵敏度与候选差异：J_w=L⁻¹J，d_w=L⁻¹(y_K2−y_Am)。使用已有两尺度差分与 SVD，d_perp=d_w−U_r(U_rᵀd_w)。分别投影初始状态/公开 nuisance 和全部候选参数切空间；后一项用于诊断 A/m 非线性族附近的补偿，不能叫作仅测量 nuisance。两模型线性化点分别报告；数值分辨率参与秩截断。不同参数点重新线性化的Fisher不要求逐项单调，PSD增量检查只用于同一点、独立新增观测。

F_data=J_wᵀJ_w 保存全部交叉项；condition number、numerical/practical rank、弱右奇异向量、初始/演化条件灵敏度、可辨识子空间相关系数一并保留。秩亏时的相关矩阵仅为该子空间，不能解释为完整后验，未识别方向方差不因伪逆变小。

| S1协议 | 整体位移SNR | Am/K2 raw norm | 初始/nuisance投影 | Am全参数投影 | 全参数投影每分量RMS |
|---|---|---|---|---|---|
| fixed_short | 120.646 | 0.0950133 | 0.0950133 | 0.0936572 | 0.00427485 |
| fixed_long | 131.344 | 0.70215 | 0.702149 | 0.699711 | 0.0319373 |
| multi_long | 85.1048 | 0.118496 | 0.118496 | 0.104794 | 0.00478319 |
| with_reaction_healthy | 85.1048 | 0.118559 | 0.118559 | 0.104882 | 0.00412973 |

整体SNR约85–131，拟合后family差异仍显著小于white noise。特别是multi60+healthy/force的全部参数满实用秩5/5，却只剩约.105的模型差异；状态信息充分与结构区分度不足可同时成立。

S1 multi60的m profile中m=1.15/1.375的数据代价255.015/255.558，仅差.543，N220误差却为.00140/.00589；c1=0/1.5的数据代价254.976/256.337，N220误差.00159/.00781。所有profile包括较差端点都保留；这些未来评分只说明外推歧义，没有用来选择m/c1或调整规则。


本轮还用真实 FEM 在合法 box 内做 bounded profile：固定 m 或 c1，TRF 重优化其他参数；初始状态做 E0–d0 非线性等价 ridge 与有健康参照的 E0 profile。保存全部 grid、未收敛、触边、FD检查与未来轨迹。它们是条件 data-cost 曲线，不是经过校准的95%区间，也不证明已找到所有全局模式。固定 correction 系数的正则常数不影响条件最优解；表中比较使用 data_cost，不把省略的常数或 box 当成 DIC 信息。

## G. Noise / model-difference sensitivity and expression limits

| 固定估计点的观测条件 | DIC标量 | σux mm | Am s_min | K2 s_min | Am切空间后差异 |
|---|---|---|---|---|---|
| primary independent | 480 | 5e-5 | 3.42042 | 3.68088 | 0.104794 |
| dense_history | 1120 | 5e-05 | 7.36602 | 8.62567 | 0.280553 |
| spatial_noise | 480 | 5e-05 | 2.18864 | 2.3553 | 0.0670553 |
| high_noise | 480 | 0.00025 | 0.684084 | 0.736175 | 0.0209589 |

这些新增噪声/dense协议不重新标定或执行选择器；避免把固定点信息评价称作noise Monte Carlo模型识别率。空间相关性使min singular约2.19/2.36，5倍σ使其约.684/.736，均低于3；dense使其约7.37/8.63，但模型差异仍只有.281。不能通过多点带来的参数满秩宣称材料律已确定。

S1在x∈[.001,15]的privileged函数近似log-factor RMSE=.01558，说明它能近似而非精确表达；S2最小倍率exp(−2)=.1353仍不能产生真阈值零。S3/S4/S5为固定x下不同R/D/坐标的函数碰撞证明，本轮未运行这些family的额外疲劳测试世界。


主噪声5e−5 mm保持，未降低到旧2e−7高SNR。独立噪声 uyσ=1.25 uxσ；空间噪声用已有长度.22 mm、分量相关协方差。帧/试件噪声在公共载荷和参数条件下独立；本轮没有新增共享配准噪声世界，其混淆证据仍来自已使用的 Phase2E。表中的整体 SNR 是64个公开评价坐标上的分量响应RMS/名义uxσ，模型差异指标为白化候选差异，二者分母定义不同，不能直接混同。

表达能力是独立限制。令x=Y_a/Y_ref，Y_ref=.025 N/mm²，A0=8e−5 cycle⁻¹、m0=1；x、D、R和c均无量纲，hazard单位cycle⁻¹。当前K2仅使用 t=x/(1+x) 的两列，hazard=A0 x^m0 exp[2 tanh(c0 t+c1 t²)]，c∈[−4,4]。S1 可在声明 x 范围近似；S2 真阈值区间的精确零 hazard 不能由正的 bounded multiplier 产生；等 x 下的 S3额外R、S4额外D、S5空间材料因子不能由 energy-only K2 同时表示。oracle 函数诊断明确标记 privileged，不用于训练、标定、加载选择或选择器。增加测量不能恢复模型族排除的依赖。本轮没有为这些缺陷新增更高K。

## H. Development pilot and frozen-rule comparison

| 世界/协议 | 私有评价family | B0 | B1 | B2 | B5 | B7 | E9 | E9部署 |
|---|---|---|---|---|---|---|---|---|
| 000-fixed_short | matched | 0.000709783 | 0.00131524 | 0.0029909 | 0.00266849 | 0.000709783 | 0.000709783 | B0 |
| 000-multi_long | matched | 0.000255161 | 0.000714563 | 0.00225131 | 0.0016718 | 0.000255161 | 0.000255161 | B0 |
| 001-fixed_short | parameter | 0.0121212 | 0.00754305 | 0.00569153 | 0.00513715 | 0.00754305 | 0.00754305 | B1 |
| 001-multi_long | parameter | 0.013221 | 0.00613972 | 0.000787302 | 0.000756839 | 0.00613972 | 0.00613972 | B1 |
| 002-fixed_long | S1 | 0.0499413 | 0.00838647 | 0.00162748 | 0.00194999 | 0.00838647 | 0.00162748 | B2 |
| 002-fixed_short | S1 | 0.0215858 | 0.00737308 | 0.00361741 | 0.00435163 | 0.00737308 | 0.00737308 | B1 |
| 002-multi_long | S1 | 0.0240288 | 0.00631003 | 0.00189435 | 0.00152102 | 0.00631003 | 0.00631003 | B1 |
| 002-unknown_material_load | S1 | 0.098081 | 0.00985955 | 0.00732779 | 0.0435957 | 0.098081 | 0.098081 | B0 |
| 002-with_reaction | S1 | 0.16643 | 0.0106476 | 0.0021513 | 0.014688 | 0.16643 | 0.16643 | B0 |
| 002-with_reaction_healthy | S1 | 0.0241038 | 0.00569306 | 0.00215136 | 0.00188828 | 0.00569306 | 0.00569306 | B1 |

表为最后horizon的damage RMSE，short为N180，long为N220；按同一行公平比较。当前filtered error、全部horizon、空间holdout位移RMSE与白化prefix residual另见原始CSV，未只报拟合位移。

匹配世界中复杂修正均比B0长期误差更大，冻结选择未激活K2；只有一个独立matched世界，不能推算可靠false-activation率。参数世界multi60的B2/B5点估计接近，不证明K2不可替代。S1 multi60 K2相对Am最后误差降低约.00037，fixed60反而更差；没有稳定优势证据。

未知E/load时，DIC+force虽改善载荷约束，却不能保证damage准确：冻结fallback B0 N220误差分别.0981/.1664；health+force将其降到.0241，仍受错误hazard形式限制。复杂候选在前两组共6份未收敛，局部弱方向保留，不能用其某个低误差点宣称唯一识别；更不能将force-only反而较差的damaged状态忽略。

预冻结cold start d0=.05/.22均重估相同前缀，未以true d0初始化；它们的参数/目标值/未来场全部保留。该对照不消除模型族误差。

![状态信息、模型区分度与外推歧义](../results/phase2f/20261010T010645Z/development_v2/analysis/information_summary.png)


3个世界种子240000/240001/240002，matched、parameter、S1各一个；十个完整拟合协议和三个 information-only 协议都是配对条件，不是13个独立世界。每方法使用相同观测、公开物理信息、原TRF及40次函数评价上限；profile另限30。没有从truth挑初始化，没有训练网络。

B0仍估计必要d0/nuisance，hazard保持先验；B1估计A，B2估计A/m，B5为原K2。B7/E9调用原冻结函数，历史保留最后一帧作验证，选择后full-prefix重估并按原稳定性检查部署。长历史改变了时间协议，属于冻结规则的**Development protocol transfer**，不继承原时间协议的独立资格。ABSTAIN 保留，接受的简单标定可部署；它不保证未来无偏，也不证明真实 law 已识别。

全部候选 forecast 文件在未来评价真值生成前写入；没有生成未来 DIC 测量。profile forecast 仅由已保存 prefix参数构建，全数保存后才读取已有 held-out truth 做评分。静态/信息诊断在公开估计点计算，不用future residual改变载荷或模型。原2D/2E世界没有进入本轮训练或新资格。

## I. Physical / numerical limitations, tests and cost

主Development生成+十协议 workflow wall **1666.32s**，诊断 **92.81s**，后分析 **25.42s**（后两项部分重叠）。主inference累计 **4193415 FEM调用**；unknownE/load两协议各约23min案例wall，表明弱方向反演的实际代价，而非神经forward latency。未知参数维数由3提高到5，不能据此给任意高维场资格。

step2/1相对step.5的damage RMSE分别 **6.78882e−5 / 2.27201e−5**；插入11.3、31.7等额外查询时刻后的最终差为0。未通过查询方式或积分步长改变材料参数。最大真损伤 **.259454**、最大预测损伤 **.291512**，均在本轮真实暴露安全范围内；60份数组重算与Fisher PSD检查通过。候选hard exception为0，不据此省略6份未收敛或13份弱秩结果。时间加密误差在damage空间，模型差异在白化位移空间，不能直接比较二者数值。本轮未对不同步长的模型区分度完整传播，也未做空间mesh refinement或完整model posterior。

| 审计项 | 结果 / 资格 |
|---|---|
| 原有完整回归 / 本轮完整回归 | 489 / **502 passed**，`-W error`；[最终日志](../results/phase2f/20261010T010645Z/pytest_final_verified.txt) |
| 原314文件 + checkpoint + Phase2E证据保护 | 314 / 161 / 49916 SHA一致；241352历史文件size/mtime不变；[复核](../results/phase2f/20261010T010645Z/preservation_verified.json) |
| 初始静态rank / 非线性等价解 / healthy信息 | PASSED（声明的均匀、健康共享E假设） |
| 真FE–damage / fixed-point Fisher / 时步与查询 | PASSED（当前网格和参数） |
| 十协议 + 三information-only / profile | EXECUTED，未收敛/弱秩保留；无候选hard exception |
| 疲劳law可靠区分 / unknownE优化 / 高噪声 | NOT QUALIFIED；不扩大 |
| 结构OOD动态、独立test、概率coverage、跨网格 | NOT EXECUTED，本轮不授资格 |

配置[固定pilot.toml](../configs/phase2f/pilot.toml)，[实际runtime源包](../results/phase2f/20261010T010645Z/development_v2/runtime_source.tar.gz)和224文件manifest保存。估计/生成runtime hash为`d11a931b23754c3ad127b942c68089499e9b359d6a0f750cd8eda713da67d8b4`（排序path:sha行定义，区别于完整源码canonical JSON哈希）；诊断格式修复有单独SHA记录，后分析脚本与最终源码一起归档。当前精确Source恢复见final_source_verified，原始工程失败见development与debug_inference；没有声称整个Development期间从未修复代码。

可复现命令（均用新输出目录；不是正式test入口）：

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/verify_phase2f.py --output <new-development-directory> --workers 4
PYTHONPATH=src python scripts/analyze_phase2f.py --development <new-development-directory>
PYTHONPATH=src python scripts/audit_phase2f.py --development <new-development-directory>
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q -W error
```


原 Tri3、transition、FatigueEvolutionSolver、材料律与 observation 行为未改变；开发前314文件逐字节保留。原回滚、损伤界限/不可逆、truth isolation 测试继续执行。新增测试验证未来请求先于measurement IO拒绝、truth字段拒绝、因果读入、健康与反力不重复、候选历史不串用、查询顺序不影响最终状态、静态实际FE秩及等价解、K2缺失物理特征的函数碰撞。

第一次 Development 接线因重建 MeshData 被原身份 guard 拒绝，十条失败记录保留；修复为拆分同一ObservationBatch后按相同种子、噪声、场景在新目录重跑。诊断合并字段的实现错误在调用前修正，版本差异有单独记录；没有修改物理、场景或选择门槛。第一次新增完整 pytest 的子进程未继承包路径而报 ModuleNotFoundError，保留失败日志；使用显式PYTHONPATH=src重跑，不改测试。不是数据筛选或科学门槛放宽。

PyTorch/CUDA build存在，但驱动不可用；没有更换驱动或环境，没有新训练。CPU4workers，SciPy/Python版本与依赖export保存。FEM调用数包含主候选历史/验证/full fit/Fisher/profile/cold-start/forecast；合成生产、后分析与数值加密另外记录，见[计算组件](../results/phase2f/20261010T010645Z/development_v2/computational_cost_components.json)。兼容base producer的完整FEM次数没有计数 instrumentation，不宣称统计了全流程每一次FE；主workflow wall包含各案例写盘/科研对照，不是部署单一模型时延。没有本轮重跑所有旧科学CLI或训练checkpoint，旧完整证据只读保存与哈希核验。

## J. Independent validation proposal

**不启动正式结构材料律独立测试矩阵。** Development 支持健康/反力对初始状态的研究方向，却没有支持当前K2族的可靠结构区分；先执行本轮信息不足停止门槛，保持3世界规模，不增加K、降噪或复用旧test取得资格。

仅提出以下**有条件的初始状态信息独立验证预注册草案**，不执行、不宣称已经冻结或获得资格。它的主目标是确认健康+反力能否在非理想健康参照条件下消除E0/D/load混淆，不能作为结构疲劳律验证：

| 项目 | 待审阅方案 |
|---|---|
| 划分 / 独立单位 | classical方法无需训练集；validation12、measurement-calibration12、独立test100个新物理世界；每世界五共材料加载试件为配对数据。预留种子250000+ / 260000+ / 270000+，尚未生成数据 |
| 几何 / 材料 / law | 当前2×1 mm、厚度.8 mm、四Tri3、ν=.3；E1800–2200 N/mm²、d0 .08–.16，公开载荷误差box±15%；matched/parameter/S1标签仅生成/评价，N0静态主目标不声称区分这些law；任意几何/空间材料不授资格 |
| 噪声与校准 | σux=5e−5/2.5e−4 mm × 独立/空间相关，四层各25个test世界；另在validation检查健康材料偏差及载荷传感器系统误差。covariance只能由calibration确定，test前冻结；不继承Phase2B未校准协方差 |
| 观测 / 暴露 | N0每世界160 DIC标量；反力额外5标量/10成本；健康额外160 DIC标量，总330成本；静态加载，不推进疲劳循环。extra信息成本明确，另需重新预注册等成本设计才可宣称预算优越 |
| 方法 | 原static TRF，4种信息条件：DIC、+force、+healthy、+both；相同bounds和优化上限。不调整B7/E9，未训练policy |
| 主要门槛 | 在未知E/载荷主组，+both相对DIC的damage RMSE绝对平均下降≥.005，world-paired bootstrap95%下界>0；load_factor绝对误差平均下降≥.01；相对下降≥25%为次要描述，baseline近零不用比值 |
| 负对照 / 失败 | 已知E/载荷的matched负对照使用绝对excess平均≤.002、max≤.01；未收敛、触边、FD失败、ABSTAIN不删除，失败按无收益计入分母。记录可辨识状态而不强制输出唯一damage |
| 统计精度 | n100总体Wilson最坏半宽约.096，各层n25约.183；五试件、帧及noise seed不增加n。主连续误差按世界配对bootstrap10000次，固定seed281001；分层结果不因总体平均成功而自动合格 |
| 计算上限 | 预计≤200000实际FEM调用、≤3 CPU core-hours，达到上限保留未完成/失败并停止；提交前用validation dry-run核定预算。没有本轮消耗该正式预算 |

正式结构材料律方案需先在另一个Development研究中满足：测量协方差有独立标定；健康/force把当前状态与载荷歧义限制在可接受范围；对拟合后复合模型在合法box内做非线性profile仍有剩余预测差异；预先声明的安全驱动/观测跨度能提供≥3的条件差异筛查值且多组估计点稳定。该值仅工程筛查，不是统计资格门槛，最终仍需独立physical-world验证。若这些条件不能实现，结构实验继续NO-GO。当前没有给出凭3个Development世界就可通过的正式结构分类矩阵。


## K. RL Go/No-Go

RL **NO-GO**。也不扩展高维 damage action、graph-conditioned correction 或更多不可辨识系数。本轮 Development 成功的状态信息诊断不等于独立结构泛化资格；材料律弱区分度和函数表达限制不能由policy复杂度补救。

Phase2F仅评价基线预检和小规模信息研究；不证明真实材料校准、裂纹路径/局部软化网格客观性、神经/物理posterior校准或所有结构失配可恢复。停止于本阶段，没有自动进行正式测试矩阵、Git提交或RL。
