# Scientific baseline and initial-state validation readiness

结论：**VALIDATION_READINESS_PARTIALLY_QUALIFIED**。

Phase 2E/2F 源码可完整恢复，两个逻辑提交版本已准备供审阅。初始损伤的小规模 Development 与计算预检完成，但独立健康试件的模量迁移误差、两试件加载差异和传感器系统偏差尚不能由当前三参数 likelihood 完整表示。因此，**不启动正式 100 世界验证**。没有执行 Git add/commit/tag/push/merge/reset/clean，没有开展新的科研神经训练或 RL，没有更改历史资格、物理模型或旧结果。回归中的既有小型神经测试按原样运行。

本轮证据目录：[baseline_validation_readiness/20261010T021219Z](../results/baseline_validation_readiness/20261010T021219Z/)。所有资格仅涉及源码与初始状态验证准备，不涉及疲劳结构损伤律的新资格。Phase 2F 保持 `PHASE2F_PILOT_PARTIALLY_QUALIFIED`，RL 保持 **NO-GO**。

## A. Git freeze readiness

branch 为 `feature/phase0-architecture`；HEAD 为 `dc239d1b7b6dd1e44d93e71bdcccccc0f12e8bc9`，仍是 initial commit，仅跟踪 `.gitignore`、LICENSE 和 README。HEAD 不能代表完整科研实现。开发前当前 **327 文件全部与 Phase 2F final manifest 逐文件 SHA256 相同**；本轮结束仍保留这些文件原字节，新代码与本报告是后续准备增量。

已读取 [Phase 2E readiness](PHASE2E_GIT_FREEZE_READINESS.md)、[Phase 2F report](PHASE2F_INFORMATION_SUFFICIENCY_REPORT.md) 及两阶段 final manifest/RECOVERY。两个可审阅版本如下；文件清单和预期改动见 [freeze_review/expected_staged_diff.json](../results/baseline_validation_readiness/20261010T021219Z/freeze_review/expected_staged_diff.json)。这些是源码版本拆分，不是声称存在的历史提交。

| 逻辑版本 | 精确集合 | 相对前一版本 | 推荐 commit message / annotated tag |
|---|---:|---|---|
| Phase 0–2E 完整科学基线 | 314 | 相对 initial HEAD：311 A、README 1 M、2 文件不变 | `chore: freeze verified Phase 0–2E scientific source baseline` / `scientific-baseline-phase2e-partially-qualified` |
| Phase 2F Development 增量 | 327 | 13 A；原 314 文件没有修改或删除 | `research: add development-only Phase 2F information sufficiency pilot` / `scientific-baseline-phase2f-pilot-partially-qualified` |

canonical source SHA256 定义为排序的 path→文件 SHA256 映射的 canonical JSON 哈希，使用逗号/冒号分隔符。它不同于 tar、manifest 文件或 Git commit 的哈希：

| 版本 | canonical source SHA256 |
|---|---|
| Phase 2E / 314 | `0f98b2593cf9f73c6b03eef462f736aeaab52cf9720f3cd986fca9224e9a2ae5` |
| Phase 2F / 327 | `9083c806cbd0c141d45c16e88a462ce5f7ee16aefdba1f24d9b74fb60da011c2` |

[Phase 2E 相对 initial 的完整 patch](../results/baseline_validation_readiness/20261010T021219Z/freeze_review/phase2e_against_initial.patch)、[Phase 2F 的 13 文件 patch](../results/baseline_validation_readiness/20261010T021219Z/freeze_review/phase2f_increment.patch) 已生成。实际在另一空目录恢复 Phase 2E 包，再应用这个新生成的 2F 增量，327 文件全数核验通过。未来提交应使用**实际授权后的提交时间**，注明 consolidated source import，不伪造此前逐阶段开发历史。两个 tag 的注释应关联各自 source SHA、原 run ID、原 PARTIALLY_QUALIFIED 和 NO-GO；不得把后续 ignore/CI 修订后的 payload 当作原 314/327 哈希。

| 类别 | Phase 2E | Phase 2F | 普通 Git 策略 |
|---|---:|---:|---|
| src | 127 | 131 | 全部基础、数值、反演、可选学习、修正及实验设计代码 |
| scripts | 56 | 60 | 科学验证、数据生成、训练/评估和恢复脚本 |
| tests | 49 | 51 | 原单元、集成、科学测试完整保存 |
| configs | 47 | 48 | 冻结种子、场景、噪声、加载、门槛 |
| docs | 30 | 32 | 所有正式报告，包括失败和限定资格 |
| 根文件 / CI | 5 | 5 | `.gitignore`、LICENSE、README、pyproject、tests.yml |

源文件最大 50,429 bytes。两个提交的精确 allowlist 均没有 results、checkpoint、archive 或大二进制。**当前 `.gitignore` 尚未排除 results，因此不能采用无差别 staging。** 尚未实施的 [gitignore.patch](../results/baseline_validation_readiness/20261010T021219Z/freeze_review/admin_proposal/gitignore.patch) 排除 results、checkpoint/archive 目录及 tar/zip/pt/pth/ckpt，不全局排除 json/csv/npz，以免丢弃小型 fixture 和 manifest。

CI 现有数值任务安装 `.[test,verification]`，有 Python 3.11/3.13 矩阵和 Phase 0–2A 科学 CLI；神经测试使用 importorskip，故这个环境不能证明完整神经回归已执行。准备了独立 [CPU neural CI patch](../results/baseline_validation_readiness/20261010T021219Z/freeze_review/admin_proposal/optional_neural_ci.patch)，安装可选 neural extra 并运行完整 pytest。两个配置 patch 在独立恢复目录 dry-run 成功，但未修改工作区原配置，也未运行远程 CI。当前环境验证了 Python 3.13；本轮没有声称独立验证了 Python 3.11 或全部历史长时 CLI。

建议在以上两个逻辑科学版本之后，单独审阅管理配置与本轮准备文件，commit `chore: protect scientific artifacts and document state validation readiness`；若需要 tag，采用 `validation-readiness-partially-qualified`。当前准备快照的 source SHA 见 [final_source_verified_v2/audit.json](../results/baseline_validation_readiness/20261010T021219Z/final_source_verified_v2/audit.json)；如果未来实际应用 ignore/CI patch 或新增 evidence index，必须重新计算最终 payload SHA，再写入 tag 注释，不能沿用此快照哈希。该第三增量不混入历史两个 source SHA。本轮仅准备审阅材料，没有生成任何 commit/tag 对象。

## B. Source and artifact integrity

两阶段均实际执行 **final archive 恢复**及 **predevelopment archive + 原 incremental.patch 恢复**，各自恢复目录独立，全路径集合和每个 SHA 全数相符。结果在 [baseline/audit.json](../results/baseline_validation_readiness/20261010T021219Z/baseline/audit.json)，另有 E→F 逻辑版本恢复日志。

| 阶段 | final tar SHA256 | final manifest 文件 SHA256 |
|---|---|---|
| 2E | `e5a6e140c6b3e6ae515b5ec03ceaf1b3d982eb2d2b29757ad34413571690214e` | `9a8de2025b36c692ef539da965feb172d5a6953b71711caadf6a7385ab599d54` |
| 2F | `8211b2ed2c259f8f2cda95a647040b0458e8da6092357dd1841dd46d3b7dbadc` | `939ef60c4c7d5e346cc27dbdff7a04ff337639eef1065e526a4c26efb8f36b06` |

Phase 2E RECOVERY 的 `../baseline` 正确指向 **同一 Phase 2E run**；记录中的 `/tmp/fatigue-phase2c-restoration-*` 是误导性的旧临时前缀，并未引用 Phase 2C 归档。已以本轮新的独立恢复证明；旧说明保持原文。Phase 2F RECOVERY 正确指向自身 run 的 baseline，不依赖过去临时目录仍然存在。

Phase 2E 的 **49,916** 项和 Phase 2F 的 **1,677** 项 delivery-ledger 资产全部实际核验 SHA。历史结果库存有 243,030 文件、1,803,682,012 bytes，另记录 161 checkpoint SHA 和 27 个 ≥5 MiB 文件，[asset_inventory.json](../results/baseline_validation_readiness/20261010T021219Z/freeze_review/asset_inventory.json)。这不是其他所有阶段数值文件的全内容证明；全内容验证范围明确为两阶段 ledger 和 checkpoint。没有移动、删除或覆盖这些资产。

源包不递归包含 results、checkpoint、旧 tar、缓存或 Git。Phase 2F 独立恢复的完整源码还在临时目录成功构建 wheel，并从 `/tmp` 直接导入该 wheel 的数值和物理预测路径；没有导入 torch，没有安装或替换工作环境。[wheel 验证](../results/baseline_validation_readiness/20261010T021219Z/baseline/wheel_validation.json)。

普通 Git 应保存精简 evidence index，关联阶段/run、source/config SHA、原资格、恢复包和完整资产 ledger SHA；[审阅用 index](../results/baseline_validation_readiness/20261010T021219Z/freeze_review/admin_proposal/SCIENTIFIC_SOURCE_EVIDENCE.json) 已准备。大派生资产保留在外部不可变科研存储；确需分发的少量模型可单独 Git LFS，不把整个 results 放入 LFS。本轮**没有确认外部备份已经落地**，本机路径不能视为长期异地备份。迁移需逐文件复核，并保留失败资产。当前完整新源码的 SHA 与恢复记录位于外部 final manifest/audit，避免报告把自己的哈希写入自身而产生循环。

所有 Phase 2A–2F PARTIALLY_QUALIFIED 保留，包括 2A 原 89/100 coverage、2B 校准/OOD 限制、2C 高 SNR 局限、2D 6/8 未达门槛、2E Fisher 1/8 与 discrimination 0/8、2F 信息不足及 RL NO-GO。准备提交不是提升历史科学资格。

## C. Initial damage observability formulation

范围仅为二维均匀标量损伤、Tri3 平面应力、已知 ν=.3 与厚度 .8 mm、相同已知 2×1 mm 几何、零位移支承和载荷控制。实际 E 未知；待估计三参数为 `[d0,E0,load_factor]`，沿用 Phase 1C 无量纲坐标、原 TRF、covariance whitening、two-scale FD 和 Fisher/SVD。不识别疲劳 A/m、K2 或演化律，也不计算 RUL。

令参考 FEM 位移模板为 v_j，健康/损伤试件模量分别 E_h/E_d、实际载荷倍率分别 f_h/f_d：

\[
u_{d,j}=v_j\frac{f_d E_{ref}}{E_d(1-d_0)},\qquad
u_{h,j}=v_j\frac{f_h E_{ref}}{E_h},\qquad
R_{d,j}=-F_{nom,j}f_d.
\]

反力在这些载荷控制边界下主要测量实际载荷，不直接区分 E 与 D。增加同形加载幅值不自动增加等效刚度参数的秩。三参数、相同 E/相同实际倍率这一**条件模型**中：

| 信息 | data rank / 3 | 可辨识内容 |
|---|---:|---|
| DIC only | 1 | `f/[E(1-D)]`；D 输出 ABSTAIN |
| DIC + reaction | 2 | f 与 Eeff=E(1-D)；D 仍 ABSTAIN |
| DIC + healthy | 2 | D 与 f/E；E/f 仍有零空间 |
| DIC + reaction + healthy | 3 | 条件模型下三个参数；不是非理想材料迁移的证明 |

例如给定 Eeff，合法 box 内不同 `E,D=1-Eeff/E` 的 DIC 和反力完全相同。普通逆矩阵不能消除这种等价性。数据 Fisher 与先验约束必须分别报告；未校准神经 covariance 未被使用。

## D. Nonideal healthy-reference assumptions

健康 D=0 的条件由独立外部已知状态给定；loader 在打开健康 DIC 前检查 certificate，拒绝从损伤观测推断的健康声明。这项 certificate **不证明 E_h=E_d**。

设 `ρ_E=E_h/E_d−1`、`g=f_h/f_d`，相同几何下的位移比例满足：

\[
q=u_h/u_d=g\frac{1-D}{1+\rho_E},\qquad
D=1-q\frac{1+\rho_E}{g}.
\]

固定 ρ=0、g=1 的模型得到 `Dhat=1−q`。在 Dtrue=.12 的无噪声机制诊断中，健康 E 差异 +1%/−1% 导致约 +.00871/−.00889 的 damage 偏差，+3%/−3% 导致 +.02563/−.02722。增加观测点不能去除这一系统误差。

允许未知 ρ 后，信息矩阵从 rank3/3 变为 rank3/4，损伤灵敏度投影到 E/f/ρ 的正交空间后约 `4.7e−14`。实际原 Tri3 计算了 D=.10/.12/.14 与 E=2004.44/2050/2097.67、对应 ρ=+.022727/0/−.022727 的等价解：健康和损伤位移最大差 ≤6.6e−19 mm、反力差 ≤4.5e−16 N。它们均位于预声明 ±3% 迁移范围。[health_transfer](../results/baseline_validation_readiness/20261010T021219Z/development_verified/summary.json) 保存了弱方向与真实 FE 对照。

独立健康信息可以显著缩小损伤可能区间，但未知迁移下不保证唯一真损伤。以 q=.88、g=1、ρ∈[−.03,.03] 为例，即使零测量噪声也有 D∈[.0936,.1464]。若用外部校准先验限制 ρ，必须标明先验贡献和迁移假设，不能称为 DIC 新增的数据秩。

未知独立健康载荷倍率也使 rank3/4 并恢复 damage 零方向。为健康和损伤两试件分别设计反力测量，在假定 E 可迁移、sensor bias 已独立校准的条件下可解除加载比退化；本轮前瞻 J 的 rank 为4/4。最小奇异值仅约1.22，说明局部满秩仍不意味着高精度。这是设计灵敏度分析，不是已经采集健康反力或完成鲁棒四参数推断。

## E. Reaction and DIC uncertainty

主试验 DIC σux 为 5e−5 或 2.5e−4 mm；独立噪声 σuy=1.25σux。空间噪声沿用原 .22 mm 坐标核、二维分量协方差。首次纯平滑核在聚集 Q64 点上出现数值奇异，Cholesky 显式拒绝，[原 Development log](../results/baseline_validation_readiness/20261010T021219Z/development.log) 和配置/代码/部分结果保留。没有静默加 variance floor。

后续诊断明确冻结 `Σ=.9Σ_spatial+.1diag(Σ_spatial)`，作为相关误差与独立测量误差的合成混合模型；生成和 likelihood 使用同一个 Σ。这个修改针对协方差合法性，没有调整恢复误差门槛，也不是原纯平滑核结果通过。两版噪声模型都未经过真实实验标定。

每次 damaged 诊断加载采集一个独立全局反力，随机 std 为 nominal force 的1%；同一世界的 sensor 乘性 bias 为0、±1%或±3%。随机误差与共享 bias 分开定义；拟合 likelihood 暂未建模这个共享 bias，所以相应误差属于 model misspecification。不能把系统 bias 当成每帧独立证据平均掉，也不能把它计入 covariance 后又重复用 nuisance prior。

若 bias 在健康/损伤加载和材料迁移中被错误假定为零，原模型可能得到收敛且很小 residual，却错误估计 E 和实际载荷。已知共同实际载荷且 E_h=E_d 时，位移比可以抵消公共倍率：本轮 healthy 两组 d0 几乎相同，反力主要改善 E/f，而非额外提高 D。跨试件加载比未知时则需要两侧独立力信息及共同/差异 sensor bias 的校准。现有 scalar ReactionObservation 不能直接代表这些完整联合系统误差；正式方案仍需补齐此 likelihood。

本轮未计算或宣称独立 posterior coverage。已知 synthetic covariance 只是生成契约，不替代 calibration specimens。空间采样密度、重复测量和健康共享误差均不能被当作新增独立物理世界。

## F. Proposed independent validation protocol

以下是**需再次审阅与预注册的条件方案**，不是已启动或已取得独立资格的矩阵。旧 2D/2E/2F 世界和本轮六世界均不能用于其最终确认；不调整冻结 B7/E9，本轮也没有调用选择器。

| 项目 | 建议及未解决条件 |
|---|---|
| 独立单位 | 一个材料世界包含独立 healthy/damaged 两试件及配对加载；3 次加载、4 信息臂、重复噪声不是4/12个世界 |
| 划分 | classical 推断无需训练；validation12、独立 measurement/material-transfer calibration24、test100；新种子与世界在审批/预注册后冻结，本轮未生成 |
| 主分层 | 2 个噪声 std × iid/spatial 四层各25个 test 世界；主 Q64、覆盖良好的预定义不规则点；E、D、载荷均未知；ν/几何/厚度固定 |
| 非理想迁移 | 同时保留 ρ_E 的小随机差异及 ±1%/±3% stress controls，公共及跨试件 loading/sensor bias；外部健康 D0 certificate，不能用 damaged DIC 建立这个 certificate |
| 外部测量 | 健康材料迁移需要独立批次/试件模量信息，sensor bias 需要独立可溯源载荷标准；两试件均记录反力；这些成本单列，不通过重复使用同一 DIC 当作 prior 获得信息 |
| 噪声校准 | 参数化空间核、分量 std/correlation、独立部分、跨帧误差；可在24校准世界内做固定8次测量重复，重复数用于噪声估计而非物理 n；不用少量试件拟合任意128×128 dense covariance |
| 方法 | 四组相同 bounds、TRF初值、停止/重试预算；保存失败/未收敛/边界/FD信息。DIC及DIC+force报告 identifiable stiffness 和 damage ABSTAIN；如需操作性点值，外部 prior/fallback 在 test 前冻结并明确其先验依赖 |
| 观测预算 | 每世界3静态 load ×Q64×2分量：damaged384标量，healthy额外384，damaged reaction3、两试件均测时共6；四组有不同额外测量成本，不能声称等预算优势 |
| Q/布局对照 | 最多在预选20个 test 世界附加 Q16 覆盖/clustered 两种配对协议；共40额外协议，不增加独立 n；纳入预算，不能按误差选世界 |
| 主要评价 | damage identified-set/profile 宽度、coverage与ABSTAIN、Eeff/load误差；damage RMSE只在预定义输出策略上评价，记录 prior-derived/observable 类型，不能用任意秩亏TRF点制造改善；两健康组单独比较 reaction 的增益 |
| 收益与失败 | 可继承原草案绝对 damage RMSE 改善 .005、世界配对95%下界>0作为待审阅门槛；不能对ABSTAIN baseline伪造唯一damage误差。初始-state negative control 使用绝对 excess容限，不用近零误差的相对比；失败、不收敛、ABSTAIN不从分母删除 |
| 统计 | 总体 n100 二项概率 Wilson 最坏半宽约.096，n25约.183；不是高精度分层资格。连续误差按世界 paired bootstrap，calibration和validation不用作test；预先冻结多重对照与主指标 |
| 时间与物理 | 全部 N0静态诊断、零疲劳暴露；不识别演化律、不读取未来DIC；同一候选重建物理请求，原求解器负责场与支承 |

若实际目标是 damage 绝对误差 .01，可先把外部 transfer/loading 系统误差预算和随机误差预算各分配 .005；这需要独立约束大约 `|ρ_E|+|g−1|≤.005`，而非当前 ±3% 未知迁移。以 damage 无量纲尺度 .14 为例，局部95%随机半宽 .005 需要条件信息范数约≥55。本轮高噪声规则/随机 Q64 的范数约38，尚不支持直接给该层精度保证；该值只是从目标推导的 Development 设计筛查，还需非线性 profile 与独立校准确认。不能临时降低 test 噪声或删除 stress controls。

进入正式test之前，至少需要：独立外部条件可信；ρ、独立载荷比及共享 sensor bias 的观测契约和 likelihood 完成；测量 covariance 在 calibration 冻结；上述输出/ABSTAIN与profile算法完成小规模原求解器对照；预算覆盖实际新增 nuisance；明确100世界的统计精度是否够用；用户另行批准。当前未满足这些条件。

## G. FEM computation cost estimation

原 Phase 2F 主拟合 4,193,415 次 FEM 含多步疲劳历史、多个模型、FD/profile/forecast；不是100世界 N0静态反演的直接单位成本。本轮实际计数，不把 optimizer nfev 当作FEM调用数。

六个独立 Development 世界，每世界4配对信息臂、同一观测分别用原/缩放 adapter。只在两个世界执行全部原 adapter 作精度和成本对照，其余世界使用诊断性缩放；这不是100世界结果。3个静态traction scales .35/.7/1.4，位于原Phase2F加载范围内；Q16/64、random/regular/clustered、E1800–2200、d0 .08–.16、actual倍率 .95–1.08。初值固定无量纲 `[0,0,0]` 和 `[-.55,.4,-.35]`，max_nfev60；仅当全部未收敛时按最低观测cost重试，cap100，不看真值。

| 项目 | 实测 |
|---|---|
| 单次原Tri3 / 4 elements | median .850 ms，p95 1.033 ms，20次 |
| 16 / 64 elements | median 1.487 / 3.803 ms；不是任意网格成本保证 |
| 原四组点估计 | 两世界各2235/2088 FE、2.240/2.091秒；合计4323 FE |
| cold Jacobian+Fisher /6 load cases | 78 FE、.077秒；原two-scale差分 |
| warm显式最终J | 0–78 FE，某些已被原optimizer sensitivity检查放入合法64-entry cache，仍实际做矩阵/SVD工作 |
| 单次满秩组不同初值 | 408 FE、8 nfev、56 residual calls；第二初值相同量级 |
| 再重复第一初值 | 仍408 FE；工作集超出有限cache，不能宣称重复优化全部免费 |
| 原optimizer cap1 / retry | 未收敛114 FE/.111秒；重试324 FE/.317秒、收敛；两次记录均保留 |
| 实际物理 D=.4、逆box≤.3诊断 | 21 FE，optimizer收敛于D=.3边界，cost380.806；**不是成功恢复** |
| 同一预测请求10次 /改变候选 | 原cache首次6 FE、重复0、改候选6 FE；合法缓存，不跨候选复用损伤历史 |
| bounded profile成本 | 一世界、两健康信息臂、各5个固定D值、2初值：原5172 FE/5.116秒；精确缩放12参考FE，profile优化约.330秒 |

计数包含 optimizer 的三点差分和独立 sensitivity检查；explicit最终J另外计数，避免重复相加。原FD检查refinements与failed候选会改变每次 nfev对应调用数。所有试验是CPU、单线程BLAS；CUDA build13.0但 `torch.cuda.is_available=False`，nvidia-smi exit9。未修改驱动/依赖。Python3.13.12、NumPy2.4.4、SciPy1.17.1及完整pip export见环境证据。

[预算投影](../results/baseline_validation_readiness/20261010T021219Z/computation_budget_projection.json) 明确区别已经执行与假设：

| 100世界方案（未运行） | FEM投影 | 当前CPU单核算法时间投影 |
|---|---:|---:|
| 四组原点估计+合成生成，无profile | 216,750 | 点拟合约217秒，另加生成/IO/校准 |
| 加两健康组各21-point原profile | 2,388,990 | 约2365秒 /39.4分钟，另加非理想模型与IO |
| 原四组点估计+精确缩放profile | 217,950 | 约355秒 /5.9分钟，另加独立加速审计/校准 |

profile时间由一个世界五点外推，不是21点实测；完整原点拟合仅两个世界，不能据此提供时延置信区间。扩大到 `[D,E,f,ρ,g,sensor_bias]` 会提高差分维数、失败与重试成本；本轮没有验证该完整模型的预算。原≤200000 FE草案已不能覆盖原四组点拟合，更不含profile。

拟议12validation+24calibration+100test，加40个Q/layout附加协议，共176个拟合协议当量但136独立世界。当前三参数原点估计+缩放profile的 nominal约383,592 FE；加2倍余量和四套原profile/场审计，预留约790,896 FE、15–30 core-minutes。这个数**尚不能作为扩展nuisance正式执行预算**，也不是已经获批的资源额度。用多个核只能减少墙钟，不降低物理FEM总预算。

最终小试验约3.18 MB、175文件，主summary序列化/写盘约.022秒；总流程约12.1秒包含原/缩放点拟合、场审计、profile和IO。100主世界粗估50–150 MB，含附加对照/历史约100–300 MB，重复校准观测另计。保存稠密 covariance、完整 TRF历史和重复summary会增加存储；建议以共享不可变covariance资产引用去重，不删失败。来源为 [Development summary](../results/baseline_validation_readiness/20261010T021219Z/development_verified/summary.json) 与 [cost_probe](../results/baseline_validation_readiness/20261010T021219Z/cost_probe.json)。

## H. Numerical acceleration and verification

新增可选 [uniform_static_scaling.py](../src/fatigue_dic_rl/inference/uniform_static_scaling.py) 仅在上述均匀线性静态条件下用原Tri3参考解缩放；不修改 solver，不创建假的 SolverResult，不缓存疲劳历史。非零prescribed displacement、Gaussian decoder、其他本构、非均匀材料、registration、非N0请求被拒绝。它直接保持 `K=E(1−D)Kref/Eref` 的精确比例；generic sparse结构/因子化缓存未实现，不声称适用于非均匀D、非线性或fatigue rollout。

位移与strain乘 `fEref/[E(1−D)]`，stress/reaction/applied force乘 f，elastic energy乘两者之积；已用原Tri3核验完整场。任意候选重新解码E、D、f，不使用另一个候选的旧state。

9个预定内部候选、6个load cases的场/白化 residual/J核验：混合物理场最大绝对差2.31e−14（各场按自身单位比较）、white residual最大差4.79e−13、解析J相对原two-scale FD最大差2.10e−8。预声明 field rtol/atol1e−10、J相对2e−5均通过。TRF仍用原three-point有限差分，解析J只是独立对照，未替换成未经验证的优化gradient。

两世界八组 original/scaled对照的目标值与位移满足声明容差；满秩 +both组三个normalized参数满足2e−5对照，healthy-only组 d0也一致。**但 DIC+reaction 的秩亏原始 d0分别相差 .01277/.00829，healthy-only E/f也可选择不同等价代表。** 因此完整参数不变这一要求在秩亏组 **NOT QUALIFIED**，不能用缩放组更好的偶然真值误差宣称加速改善精度。报告保留两个原始解；初步正式方案继续使用原四组点拟合。缩放可用于已验证的满秩组或只输出profile cost/物理场的计算；后者10点目标值相对差最大2.41e−14，但不把profile中的不可辨识nuisance代表当作唯一参数估计。

若未来统一使用缩放点估计，应先固定商空间/等价解输出规则，并证明合法有界profile及先验模型等价；不能事后按damage真值挑选ridge成员。健康/载荷独立nuisance、非均匀ν/厚度或新geometry需要重新进行原solver验证，不继承本轮窄范围结果。

## I. Scientific risks and executed checks

Development 六世界 +both d0估计及评价残差如下；原真值只在生成和拟合后评价读入，未选择初始化、模型或结果。重复运行同一六种子不增加 n。σ/采样/迁移等因素在小矩阵内共同变化，不能凭该表估计任何单因素因果效应或统计泛化。

| world | σ / Q /布局 | ρE /sensor bias | Dtrue | +both估计 | error | damage条件信息 |
|---|---|---|---:|---:|---:|---:|
| 00 | 5e−5 /16 random iid | 0 /0 | .15863 | .16052 | +.00189 | 76.8 |
| 01 | 5e−5 /16 regular iid | +1% /0 | .14771 | .15520 | +.00749 | 98.5 |
| 02 | 2.5e−4 /64 random iid | −1% /+1% | .13966 | .13046 | −.00920 | 37.7 |
| 03 | 5e−5 /64 clustered spatial | +3% /−1% | .08646 | .08463 | −.00183 | 8.48 |
| 04 | 2.5e−4 /16 clustered spatial | −3% /+3% | .13389 | .05912 | −.07477 | 1.16 /ABSTAIN |
| 05 | 2.5e−4 /64 regular iid | 0 /−3% | .08993 | .08868 | −.00125 | 38.8 |

所有四组的两个初值原始优化history、J/F、status均保留；修订后24信息臂的选定结果收敛、未触边、未发生主case硬FE失败。world04满数值秩仍弱信息且误差大，不能删除。world03的measurement noise抵消部分系统偏差，也不能据此认为3%迁移无害。专门cap1未收敛和box-boundary诊断与首次协方差拒绝另行保留。

| 验收 | 结果 |
|---|---|
| 原 Phase0–2F 回归 | **502 passed，366.02秒** |
| 新窄范围科学/接口测试 | **12 passed**；含4信息rank、场/J、禁止非零BC/非uniform decoder、健康迁移null、truth/辅助IO隔离、外部certificate、formal100入口拒绝 |
| 最终全量回归 | **514 passed，368.30秒**，`-W error`，不改旧tests |
| Phase2E/2F两个恢复路径及E→F逻辑恢复 | 全文件与SHA通过 |
| 2F源码构建wheel/数值路径无torch | 通过；工作环境未安装wheel |
| healthy D0外部条件、unknown E-D退化、实际bounded等价FE | 通过 |
| 六Development世界和成本/失败/重试 | 已执行；不是独立test资格 |
| 扩展ρ/g/bias probabilistic inference与coverage | **NOT QUALIFIED / NOT EXECUTED** |
| 跨几何/ν/空间损伤/疲劳律识别 | 本轮 **NOT EXECUTED** |
| 远程完整CI /旧阶段全部长时科学CLI重跑 | **NOT EXECUTED**，保留旧证据和源码SHA |

复现只会创建新的Development输出目录：

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -W error scripts/verify_validation_readiness.py --output <new-development-directory>
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -W error scripts/probe_static_identification_cost.py --observations <new-development-directory>/observations/development-00 --output <new-cost-json>
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q -W error
```

Development CLI拒绝超过6世界及非development scope。原纯平滑失败、混合协方差修订、诊断与source归档均保存；后续重复冻结种子仅验证可复现，不把重复seed当新统计样本。原Tri3、DamageTransition、FatigueEvolutionSolver、本构、old observation接口与327基线完全不变。没有使用Phase2B covariance，没有新未来观测、K2规则或policy。

## J. Recommended Go / No-Go

**源码冻结审阅 GO**：314基线、13增量、归档恢复、精确SHA、构建和回归均具备；实际提交需用户另行授权。普通Git只按精确源码allowlist，ignore/可选neural CI/evidence index需单独审阅；外部资产备份和CI跨环境结果仍需确认。

**初始损伤研究条件 GO，正式100世界暂 NO-GO**：健康信息方向有明确数据与解析支持；反力解除载荷混淆，却不单独解除材料模量–损伤混淆。健康D0外部已知仍不足以确认跨试件模量/载荷迁移。先完成独立迁移及sensor calibration、联合nuisance likelihood、弱方向/ABSTAIN协议和完整成本小试，再审阅正式预注册与预算。不能仅因三参数Fisher满秩、TRF收敛或Development低RMSE立即启动100世界。

**疲劳结构修正、神经高维damage与RL保持 NO-GO**；本轮没有建立新的必要性证据。完成后停止，没有自动提交Git、启动独立正式验证或进入下一科研阶段。
