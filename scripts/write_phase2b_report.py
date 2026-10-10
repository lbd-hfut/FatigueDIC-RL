"""Render the final report from actual frozen verification assets."""
from pathlib import Path
import json,numpy as np
root=Path('results/phase2b/20261008T114713Z')
def read(name):return json.loads((root/name).read_text())
def table(methods):
    lines=['| 方法/seed | d0 RMSE | A RMSE | Current D RMSE | D40 RMSE | Online s | FEM solves |','|---|---:|---:|---:|---:|---:|---:|']
    for k,v in methods.items():
        p=v.get('parameters',{}).get('rmse');f=v.get('forecast_damage_rmse_mean');fmt=lambda x:'N/A' if x is None else f'{x:.6g}'
        lines.append('| '+k+' | '+' | '.join(fmt(x) for x in [p[0] if p else None,p[1] if p else None,v.get('current_damage_rmse_mean'),f[1] if f else None,v.get('online_seconds_mean'),v.get('fem_solves_mean')])+' |')
    return '\n'.join(lines)
if __name__=='__main__':
    test=read('test_final/summary.json');gauss=read('gaussian_test_final/summary.json');ood=read('generalization_final/summary.json');noise=read('fixed_noise_final/summary.json');science=read('scientific_final/verification.json');cal=read('calibration_final/calibration.json');pyt=read('pytest_final_execution.json');reg=read('historic_scientific_regression/matrix.json');audit=read('baseline/audit.json');selected=read('selected/checkpoint_selection.json')
    parts=[f'''# Phase 2B Graph + GRU sequential damage estimator

日期 2026-10-08；资格：**{science['qualification']}**。
本阶段为 synthetic-supervised neural amortized sequential inference；完成后停止，无神经物理 correction/RL训练。
资格限于声明的已知 E0/BC、低维参数、合成采集分布；不等于任意G维damage posterior、真实材料校准或RUL。

## A. Git/source/environment audit

分支feature/phase0-architecture，HEAD dc239d1b7b6dd1e44d93e71bdcccccc0f12e8bc9仅Initial commit。
未提交工作树包含完整已验证阶段代码，HEAD不能代表科研基线；无commit/push/merge/reset/clean。
开发前169文件可恢复tar SHA256 `577b52dd46094a97d78f263228cd0db87f38910d02a08aaebc7ae3c7d678b849`，
manifest SHA256 `a2af2c71225546120baff04ca8ff846c4541b259b7223db9aaba0e7e1e02849a`；逐项恢复校验169/169。
历史10133结果文件独立登记哈希，源码快照排除results/cache/旧tar。原数值求解器及历史报告保留。
最终完整manifest、增量patch、恢复tar及历史资产复核在本run `delivery/`，其哈希由`delivery/manifest.json`登记。
Python3.13.12/numpy2.4.4/scipy1.17.1/torch2.13.0/pytest9.0.3/matplotlib3.10.9。
CUDA编译版本13.0，但nvidia-smi无法连接driver、torch CUDA unavailable、0devices，正式训练采用CPU。
16CPU约62GBRAM；每worker numerical/torch thread1；没有破坏原环境、安装PyG或大型RL依赖。
[完整baseline](../{root}/baseline/audit.json)，[Git基线建议](GIT_BASELINE_RECOMMENDATION.md)。

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
AdamW1e−3、weight decay1e−4、clip1、batch32、validation early stopping；所有五固定seeds保留。
point epochs上限160(uniform)/120(Gauss)，covariance60/40；cov阶段冻结point/encoder/GRU，仅训练uncertainty。
E observation-only控制用N0观测TRF pseudo-label，A声明先验，无内部damage labels；它无joint fatigue监督。
CPU实际offline生成/每seed训练、checkpoint/hash、loss histories、validation选epoch及最初失败日志均保留。
GPU不可用，未记录虚构GPU memory或训练加速。最终cohort选择由validation在校准/test前冻结。
''']
    parts.append('## I. Controlled parameter/state recovery\n\n全128 test cases的每seed指标（全部权重冻结后首次打开test评价；不挑最优seed）：\n\n'+table(test['methods']))
    parts.append('## J. Multi-step forecasting\n\nN≤20 assimilation与N30/40 open-loop严格分离。E/control只用N0，forecasts origin0；F/G/H/MAP使用全prefix，origin20。\nindependent current-frame inverse没有初始d0/A识别，forecast为N/A。所有current/displacement/heldout spatial/D30/D40原始每case结果\n在test_final/records.json与raw/；forecast未持续读取未来DIC修正。\n\n![open-loop](../'+str(root)+'/figures/open_loop_forecasts.png)')
    covrows=['| seed | d0 raw/cal coverage | A raw/cal coverage | Current D coverage | D30/D40 coverage |','|---|---|---|---:|---|']
    for k,v in test['methods'].items():
        if not k.startswith('physics_shape_uncertainty'):continue
        raw=v['raw_parameters']['coverage'];c=v['parameters']['coverage'];dc=v['damage_coverage_mean'];covrows.append(f"| {k} | {raw[0]['coverage']:.4f}/{c[0]['coverage']:.4f} | {raw[1]['coverage']:.4f}/{c[1]['coverage']:.4f} | {v['current_damage_coverage_mean']:.4f} | {dc} |")
    parts.append('## K. Noise/registration calibration\n\n每seed独立cal128，以每prefix/参数标准化绝对error的预定95%分位数确定D尺度，Ccal=D Craw D；相关项保留。\n它是learned marginal error calibration，不是精确联合posterior/全条件coverage保证；未在test上调SD。\nfull noise R用于innovation和predictive NIS，field点均值不冒充独立Bernoulli样本。Wilson间/width/standardizedRMSE\n在summary，未来damage、mean displacement与predictive observation分开。N0弱A方向必须保持prior依赖。\n\n'+'\n'.join(covrows)+'\n\n![calibration](../'+str(root)+'/figures/parameter_calibration.png)')
    nr=['| Noise scenario/seed | d0/A coverage | D40 coverage | NIS30/40 |','|---|---|---:|---|']
    for s,methods in noise['scenarios'].items():
        for k,v in methods.items():
            if k.startswith('physics_shape_uncertainty') and v.get('parameters'):
                cs=[c['coverage'] for c in v['parameters']['coverage']];nr.append(f"| {s}/{k.rsplit('_',1)[1]} | {cs} | {v['damage_coverage_mean'][1]:.4f} | {v['normalized_predictive_nis_mean']} |")
    parts.append('固定mechanics .12/.0015、seeds53000–53099，每场景100独立采集：\n\n'+'\n'.join(nr)+'\n\n配准bias对K2模型和K2 MAP均未建nuisance；与Phase2A K5共享配准复核物理信息不同，不能声称公平同等nuisance识别。')
    orows=['| OOD scenario | H D40 RMSE across5 seeds min/max | H failure total |','|---|---|---:|']
    for s,methods in ood['scenarios'].items():
        hs=[v for k,v in methods.items() if k.startswith('physics_shape_uncertainty')];vs=[v['forecast_damage_rmse_mean'][1] for v in hs if v.get('forecast_damage_rmse_mean')];orows.append(f"| {s} | {min(vs):.6g}/{max(vs):.6g} | {sum(v['count']-v['success_count'] for v in hs)} |" if vs else f'| {s} | N/A | all |')
    parts.append('## L. Cross-mesh/OOD/Gaussian\n\n17预声明场景×12独立case，加12paired cross-mesh reference（不是额外独立试验）；包含Q16/64/256/37、\nregular/clustered、independent/spatial、registration、unknown load1.1、nx8 mesh、E2600、3×.7 geometry、\nm=2 model discrepancy、no noise、missing45%、irregular N0/5/20。ID calibration原样transfer，不用OOD拟合。\npaired mesh保持同参数/点/噪声，truth displacement差<1e−12；原始paired每case gap见summary。\n\n'+'\n'.join(orows)+'\n\n![OOD](../'+str(root)+'/figures/OOD_forecast_diagnostics.png)')
    grows=['| Gaussian seed | 6-parameter RMSE | Current D RMSE | D40 RMSE | coverage |','|---|---|---:|---:|---|']
    for k,v in gauss['methods'].items():
        if not k.startswith('physics_shape_uncertainty'):continue
        grows.append(f"| {k} | {v['parameters']['rmse']} | {v['current_damage_rmse_mean']:.6g} | {v['forecast_damage_rmse_mean'][1]:.6g} | {[c['coverage'] for c in v['parameters']['coverage']]} |")
    parts.append('Gaussian32独立test、cal32的小样本探索，完整6×6cov；较窄training priors及弱方向使参数恢复有限，\n没有将小damage RMSE解释为六参数可靠恢复或高维posterior资格。每seed/经典4例结果保存：\n\n'+'\n'.join(grows))
    parts.append('## M. Classical MAP comparison\n\n以下只比较预冻结前16个test cases，全部方法相同observations/noise/public inputs（E/control仅N0作为结构性baseline）：\n\n'+table(test['fair_comparison_first_frozen_cases'])+'\n\nTRF复用Phase1C，prefix冷/暖均只用当前prefix，保存Jacobian/Fisher/历次optimization；未收敛保留且单列。\n只比较同case cost，不把离线生成/训练成本隐藏于在线latency。online包含observations加载/物理rollout与diagnostic写入，\nnetwork_seconds另列；forecast mean/uncertainty成本分开，FEMsolve统计包括全部疲劳endpoint求解。\nSciPy replay仍限制physics-guided端到端速度；NNforward快不能等价宣称全流程相同比例加速。\n\n![accuracy cost](../'+str(root)+'/figures/fair_accuracy_cost.png)')
    parts.append('## N. Ablation and failed experiments\n\nA prior，B per-frame current inverse，C cold prefixMAP，D warmMAP，E graphN0+obs-onlypseudo监督控制，\nF无physics GRU，G global/shape physics-GRU，H shape+fullcov；H point weights固定G，故H/G相同点误差，\n不是额外独立算法点准确性提升。global/shape都训练五seed；不预设shape或G优于F/MAP。\n初始teacher-cohort、deployment validation偏移、seed41004早停和orchestrationhandoff/测试失败日志均保存，\n最终selection不换seed、不按test最好模型挑选。完整每seed loss和代价在training*/summary/history.json。\nGaussian训练规模小，teacher→deployment偏移仍为限定风险，没有看test后追加调参。')
    parts.append(f'''## O. Full tests/scientific verification

开发前357 passed，exit0；正式完整pytest exit{pyt['returncode']}、wall{pyt['wall_seconds']:.2f}s，
实际计数见[完整日志](../{root}/pytest_final.txt)。历史全部{len(reg)}个CLI（含Phase0 smoke）实际重跑，
失败{sum(r['returncode']!=0 for r in reg)}；原Phase2A全部14类科学场景重新运行，并不改其资格。
正式Phase2B A–P全部执行/登记，trained五seed permutation/checkpoint因果检查、Q/V/G、fusion、no-leakage、
rollback、uncertainty/calibration、MAPcomparison、跨mesh/geometry、model discrepancy、成本、truth隔离均在
[科学验收](../{root}/scientific_final/verification.json)。长训练和统计CLI不放进普通pytest。
controlled point gate={science['controlled_point_gate']}；neural coverage gate={science['neural_parameter_coverage_gate']}；regression gate={science['regression_gate']}。

## P. Failure modes/limits

本阶段不保证GNN更准确，经典低维MAP在matched model中通常更精确；physics innovation的额外收益需独立统计。
learned Gaussian与fullcov只给近似条件误差描述，prior支撑单N0 A，不同case条件和OOD不能由总体coverage替代。
局部线性field pushforward未证明global非高斯/多峰/边界coverage；Gaussian6参数样本32统计精度有限。
未知载荷、固定未建registration及fatigue函数形式误差能造成persistent innovation/forecast偏差及CI失效，
不能靠任意改damage吸收后声称解决物理失配；unknownE严格退化不输出唯一真实D。
物理guidance保留大量FEMreplay；CPU训练预算和小数据对泛化有限，GPU加速尚未实际验证。
qualification是部分资格；无任意G维损伤、真实实验、RUL、RL或physicscorrection实现。

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
''')
    Path('docs/PHASE2B_GRAPH_RECURRENT_ESTIMATOR_REPORT.md').write_text('\n\n'.join(parts)+'\n')
