"""Produce the scientific report only from completed frozen evidence."""
import argparse,json,re
from pathlib import Path
import numpy as np


def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(str(x) for x in row)+' |' for row in rows])

def fmt(v):return 'N/A' if v is None else f'{v:.6g}'

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--audit',required=True);ap.add_argument('--output',default='docs/PHASE2D_ROBUSTNESS_AND_MODEL_SELECTION_REPORT.md');a=ap.parse_args();r=Path(a.root);dest=Path(a.output)
    if dest.exists():raise FileExistsError('reports are not overwritten')
    audit=json.loads((Path(a.audit)/'audit.json').read_text());records=json.loads((r/'evaluation/records.json').read_text());good=[x for x in records if x['status']!='FAILED'];rel='../'+str(r);source=json.loads((r/'baseline/audit.json').read_text());env=json.loads((r/'baseline/environment.json').read_text());cost=json.loads((r/'serial_deployment_cost/summary.json').read_text());functions=json.loads((r/'function_family_diagnostics/summary.json').read_text());state=json.loads((r/'state_law_diagnostic/summary.json').read_text());fulltest=r/'pytest_final_v2.txt' if (r/'pytest_final_v2.txt').exists() else r/'pytest_final.txt';testcount=re.findall(r'(\d+) passed',fulltest.read_text())[-1]
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained');colors={'B0':'black','B1':'tab:blue','B2':'tab:orange','B5':'tab:red','B7':'tab:green','B8':'tab:purple'}
    for ax,family in zip(axes.ravel()[:3],('matched','parameter','S1')):
        for m in colors:
            xs=[];ys=[]
            for sigma in sorted({x['sigma_mm'] for x in good if x['split']=='test'}):
                g=[x['methods'][m]['forecast_damage_rmse'][-1] for x in good if x['split']=='test' and x['family']==family and x['sigma_mm']==sigma and m in x['methods']]
                xs.append(sigma);ys.append(np.mean(g) if g else np.nan)
            ax.loglog(xs,ys,marker='o',label=m,color=colors[m]);ax.set(title=family,xlabel='Synthetic sigma_x [mm]',ylabel='Mean N220 damage RMSE');ax.grid(alpha=.25)
    ax=axes.ravel()[3]
    for k in (2,4,8):
        xs=[];ys=[]
        for row in audit['fixed_point_snr']:
            q=next(v for v in row['rows'] if v['K']==k);xs.append(row['sigma_mm']);ys.append(np.median(q['minimum_singular_values']))
        ax.loglog(xs,ys,marker='o',label=f'K={k}')
    ax.axhline(3.,ls='--',color='black',label='Practical gate');ax.set(xlabel='Synthetic sigma_x [mm]',ylabel='Median minimum singular value',title='Data-only box-scaled sensitivity');ax.grid(alpha=.25);ax.legend();axes[0,0].legend(ncol=3);fig.savefig(Path(a.audit)/'comparison.png',dpi=180);fig.savefig(Path(a.audit)/'comparison.svg');plt.close(fig)
    noise_rows=[]
    for row in audit['noise']:
        noise_rows.append([fmt(row['sigma_mm']),row['independent_cases'],row['history_K2_eligible'],str(row['selected_counts'])])
    means=[]
    for sigma in sorted({x['sigma_mm'] for x in good if x['split']=='test'}):
        for family in ('matched','parameter','S1'):
            group=[x for x in good if x['split']=='test' and x['sigma_mm']==sigma and x['family']==family]
            means.append([fmt(sigma),family,len(group)]+[fmt(np.mean([x['methods'][m]['forecast_damage_rmse'][-1] for x in group if m in x['methods']])) for m in ('B0','B1','B2','B5','B7','B8')])
    ood=[]
    for family in ('S2','S3','S4','S5'):
        group=[x for x in good if x['split']=='ood' and x['family']==family];ood.append([family,len(group)]+[fmt(np.mean([x['methods'][m]['forecast_damage_rmse'][-1] for x in group if m in x['methods']])) for m in ('B0','B2','B5','B7','B8')]+[str({n:sum(x['inference']['selection']['selected']==n for x in group) for n in ('B0','B1','B2','B5')})])
    diagnostics=[]
    for label in sorted({x['spec'].get('diagnostic') for x in good if x['split']=='diagnostic'}):
        group=[x for x in good if x['split']=='diagnostic' and x['spec'].get('diagnostic')==label];diagnostics.append([label,len(group)]+[fmt(np.mean([x['methods'][m]['forecast_damage_rmse'][-1] for x in group if m in x['methods']])) for m in ('B0','B2','B5','B7')])
    mesh_rows=[[x['seed'],str((x['spec'].get('nx'),x['spec'].get('ny'))),x['inference']['selection']['selected'],fmt(x['methods']['B7']['current_damage']['rmse']),fmt(x['methods']['B7']['forecast_damage_rmse'][-1]),fmt(x['methods']['B7']['spatial_heldout_displacement_rmse_mm'][-1])] for x in good if x['split']=='diagnostic' and x['spec'].get('diagnostic')=='paired_spatial_mesh']
    negative=[]
    for name,g in audit['matched_gates'].items():negative.append([name,g['count'],fmt(g['mean']),str(g['bootstrap95']),fmt(g['maximum']),g['false_correction_activations'],str(g['false_activation_wilson95']),g['passed']])
    snr_rows=[];horizon_rows=[];observation_rows=[]
    for sigma in sorted({x['sigma_mm'] for x in good if x['split']=='test'}):
        group=[x for x in good if x['split']=='test' and x['sigma_mm']==sigma]
        ds=[x['inference']['candidates']['B5']['observation_diagnostics'] for x in group if 'observation_diagnostics' in x['inference']['candidates'].get('B5',{})]
        snr_rows.append([fmt(sigma)]+[fmt(np.median([d[key] for d in ds])) for key in ('displacement_feature_rms_mm','displacement_snr','d0_change_0_01_whitened_signal_rms','whitened_prefix_residual_rms')])
        for name in ('B0','B2','B5','B7','B8'):
            methods=[x['methods'][name] for x in group if name in x['methods']]
            horizon_rows.append([fmt(sigma),name]+[fmt(v) for v in np.mean([d['forecast_damage_rmse'] for d in methods],axis=0)])
            observation_rows.append([fmt(sigma),name,fmt(np.mean([d['current_damage']['rmse'] for d in methods])),fmt(np.mean([d['forecast_physical_displacement_rmse_mm'][-1] for d in methods])),fmt(np.mean([d['spatial_heldout_displacement_rmse_mm'][-1] for d in methods])),fmt(np.mean([d['heldout_noisy_observation_rmse_mm'] for d in methods]))])
    structural=audit['structural_gates']['B7'];preference=audit['parameter_preference'];structural_failures=[]
    for row in good:
        if row['split']=='test' and row['family']=='S1' and row['inference']['selection']['eligibility'].get('B5'):
            ratio=row['methods']['B7']['forecast_damage_rmse'][-1]/max(row['methods']['B2']['forecast_damage_rmse'][-1],1e-15)
            if ratio>=1:structural_failures.append([row['case_id'],fmt(row['sigma_mm']),row['inference']['selection']['selected'],fmt(ratio),fmt(row['methods']['B7']['forecast_damage_rmse'][-1]),fmt(row['methods']['B2']['forecast_damage_rmse'][-1])])
    functionrows=[[x['label'],fmt(x['oracle_function_fit_A']),fmt(x['oracle_function_fit_m']),fmt(x['positive_hazard_relative_rmse']),fmt(x['zero_hazard_max_false_rate'])] for x in functions['rows']]
    state_rows=[]
    for row in state['rows']:
        state_rows.append([row['case_id'],row['family'],fmt(row['sigma_mm'])]+[fmt(row['forecast_damage_rmse'].get(n,[None])[-1]) for n in ('estimated_state_B2','oracle_state_B2','estimated_state_correct_law_B2','oracle_state_correct_law')])
    step_rows=[[case['case_id'],case['family'],v['method'],fmt(v['step_cycles']),fmt(v['damage_rmse_vs_fine_oracle'][-1]),fmt(v.get('N220_difference_to_next_finer'))] for case in audit['frozen_parameter_steps']['rows'] for v in case['rows']]
    matrix=[('A','Phase2C core numerical replay','PASSED','100 rows exactly equal in parameter/cost/rank/damage fields'),('B','Noise-level sensitivity','PARTIALLY QUALIFIED','20 independent cases per level, synthetic covariance known'),('C','Fisher vs SNR','PASSED','100 cases, K2/4/8; practical rank separated from numerical rank'),('D','Matched negative control','PASSED' if all(x['passed'] for x in audit['matched_gates'].values()) else 'NOT QUALIFIED','40 main matched cases; additional wrong-state/boundary diagnostics reported separately'),('E','A/m preference','PASSED' if audit['parameter_preference']['passed'] else 'NOT QUALIFIED',str(audit['parameter_preference'])),('F','Structural family generalization','PARTIALLY QUALIFIED','S1 controlled benefit; S2/S3/S4/S5 have mechanism/information limits'),('G','Unseen structural OOD','NOT QUALIFIED','Independent families; no test tuning or universal recovery claim'),('H','Registration/loading uncertainty','PARTIALLY QUALIFIED','Shared nuisance and independent reaction contrast; unknown E ambiguity remains'),('I','Coverage/clustering','PARTIALLY QUALIFIED','Irregular Q16/Q64, cluster and missing observations; small diagnostic cohorts'),('J','State-law separation','PASSED','Four oracle-only restart diagnostics and biased-state negative control'),('K','Correction identifiability','PARTIALLY QUALIFIED','K2 strong under high SNR; K4/8 remain weak/null'),('L','Model selection reliability','PARTIALLY QUALIFIED','Frozen engineering score; safety is narrower than correct constitutive discovery'),('M','Temporal convergence','PASSED','Seven reference laws plus three gates, independent DOP853 and query invariance'),('N','Cross-mesh','PARTIALLY QUALIFIED','Two spatial cases each on three meshes; not six independent specimens'),('O','Long-horizon forecasting','PASSED','N30/40/70/120/220, two public future load factors, no future data update'),('P','Accuracy-cost','PASSED','Actual serial B7 deployment and full comparison costs recorded'),('Q','Truth/time isolation','PASSED','Strict anonymous public interface; deleted-truth integration test; independent prefix residual replay'),('R','Phase0–2C regression','PASSED','Full pytest, all37 historic CLI,100 Phase2C baseline replay and frozen Phase2B regression')]
    text=f'''# Phase 2D — Robust and Identifiability-Aware Physics Correction

日期：2026-10-09。正式资格：**{audit['qualification']}**。RL：**NO-GO**。

本阶段完成经典低维、可辨识性感知的历史模型选择与门控修正，没有新增 Phase2D 神经训练或 RL。Phase2C 五个 seed 的重新训练仅用于历史可重复性回归。
受控高 SNR 下可以区分匹配物理、A/m 参数失配和 S1 结构修正；观测信息下降后，保留简单模型的策略减少不必要的修正，却不能保证未知材料律的长期恢复。
**结构选择未达冻结门槛**：历史 K2 eligible 的 {structural['count']} 个 S1 案例中，长期预测改善比例为 {100*structural['improved_fraction']:.1f}%，低于80%；两个50 nm案例的 B7/B8 误差反而高于 B2。40个 matched 案例的负对照通过，参数偏好在 {preference['count']} 个历史 B2 eligible 案例中通过，不能抵消这个结构选择失败。
数值满秩、低 DIC 残差和优化收敛均不代表模型失配已经解决。
证据：[最终审计]({rel}/final_audit/audit.json)、[全部原始案例]({rel}/evaluation/records.json)、[冻结设计]({rel}/frozen_design/frozen_plan.json)。

## A. Git/source/environment audit

分支 feature/phase0-architecture；HEAD dc239d1b7b6dd1e44d93e71bdcccccc0f12e8bc9 仍是初始提交，不能代表完整已验证代码。
开发前 {source['source_files']} 项源码/测试/配置/文档与 Phase2C 最终 manifest 完全一致，canonical SHA256 为 `{source['source_sha256']}`。
轻量 tar 逐文件验证，保存 manifest、包含未提交文件的 HEAD patch、恢复说明、Git 状态和依赖版本。
旧 {source['historic_asset_count']} 项科研资产 path/size/mtime 及 {source['checkpoints']} 个 checkpoint SHA 单独保护；最终核对见 audit。
未修改原 Tri3、FatigueEvolutionSolver、covariance、TRF 或 Phase0–2C 报告；无 Git 写操作、驱动修改或依赖安装。
完整最终 SHA、源码 tar、增量 patch 与实际恢复检验见 [最终恢复审计]({rel}/final_source_verified/audit.json)，避免报告自身哈希循环。
各运行日志的 source_sha256 是相应时刻 src/scripts/config 集合的指纹；候选推断的实际冻结文件以 frozen_inference_manifest 为准。部署收敛 guard 的单项补充有独立 addendum，全部160案例确认不会改变实际选择。不能将运行结束时新增脚本的集合 hash 或 initial HEAD 冒充 worker 使用的完整版本。

Python {env['python'].split()[0]}、NumPy {env['numpy']}、SciPy {env['scipy']}、torch {env['torch']}、pytest {env['pytest']}。
CUDA 编译版本 {env['cuda_build']}，GPU driver unavailable；CPU 单线程 BLAS、科学 CLI 多进程。
GPU 不构成资格条件。源码归档不包含 results、旧 tar、checkpoint 或缓存。
[Git staging proposal](GIT_PHASE2D_BASELINE_STAGING_PROPOSAL.md) 只提出人工分阶段整理建议。

## B. Frozen physical and observation scenarios

M0 matched，M1 parameter mismatch，M2 structural discrepancy，M3 observation/boundary/material uncertainty 分开评价。
prior λ=A x^m，x=Ya/.025≥0；Ya 为 N/mm²，A/λ 为 cycle⁻¹，m、D、R 和修正系数无量纲。
known E 1800–2200、d0 .08–.16、五个公开比例轴向载荷幅值，以及 future factor .25/.4 均由冻结种子生成。
主测点 Q16 不规则坐标；额外 Q64、cluster、missing、spatial covariance 和共享 registration。
train8/validation8/calibration12 只生成 N0/10/20。方法是固定经典选择规则，不需要神经训练，calibration split 未用于调阈值或概率校准。
正式100个独立主测试案例：每噪声20，matched8/parameter6/S1六；OOD24和diagnostic36另列。
188 是数值数据集数；配对空间网格复用两个物理 seed，不能按188个完全独立物理重复统计。
selector 接口只有匿名 observation acquisition，不读取真实 family 标签。

## C. Noise and SNR study

噪声 x-component [2e-7,5e-5,2.5e-4,7.5e-4,1.25e-3] mm，分别为 .2/50/250/750/1250 nm；均为合成假设，没有实际 DIC 标定资格。
独立 y-component std=1.25σ；空间模型使用完整 .22mm covariance kernel 与 component correlation。
位移 RMS/SNR、0.01 d0 变化的白化信号、Fisher singular values、condition、practical rank、残差及优化成本逐案例保存。
位移特征有较高 SNR，仍可能没有足够信息识别材料修正的第二方向。

{table(['sigma_x mm','独立主案例','history K2 eligible','选择计数'],noise_rows)}

{table(['sigma mm','median displacement RMS mm','median displacement SNR','median Δd0=.01 white signal','median B5 prefix white residual RMS'],snr_rows)}

固定 public nominal 线性化的100案例 K2 numerical rank全部为3；实际 rank 依据 min singular value≥3，数值秩与实践秩分开。
这些方向是在参数箱尺度下定义，不是已校准的 posterior 标准差声明。
不同噪声等级是独立物理 draws；只在同一案例内作 paired method comparison。

## D. New structural law families

S1：λ=A x^m(1+3x/(1+x))。S2：λ=A max(x−.5,0)^m。
S3：λ=A x^m exp(.55(R+1))；S4：λ=A x^m exp(3D)；S5：λ=A x^m exp(.3 cos(πX/Lx))。
以上 synthetic oracle 不是文献经典式或真实材料定律。非负、零驱動不增长、不可逆与时间收敛均实际验证。
S2 是相对于冻结 A/m 家族的结构失配；在支持阈值的更大物理家族中可以优先标定 η。
S3 在固定 R 下与 A 重标定严格等价；本试验没有把不同试件的 R 冒充共享材料的 joint-R 识别。
S4/S5 可在窄状态/空间区间被简单参数近似，不能仅凭数学上不属于 A/m 就声称 DIC 可区分。
纯函数 oracle 诊断使用 log-positive hazard 线性拟合，无 epsilon，**不是公平 DIC 基线**：

{table(['law','oracle A','oracle m','positive relative RMSE','zero false hazard'],functionrows)}

S2–S5 在正式测试前指定为结构 OOD，未用于调整选择规则或 basis。

## E. Classical parameter-refit comparison

B0：原 hazard、d0由同一 prefix MAP 估计。B1：d0/A；B2：d0/A/m；B5：d0/c0/c1。
所有方法共用 FEM、DIC、covariance、nominal 初始化、TRF80、候选一致历史和 time split。
每种方法保留 N0/10 历史 fit 与 N0/10/20 warm refit；不保证不同局部解与 cold optimum 相同。
B5 coefficient ridge .001 仅加入一次，Fisher 不包含它的约束。
以下为全部有限结果的 N220 平均 damage RMSE，不按真值误差筛选：

{table(['sigma','family','n','B0','B1','B2','B5','B7','B8'],means)}

## F. Correction identifiability

生产 correction 为 K2 energy basis φ=[t,t²]，t=x/(1+x)，固定全局系数 |c|≤4。
参数在各自 box 内映射到无量纲坐标：d0∈[.02,.30]、A∈[1e−5,5e−4] cycle⁻¹、m∈[.6,1.6]、c∈[−4,4]；共享 tx/ty∈[−.002,.002] mm、θ∈[−.002,.002] rad、load_factor∈[.85,1.15]。
B2 和 B5 都有三个主要未知量（包括 d0），K2 并不天然增加两个相对于 B2 的可辨识自由度。B5 固定先验 A/m，以避免同时放开 A/m/c 的冗余；原联合混淆科学实验另行重放。
conditional sensitivity 投影移除 d0/nuisance，保留全部参数交叉项；原 data Fisher、null/weak directions、差分误差另存。
K4/8 仅执行固定点信息对照，没有扩大网络或拟合高维 action。
低 SNR 下 min singular value不足时拒绝自由修正；box或ridge不被解释为新增 DIC 信息。
当前仅保存局部信息与 point plug-in damage，不提供已校准参数/损伤/预测覆盖率。

## G. Model selection / shrinkage formulation

fit=N0/10；独立历史验证=N20；然后冻结选择，允许 full-prefix=N0/10/20 refit。
score=SSE_history-validation+2 practical rank；额外收益需大于 max(9.21,2√(2M))。
eligible 要求全部 data singular values≥3、两尺度差分可靠、无 .995 活动边界、optimizer收敛。
优先 physics→A→A/m，只有 K2 对已验证简单模型有额外独立历史收益才启用。
该分数是预冻结工程规则，不声称是精确 Bayes factor 或非嵌套 χ²检验。

B8：λ=λphys exp(gate·2 tanh(Φc))，gate-grid=[0,.25,.5,.75,1]。
历史 score 添加 2 gate²||c||²；gate冻结后若为中间值，固定 full-prefix c 并仅重新估计 d0/nuisance。
没有修正证据时保留已验证 physics/parameter model。gate=0 保留相应物理 baseline，zero hazard始终为zero。
乘性模型不能创造原 hazard=0 区域的新损伤成核。
本次 gate 值计数为 {audit['gate_values']}；中间 gate 是否改善不能由实现支持直接推断。B8 与 B7 的成绩应按实际选出的 gate 解释。

部署审计补充了“full-prefix未收敛即拒绝部署”的条件，未改变噪声/统计门槛。
全部已选模型的收敛状态检查及原记录是否受影响见 deployment_guard_addendum；没有用未来误差决定修复或选型。

## H. Correct-physics negative controls

冻结门槛：paired N220 mean excess 的 bootstrap95上界≤.001，单案例 excess≤.01，false K2 activation Wilson95上界≤.10。
不用 Physics-only 接近零时不稳定的相对误差。40独立matched主案例，bootstrap seed94991/repeats10000。
每噪声只有8个matched，零激活的单层 Wilson上界仍超过.10，不能声称所有层都有充分统计精度。

{table(['method','n','mean excess','bootstrap95','max excess','false K2','Wilson95','gate'],negative)}

固定偏差初态的 oracle-only 负对照另外执行；其演化修正不能消除 N0 错误，并可能恶化长期误差。
实际固定 d0 偏低 .01 的诊断中，K2 把 c0 推到边界；N220 误差从不修正的 {state['fixed_biased_state']['forecast_damage_rmse']['wrong_state_no_correction'][-1]:.6g} 增至 {state['fixed_biased_state']['forecast_damage_rmse']['wrong_state_fitted_correction'][-1]:.6g}，虽然短期误差下降。该诊断明确显示状态误差可以诱发错误材料修正。
matched gate只覆盖声明的已知材料/边界分布，不覆盖未标定 E/loading/registration。

## I. State-law-boundary ambiguity

shared tx/ty/θ 在所有帧只声明一次，measurement covariance 是给定 nuisance 后的噪声，不重复计数配准方差。
unknown load 对照加入独立 N0 reaction，全部参数 Fisher cross blocks 保留；这种反力是额外物理信息，不等同纯 DIC。
wrong E 仍可能与 damage 混淆；Phase1C/2A healthy-baseline/force诊断复跑，不授予新的 dynamic unknown-E 资格。
不同 nuisance 设置的初始 diagnostic cohorts不是全部配对，不能把其均值差解释成精确因果收益。

{table(['diagnostic','数值数据集','B0','B2','B5','B7'],diagnostics)}

四个正式案例完成后进行 memoryless oracle-current-state restart；这些结果仅用于误差分离：

{table(['case','family','sigma','estimated B2','oracle-state B2','estimated-state correct-law','oracle-state correct-law'],state_rows)}

真实 law/内部 state 不进入可部署选择器；不能把 oracle 改善归功于 correction 方法。
高 SNR S1 案例 p2d-0042 中，B2 仅替换真实当前状态把 N220 RMSE 从 .002058 降到 .001992；保留估计状态而替换正确材料律则降至 .000227。该有限 oracle 诊断中，演化律误差比当前状态误差更主导长期偏差。高噪声下两者均存在，不能把一个更准确 estimator 等同于已经修复材料规律。

## J. Long-horizon prediction / time leakage

所有候选 N30/40/70/120/220 完整开环预测完成，评价器才读 synthetic truth、family 和未来 DIC。
预测跨度10/20/50/100/200 cycle；两种公开 future amplitude factor，不能按 observation index 代替 cycles。
每个新参数都从一致N0初态重演；新damage重新求FEM；future designs由已知网格/坐标/载荷生成。
输出复用 MechanicsBelief、PredictedObservation、StatePrediction、ForecastResult。
filtered、history-only tentative prediction、forecast标签保持区分，physical/registered 位移指标分开。
删除全部真值与future DIC后仍可完成实际 forecast 的集成测试通过；独立 prefix残差重放核对选择证据。

10组 time-law/gate对独立DOP853全部通过一阶收敛与额外查询不改变终态。
具体 dt4/2/1/.5、误差、order和场见 [time reference]({rel}/scientific_diagnostics/temporal/summary.json)。
这是同一 FE 空间离散的时间验证，不是连续空间真解或模型正确性的证明。
另外针对预声明排序中的第一个 matched、前两个 high-SNR S1 案例，固定已选系数，在 dt2/1/.5 与 oracle dt.25 上独立重演完整轨迹。该后续数值诊断不重拟合或改变 selector：

{table(['case','family','method','step cycle','N220 RMSE vs fine oracle','difference to next finer'],step_rows)}

两个 S1 案例在 dt.5 的 B5/B2 RMSE 分别约 .170 和 .180，优势没有因步长加密消失；dt2→1→.5 的相邻 trajectory 差约减半。matched 案例中原来按 dt2 拟合的参数含微小离散偏差，固定参数后对 fine-oracle 的误差不必单调。这项检查仅覆盖三个诊断案例，不能证明所有 OOD/noise 场景都有同样结论。

全部主案例的各 horizon damage RMSE 平均值如下；同一案例的五个时刻是配对轨迹，不能作为独立统计样本：

{table(['sigma mm','method','N30','N40','N70','N120','N220'],horizon_rows)}

当前 filtered state 与未来位移分开报告。physical 与 spatial-heldout 指标对无噪声参考；noisy observation 指标包含测量噪声，不能直接作为内部损伤恢复证据：

{table(['sigma mm','method','current D RMSE','N220 physical u RMSE mm','N220 spatial-heldout u RMSE mm','future noisy observation RMSE mm'],observation_rows)}

## K. Cross-mesh / law-family OOD

{table(['family','n','B0','B2','B5','B7','B8','选择'],ood)}

S3固定R的 A 等价不是新的机制识别；S2 threshold、S4 state-dependence、S5 spatial变化不能据本轮声明通用修复。
空间S5两物理cases×三网格，实际非均匀FEM；配对 common DIC 和 area-weighted指标保存。
未把不同G向量直接对比，没有平滑、graph-conditioned damage或网格客观裂纹资格。
跨geometry仅两个小规模诊断case，统计资格有限。

{table(['physical seed','mesh nx,ny','selected','current damage RMSE','N220 damage RMSE','N220 spatial-heldout u RMSE mm'],mesh_rows)}

## L. Accuracy/cost tradeoff

{table(['case','selected','history选择s','selected fit s','selected rollout s','total online s','FEM'],[[x['case_id'],x['selected'],fmt(x['history_selection_seconds']),fmt(x['final_selected_fit_seconds']),fmt(x['selected_forecast_seconds']),fmt(x['total_online_seconds']),x['fem_solves']] for x in cost])}

这是实际串行B7部署：四历史候选、一个最终选中模型refit、一次完整物理rollout，不执行未选模型 full refit/forecast。
研究矩阵 total_online_seconds 另外包含四个baseline full refit/forecast、gate验证和所有Fisher成本，不能与单模型部署直接混比。
FEM solves、nfev、evaluation history、失败、条件数、selection/forecast wall逐例记录；wall受并发任务影响。
Phase2D 方法无神经训练，neural-forward latency/GPU-memory为N/A。离线数据生成成本见 dataset/manifest，不能隐藏为在线收益。
此处无神经训练指 Phase2D 方法本身；历史 Phase2C 五-seed 复跑费用单独保存，不计为选择器训练收益。
主矩阵三 worker 的实际 wall 为 {audit['evaluation_execution']['wall_seconds']:.2f}s，160案例；源数据生成 wall 为 {json.loads((r/'dataset/manifest.json').read_text())['wall_seconds']:.2f}s。固定点 K2/4/8 信息诊断与十组独立时间解另耗924.64s；三个正式案例固定系数加密另耗 {audit['frozen_parameter_steps']['wall_seconds']:.2f}s。不能将并行批次 wall 当作单次在线成本。

## M. Statistical confidence and failure cases

硬失败 {audit['failed']}，candidate失败 {audit['candidate_failures']}，未收敛候选 {len(audit['unconverged_candidates'])}，活动边界解 {len(audit['boundary_solutions'])}，均保留。
conditional-identified structural gate：{audit['structural_gates']}。
parameter preference：{audit['parameter_preference']}。
结构门槛同时要求 median ratio≤.9 与改善比例≥.8，当前仅前者满足，因此为 **NOT QUALIFIED**。没有将 eligible 子集缩减为 high-SNR 或最终选中 correction 的案例。

{table(['case','sigma mm','selected','B7/B2 N220 ratio','B7 RMSE','B2 RMSE'],structural_failures)}

这两个失败说明：局部 data rank 合格并不保证短历史能够区分不同材料律在改变载荷后的行为。模型选择需要独立的辨别加载信息，不应通过 test 真值修改 gate。

{table(['evaluation-only family','n','Physics','parameter refit','K2'],[[x['family'],x['case_count'],x['physics'],x['parameter'],x['correction']] for x in audit['main_family_selection_concordance']])}

这是预测完成后才读取的 family 标签一致性表；低信息下保留 physics 并不是识别出了正确 law。只有8个历史 eligible 的结构案例，改善比例的 Wilson95 区间为 {structural['improved_wilson95']}，统计精度也不足以支持普遍可靠性。
这些子集用历史data诊断定义，不按未来误差筛选；全部noise层的失败/误差仍报告。
同一case的五个载荷specimens、时间帧、测点、配对网格均不是额外统计重复。
未来hazard误差含 state 与law误差；coefficient真值不唯一，coefficient RMSE为N/A。
Fisher不是已校准风险权重，未继承Phase2B covariance；Phase2A原89/100和后续独立97/100仍保持各自原证据。

## N. Full numerical tests and scientific CLI

开发前426 passed；最终完整 **{testcount} passed**，旧测试未弱化或删改。
Phase2C四核心方法×25cases共100条实际重放，非时间参数/目标值/rank/damage字段逐位相同。
37个Phase0–2A历史科学CLI全部exit0；Phase2B冻结原数据/checkpoint完整复跑，不重训或调calibration。实际评价9968条，原8888条全部非时间数值和状态一致，额外1080条单独保留。
Phase2C constant/zero/semigroup、独立时间解、joint identifiability、spatial mesh/permutation 四套原科学实验实际通过；五个训练 seed 的独立重训系数差均为0，原训练案例 prefix 重新生成也逐位相同。
new focused contracts、真实solver、covariance、truth-isolation、practical-rank、未收敛部署拒绝测试均通过。

{table(['ID','验收','状态','范围'],matrix)}

此表的 PASSED 可指物理/接口/隔离或数值复现检查通过；未知材料律恢复和噪声鲁棒性的科学资格以相应 PARTIALLY/NOT QUALIFIED 与冻结统计门槛为准。

复现需新输出目录，先设置 PYTHONPATH=src 与 OMP/OPENBLAS/MKL线程1。

```bash
python scripts/verify_phase2d.py --stage freeze --output results/reproduce2d/design
python scripts/verify_phase2d.py --stage generate --plan results/reproduce2d/design/frozen_plan.json --output results/reproduce2d/dataset
python scripts/diagnose_phase2d.py --dataset results/reproduce2d/dataset --output results/reproduce2d/diagnostics
python scripts/evaluate_phase2d.py --dataset results/reproduce2d/dataset --output results/reproduce2d/evaluation
python scripts/audit_phase2d_prefix_replay.py --dataset results/reproduce2d/dataset --evaluation results/reproduce2d/evaluation --output results/reproduce2d/prefix_replay
python scripts/diagnose_phase2d_state_law.py --dataset results/reproduce2d/dataset --evaluation results/reproduce2d/evaluation --output results/reproduce2d/state_law
python scripts/diagnose_phase2d_functions.py --output results/reproduce2d/functions
python scripts/diagnose_phase2d_frozen_steps.py --dataset results/reproduce2d/dataset --evaluation results/reproduce2d/evaluation --output results/reproduce2d/frozen_steps
python scripts/benchmark_phase2d.py --dataset results/reproduce2d/dataset --evaluation results/reproduce2d/evaluation --output results/reproduce2d/cost
python -m pytest -q -W error
```

本run最终audit脚本还核对baseline/历史回归目录布局；复现实验应分别提供这些回归证据，不假定旧结果已经重跑。

![Accuracy and information]({rel}/final_audit/comparison.png)

## O. Generalization limitations

本阶段是有限合成、二维小应变、平面应力、比例载荷和局部 scalar damage 验证。
noise/covariance未真实实验标定；方法没有概率覆盖率资格。弱方向、未知材料/边界和model-form OOD仍会导致未来偏差。
shrinkage未必优于离散选择，不能只因gate更连续就宣称更先进。
没有高维damage posterior、graph-conditioned action、裂纹路径、RUL、真实材料校准或mesh-objective localization。
对真实观测信息不足的案例，选择简单模型保护负对照，同时保留无法恢复未知机制的限制。

## P. Updated RL Go/No-Go

**NO-GO。** 当前主要瓶颈是材料机制/状态/载荷的信息混淆与观测SNR，RL没有增加独立测量信息。
优先做独立载荷/反力与known-healthy E校准、共同材料的多R/阈值诊断、低驱动区更长独立历史以及DIC metrology。
为可区分的机制增补少量传统η/R/D/spatial参数，并先与A/m和固定K2公平比较；不扩大弱可辨识K。
需要独立calibration/test的预测风险验证后，才能讨论未来学习决策的额外科学价值。
本轮停在Phase2D，不进入RL。
'''
    dest.write_text(text);print(dest,flush=True)
