# Phase 2A 共享配准不确定性独立复核

日期：2026-10-08。只读审计 Phase 2A 资产；**不更改原部分资格**。
结果位置：[独立复核](../results/phase2b/20261008T114713Z/uncertainty_followup_v2/summary.json)、
[曲率/profile/bootstrap](../results/phase2b/20261008T114713Z/uncertainty_followup_v2/diagnostic.json)、
[原100例审计](../results/phase2b/20261008T114713Z/original_registration_audit.json)。

原结果使用 seeds 1000–1099，d0 的95%名义区间覆盖 **89/100**，Wilson95%区间
**[81.369%,93.746%]**。A/tx/ty/θ 覆盖分别94/92/95/92%；未来 D30/D40 为95/95%，
predictive normalized NIS 为1.027/.959。参数区间、damage pushforward区间和预测观测NIS分别计算。

新 seeds **22000–22099**、profile seed22000、bootstrap seeds23000–23015 在运行前写入
[冻结配置](../configs/phase2b/uncertainty_followup.toml)，没有根据结果替换种子或缩放标准差。
前缀仅N=0/10/20，N=30/40只作评价。匹配物理模型、Q=16、σx=5e−5 mm、σy=1.25σx。
测量误差在给定共享参数后各帧独立；全前缀只有一个tx、ty、θ，未把共享配准再加入测量R。

| 参数 | 原coverage | 新coverage | 新Wilson95% | 新bias | 新RMSE |
|---|---:|---:|---|---:|---:|
| d0 | .89 | .97 | [.9155,.9897] | −9.7491e−5 | .0022359 |
| fatigue A | .94 | .96 | [.9016,.9843] | −1.5287e−6 | 9.6174e−5 |
| tx (mm) | .92 | .95 | [.8882,.9785] | 5.6099e−7 | 1.7426e−5 |
| ty (mm) | .95 | .94 | [.8752,.9722] | 1.1109e−6 | 1.9607e−5 |
| θ (rad) | .92 | .96 | [.9016,.9843] | −1.9640e−6 | 1.3765e−5 |

新100例全部收敛；D30/D40覆盖.95/.94，预测normalized NIS1.0280/1.0253。
新d0 error/SD的RMS=.9691，原值1.1084；原d0 SD范围.002276–.002339。
原data Fisher与JᵀJ逐元素重建相同；prior Fisher为零；damage/fatigue与registration交叉块
范数148549–152362，未遗漏不同参数间信息。原d0–A后验相关为−.5615至−.5372。
全部原拟合距normalized参数边界至少.2983，没有明显边界截断原因。

首个新seed的full observed Hessian采用标量NLL中心二阶差分(h=2e−4)，包含残差曲率。
相对GN差异范数3.175e−6，全部特征值正；d0 SD为GN .002284389、full Hessian .002283834。
Profile每个固定d0重新TRF优化[A,tx,ty,θ]，以ΔNLL=1.920729为局部95%似然阈值，
25点网格线性插值给[.1191514,.1280894]，Gaussian给[.1191669,.1281216]。
该首例未显示明显曲率导致的区间缩窄；profile的Wilks阈值也不是全局/小样本精确保证。

参数bootstrap在拟合物理参数及同一个拟合配准下重新采样独立测量噪声，再优化完整前缀。
16次全部保存，d0 percentile区间[.1198229,.1275624]。只有16次、尾部分位数统计精度很低，
它是形状诊断，**不是**bootstrap覆盖率资格或可替代100seed的验证。

独立复核没有再次显示明显d0欠覆盖，尚无共享nuisance/covariance交叉项实现错误的证据。
这不撤销原89/100，也不能由两次100seed或多个情景中较好的一个结果证明普遍95%校准。
抽样波动、多情景比较、非线性/有界近似和模型失配仍需更大预声明实验。
没有在同一测试集事后乘SD系数。Neural uncertainty将另用train/validation/calibration/test评价，
不监督拟合本阶段Gaussian covariance。

命令：`PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/audit_phase2a_uncertainty.py --output <new-directory>`。
实际wall303.92s，exit0。首次辅助配置缺geometry导致KeyError的100条失败记录保留在
`uncertainty_followup/`；修复配置后以新目录v2完整复跑，没有改原物理或统计标准。
