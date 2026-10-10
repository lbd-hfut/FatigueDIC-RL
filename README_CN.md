# FatigueDIC-RL

**DIC-Observed Physics-Guided Fatigue Damage Identification and Evolution**<br/>
**DIC观测下的物理引导疲劳损伤识别与演化**

[English](README.md) · [科学报告](docs/PHASE2F_INFORMATION_SUFFICIENCY_REPORT.md) · [验证准备](docs/SCIENTIFIC_BASELINE_AND_VALIDATION_READINESS.md)

本项目研究如何从不规则DIC位移观测识别低维损伤参数，并通过有限元与疲劳模型追踪损伤、诊断可辨识性和模型失配、预测未来响应。核心目标是区分“观测实际支持的识别结果”与“依赖材料先验的推断”。项目名保留早期RL研究设想；**目前RL为NO-GO，没有经验证的RL疲劳预测器**。全部公开科学结果来自合成数值试验，未完成真实材料或相机误差标定。

## 整体框架与数据契约

![物理引导状态识别与未来预测框架](docs/assets/readme/architecture.svg)

几何与网格 → FEM与本构 → 疲劳损伤演化 → FE-to-DIC观测 → 可辨识性与反演 → 序贯参数belief → 物理修正 → 未来开环预测。经典MAP与Graph+GRU共用观测、belief和forecast契约；真值生成/评分路径与Observation-only推断隔离。修正是可选择的低维材料律研究模块。

| 支撑空间 | 数据与含义 |
|---|---|
| FE节点 $V$ | 位移 $u[V,2]$、物理坐标 |
| FE单元 $E$ | Tri3连接关系 $[E,3]$、单元响应 |
| 积分点 $G$ | 损伤/材料历史 $D[G]$，应变与应力 $[G,3]$ |
| DIC观测点 $Q$ | 任意物理坐标、位移 $[Q,2]$、mask与协方差 |

这些位置与数量不能互相reshape替代；单个Tri3算例中恰好 $G=E$ 不代表一般一一对应。详见[Phase 0](docs/PHASE0_IMPLEMENTATION_REPORT.md)与[Tri3契约](docs/08_TRI3_MECHANICS_CONTRACTS.md)。

## 有限元与损伤本构

首版为二维、小应变、各向同性标量损伤、非结构Tri3平面应力模型：

$$
\varepsilon=\tfrac12(\nabla u+\nabla u^\top),\quad
\nabla\cdot\sigma+b=0,\quad
\sigma=(1-D)C_0:\varepsilon,\quad 0\le D<D_{max}<1.
$$

$$
K(D)u=f,\qquad K(D)=\sum_e\sum_{g\in e}t\,w_gB_e^\top C(D_g)B_e,
\qquad
C_0=\frac{E_0}{1-\nu^2}
\begin{bmatrix}1&\nu&0\\\nu&1&0\\0&0&(1-\nu)/2\end{bmatrix}.
$$

$t$为厚度，$w_g$为物理积分面积，$B_e$对应工程剪应变 $\gamma_{xy}$。位移用mm、应力及能量密度用N/mm²、反力用N。求解器执行稀疏装配、Dirichlet消元、反力/平衡/能量检查。[Tri3源码](src/fatigue_dic_rl/mechanics/tri3.py)与[Phase 1A报告](docs/PHASE1A_MECHANICS_VERIFICATION_REPORT.md)给出范围限定的验证。

## 疲劳驱动力与不可逆演化

实际求解当前损伤下的比例载荷两个端点：

$$
\varepsilon_a=(\varepsilon_{max}-\varepsilon_{min})/2,\qquad
Y_a=\tfrac12\varepsilon_a^\top C_0\varepsilon_a,
$$

$$
\lambda_{phys}=A\left\langle Y_a/Y_{ref}-\eta\right\rangle_+^m\exp(\beta_RR),\qquad
D_{k+1}=D_{max}-(D_{max}-D_k)\exp(-\Delta N_k\lambda_k).
$$

$A$单位cycle⁻¹，$Y_a,Y_{ref}$单位N/mm²，其余参数无量纲；$R$是公开的比例循环载荷最小/最大倍率。不能用端点能量差代替应变振幅驱动力，否则完全反向载荷会被错误认为无疲劳驱动。

每个接受增量只更新一次损伤，并使用**新损伤重新求解FEM**。材料历史绑定同一积分点布局；非有限hazard、力学求解失败和非法状态显式拒绝并原子回滚。指数更新只对增量内冻结hazard精确，耦合演化需要时间步收敛检验。这是可标定的**合成参考律**，不宣称是已验证的经典文献材料定律。[模型源码](src/fatigue_dic_rl/fatigue/models.py)、[Phase 1B](docs/PHASE1B_FATIGUE_EVOLUTION_REPORT.md)。

## DIC观测与统计反演

$$
y_k=P_{FE\to DIC}u_k+e_k,\qquad \Sigma_k=L_kL_k^\top,
$$

$$
-\log p(y\mid\theta)=\tfrac12\|L^{-1}(\widehat y(\theta)-y)\|^2
+\tfrac12\log\det\Sigma+\tfrac{n_{obs}}2\log(2\pi).
$$

Tri3形函数在任意物理坐标插值，不要求规则H×W图像。ROI与validity共同定义有效点；先取有效协方差子矩阵，再白化残差。独立反力残差按自身N单位不确定性白化。支持逐分量误差、空间协方差和共享配准nuisance；confidence不是方差倒数，共享registration不能被重复计入独立帧证据。真实DIC接入仍需外部单位/坐标/载荷/误差标定，本项目不执行原始图像相关、畸变校正或双目重建。[Phase 1C](docs/PHASE1C_IDENTIFIABILITY_REPORT.md)。

## 可辨识性与参数后验

$$
J_w=L^{-1}\frac{\partial\widehat y}{\partial\theta},\quad
F_{data}=J_w^\top J_w,\quad J_w=USV^\top,\quad
F_{post}=F_{data}+F_{prior}.
$$

中心有限差分在无量纲有界参数空间执行，分别报告data-only rank、弱/零方向与先验约束rank。先验不等于新增DIC信息；秩亏子空间伪逆不能被当成所有参数方向都有小方差的后验。**满数值秩不保证可靠恢复。** 均匀损伤下 $E_{eff}=E_0(1-D)$ 导致严格模量–损伤等价解。不同初值可收敛到相同观测拟合、不同真实损伤的解。

## 序贯belief与神经摊销估计

首版 $\theta=[d_0,A]$，已知 $E_0$ 和边界；Gaussian初始损伤扩展为六参数空间。参数belief与当前损伤估计不同：

$$
b_k(\theta)=p(\theta\mid o_{0:k}),\quad
\widehat\theta_k=\arg\min_\theta\left[
\tfrac12\sum_{j=0}^k\|L_j^{-1}(\widehat o_j(\theta)-o_j)\|^2+\mathcal R_{prior}(\theta)\right].
$$

Prefix MAP复用有界SciPy TRF。Warm-start只改变初值，不把旧posterior重复当先验；每次完整prefix目标只纳入各观测和原先验一次。**每个候选参数都从一致初态重新构建材料历史**。参数均值经非线性decoder得到的损伤是plug-in estimate，不严格等于损伤后验均值。

`MechanicsBelief`、`PredictedObservation`、`Innovation`、`ForecastResult`分别保留参数/不确定性、预测、innovation及未来状态。Prediction-before-observation、filtered和future-open-loop标签分离。例如N0/10/20同化，N30/40未来观测不得再修正预测。[Phase 2A](docs/PHASE2A_SEQUENTIAL_BELIEF_REPORT.md)共享配准情景原95%区间仅89/100覆盖；[独立follow-up](docs/PHASE2A_UNCERTAINTY_FOLLOWUP.md)保留原结果。

Graph+GRU使用坐标图、FE连接图和形函数加权特征融合：

$$
\widetilde z_v=\frac{\sum_qP_{qv}z_q}{\sum_qP_{qv}},\quad
r_k=L_k^{-1}(o_k-\widehat o_k^-),
$$

$$
z_k=Encoder(G_{DIC},G_{FE},r_k,\widehat o_k^-,\ell_k,N_k,\Delta N_k),\quad h_k=GRU(h_{k-1},z_k),
$$

$$
\widehat\zeta_k=\tanh(Head_\theta(h_k)),\quad
\Delta\zeta_k=\widehat\zeta_k-\widehat\zeta_{k-1},\quad
\widehat\theta_k=c+s\odot\widehat\zeta_k,\quad C_\zeta=TT^\top.
$$

无量纲坐标 $\zeta\in[-1,1]^K$保证参数有界；Cholesky head保证PSD协方差，但小方差不等于后验经过校准。融合是特征归一化，不是观测精度求和。可选native-PyTorch路径使用SiLU图消息传递和因果graph-level GRU，不要求PyG。合成参数标签属于额外监督信息；SciPy FEM只提供停止梯度的物理特征，**没有伪装端到端可微FEM**。更新参数后重新建立物理历史。[Phase 2B](docs/PHASE2B_GRAPH_RECURRENT_ESTIMATOR_REPORT.md)。

## 物理修正、选择与实验设计

低维修正保留不可逆DamageTransition：

$$
\lambda_{hybrid}=\lambda_{prior}\exp\left[\alpha\tanh\left(\sum_{j=1}^Ka_j\phi_j(x,S_k)\right)\right].
$$

实际energy basis为 $t_x=x/(1+x)$、$\phi_j=t_x^j$（j=1…K）。主研究 $K=2$，不直接输出任意G维damage action。原hazard为零时乘性修正仍为零，不能表达新的成核机制。必须分开评价：参数错误（优先A/m标定）、材料律结构错误、状态估计错误。Oracle hazard监督与纯DIC方法的信息条件不同。[Phase 2C](docs/PHASE2C_PHYSICS_CORRECTION_REPORT.md)。

冻结B7/B8使用历史验证、信息rank和残差诊断选择/抑制修正；Phase 2E的E9独立命名，不修改旧规则。候选设计比较规则/随机点、Fisher和model-discrimination设计。对模型差异及nuisance白化灵敏度：

$$
d_w=L^{-1}(y_1-y_2),\qquad d_\perp=(I-U_nU_n^\top)d_w.
$$

$U_n$是nuisance Jacobian的已解析SVD基。原始模型差异可能被载荷、模量、配准调整吸收；大局部差异不直接等于统计显著性或选择准确率。信息不足允许ABSTAIN。静态加载识别当前刚度；疲劳激励必须实际推进循环数、改变损伤，并计入试验成本。[Phase 2D](docs/PHASE2D_ROBUSTNESS_AND_MODEL_SELECTION_REPORT.md)、[Phase 2E](docs/PHASE2E_OBSERVABILITY_AND_DESIGN_REPORT.md)。

## 已验证的有限范围结果

最新验证准备记录为**514项测试通过**。测试通过不代表全部科研假设成立；原PARTIALLY_QUALIFIED、失败及NO-GO结论全部保留。

![Tri3制造解收敛](docs/assets/readme/tri3_convergence.png)

Phase 1A最后两层位移/应变收敛率约2.130/1.055，符合光滑问题的 $O(h^2)$ / $O(h)$。不证明局部软化损伤的网格客观性。

![真实耦合损伤场及累计增量](docs/assets/readme/damage_evolution.png)

Phase 1B的16单元/16积分点局部损伤场，从N0推进至N40，显示实际FEM耦合演化；粗网格与局部本构仍有网格敏感性。

![Gaussian初始参数位移灵敏度](docs/assets/readme/damage_identifiability.png)

Phase 1C在64个不规则DIC点上记录六参数灵敏度。[奇异值图](docs/assets/readme/identifiability_spectrum.svg)数值rank6，但仅四个奇异值超过单位噪声尺度；未知E0–damage仍可严格退化。

![结构失配高SNR基线比较](docs/assets/readme/phase2c_structural_benchmark.png)

同一结构失配family的五个配对测试case，N220平均damage RMSE：B0物理先验0.0196014，B2重标定A/m为0.00201021，B5 DIC-only K2为0.000350085；K2相对B2中位降低约82.7%。主噪声 $2\times10^{-7}$ mm（**0.2nm**）只是极高SNR合成设置，绝不是实际DIC精度。五case不是五个独立材料机制。参数失配中A/m重标定更准确；正确物理模型下多余修正会退化。

![初始状态信息秩](docs/assets/readme/phase2f_information_rank.png)

在健康D=0外部已知、健康/损伤模量可迁移、共享且已标定载荷的三参数诊断中，DIC、DIC+reaction、DIC+healthy、两者均加入时rank分别1/3、2/3、2/3、3/3。未知健康模量迁移或载荷比会重新引入damage零方向，不能将条件满秩描述为一般唯一恢复。[Phase 2F](docs/PHASE2F_INFORMATION_SUFFICIENCY_REPORT.md)与[Readiness](docs/SCIENTIFIC_BASELINE_AND_VALIDATION_READINESS.md)。

| 阶段 | 科学结论与限制 |
|---|---|
| Phase 2A | 经典prefix同化与物理forecast对接；89/100区间欠覆盖，PARTIALLY_QUALIFIED |
| Phase 2B | 指定隔离CPU参数belief基准约18.5×提速，但精度不如MAP；固定噪声下A覆盖最低76%、未来D40最低55%；神经不确定性与OOD未可靠校准 |
| Phase 2C | K2有上述有限结构失配收益；参数失配应优先传统标定，matched负对照保留退化 |
| Phase 2D | 160科学案例，40matched未错误激活K2；结构6/8改善（75%），未达到80%门槛；B8未证明优于B7 |
| Phase 2E | 40独立世界、204观测臂；结构Fisher设计仅1/8达到收益门槛，区分度设计0/8，不能宣称可靠识别K2 |
| Phase 2F | Development初态信息改善；疲劳结构律信息不足，PHASE2F_PILOT_PARTIALLY_QUALIFIED |
| Validation readiness | VALIDATION_READINESS_PARTIALLY_QUALIFIED；未运行正式100世界验证，健康迁移/sensor/联合nuisance与预算仍需审阅 |

## 安装与快速使用

要求Python ≥3.11。基础依赖NumPy；`mechanics`增加SciPy；`verification`增加Matplotlib；`test`增加pytest；`neural`为可选PyTorch。实际下限见[pyproject.toml](pyproject.toml)，并非完整锁定环境。发布回归环境Linux/Python3.13.12、NumPy2.4.4、SciPy1.17.1。

```bash
python -m venv .venv-numerics
source .venv-numerics/bin/activate
python -m pip install -e '.[test,verification]'
PYTHONPATH=src python -m pytest -q -W error
```

未安装PyTorch时神经测试会报告SKIP，不能宣称完整514回归。完整回归使用独立神经环境：

```bash
python -m venv .venv-neural
source .venv-neural/bin/activate
python -m pip install -e '.[test,verification,neural]'
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q -W error
```

CUDA不是必要条件，历史正式神经训练没有可用GPU。以下三个CLI无需训练，输出必须是新目录：

```bash
FATIGUEDIC_DEMO=$(mktemp -d /tmp/fatiguedic-demo.XXXXXX)
PYTHONPATH=src python -W error scripts/verify_phase1a.py \
  --config configs/phase1/tri3_patch_test.toml --output "$FATIGUEDIC_DEMO/tri3-patch"
PYTHONPATH=src python -W error scripts/verify_phase1b.py \
  --config configs/phase1b/uniform_damage.toml --output "$FATIGUEDIC_DEMO/uniform-fatigue"
PYTHONPATH=src python -W error scripts/verify_phase1c.py \
  --config configs/phase1c/uniform_confounding.toml --output "$FATIGUEDIC_DEMO/uniform-identifiability"
```

每个输出含summary、复制配置、日志；按算例还有CSV/NPZ数值场及PNG/SVG。疲劳例保存真实力学/材料轨迹，反演例保存灵敏度和等价解。可辨识性诊断PASSED表示正确发现退化，不表示恢复唯一真损伤。历史长统计矩阵、训练和正式100世界测试不由这些命令启动。CI分数值与显式PyTorch任务，远程CI结果需单独检查。

## 项目结构

```text
src/fatigue_dic_rl/
  core/ geometry/             # 显式场、状态、非结构网格与积分点契约
  mechanics/ materials/       # Tri3与标量损伤本构
  fatigue/                    # 比例循环、历史、hazard与不可逆演化
  observations/ inverse/      # FE-DIC、协方差、Fisher、TRF
  inference/                  # 经典与可替换神经belief接口
  networks/ training/         # 可选图/GRU、合成训练
  correction/ evaluation/     # 低维修正、信息/设计诊断与科学评价
configs/ scripts/ tests/      # 冻结配置、CLI、完整回归
docs/                        # 原科学报告、轻量图与证据索引
```

## 科学范围与限制

首版不包含经验证的3D、塑性、接触、裂纹路径、真实疲劳寿命/RUL或高维自由损伤后验。局部损伤网格敏感性、材料/载荷/配准混淆、相关或聚集DIC和结构OOD失败均保留。健康D0已知不等于健康模量能无误迁移；神经confidence不等于经过校准的物理后验。测试通过和Git发布不会提升历史科研资格。**正式100世界、结构修正扩展及RL继续NO-GO。**

## 复现、科研资产与引用

[科学证据索引](docs/reproducibility/SCIENTIFIC_SOURCE_EVIDENCE.json)将两个实际日期合并式Git基线关联到原run、源码及恢复包SHA256；不伪造过去逐阶段提交。[图片出处](docs/assets/readme/ASSET_PROVENANCE.md)和[绘图代码](scripts/plot_readme_figures.py)保留准确数值摘录及原数据哈希。原报告中的`results/`链接对应外部科研资产，不在普通Git内分发；完整外部备份/公开下载位置尚未确认，不能把本机保留当作长期异地备份。

完整FE场、优化历史、训练权重和历史归档均保留本地，通过外部manifest关联，不上传整个results或checkpoint。原PARTIALLY_QUALIFIED、失败和NO-GO报告保持不变。[发布审计](docs/GITHUB_PUBLICATION_REPORT.md)。

代码与项目自生成图片采用[MIT License](LICENSE)，没有使用未经许可的论文图片。尚无已声明的论文DOI；引用仓库并标明实际commit/tag，可采用：

```bibtex
@misc{fatiguedicrl2026,
  author = {FatigueDIC-RL contributors},
  title = {FatigueDIC-RL: DIC-Observed Physics-Guided Fatigue Damage Identification and Evolution},
  year = {2026},
  howpublished = {\url{https://github.com/lbd-hfut/FatigueDIC-RL}},
  note = {Research software; cite the source commit or scientific baseline tag}
}
```
