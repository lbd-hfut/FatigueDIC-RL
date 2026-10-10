# 00 — 科学问题与数学表述

## 隐状态与观测

试件真实物理状态记为
\(s_N^*=(D^*(\mathbf x,N),\mathbf u^*,\boldsymbol\varepsilon^*,
\boldsymbol\sigma^*,z^*,\theta^*,\ell_N)\)，其中 \(z^*\) 是内部变量，
\(\theta^*\) 是材料/几何参数，\(\ell_N\) 是外部加载条件。
疲劳循环数可以连续化，记录单位为 cycle。

真实损伤演化为未知的 \(s_{N+\Delta N}^*=F^*(s_N^*,\ell,\Delta N)\)。
它由试件与外部加载决定，网络内部动作 \(a_k\) 不作为其控制输入。
未来若做主动试验设计，改变外部加载属于独立的问题，不属于当前 latent correction。

正式观测主要为
\[
o_k=(\mathbf q_k,\mathbf y_k,m_k,c_k,\Sigma_k,N_k,\phi_k,\ell_k),\qquad
\mathbf y_k=H(\mathbf u^*_{N_k})+\eta_k.
\]
\(\mathbf q_k\) 是任意观测坐标，\(m_k\) 是有效掩膜，\(c_k\) 是质量置信度，
\(\Sigma_k\) 是可选位移协方差。质量分数不是已校准的概率；本轮不定义两者的转换。
二维模型需已完成像素到试件坐标/长度单位的标定，三维离面运动不在当前接口内。

估计器构建 \(b_k=p(s_{N_k}^*,\theta\mid o_{0:k},\ell_{0:k})\)。
Phase 0 的 `StateRole.BELIEF` 只保存点估计，不实现概率后验或不确定性量化。
`GROUND_TRUTH`、`BELIEF`、`PREDICTION` 在接口中显式区分。

## 力学关联

未来准静态小应变示例满足
\[
\nabla\!\cdot\!\boldsymbol\sigma+\mathbf f=0,\quad
\boldsymbol\varepsilon=\tfrac12(\nabla\mathbf u+\nabla\mathbf u^T),\quad
\boldsymbol\sigma=\mathcal C(D,\boldsymbol\varepsilon,z;\theta),
\]
并带有位移和牵引边界条件。FE solver 给出损伤假设与位移观测之间的前向关系。
本轮没有组装或求解这些方程。

示例本构 `IsotropicPlaneStress` 使用 \(\mathbf C(D)=(1-D)\mathbf C_0\)，
该关系属于这个具体模型。core 的 damage 字段不决定本构关系。
各向异性张量损伤、弹塑性与相场将需要自己的字段、历史变量和演化契约；
当前标量 \(D\in[0,1]\) 不是对它们的通用表示。

## Damage rate 与 hazard

定义 \(D_{\max}\in(0,1]\)，\(\lambda\ge0\)，其单位为 cycle\(^{-1}\)。
\[
\frac{dD}{dN}=(D_{\max}-D)\lambda.
\]
**damage rate** 是 \((D_{\max}-D)\lambda\)，**hazard** 是 \(\lambda\)，
两者不可用相同含义调用，即使它们的量纲均可写成 cycle\(^{-1}\)。
靠近饱和时，有限 hazard 仍可非零，但 damage rate 趋于零。

在一步内固定 hazard，分离变量或直接解线性 ODE 得
\[
D_{k+1}=D_{\max}-(D_{\max}-D_k)e^{-\Delta N_k\lambda_k}.
\]
因为指数因子属于 \([0,1]\)，有
\(D_k\le D_{k+1}\le D_{\max}\)。当 hazard 或步长为零时，损伤保持不变。
有限常 hazard 下损伤渐近达到 \(D_{\max}\)，不保证有限循环内严格达到 1；
失效/RUL 需要另行规定物理失效阈值。

实现使用等价表达
\(D_{k+1}=D_k+(D_{\max}-D_k)[-\operatorname{expm1}(-\Delta N_k\lambda_k)]\)，
减小极小 exposure 的数值消减。输入先严格验证，再限制浮点舍入造成的越界。
有限极大输入乘积溢出时使用指数的饱和极限。

这对冻结的 hazard 是解析解；若 hazard 随损伤、应力或循环内相位变化，
冻结近似不是完整耦合系统的解析解。后续必须开展步长收敛与重求解研究。

## 模型修正与约束

未来动作修正估计模型：
\[
\widehat\lambda_{\rm hybrid}=
\widehat\lambda_{\rm physics}\exp[\alpha\tanh(Ba)],\quad a\in\mathbb R^K,
\quad K=8\text{（初始设计）},\ \alpha\ge0.
\]
\(B\) 的每一行显式对应 damage/hazard 支撑点。修正因子位于
\([e^{-\alpha},e^{\alpha}]\)，不会改变 hazard 的非负性；零物理 hazard 无法被这种
乘法修正激活。这是需要检验的建模限制。

Hard constraints：损伤界、不可逆、非负 hazard、非负步长和合法数据契约；
违反时应拒绝执行。Soft losses：位移残差、平衡残差、空间正则化、能量/耗散惩罚；
它们不能替代硬约束。损伤单调本身不证明热力学一致性；还需验证自由能与耗散。

## Belief update 不是物理演化

先验预测可用物理演化提高循环数，再通过观测修正 belief。
新数据可能让估计 damage 下降，这是估计修正，不应冒充真实材料的损伤恢复。
DamageTransition 强制单条演化轨迹不可逆；估计器与整段轨迹平滑的可容许集合应独立设计。
当前仅提供 StateEstimator 接口，不实现这些算法。

正式观测坚持位移为基础数据，是因为 DIC 位移与派生应变存在不同的噪声和空间分辨率权衡。
此处依据 [NIST DIC Challenge 2.0](https://www.nist.gov/publications/dic-challenge-20-developing-images-and-guidelines-evaluating-accuracy-and-resolution-2d)，
本项目的反演可识别性结论仍需实验支持。
