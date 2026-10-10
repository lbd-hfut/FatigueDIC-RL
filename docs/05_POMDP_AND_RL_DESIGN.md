# 05 — POMDP 与 RL 研究设计

## 两种演化不能混淆

真实物理轨迹满足 \(s^*_{k+1}\sim p^*(s^*_{k+1}\mid s^*_k,\ell_k)\)，
观测满足 \(o_{k+1}\sim p(o\mid s^*_{k+1},\ell_{k+1})\)。
物理加载 \(\ell\) 是已知条件，模型修正动作 \(a\) 不进入真实试件转移。

模型预测轨迹则为
\[
\widehat s_{k+1}^-=\widehat F(\widehat s_k^+,\ell_k,\Delta N_k,a_k),\quad
b_{k+1}=\mathcal A(b_{k+1}^-,o_{k+1}).
\]
\(\mathcal A\) 是 belief update，\(\widehat F\) 是估计演化，二者职责不同。
在 POMDP 的估计模型决策问题中，动作控制内部预测/校正，不能据此声称控制了疲劳过程。
若状态特征缺少加载历史、内部变量或参数后验，模型环境未必满足 Markov 性。
需要保留历史或递归状态并进行验证，不能直接宣称一个 damage 图就是完备状态。

## 未来状态、动作与策略

图状态支持 FE 节点/单元邻接及积分点到图的显式映射。
输入候选为估计 damage、已求解应力/应变、载荷、位移残差与观测质量，
并通过 GNN/GRU 表达空间和时间信息；没有固定 H×W 限制。
图节点数量不自动等于积分点或 DIC 点数量。
Phase 0 只用 MechanicsState(role=BELIEF) 作为点估计接口，未实现网络或概率 belief。

策略 \(a_k=\pi(b_k,o_k)\in\mathbb R^8\)；Action Decoder/Basis 产生
damage/hazard 支撑上的 \(Ba\)。当前 `LatentAction` 与带空间引用的 basis FieldData
仅定义形状、单位和有限性；尚未训练 Decoder/Actor，也未固定 action 的额外幅值边界。
指数修正在 [00](00_SCIENTIFIC_FORMULATION.md) 中定义；它维持非负驱动，
但无法激活物理模型的零 hazard，且动作基底可能无法表达局部损伤模式。

## Reward 与多步 Critic

候选 reward 为
\[
r_k=-\mathcal L_{DIC}(\widehat u_{k+1},o_{k+1})
-\beta_a\|a_k\|^2-\beta_r\mathcal R(\widehat D_{k+1}),
\]
其中 mask 必须取测量与预测有效区域的交集，covariance 的处理必须明确。
可以增加经验证的平衡/能量残差 soft losses；硬物理约束仍由 transition 检查。
这个形式只是设计候选，权重、折扣、终止条件和失效阈值尚未确定。
future observations 只能用于训练/事后评价 reward，在线策略不能看到未来信息。

Critic 学习
\[
Q^\pi(b_k,a_k)=\mathbb E\left[\sum_{j=0}^{H-1}\gamma^j r_{k+j}
\mid b_k,a_k,\pi\right],
\]
通过未来多步模型演化评价修正动作；它不是直接预测剩余寿命的模型。
RUL predictor 要在指定未来载荷和失效定义下预测首次通过阈值的循环数分布。
即使 Critic 与 RUL 都使用长时间信息，其监督目标、单位与校准要求不同，不能互换。

## 监督 baseline 与真正 RL

监督 baseline 可拟合已知 hazard/参数/专家修正，或最小化位移反演与多步预测误差。
多步监督损失、可微优化、behavior cloning 均不会自动成为 RL。
真正 RL 还需要明确的可行动作、动作影响的模型状态转移、reward、
多步价值/Bellman 目标、决策策略评价和有效的动作数据支持。
固定实验疲劳轨迹没有天然记录“网络内部修正动作”，因此不能直接当作
TD3+BC 的有效离线 action dataset。
需先通过经验证的模型环境或独立校正策略构建有 provenance 的转移和动作数据。

TD3+BC 将行为克隆正则与 TD3 策略学习结合，是未来可考虑的离线基线。
该方法的公开结果不能证明它对本问题有效，参见
[Fujimoto 与 Gu 的原始论文](https://arxiv.org/abs/2106.06860)。
本轮没有 TD3、SAC、PPO、Actor、Critic、replay buffer 或训练循环。

## 为什么 RL 仍是研究假设

1. DIC 到 hidden damage 的可识别性尚未成立，价值学习无法补救结构性非唯一。
2. 模型动作改善未来预测的因果关系与多步收益尚未验证。
3. 离线动作覆盖不足会造成值函数外推误差，当前没有合格的 action dataset。
4. 模型误差、奖励设定和正则可能让策略补偿错误边界/参数，而非真实损伤演化。
5. 纯物理、参数校准、数据同化和监督多步 baseline 可能已经足够。

只有在可信前向模型、观测模型、数据隔离与公平 baseline 成立后，
RL 对独立试件/几何/载荷上的多步收益才值得检验。
如果没有优势，应保留 physics-only 或更简单的监督/同化方案。
