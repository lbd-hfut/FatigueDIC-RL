# Environments — 尚未实现

未来真实/可信 synthetic physical environment 按外部载荷演化，产生位移观测；
model prediction environment 接受 latent action，只修改内部估计演化。
真实损伤不受内部策略动作控制，两种环境不能共享可写状态。
本轮不提供 Gym wrapper 或虚假 step 函数。见 [系统架构](../../../docs/01_SYSTEM_ARCHITECTURE.md)。
