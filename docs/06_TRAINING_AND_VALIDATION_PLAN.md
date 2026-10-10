# 06 — 训练与科学验证计划

## Phase 0 的实际验证

单元测试验证数据形状/空间引用/单位、拓扑、协方差、外点 mask、非法参数、
solver 异常契约和 truth 隔离。科学测试验证冻结 hazard 的解析解/半群、
极小与极大 exposure、不可逆和界限、示例本构的正刚度/正能量、
不同尺度和朝向的线性三角形形函数再现。
集成测试组合解析仿射位移、观测映射、物理 hazard 和多个循环步长。
以上都不是 FE 方程求解或真实疲劳材料验证。

主命令：`python -m pytest -q -W error`。
Smoke：`PYTHONPATH=src python scripts/phase0_smoke.py --config configs/phase0.toml`。
有固定随机种子的科学测试可重复；最终真实输出在实施报告与 results/phase0 中保存。
CI 配置计划测试 Python 3.11/3.13；本轮本地只在 Python 3.13.12 上执行，
远程 CI 是否通过必须等待实际运行，不能从配置推断。

## Phase 1 建议顺序（本轮不执行）

1. 实现 NumPy tri3 小应变准静态 solver，显式 plane stress/strain、厚度和 SI 转换。
   先进行 patch test、刚体模态/约束、力与反力平衡、能量与网格收敛，建立参考解。
2. 定义完整 quadrature layout、区域材料映射、历史变量更新与应力到 damage 支撑的投影。
3. 加入有出处且可校准的 stress/energy-driven DamageLaw，验证循环步长、载荷相位和耦合误差。
4. 加入 DIC 标定/配准、质量/协方差处理与受控噪声试验；确认测量模型的有效分辨率。
5. 先做参数估计/变分反演/滤波等 state-estimation baseline，检查可识别性与不确定性。
6. 在已经验证的物理模型上建立小规模、可追溯 synthetic dataset；独立保存 truth。
7. 比较 physics-only、参数校准、数据同化、监督单步/多步预测 baseline。
8. 满足 RL 数据与因果决策前提后，才另行考虑 model environment 和 TD3+BC 研究。

每一步先给科学验收门槛，再扩展功能。本轮停止于 Phase 0。

## 数据分割与防泄漏

按完整 specimen/run 划分 train/validation/test，避免同一试件重叠时间窗口跨集合。
保留跨几何、材料分区、应力比、幅值、载荷谱和初始缺陷的域外测试。
所有标准化参数、基底、参数标定只用训练集合，模型选择使用 validation，
独立 test 仅作最终评价。记录随机种子、数据来源、mesh_id、空间布局和加载条件。
不得将 synthetic damage truth、未来 DIC、失效时间或测试集合拟合参数输入正式策略。
smoke 没有生成 dataset；它只是几组确定性的解析数值。

## 评价层级

| 层级 | 指标/检查 | 可支持的结论 |
|---|---|---|
| 数据契约 | shape、support、unit、mask、role | 模块输入/输出合法 |
| 数值算子 | 解析误差、patch test、网格/步长收敛 | 离散算法实现可信 |
| 物理前向 | 力平衡、能量、参考解、材料标定 | 指定模型和范围内前向可信 |
| 隐状态估计 | truth 误差、后验覆盖、敏感度、非唯一性 | 受控范围的反演能力 |
| 多步预测 | 留出载荷/试件的位移、damage 与失效指标 | 泛化预测能力 |
| RL 收益 | 相同数据预算下的 baseline/动作/价值消融 | 对定义的决策任务有额外收益 |

真实实验通常没有完整 damage truth；位移拟合良好不能单独证明隐损伤正确。
需独立测量或经验证的 synthetic truth 支持 hidden-state 评价。
对 RL 需多随机种子、置信区间、训练预算、失败案例与域外行为报告。
对 RUL 需单独定义失效阈值、未来载荷假设、删失处理和概率校准。
