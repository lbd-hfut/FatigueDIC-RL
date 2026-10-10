# World models — 尚未实现

物理代理模型应复用 MechanicsSolver 和 FieldData 的输入输出契约，并报告前向误差、
域外检测与多步误差。可微代理还需要独立梯度验证，不能因可训练就替代科学验证。
本轮没有 surrogate 或生成模型。见 [验证计划](../../../docs/06_TRAINING_AND_VALIDATION_PLAN.md)。
