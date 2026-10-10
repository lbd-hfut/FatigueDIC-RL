# Phase 2C Git baseline freeze recommendation

HEAD仍为dc239d1初始提交。工作区中的科研基线远超过HEAD；禁止用reset/clean准备提交。
本阶段没有执行任何Git写操作。旧报告、checkpoint与结果保留在原位。

建议维护者先在独立恢复目录根据Phase2B交付manifest核验Phase0–2B基线，执行完整pytest与37个历史科学CLI，
复验冻结2B checkpoint及qualification限制。由维护者人工review后决定保存经典物理、序贯估计与神经估计的分阶段commit/tag。
随后独立reviewPhase2C低维修正、固定实验契约、训练信息权限、Fisher弱方向、失败与RL Go/No-Go依据，再决定增量保存。

应进入Git：src/tests/configs/scripts/docs、工程元数据、README、LICENSE、CI、小型固定reference与manifest。
外部科研资产：全部逐案例DIC/truth NPZ、CPU/GPU checkpoint、seed级指标/历史、图表、长日志、tar与重复运行结果。
使用不可变run ID与逐文件SHA256关联源码/config/environment；备份校验完成前不移动、删除或忽略唯一资产。
本次轻量恢复tar排除results、历史tar与模型权重，incremental patch相对开发前源码快照，不能错误地视为相对initial HEAD的完整科研diff。

这些是人工冻结建议；本轮不创建commit、push、merge或标签。
