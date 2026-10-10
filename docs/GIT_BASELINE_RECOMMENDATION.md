# Git 科研基线整理建议（不执行 Git 写操作）

当前 `feature/phase0-architecture` 的 HEAD `dc239d1b7b6dd1e44d93e71bdcccccc0f12e8bc9`
仅是 Initial commit。未提交工作树才包含 Phase 0–2A 已验证实现。不能通过 HEAD 单独复现。

建议人工审查后分三步正式保存：

1. 根据各阶段开发前快照、增量 patch 和 SHA256 manifest，检查 Phase 0–1C 源码、
   测试、配置、CLI、pyproject、README、CI 与正式文档；建立一个完整、可复验的经典物理基线。
2. 独立审查 Phase 2A 的协议、Prefix MAP、rank-aware covariance、时间留出测试和报告。
   保存其部分资格及原始 89/100 覆盖率，不把补充诊断追溯改写为原阶段资格。
3. 单独审查 Phase 2B 神经路径、可选 torch 依赖、数据拆分、训练/测试脚本和报告。
   明确五个训练 seed、每次运行的源码/config hash、模型权重来源和失败实验。

每步都应在独立恢复目录核对完整清单，并实际运行对应回归后由维护者决定提交/标签。
历史 snapshot 可用于 review，不能将 live 工作树 reset 回 Initial commit 后丢弃未跟踪研究资产。

应进入版本控制：`src/`、`tests/`、`configs/`、`scripts/`、`docs/`、`pyproject.toml`、
README、LICENSE、`.github/`；保留全部正式历史报告。少量固定数值 reference/manifest 可以入库。

大型派生科研资产应在外部受管理存储保留：逐轨迹 NPZ、DIC acquisition、truth labels、
GPU/CPU checkpoints、每 seed 历史、图表、长日志、tar 快照和重复运行结果。
使用不可变 run ID、逐文件 SHA256、源码/config hash 和复制校验清单关联；必要时配置
Git LFS/DVC 或机构数据存储，但本轮不安装、不移动、不删除现有资产。

不要递归保存历史 results 中的 tar 到新 tar。快照只打包源码、测试、配置、脚本、文档和工程文件，
结果留在原位并独立登记哈希。不要在尚未核验真实保存位置之前忽略或清理旧结果。

本文件是操作建议，不代表已创建 commit、push、merge、reset、clean 或标签。
