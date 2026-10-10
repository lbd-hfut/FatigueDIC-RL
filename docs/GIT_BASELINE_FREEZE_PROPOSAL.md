# Git baseline freeze proposal — Phase 2E

本文件是可审阅计划，没有执行 staging、commit、push、merge、reset 或 clean。
当前 `feature/phase0-architecture` 的 HEAD 仍为 initial commit；已验证工作树与科研资产必须一起引用，不能只引用 HEAD。
Phase 2E 开发前源码与 Phase 2D 最终源码逐文件一致，292 文件的 canonical SHA256：
`d37ba93feb411397cf28a41063b373b650ddcf13232764fb76d45e3d581ed9bf`。
可恢复证据位于 `results/phase2e/20261009T061530Z/baseline/`；最终交付使用独立 final source manifest。

| 类别 | 建议纳入版本控制 | 资产策略 |
|---|---|---|
| Source | `src/`、`scripts/`、项目依赖声明与根 README | 普通 Git；保证 NumPy/SciPy 基础路径不强制引入 torch |
| Tests | 全部 unit/integration/scientific pytest | 普通 Git；保留旧失败与资格限制，不删弱化旧测试 |
| Configs | `configs/phase0*` 至 `configs/phase2e/` | 普通 Git；正式运行的冻结配置另存 SHA256 |
| Scientific reports | `docs/` 的阶段报告、follow-up、恢复/资产说明 | 普通 Git；报告链接外部证据并说明其位置 |
| Reproducibility manifests | 环境 lock/export、种子、划分、冻结规则、source/config 哈希、可复现命令 | 普通 Git 中保存精简索引；巨大逐文件 ledger 作为外部不可变资产并在索引记 SHA256 |
| Large numerical artifacts | 全量 results、逐帧观测、FE 场、统计原始记录、归档 tar | 带内容哈希和不可变 run ID 的外部科研存储；保留原始失败记录及数据权限，报告中可解析链接 |
| Model checkpoints | Phase 2B/2C 已冻结的 `.pt` 等模型 | 优先外部模型资产；确需随代码分发的小量模型可用 Git LFS，记录对应训练 source/config SHA 和依赖版本 |

建议先建立**当前完整基线**，其后按经审计的阶段增量保存。因为历史源码大部分没有提交，不能据推测重建每阶段 Git 历史。先核对本轮快照、历史保护清单和最终测试证据，再由用户批准并执行提交。阶段资格文字应与当时证据一致，保留所有 PARTIALLY_QUALIFIED 和 NO-GO。

建议 staged diff 审阅顺序：基础契约/求解器、inverse/inference/observation、可选 networks/training、correction/design、测试、配置与脚本、报告与资产索引。完整源码基线应最终作为一个可重建的集合验证，不以拆分 staging 的局部测试代替完整回归。提交前在空目录恢复 snapshot，运行声明的测试和科学 CLI；只读检查 Git diff 与 manifest。一旦实际冻结，请记录新 commit 与完整 source hash 的对应关系。

建议 `.gitignore` 条目如下，仅供审阅，未写入当前 `.gitignore`：

```gitignore
__pycache__/
*.py[cod]
.pytest_cache/
.mypy_cache/
.ruff_cache/
.venv/
venv/
results/
*.tar
*.tar.gz
*.tgz
*.zip
*.pt
*.pth
*.ckpt
```

不要无差别忽略 `.npz`、`.json` 或 `.csv`：小型科学测试 fixture 和可审阅 manifest 可能应进入 Git。LFS 文件如果启用，应从相应忽略规则中精确排除，并审阅 `.gitattributes`。不要对现存大文件执行历史重写；现有科研结果、旧 tar 和 checkpoint 均保留。恢复归档必须排除 results/checkpoint/cache/旧归档，防止递归膨胀。外部存储索引至少包含 relative path、字节数、SHA256、run ID、来源 source/config hash、种子/单位/划分、保留策略和恢复步骤；仅有目录大小或修改时间不足以校验资产内容。

这份计划不授权任何 Git 写操作。
