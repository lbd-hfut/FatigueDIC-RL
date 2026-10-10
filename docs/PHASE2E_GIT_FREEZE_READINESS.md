# Phase 2E Git baseline freeze readiness — Phase 2F Part A

结论：**Phase 2E 的 314 文件源码集合可以完整恢复，具备供用户审阅的基线冻结条件。Git 尚未冻结。** 本轮仅进行了只读 Git 审计、归档恢复与清单生成；没有执行 add、commit、tag、push、merge、reset 或 clean，未修改 `.gitignore`、旧报告或历史资格。

## 已验证源码与 Git 的关系

branch `feature/phase0-architecture`，HEAD `dc239d1b7b6dd1e44d93e71bdcccccc0f12e8bc9` 是 initial commit。HEAD 只跟踪 `.gitignore`、`LICENSE`、`README.md`；README 已修改，其他已验证科研源码主要未跟踪。因此 HEAD 不能代表完整已验证版本。[实际 Git 状态及环境](../results/phase2f/20261010T010645Z/baseline/)、[审计结论](../results/phase2f/20261010T010645Z/baseline/audit.json)。

本轮开发前工作树的 314 个文件与 Phase 2E final manifest **逐文件 SHA256 相同**。canonical 完整源码 SHA256（排序的 path→SHA256 映射，canonical JSON，分隔符为逗号/冒号）：

`0f98b2593cf9f73c6b03eef462f736aeaab52cf9720f3cd986fca9224e9a2ae5`

Phase 2E 最终源码归档 SHA256：

`e5a6e140c6b3e6ae515b5ec03ceaf1b3d982eb2d2b29757ad34413571690214e`

final manifest SHA256：`9a8de2025b36c692ef539da965feb172d5a6953b71711caadf6a7385ab599d54`；incremental patch SHA256：`7b3f86049092483e92ead7139aa160d4b08c21c45df44d31bbc6b4312d372aaa`。报告文本与 manifest 不互相自引用计算哈希；最终资产哈希由外部清单提供。

## 独立恢复验证

实际执行两条独立恢复路径，均只操作新建空目录：

1. 将 `results/phase2e/20261009T061530Z/final_source_verified/source_final.tar.gz` 解包到 `/tmp/fatigue-phase2e-audit-archive-dg45e4rs`，逐文件检查 314 个哈希，全数一致。
2. 将 **Phase 2E 同一 run** 的 `baseline/source_predevelopment.tar.gz` 解包到 `/tmp/fatigue-phase2e-audit-patch-75975sft`，应用同一 run 的 `final_source_verified/incremental.patch`，逐文件检查最终 manifest，全数一致。[实际 patch 日志](../results/phase2f/20261010T010645Z/baseline/independent_patch_restoration.log)。

旧 [RECOVERY.md](../results/phase2e/20261009T061530Z/final_source_verified/RECOVERY.md) 的 `../baseline/` 解析到 Phase 2E run，**没有误引用 Phase 2C 归档**。它记录的旧临时恢复目录名包含 `fatigue-phase2c-restoration`，这是复用脚本留下的误导性临时前缀，不能作为 Phase 2C 来源证据，也不需要存在该目录才能恢复。原说明保留；本文件及新 [恢复说明](../results/phase2f/20261010T010645Z/baseline/RECOVERY.md) 给出澄清。

归档仅含源码、测试、配置、文档和项目根文件；没有 results、旧 tar、checkpoint 或缓存递归打包。开发前源包是已验证 Phase 2E final archive 的独立副本，便于 Phase 2F 增量恢复。

## 建议纳入 Git 的精确集合

以下是 **Phase 2E 完整基线**；本轮 Phase 2F 新增文件另作后续增量，不混入旧 Phase 2E 的源码哈希。

| 类别 | 文件数 | 建议 |
|---|---:|---|
| `src/` | 127 | 全部基础契约、FEM、fatigue、observation、inverse/inference、可选 neural、correction/design 路径 |
| `scripts/` | 56 | 全部验证、训练、重放、评估及源码导出脚本 |
| `tests/` | 49 | 全部原单元、集成、科学测试，不删失败控制或弱化断言 |
| `configs/` | 47 | Phase 0–2E 配置、种子、加载及冻结门槛 |
| `docs/` | 30 | 原阶段正式报告、follow-up、恢复/基线方案；保留资格限制 |
| 根文件及 CI | 5 | `.gitignore`、LICENSE、README、pyproject.toml、`.github/workflows/tests.yml` |

[完整 314 路径清单](../results/phase2f/20261010T010645Z/phase2e_submit_paths.txt)、[逐文件 SHA 和预期 diff](../results/phase2f/20261010T010645Z/git_expected_source_diff.json)。相对当前 HEAD，预期 **311 新增文件、1 个修改 README、2 个原文件不变**。这是根据 HEAD blob 与完整文件内容推导的审阅集合，没有写入 index，也不是当前目录所有未跟踪文件的盲目 staging。授权前应重新核对源码/测试与依赖导出，并审核尚未版本控制的完整集合。

## 排除项和资产关联

排除全部 `results/`（逐帧 DIC、FE 场、拟合历史、失败案例、coverage 原始数据）、checkpoint、archive、旧 tar/zip、缓存、虚拟环境和本机私有配置。历史资产共 241352 个文件的大小/mtime 清单已保存，另核验 161 个 checkpoint 的 SHA；Phase 2E delivery ledger 的 49916 资产 SHA 实际全量核验通过。[大文件和总大小](../results/phase2f/20261010T010645Z/historic_asset_sizes.json)。大小/mtime 检查不是旧阶段所有数值资产的内容哈希证明；完整内容校验的本轮范围明确为 Phase 2E delivery ledger 与 checkpoint。

建议在普通 Git 保存精简 evidence index，关联阶段名、不可变 run ID、source/config SHA、原资格、环境导出及外部 ledger SHA。源包与数值资产的恢复用途不同，不能把只恢复源码称为恢复了所有观测或 checkpoint。Phase 2E evidence run 为 `20261009T061530Z`；delivery ledger SHA 为 `c6fdd2f81f8ca7b2e9e2c1fba6ddaf0fe49176d80e60ea68f8ce217146c50cde`。复制外部科研存储后应逐文件重新校验，保留失败结果，不改原 run。精简 index 可引用外部完整 ledger；本轮未搬迁或删除资产。

## `.gitignore` 建议（未实施）

当前 ignore 已覆盖常见 Python 缓存与环境，但**没有排除 results**。建议审阅后增补：

```gitignore
results/
checkpoints/
archives/
*.tar
*.tar.gz
*.tgz
*.zip
*.pt
*.pth
*.ckpt
```

不要无差别忽略 `.json`、`.csv`、`.npz`；小型可审阅 fixture、配置及复现清单可能需要 Git。大型数值资产优先保存在外部不可变科研存储。确有分发必要的小量 checkpoint 可用 Git LFS，精确豁免对应 ignore 并记录 `.gitattributes`；不要自动改写现有历史或把全部 results 纳入 LFS。

## 冻结步骤建议及必须保留的结论

建议先保存当前完整可恢复基线，随后按经审计的阶段增量保存；不要凭推测构造不存在的每阶段提交历史。分批审阅 solver/contracts、inverse/inference、可选 networks、correction/design、测试、配置和报告，但最终确认完整集合可恢复并回归通过。本轮原有完整 pytest **489 passed**，没有重新训练 Phase 2B 或重跑所有旧科学优化矩阵。

应保留 Phase 2A–2E 的 `PARTIALLY_QUALIFIED`、Phase 2A 原 89/100 coverage、Phase 2B 校准/OOD 限制、Phase 2C 高 SNR 限制、Phase 2D 6/8 未达 80% 和 Phase 2E Fisher 1/8 / discrimination 0/8；**RL NO-GO 不变**。新 Development 结果不改旧资格。

供授权后使用的推荐消息：`chore: freeze verified Phase 0–2E scientific source baseline`；推荐 annotated tag：`scientific-baseline-phase2e-partially-qualified`，tag 描述应关联上述 canonical SHA、证据 run 和 NO-GO。若同时提交 Phase 2F，应另立 `research: add development-only fatigue-law information pilot`，不得将它称为 Phase 2E 的历史源码。**本文件仅为冻结预检方案，不授权任何 Git 写操作。**
