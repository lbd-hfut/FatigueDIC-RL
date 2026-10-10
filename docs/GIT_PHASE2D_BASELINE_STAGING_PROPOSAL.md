# Git baseline staging proposal — Phase 2D

本文件仅提出人工分阶段冻结方案，不执行 git add/commit/push/merge/reset/clean。
当前 HEAD `dc239d1b7b6dd1e44d93e71bdcccccc0f12e8bc9` 是初始提交；不能作为完整验证版本。
Phase 2D 开发前 263 文件 manifest 的 canonical SHA256 为
`13d4e1b0679b56cfbcf397a0fe99b25a26ac0616434fc10890ed684a66d58456`，与 Phase 2C 最终源码一致。

建议先在独立临时目录解包开发前轻量源码，逐项核对 SHA 与恢复 patch；不得覆盖现有工作区或历史结果。
然后人工审核以下逻辑分组，每次先完整检查 staging diff，再运行该组新增及完整回归：

1. Phase 0–1B：基础契约、Tri3、材料与 fatigue solver、配置、测试、报告、pyproject、LICENSE 与 CI。
2. Phase 1C–2A：不规则 DIC、完整 covariance、经典 TRF/Fisher、belief/forecast、隔离测试与科学 CLI。
3. Phase 2B：可选 torch 路径、图/GRU、训练和评估程序、配置、接口文档及未校准的失败结果说明。
4. Phase 2C：固定低维 correction、经典/特权监督对照、恢复审计、全部测试、科学配置和正式部分资格报告。
5. Phase 2D：参考 law families、可辨识性/gate/selector、冻结数据方案、鲁棒性与长期预测评估、测试、正式报告。

这些是历史科学阶段的整理顺序，不建议伪造过去日期或将当前混合代码冒充当时的 commit。
应在待提交说明中记录“从已验证未提交快照整理”，附来源 manifest 和实际检查日志。
README、配置、测试、脚本、接口/科学报告应纳入版本控制；报告中的相对结果链接应由外部资产目录配套提供。
`src/*.egg-info` 为派生安装元数据，保留快照证据但通常不进入正式版本控制。

大型 `results/`、训练 checkpoint、NPZ 轨迹、tar、完整优化日志、缓存和科学图片建议保留在外部科研资产存储。
为每批资产保存不可变 manifest（相对路径、字节数、SHA256）、source/config hash、种子、依赖版本、命令、状态与恢复说明。
原始失败、未收敛和所有 seed 必须保留，不能只归档成功或最佳模型。可将小型资产索引及 qualification 摘要纳入 Git。
外部资产迁移需要另行授权；本轮不移动、删除或改写这些文件。

最终提交/推送仍由用户决定。建立新完整基线后再考虑只读标签引用；当前不会执行任何 Git 写操作。
