# 项目记忆：COCO / Qwen3-VL-2B 定位实验

## 用户评测要求：直接遵循 Rex-Omni / GroundingPI 原作，禁止自行设计指标

用户已明确反对按生成顺序、坐标概率、投票强度人为赋分来计算排序 AP。
2026-11-02 用户进一步澄清：已有工作怎么评测，我们就怎么评测。
不得自行发明“不排序AP”。此前提出的 `AP_set=P*R` 已撤回，不能当主结果。

### 核实的原作协议

- Rex-Omni 论文5.1明确说明生成模型缺少可靠置信度，AP不适用，采用 Recall、
  Precision、F1；按类别汇总，报告IoU=.50、.95和.50:.05:.95的结果。
- 直接复用官方源码commit `6508981c1e0c3fbb2dbe7b962a4bb745005f3e2e` 的
  FastEval C++和 `evaluation/metrics/coco_lvis_metric.py` 的 `safe_mean`/`f1`。
  原作者转换脚本为所有生成框填固定 `conf=0.2`。不得替换为递减或模型分数。
- 原版实现内部仍有固定分数的同分处理、排序与maxDets，`safe_mean`只平均
  大于0的值。必须忠实记录这些细节，不能声称它完全排列不变，不能自行改成
  Hungarian或另一个微平均F1后仍叫“官方结果”。
- GroundingPI论文也报告R/P/F1，检测主指标是F1mIoU。当前官方代码commit
  `98641f0b8ed94a091f1457e0e624c49c0ebf3876` 的GAM类别路径使用逐图类别约束匹配，
  包含精确重复框去重，再平均逐图P/R后计算F1，各IoU的F1再平均。
  它与Rex FastEval的聚合实现不完全相同，不能混用名称或分数。

### 当前实验与原作提示协议的差异

我们固定实验查询全部COCO80类，已有全部输出必须保留，不能用GT偷偷过滤误检。
Rex官方 `eval_coco.sh` 则使用 `--positive_only`，原论文给模型GT类别提示。
因此当前回放把全部输出交给官方计分核心，另列官方positive-only过滤参考结果。
两个口径必须分开标注；仅回放计分不等于完成原作GT提示推理的全面复现。

### 当前入口与文件

- `official_metrics.py`：调用原版Rex FastEval及作者reducers；不报告AP/AP_set。
- `rescore_official.py`：重算全部已有预测；`rescore_unranked.py`保留为兼容入口。
- 新结果：`rex_official_metrics.json`；`set_metrics.json`仅为同一结果的兼容副本。
- 总表：`evaluation/rex_official_summary.json`和`.md`。
- 固定官方源码及许可证：`official_eval/upstream/`；编译产物不入Git。
- 排序AP、AP_set、之前自写最大匹配F1的文件均为历史诊断，保留追溯，禁止继续
  据其声称“和Rex/GroundingPI一致”。M45旧的39.94 AP/+4.50提升结论撤回。
- 不重新训练、不改已有预测来迁就指标；已有预测优先CPU重算。

### 参考

- https://arxiv.org/html/2510.12798v1#S5.SS1
- https://github.com/IDEA-Research/Rex-Omni/blob/6508981c1e0c3fbb2dbe7b962a4bb745005f3e2e/evaluation/metrics/coco_lvis_metric.py
- https://arxiv.org/html/2609.39601#S12.SS1
- https://github.com/groundingpi/GroundingPI/blob/98641f0b8ed94a091f1457e0e624c49c0ebf3876/eval/utils/detection_utils.py

### 实验条件

HPC `/home/crsun/sunchurui/COCO_Qwen3VL2B_LoRA`，模型Qwen3-VL-2B LoRA，Rex原ID，
基座 `rex_id_sft_outputs_a800_80`。COCO val5000，固定子集256，seed42。
不同图片集合不直接比较。训练GPU必须>=80GB，不删除数据/模型/检查点。
GitHub `sunchurui/locate/COCO_Qwen3VL2B_LoRA`。不要擅自恢复Codex周期自动检查。
