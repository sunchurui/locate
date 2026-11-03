## Current decision2026-11-02: global endpoint micro P/R/F1

The user explicitly authorized replacing ALL primary results with audited variant E.
This supersedes earlier exact-author-reducer instructions in the history below.
Protocol rex_matching_global_endpoint_micro_v1. Per IoU=.50:.05:.95 pool original
Rex native TP/FP/FN across images AND categories. P=TP/(TP+FP), R=TP/(TP+FN).
Zeros remain. meanIoU F1=harmonic(mean(P_t),mean(R_t)), NOT mean(F1_t). No PR
extraction, zero dropping, GT-positive filtering or fabricated confidence. This
is a PROJECT metric retaining Rex matching, NOT original Rex F1/GroundingPI GAM.
Similar subset/full scores are an observation, never a requirement of scoring.

Code micro_metrics.py/rescore_micro.py/official_eval/build_counts_engine.py.
Old official_metrics/unranked APIs import current code for live eval. Author
reproduction remains legacy_rex_metrics.py. Current micro_metrics.json,
set_metrics.json and evaluation/micro_summary.*. Keep historical author results,
but never mix their curves/deltas with current metrics. Read EVALUATION_PROTOCOL.md.
Native matching/crowd/ignore/maxDets100 remain. Models/data/checkpoints untouched.
No new recurring Codex monitoring. Historical decisions below do not override this latest user authorization.

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


## 2026-11-02：第三轮 SFT 基座的新 RL 对照

用户选定 `rex_id_sft_outputs_5epoch_b64/checkpoint-5547`（epoch3）为新 RL 的共同起点。
整段奖励和响应内标准化物体信用两版的 policy 与 frozen reference 都从该 SFT 初始化。
两版均 batch32×accumulation8、每次32图×8回答、固定train10000/val256、seed20260812、
LR5e-7、1epoch/313updates。旧 RL 仍归属旧单轮 SFT，不能混作这组的基线。
用户在本聊天确认以最新项目记忆的 Rex 官方 P/R/F1 为准，撤回AP_set；
所有查询类输出交给原版计分核心，同分处理/maxDets等原作细节须保留说明。
脚本 `rex_id_epoch3_rl/`，注册表 `epoch3_rl_jobs.json`。
训练207728/207730，配套完整评测207729/207731，第三轮SFT完整基线评测207732。
100/200/300步做同步val256生成评测，50步保存检查点，不新建周期Codex自动任务。


## 2026-11-02：训练轮次比较必须并列同一范围的旧基线

旧原ID完整5000张F1mIoU41.84，独立val256为49.15，旧全量预测切出同256为49.62。
新五轮SFT第一轮val256为47.55，低于旧同范围基线约1.6点，不是大幅提升。
旧有效batch8/14786updates，新batch64/每轮1849updates，五轮总调度，非相同配置重跑。
原版F1的50份随机256子集均值50.39、范围46.16~55.07，存在小样本乐观偏差。
作者safe_mean排除零项；子集IoU.95有31个有GT类别零召回被跳过。不能据此解释全部差距。
以后训练曲线同表列同范围baseline。详见 audits/SFT_SUBSET_COMPARISON_20261102.md。


## 2026-11-02：原版Rex F1根因已经量化

详见 audits/REX_F1_ROOT_CAUSE_20261102.md。原版有precision变量跨类别/IoU沿用的缺陷，
safe_mean跳过零项，且precision从插值PR数组提取。隔离副本复现，原版源码未改。
同一原版匹配按零项保留、修复重置、终点宏平均、总体计数依次诊断，子集-全量
差距7.7820→7.0254→5.4728→2.1565→-0.0300；步骤有交互，非唯一因果归因。
原版官方转换和评测CLI已直接执行；相同转换内容的JSON/TSV六组PR数组完全一致。
不能再用“普通抽样波动”概括；不得擅自把诊断改为新的主指标。


## 2026-11-02: M30 order curriculum and M45 instance-selection repair

User authorized these two only; additional SFT ideas remain proposals in
sft_curriculum/NEXT_SFT_CANDIDATES.md. COCO max27/class,90/image means original
50/500 curriculum thresholds do not divide this dataset. Frozen training-only
fallback75th-percentile threshold5; phase1 first59136 sparse images, phase2
remaining30249 sparse+28902 dense shuffled; every118287 image exactly once.
Paired random-order control, seed42, fresh2B+rank16 LoRA/original Rex IDs,
4x2 effective8,14786updates, same LR1e-4 and fixed visual/context budgets.
Train207964/207966, eval207965/207967,80GB only; preflight207977 passed actual
images/tokenizer/collator/order.2 real smoke updates are required on allocation
before formal training. At submission both train jobs pending, not training yet.
M45 cache replay207968 completed256 images with frozen support/NMS choices;
primary T.7 author-vote support16+NMS.7 F1mIoU25.30 vsbaseline35.48, despite
improvement from unfiltered6.93. Member-box extent diagnostic T.7 support16
39.65, T.5 support16 40.20; support32/T.5 exploratory42.49, NOT independent
confirmation. Support/NMS are inference decisions, not evaluation confidence.
No new model generation or GT-guided output choice. Keep original results.


## 2026-11-02: four additional source-grounded SFT ablations authorized/submitted

User approved image-mean loss, projector LoRA, actual higher resolution, and
class-balanced whole-image sampling. Code/config/manifests under sft_followups/.
Paper evidence: Nemotron3 Super sec3.1 stage2 sample-mean equation; LLaVA
projector+language training (our projector is low-rank) and Swift aligner-LoRA;
WeVisOne stage2 resolution factor isolated; pinned Detectron2/LVIS RFS reused.
These are component ablations, not reproductions of complete paper recipes.
All compare against M30 random control207966 (native full eval207967); same
fresh Qwen3-VL-2B, seed42, rank16/alpha32, original Rex IDs,118287 draws,
14786 updates, effectivebatch8, existing LR1e-4/cosine schedule. Separate runs.
Image-mean uses image count over full GA window; real Trainer micro3/acc1 vs
micro1/acc3 test max parameter difference2.98e-8 incl partial last window.
Projector extends8 merger linears (main+3 DeepStack), freezes vision blocks;
shared language/token initial trainables restored bit-exactly after extension.
Higher resolution uses actual patch16*merge2 pixel-area conversion; real images
300/260 native visual tokens become1036/1080; supervised targets unchanged.
Primary generation comparisons use native inference; highres model also gets
highres inference and control gets matched highres eval208050.
Original RFS t=.001 repeats0 COCO images. Predeclared training-frequency-median
fallback t=.02605527; direct original stochastic-repeat/shuffle stream,
first118287 draws,106196 unique images, full GT retained on each draw.
Train/eval: image_mean208042/208043, projector208044/208045,
highres208046/208047, class_rfs208048/208049. CPUprepare208033/preflight208026
completed. At submission all4 train jobs PENDING, not actually training yet.
On>=80GB allocation do2 real gradient smoke steps before fresh formal training;
loss validation every1000, fixed256 generation5000/10000, full5000 afterok.
Current micro protocol unchanged, no AP/confidence ranking, no checkpoints
removed, no new Codex recurring monitor. Jobs/source locks in package.


## 2026-11-03 09:15 CST: full SFT comparison and follow-up launch recovery

Both full5000 generation jobs completed and were rescored with current micro protocol:
epoch3=40.08 F1mIoU (P42.00/R38.33), epoch5=38.81 (P39.51/R38.14).
Old Rex original-ID full5000=35.51; epoch3 gain+4.57, epoch5+3.30.
This is fresh five-epoch batch64 recipe vs old batch8, not an epochs-only ablation.
M30 paired launches replaced by pathfix208920/208921 and208922/208923.
Follow-ups old highres208046/class_rfs208048 failed before training because
the submitted Slurm snapshot sourced the venv under the submission checkout.
On-disk scripts were already fixed but pending snapshots208042/208044 were stale.
Replaced only those follow-up jobs and associated dependent evaluations:
image_mean208925/208926, projector_lora208927/208928,
highres208929/208930, class_rfs208931/208932; control highres eval208933
after random_control208922. All PENDING at this check, no training steps.
Explicit EXPERIMENT_ROOT, absolute activation path, bash syntax/venv existence
checked. Same source/config/budget, training>=80GB, no model/data/checkpoint
removal, no recurring automation. sft_followups/jobs.json retains history.
Epoch3 RL207728/207730 and M38 retry207004 still PENDING.


## 2026-11-03: user-requested M45 support48 cache extension

Same frozen256 images and64 T=.5 generations, member-box extent/NMS.7.
Support16 P42.68/R37.99/F140.20,1754boxes; support32 P53.46/R35.25/F142.49,
1273boxes; support48 P60.58/R32.86/F142.61,1030boxes. Baseline35.48.
48 improves only0.12F1 over32; F1@.50 drops64.38 to61.76, @.95 rises9.11
to9.65. This threshold was requested after seeing earlier results and is
post-hoc/exploratory, not independent confirmation or full5000 performance.
No new inference or model changes. Script inference_ablation/m45_support48.py;
report reports/m45_support48_v1.md; exact16/32 predictions match earlier cache.
