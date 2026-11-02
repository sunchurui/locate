# 无排序集合评测 / AP_set v1

2026-11-02 用户要求：全部评测不按分数排序，统计全部预测框，并获得修改后AP。
项目记忆和执行约束分别保存在 `PROJECT_MEMORY.md` 与 `AGENTS.md`。

## 定义

每张图、每个类别，在给定 IoU 阈值 t 下，预测框和非crowd GT框做一对一
匹配。首先最大化有效匹配数量，再在同数量匹配中最大化 IoU 总和。
匹配数为 TP；其他预测为 FP；没有匹配的非crowd GT为 FN。重复框也为 FP。
置信度、模型概率、输出顺序都不参与；不做 top-k、NMS、分数阈值或数量截断。

对一个类别，汇总该评测图片集合内全部 TP/FP/FN：

```
P = TP / (TP + FP)
R = TP / (TP + FN)
AP_set(class, t) = P * R
F1 = 2*TP / (2*TP + FP + FN)
```

分母为0时相应值取0。AP_set 是把全部框当作同一个工作点时的单点 PR 面积。
先对该次有GT或有预测的类别做等权平均，再对10个IoU阈值
0.50、0.55、...、0.95平均，得到 `AP_set_50_95`。
另列 `AP_set_50`、`AP_set_75`。有预测但没有GT的类别计0；两者皆无的类别不参与。

这是显式的自定义无排序指标，**不是标准 COCO 排序 AP**，不能与旧AP直接比较。
公式已向用户提出核对；如用户另定无排序AP公式，以其要求更新并再次重算。
同时保留 micro/macro P/R/F1、各类别 TP/FP/FN、crowd忽略数量。

## Crowd 和输入

先匹配非crowd GT。剩余预测如被同类别crowd框覆盖，使用COCO crowd的
交集/预测框面积规则判断 IoA>=t；这些框忽略并单独记录。其余全部为FP。
输入为原评测保存的 COCO `[x,y,w,h]` 有效检测框；解析和坐标格式不变。
几何排序仅用于确定匹配平局，不用于生成精确率曲线或截断预测。

## 防回归要求

- 打乱输出顺序、删除/修改所有score字段，结果应完全相同。
- 增加重复框或背景框必须增加FP。
- 1个GT，1个正确框和999个背景框：TP=1、FP=999、AP_set=0.001。
- 空预测不能遗漏FN；crowd忽略要有明确统计。
- 集合匹配最大化TP数，不能用可能损失匹配数的简单贪心或纯IoU最大化替代。

`unranked_metrics.py` 的自检覆盖上述情况；实际数据及CLI入口另外复算验证。

## 文件与复现

- 新结果：各评测目录的 `set_metrics.json`。
- 全部重要结果索引：`evaluation/unranked_summary.json` 与 `.md`。
- 推理方法汇总：`inference_ablation/results/set_summary.json` 与 `.md`。
- 旧 `coco_metrics.json`、旧排序AP报告保留追溯，已被本协议取代。
- `sft_ablation/baseline_evaluate.py` 为带哈希校验的历史解析参考，不作为新评测入口。
- `audit_subset.py`、`audit_ap_size.py` 是旧AP机制审计，输出不属于当前正式指标。

```bash
venv/bin/python rescore_unranked.py --root "$PWD"
venv/bin/python inference_ablation/analyze.py --root "$PWD"
```

重算读取已有预测，不重新推理，不改训练权重。正在运行的旧进程已加载的
代码无法靠改文件更新，完成后必须经过无排序重算；待启动入口直接使用本协议。
