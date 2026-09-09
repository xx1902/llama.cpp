# v1 前缀切分消融实验 1_1

本文补充旧版 `Reuse-aware v1` 没有做过的信号消融。旧的
`PREFIX_SPLITTING_ALGORITHM_ANALYSIS.md` 和原结果不修改；本实验使用新的 method ID、
输出目录和 `_1_1` 文件后缀。

## 1. 实验问题

v1 的原始实现只显式组合了两类候选：自然边界和历史 reuse point。`Sentence`、
`Keyword` 在旧实验中是独立方法，不是 Reuse-aware 的可开关组件。因此本次先把它们
重构成统一候选集合，才能回答“去掉 Sentence”“去掉 Keyword”“只保留 Reuse”分别
会发生什么。

## 2. 机制和算法

候选集合定义为：

```text
B_segment  = workload 已标注的 system/context/history/source 结束位置
B_sentence = 句末标点和换行位置
B_keyword  = User/Assistant/Task/Article/Question 等关键词开始位置
B_reuse    = 同 context 中已经出现过的严格 prefix 终点
```

对起点 `s` 和终点 `e`，v1 仍使用原始评分：

```text
score_v1(s,e) = |estimated_tokens(text[s:e]) - 128|
                - 72 * I(e in B_reuse)
```

只接受 `64 <= chunk_tokens <= 192` 的候选；没有候选时退化为 fixed-128。

| 方法 | Sentence/segment | Keyword | Reuse point | 说明 |
|---|---:|---:|---:|---|
| v1 full signals | 是 | 是 | 是 | 新增的全信号组合 |
| v1 no Sentence | 否（保留 segment） | 是 | 是 | 测试句末边界的贡献 |
| v1 no Keyword (original) | 是 | 否 | 是 | 等价于原 Reuse-aware 候选机制 |
| v1 Reuse only | 否 | 否 | 是 | 无 reuse point 时退化为 fixed-128 |

注意：这里的 Keyword 只是边界触发器，不进行语义摘要，也不会改变 prompt 文本。

## 3. 具体例子

对一个约 602-token 的 XSum 文章前缀，四种切法为：

```text
v1 full signals          -> [126, 113, 136, 138, 91]
v1 no Sentence           -> [128, 82, 134, 128, 128, 3]
v1 no Keyword (original) -> [126, 113, 136, 138, 91]
v1 Reuse only            -> [128, 128, 128, 128, 90]
```

该样本中关键词没有增加更优候选，所以 full 与 no Keyword 完全相同。no Sentence 出现
3-token 尾块，说明只靠关键词和长度容易产生碎片尾部；Reuse only 最规则，但缺少自然
编辑边界。

ShareGPT 5 请求结构结果也显示：full/no Keyword 平均 `6.6` chunks、节点峰值 `17`；
no Sentence 平均 `6.2` chunks、节点峰值 `13`。这说明去掉句末候选减少了节点，但不
代表语义边界更好。

## 4. 在线实验结果

TTFT change 为正表示比同 cell full-prefill 更快。MSC、Taskmaster、LMSYS、ShareGPT、
XSum 使用 5 请求 smoke；OPUS-100 使用 100 请求。

| 数据集 | Full signals | No Sentence | No Keyword / original | Reuse only |
|---|---:|---:|---:|---:|
| MSC | +69.59% +/- 4.10 | +67.41% +/- 7.82 | +65.92% +/- 7.81 | +65.64% +/- 10.39 |
| Taskmaster | +62.89% +/- 14.53 | +65.88% +/- 13.48 | +64.24% +/- 13.92 | +64.05% +/- 14.55 |
| LMSYS-33K | +11.46% +/- 42.50 | +9.53% +/- 51.33 | +9.08% +/- 46.38 | +8.36% +/- 44.60 |
| ShareGPT52K | -46.36% +/- 96.42 | -0.26% +/- 75.82 | -38.12% +/- 86.57 | -36.48% +/- 84.73 |
| XSum | -561.71% +/- 448.98 | -611.11% +/- 467.45 | -550.90% +/- 427.96 | -538.05% +/- 434.41 |
| OPUS-100 | +42.10% +/- 2.39 | +40.91% +/- 2.66 | +40.60% +/- 2.58 | +41.55% +/- 2.34 |

## 5. 结论

1. MSC/Taskmaster 四种方法的 CI 大量重叠，主要收益来自 append-only 历史复用，不是某个关键词规则。
2. LMSYS 的 5 请求 CI 很宽，不能证明 Sentence 或 Keyword 的稳定贡献。
3. ShareGPT 的 no Sentence 节点更少、回退更小，但 CI 仍跨越很大范围；长轮次和节点数量比关键词本身更重要。
4. XSum 四种 v1 粒度都产生 5~6 个长文章节点，无法解决跨 LoRA 物化和两 chunk 预取上限。
5. OPUS-100 所有方法都是 99.1% token coverage，TTFT 约 41%；短 exact-prefix pair 对边界规则不敏感。
6. 原始 v1 实际没有 Keyword 组件；因此“no Keyword”不是删掉一个已有优化，而是原算法 reference。

## 6. 图和逐数据集报告

- [MSC 1_1 图](output/ablation_v1_5/msc_continuous_prefix_method_effect_1_1.png)
- [Taskmaster 1_1 图](output/ablation_v1_5/taskmaster_continuous_prefix_method_effect_1_1.png)
- [LMSYS-33K 1_1 图](output/ablation_v1_5/lmsys_continuous_prefix_method_effect_1_1.png)
- [ShareGPT52K 1_1 图](output/ablation_v1_5/sharegpt_continuous_prefix_method_effect_1_1.png)
- [XSum 1_1 图](output/ablation_v1_5/xsum_parallel_prefix_method_effect_1_1.png)
- [OPUS-100 1_1 图](output/ablation_v1_100/opus100_parallel_prefix_method_effect_1_1.png)

原始 CSV、日志和逐数据集 Markdown 位于 `output/ablation_v1_5/` 和
`output/ablation_v1_100/`。

