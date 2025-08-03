## 项目简介
本项目源于我在系统学习 Transformer 结构时的一个观察与改进尝试。
在整理 transformer_copy 项目过程中，我注意到其中基于教程提供的注意力可视化示例代码，其输出结果往往呈现出“随机图”般的混乱状态，完全无法展现出有意义的语义对齐模式。
起初我以为问题出在数据集本身——示例采用了随机数字序列，缺乏真实语义。但深入分析后发现，该教程的可视化实际上并未基于模型训练后的真实权重，而是来自另外使用MutiHeadAttention虚拟构造模型，且未经训练的状态。
为了解决这个问题，我采用了自己的方法实现了训练后模型的 attention 可视化。于此同时，我也着手了本项目。

## 项目目标
本项目以英语转 Pig Latin 的简单 Seq2Seq 任务为实验平台，构建了具备真实语义对齐关系的训练数据集，并完成以下工作：
构建符合标准 torch.utils.data.Dataset 结构的 Pig Latin 数据集；
训练基于 Transformer 架构的编码器-解码器模型；
实现并展示以下注意力机制的热力图可视化：
Encoder Self-Attention
Decoder Self-Attention
Decoder Cross-Attention

## 可视化效果
通过本项目展示的 Attention Heatmap，可清晰观察到 Transformer 模型在处理词语转换时的关注分布，从而更直观地理解其内部机制。例如：

