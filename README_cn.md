# Transformer Attention Visualization Experiment — Pig Latin Seq2Seq Task  

## ✨ 项目简介
本项目源于我在系统学习 Transformer 结构时的一个观察与改进尝试。
在整理 transformer_copy 项目过程中，我注意到其中基于教程提供的注意力可视化示例代码，其输出结果往往呈现出“随机图”般的混乱状态，完全无法展现出有意义的语义对齐模式。
起初我以为问题出在数据集本身——示例采用了随机数字序列，缺乏真实语义。于是，我着手构建基于新的数据集的模型来验证可视化效果，便是本项目。  
虽然在深入分析后发现，该教程的可视化实际上并未基于模型训练后的真实权重，而是来自另外使用MutiHeadAttention虚拟构造模型，且未经训练的状态，
我也采用了自己的方法实现了训练后模型的 attention 可视化。但我仍做完了这一基于真实语义数据的实验，呈现于此。

## 💡 项目目标
本项目以英语转 Pig Latin 的简单 Seq2Seq 任务为实验平台，构建了具备真实语义对齐关系的训练数据集，并完成以下工作：
构建符合标准 torch.utils.data.Dataset 结构的 Pig Latin 数据集；
训练基于 Transformer 架构的编码器-解码器模型；
实现并展示以下注意力机制的热力图可视化：
- Encoder Self-Attention
- Decoder Self-Attention
- Decoder Cross-Attention

## 🔍 可视化概述
通过本项目展示的 Attention Heatmap，可清晰观察到 Transformer 模型在处理词语转换时的关注分布，从而更直观地理解其内部机制。例如：
- Encoder 如何聚焦于当前词及上下文；  
- Decoder 如何通过 Self-Attention 管理已有输出；  
- Cross-Attention 如何在编码器输出上定位对应的源词。  

## 🔬 试验记录
鉴于这对transformer来说是一个非常简单的任务，在验证集使用了early stop之后训练了7个epoch就停止了。
验证集的loss从最初就很低，主要是因为`model.train()` 与 `model.eval()`模式下dropout的开启与否。

最终测试结果：
```
Token-level Accuracy: 0.9999
Sequence-level Accuracy: 0.9995
BLEU Score: 0.9997
```
训练中的损失曲线：
<div style="text-align: center;">
  <img alt="image" src="https://github.com/user-attachments/assets/f10ccab3-8edf-4c23-aa93-b58b14dd1a0d" width="80%" />
</div>

## 💻 可视化解读
通过将6个不同类型的单词数据输入模型，我们可以清晰地看见注意力的工作机制。  
6个单词分别是`bassinet`，`bilaminar`，`muse`，`oceanwards`，`postverbal`，`tromp`。    
- Encoder Self-Attention
  1. 前缀聚焦：一些 head 对前几个 token（尤其是位置 0~2）有偏置，根据不同单词（1位辅音或2位辅音）显示出模型对输入前缀（辅音 cluster）聚焦。  
  2. 对角线模式：一些head出现了对角线/偏置对角线模式；  
  3. 可见有些 head 在输出末尾几位（添加的a, y词尾）上的 attention 分散分布，在短词上尤为明显；  
  4. 各个head捕捉到了不同区域的规律。  
- Decoder Self-Attention
  1. 典型的下三角结构，体现了masked input机制；  
  2. 可见明显的对角线/偏置对角线模式；  
  3. 可见有些 head 在输出末尾几位（添加的a, y词尾）上的 attention 分散分布，在短词上尤为明显。   
- Decoder Cross-Attention
  1. 亮点基本位于一条折线形路径，非对角线，而是随单词的开头特征（是否元音，辅音位数）有不同的偏置。
  2. 可见有些 head 在输出末尾几位（添加的a, y词尾）上的 attention 分散分布，在短词上尤为明显；
  3. 在双辅音开头的`tromp`上，可见明显的3-4-5-1-2的关注逻辑：上偏移2位的对角线+对起始辅音的重排；  
     在元音开头的`oceanwards`上，可见正对角线的一对一关注逻辑；  
     在单辅音开头的`bassinet`，`bilaminar`，`muse`，`postverbal`上，可见上偏移1位的对角线。    
     清晰体现了从输入字符重排到 Pig Latin 输出的映射。  
  

<table>
  <tr>
    <td align="center"><strong>Bassinet</strong></td>
    <td align="center"><strong>Bilaminar</strong></td>
  </tr>
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/707b4c27-841c-4e69-84a7-c87590e3b3a8" width="90%" title="bassinet"/>
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/6cc52c48-0fae-430d-9624-db20c3eef229" width="90%" title="bilaminar"/>
    </td>
  </tr>
</table>

<table>
  <tr>
    <td align="center"><strong>muse</strong></td>
    <td align="center"><strong>oceanwards</strong></td>
  </tr>
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/3036722e-e2ee-4a59-94a3-d2ffd0241560" width="90%" title="muse"/>
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/a01e9a31-6403-4729-b3ac-320ddd12e95b" width="90%" title="oceanwards"/>
    </td>
  </tr>
</table>

<table>
  <tr>
    <td align="center"><strong>postverbal</strong></td>
    <td align="center"><strong>tromp</strong></td>
  </tr>
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/0444e6d5-9bde-442b-8e4b-8997d510c611" width="90%" title="postverbal"/>
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/4f92a472-78b2-4dec-a4b6-65914fc65d25" width="90%" title="tromp"/>
    </td>
  </tr>
</table>

## 📜 License

MIT License © 2025 Peng Tang
