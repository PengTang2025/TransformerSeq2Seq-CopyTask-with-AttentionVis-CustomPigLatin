# Transformer Attention Visualization Experiment — Pig Latin Seq2Seq Task

中文版：[README_cn.md](https://github.com/PengTang2025/TransformerSeq2Seq-CopyTask-with-AttentionVis-CustomPigLatin/blob/main/README_cn.md)

## ✨ Project Overview

This project originated from my attempt to systematically study and improve the understanding of Transformer architecture.

While working on the [transformer_copy](https://github.com/PengTang2025/transformer_copy) project, I noticed that the attention visualization code provided by the tutorial produced results that looked like random noise, showing no meaningful semantic alignment patterns.

At first, I suspected the problem lay in the dataset itself—the tutorial used sequences of random numbers lacking real semantic content. To verify the visualization effect, I created a new dataset and trained models on it, which led to this project.

Further analysis revealed that the tutorial’s visualization was not based on actual trained model weights but rather derived from a separately constructed MultiHeadAttention module that had not undergone training. I implemented my own method to visualize attention using trained models. This project presents the results of experiments on real semantic data accordingly.

## 💡 Project Goals

Using a simple Seq2Seq task converting English to Pig Latin as the experimental platform, this project:

- Constructs a Pig Latin dataset conforming to the standard `torch.utils.data.Dataset` interface;
- Trains a Transformer-based encoder-decoder model;
- Implements and visualizes heatmaps for the following attention mechanisms:
  - Encoder Self-Attention
  - Decoder Self-Attention
  - Decoder Cross-Attention

## 🔍 Visualization Overview

The Attention Heatmaps produced in this project clearly illustrate how the Transformer model distributes attention when processing word conversions, providing intuitive insight into its inner workings. For example:

- How the encoder focuses on the current token and its context;
- How the decoder manages previously generated tokens via self-attention;
- How cross-attention in the decoder aligns output tokens with corresponding source tokens.

## 💻 Visualization Interpretation

By feeding six different types of words into the model, we can clearly observe the attention mechanism at work. The six words are:

`bassinet`, `bilaminar`, `muse`, `oceanwards`, `postverbal`, and `tromp`.

### Encoder Self-Attention

- **Prefix focus:** Some attention heads show bias toward the first few tokens (especially positions 0–2), highlighting consonant clusters at the word’s start depending on whether it begins with one or two consonants.
- **Diagonal patterns:** Some heads exhibit diagonal or offset diagonal patterns.
- **Output suffix attention:** Attention spreads over suffix tokens (like added ‘a’ and ‘y’ in Pig Latin), especially noticeable for short words.
- Each head captures distinct local patterns.

### Decoder Self-Attention

- Exhibits a typical lower-triangular masked structure reflecting the causal masking mechanism.
- Clear diagonal or offset diagonal patterns are visible.
- Similar to encoder, attention on suffix tokens shows dispersed distributions, especially on shorter outputs.

### Decoder Cross-Attention

- Highlights primarily form a bent-line pattern rather than a strict diagonal, biased according to the initial phonetic features of the word (vowel vs. consonant start, single vs. double consonants).
- Attention on suffix tokens is again spread out for short words.
- For `tromp` (double consonant start), a distinct attention logic is visible: a shifted diagonal plus reordering focusing on initial consonants.
- For `oceanwards` (vowel start), one-to-one strict diagonal attention is observed.
- For single-consonant starts (`bassinet`, `bilaminar`, `muse`, `postverbal`), attention follows a slightly offset diagonal.
- This clearly reflects the character rearrangement mapping from input English words to Pig Latin output.


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
