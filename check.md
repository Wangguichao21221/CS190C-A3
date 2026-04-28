### 1.What do alpha and r mean in LoRA, and how did you choose their values?
在你的实现里，r 是低秩分解的秩（adapter bottleneck 维度），alpha 是 LoRA 分支的缩放系数，实际缩放是 alpha / r
你训练和评测脚本里实际使用的是 r=16, alpha=32
选择方式从项目看是“手工指定而非自动搜索”：README 固化了这组超参并报告 81/100。见 README.md:11、README.md:12、README.md:5。
从数值关系看你用了常见的 alpha=2r（32=2×16），在容量与稳定性之间做了折中。
### 2.How are lora_A and lora_B initialized, and why?
初始化方式：
lora_A: 随机初始化（Kaiming Uniform）
lora_B: 全零初始化
这样做的作用：
由于 lora_B 初始为 0，初始时 LoRA 分支输出近似 0，模型一开始等价于原始基座层，训练更稳定。
lora_A 用随机值让低秩子空间可学习，不会退化成全零方向。
另外你冻结了 base linear 参数，只训练 LoRA 参数
### 3.Show your LoRA injection logic. Which modules did you inject into, and why those?
注入逻辑在 lora.py:61 到 lora.py:79：遍历 model.named_modules()，筛选“模块名后缀在 target_modules 且类型是 nn.Linear”的层，再用 LoRALinear 替换。

注入模块列表在：

q_proj, k_proj, v_proj, o_proj
gate_proj, up_proj, down_proj
为什么是这些：它们覆盖了 Transformer 的注意力投影与 MLP 投影主干，是参数效率微调最常见、对任务效果影响最大的线性层集合；你 README 也记录了同一目标模块配置，见 README.md:14。

### 4.How did you format the training and validation data for the model?
训练样本格式：
完整训练文本 = prompt + 参考答案 + eos，见 train.py:21。
训练标签构造：
你先分别 tokenize prompt 与完整文本，再把 prompt 对应位置的 labels 置为 -100，只让模型学习 answer 部分
验证数据格式：
从 jsonl 读取 question/answer，见 eval.py:39。
推理时仅输入 question 构造的 prompt，不把参考答案喂给模型
### 5.Training sequences have variable lengths — is that a problem in your implementation, and if so, how did you handle it?
不是问题，你已经处理了变长。
处理方式：
tokenize 时不做固定长度 padding（padding=False）
collator 里动态 pad 输入到 batch 内最大长度
labels 用 pad_sequence 补齐，并用 -100 作为忽略位
这保证了变长 batch 可训练，同时 loss 不会在 padding 或 prompt 区域上计算。
### 6.How did you set the maximum generation length at evaluation time, and how did you extract the final answer from the model output?
最大生成长度：
eval 参数定义为 max_new_tokens
实际脚本传入 512
generate 调用使用 max_new_tokens=max_new_tokens
输出截取与最终答案提取：
先根据 attention_mask 计算每条 prompt 长度，只解码 continuation（即新生成部分），见 eval.py:156、eval.py:169。
再从文本中取第一个 #### 后的答案块，防止串到下一题，见 eval.py:79。
用正则抓数字并做数值规范化（去逗号、统一整数/小数形式），见 eval.py:17、eval.py:55、eval.py:95。
ground truth 也走同一解析逻辑再比较，见 eval.py:103、eval.py:225。