---
pipeline_tag: text-generation
license: other
inference: false
---

# Tongyi-Finance-14B-Chat

## 介绍 (Introduction)

**通义金融-14B**（**Tongyi-Finance-14B**）是针对对金融行业推出的大语言模型，基于通义千问基础模型进行行业语料增量学习，强化金融领域知识和场景应用能力，覆盖金融知识问答、文本分类、信息抽取、文本创作、阅读理解、逻辑推理、多模态、Coding等能力象限。同时，在Tongyi-Finance-14B的基础上，我们使用对齐机制打造了基于大语言模型的AI助手Tongyi-Finance-14B-Chat。本仓库为Tongyi-Finance-14B-Chat的仓库。

<br>

## 要求（Requirements）和 依赖项 (Dependency)
* python 3.8及以上版本
* pytorch 1.12及以上版本，推荐2.0及以上版本
* 建议使用CUDA 11.4及以上（GPU用户、flash-attention用户等需考虑此选项）
<br>
请确保满足上述要求，再执行以下pip命令安装依赖库

```bash
pip install transformers_stream_generator==0.0.4
pip install modelscope>=1.9.0
pip install transformers>=4.32.0
```

另外，推荐安装`flash-attention`库，以实现更高的效率和更低的显存占用。

```bash
git clone -b v1.0.8 https://github.com/Dao-AILab/flash-attention
cd flash-attention && pip install .
# 下方安装可选，安装可能比较缓慢。
# Below are optional. Installing them might be slow.
# pip install csrc/layer_norm
# pip install csrc/rotary
```
<br>

更详细的要求和依赖项内容请参考基座模型[通义千问-14B](https://modelscope.cn/models/qwen/Qwen-14B)仓库。

## 快速使用（Quickstart）

您可以通过以下代码轻松调用：

```python
from modelscope import AutoModelForCausalLM, AutoTokenizer, snapshot_download
from modelscope import GenerationConfig

model_dir = snapshot_download('TongyiFinance/Tongyi-Finance-14B-Chat')

# Note: The default behavior now has injection attack prevention off.
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

# use bf16
# model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="cuda:0", trust_remote_code=True, bf16=True).eval()
# use cpu only
# model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="cpu", trust_remote_code=True).eval()
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="cuda:0", trust_remote_code=True).eval()
# 模型加载指定device_map='cuda:0'，更改成device_map='auto'会使用所有可用显卡

# Specify hyperparameters for generation
model.generation_config = GenerationConfig.from_pretrained(model_dir, trust_remote_code=True)

response, history = model.chat(tokenizer, "请解释一下资产负债率", history=None)
print(response)
# 资产负债率是一个财务比率，用来衡量一个企业的负债水平。它是用一个企业负债总额除以其资产总额的百分比来表示的。它的计算公式是：资产负债率 = 负债总额 / 资产总额。它能够反映一个企业的财务状况，以及它是否具有足够的资产来抵偿其债务。
```

<br>

## 模型细节 (Model)

通义金融-14B模型规模基本情况如下所示：

| Hyperparameter  |  Value |
|:----------------|:-------|
|    n_layers     |     40 |
|     n_heads     |     40 |
|     d_model     |   5120 |
|   vocab size    | 154112 |
| sequence length |  16384 |

在位置编码、FFN激活函数和normalization的实现方式上，我们也采用了目前最流行的做法，
即RoPE相对位置编码、SwiGLU激活函数、RMSNorm（可选安装flash-attention加速）。

在分词器方面，相比目前主流开源模型以中英词表为主，Tongyi-Finance-14B在Qwen-14B扩展了金融行业词汇，词表大小15万。 该词表在GPT-4使用的BPE词表`cl100k_base`基础上，对中文、多语言进行了优化，在对中、英、代码数据的高效编解码的基础上，对部分多语言更加友好，方便用户在不扩展词表的情况下对部分语种进行能力增强。
词表对数字按单个数字位切分。调用较为高效的[tiktoken分词库](https://github.com/openai/tiktoken)进行分词。

<br>

## 使用协议（License Agreement）

我们的代码和模型权重对学术研究完全开放，并支持商用。请查看[LICENSE](https://modelscope.cn/models/TongyiFinance/Tongyi-Finance-14B-Chat/file/view/master/LICENSE.md)了解具体的开源协议细节。

如需商用，请填写[问卷](https://dashscope.console.aliyun.com/openModelApply/Tongyi-Finance-14B)申请。如果想给我们的研发团队和产品团队留言，请通过邮件（tongyifinance@gmail.com）联系我们。

