# 实验记录

从零写一个小型 LLM 推理引擎（mini vLLM），在一张 12 GB 的笔记本 GPU 上服务 Qwen3-0.6B，逐项对比 KV cache 的管理方式、调度策略和注意力实现对吞吐和延迟的影响。

目前完成了 E0（正确性）和 E1（基线）。在固定的离线负载 W1（256 条请求）上，静态批处理的起点是 1760 输出 token/s，是 HF transformers 静态批处理（160 token/s）的 11.0 倍；但 KV 池里平均只有 26.2% 的槽位存着 token，后面的实验从这里开始改进，见第 2 节的 E1。

数字的来源：

- 第 2 节的数字都来自 [results/](results/) 下的 JSON 文件：E0 由 [tests/test_qwen3_hf.py](tests/test_qwen3_hf.py) 和 [tests/test_gpt2_nanogpt.py](tests/test_gpt2_nanogpt.py) 生成，E1 由 [bench/bench_offline.py](bench/bench_offline.py) 和 [bench/bench_hf.py](bench/bench_hf.py) 生成。
- 负载文件在 [bench/workloads/](bench/workloads/)，由 [bench/workloads.py](bench/workloads.py) 用固定的 seed 生成。
- 第 3 节记录实现上的设计决策，其中的数字来自单独的检查脚本（没有放进仓库），只用来说明设计的原因；需要作为结论的数字，会在后面的实验里正式测量。

## 1. 实验设置

### 硬件与软件

| 项目 | 配置 |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 Ti Laptop GPU（12 GB 显存，46 个 SM，计算能力 12.0） |
| 系统 | Windows 11 |
| 软件 | Python 3.11，PyTorch 2.11.0+cu128，triton-windows 3.6.0.post26；transformers 5.17.0 只用于对照 |

### 模型

模型的前向都是自己写的，在 [minivllm/models/](minivllm/models/)，不依赖 transformers。

- **Qwen3-0.6B（主模型）：**
  - 28 层，hidden 1024，16 个 query 头、8 个 KV 头（GQA），head_dim 128，词表 151,936。
  - bf16，约 5.96 亿个参数（共享的 embedding 只算一次），权重 1.11 GiB。
  - 每个 token 的 KV 是 2 × 28 × 8 × 128 × 2 字节 = 112 KiB。
- **GPT-2 124M（第二个模型）：** nanoGPT 项目训练的 checkpoint（pre-LN、GELU、RoPE），后面的投机解码实验用它。E0 检查了它的移植。

### 引擎

代码在 [minivllm/](minivllm/)：

- **输入布局：** 一步里所有序列的新 token 拼成一维，不补齐，再配上每个 token 的位置和 KV 槽位（slot_mapping）。prefill 和 decode 共用这一套。
- **KV 池：** 启动时一次分配，每层 K、V 各一块，形状是 [块数, 16, KV 头数, head_dim]。所有对比实验都用 4096 块，即 65,536 个 token；Qwen3 下是 7.0 GiB。
- **实验要比较的几个开关：**
  - `kv=contiguous`：每条序列固定占一块能放 max_model_len（2048）个 token 的连续区域，相当于静态 KV cache，池里最多同时放 32 条序列。
  - `kv=paged`：按需一块一块地分配。`admission=reserve` 时，只有 prompt 加 max_tokens 需要的块都还没被预留，才让请求进入，所以不会抢占；`admission=optimistic` 时，prompt 放得下就进入，块不够时抢占最后进入的序列，之后重新计算。
  - `batching=static`：一批全部结束后才组下一批；`batching=continuous`：每一步都可以补进新请求，prefill 优先。
- **其他固定设置：** 每一步要么全是 prefill，要么全是 decode；prefill 每步最多 16,384 个 token；最多同时运行 256 条序列；max_model_len 2048。
- **注意力：** 用 FlexAttention。prefill 一次处理打平的多条序列（document mask），decode 按 block table 直接读 KV 池，见 3.1。
- **CUDA graph：** decode 按 batch 大小分桶（1、2、4、8，再从 16 到 256 每隔 16 一个，共 20 个）录图，batch 补齐到桶的大小，见 3.2。
- **采样：** 每条请求可以有自己的 temperature、top-k 和 top-p。
- **和最初计划的不同：** 计划里 FlexAttention 和 CUDA graph 分别是 E4、E5 的内容。但不录图时，decode 每一步的耗时几乎不随 batch 大小变化（见 3.4），会把批处理带来的收益放大。所以 E1–E3 都在开着这两项的引擎上测；E4、E5 改成在最终引擎上的消融实验：换掉注意力实现，或者关掉 CUDA graph。

### 负载

- **W1（离线吞吐）：** 256 条请求，在 t = 0 同时到达。
  - prompt 是 [0, 10000) 里随机的 token id，长度在 100–1024 之间均匀随机；输出长度也在 100–1024 之间均匀随机，忽略 EOS，所以每种配置算的 token 数完全相同。
  - temperature 0.6，不用 top-k 和 top-p。
  - 合计 prompt 147,986 个 token、输出 146,005 个 token，最长的一条请求 1998 个 token。
  - 设置和 nano-vllm 的基准测试相同。
- **W2（在线延迟）、W3（共享前缀）：** 已经生成，用到时再介绍。

### 指标

- **输出吞吐：** 输出 token 总数 ÷ 总耗时，总耗时从加入全部请求算到最后一条请求完成。总吞吐把 prompt token 也算进去。
- **TTFT：** 请求到达到生成第一个 token 的时间。**TPOT：** 第一个 token 之后，平均每个 token 的间隔，每条请求算一个值。都取全部请求的 p50 和 p99。W1 的请求同时到达，TTFT 主要是排队的时间。
- **KV 指标：** 按 decode 步的耗时加权平均。
  - **KV 利用率：** 存着 token 的槽位 ÷ 已经分配给序列的槽位。
  - **池的分配比例：** 已经分配的槽位 ÷ 池的全部槽位。
  - **池里 token 的比例：** 存着 token 的槽位 ÷ 池的全部槽位，约等于前两项的乘积。
- **decode batch：** 每个 decode 步里的序列数，按耗时加权平均。

### 对比规则

- 每种配置先预热一次（编译 FlexAttention、录 CUDA graph、跑一遍小负载），再把整个负载跑 3 次，取中位数。
- 比较两种配置时，对照组在同一次测试里、改动的前后各跑一次。
- 没有提升的结果也照样记录。

## 2. 结果

### 总表

| 实验 | 日期 | 代码版本 | 配置 | 输出吞吐 | decode batch（平均 / 最多） | KV 利用率 | 池里 token 的比例 | 显存峰值 |
|---|---|---|---|---|---|---|---|---|
| E1：HF | 2026-10-06 | `192d2b2` | transformers 的 generate，静态批处理，batch 24 | 160 token/s | 24 / 24 | — | — | 6.77 GiB |
| E1：我们的引擎 | 2026-10-06 | `192d2b2` | `kv=contiguous`，`batching=static` | **1760 token/s** | 18.8 / 32 | 0.521 | 0.262 | 8.61 GiB |

- **输出吞吐：** 我们的引擎是 3 次的中位数，HF 只跑了 1 次（见 E1）。
- **显存峰值：** PyTorch 已分配显存的峰值（max_memory_allocated）。

### E0：正确性

**Qwen3 和 HF transformers 对比：**

- **做法：**
  - 4 条真实文本当 prompt：取 FineWeb-Edu 验证集里文档的开头，用 Qwen3 的 tokenizer 重新编码，长度 17、120、517、1000。
  - 4 条打包成一次 prefill，再一起贪心 decode 64 个 token；KV 写在打乱了顺序的分页池里。
  - 参照是 HF 的 fp32 模型；HF 自己的 bf16 模型给出 bf16 下正常的误差水平。
  - decode 的误差用 teacher forcing 计算：把我们生成的序列整段输入参照模型，比较每一步的 logits。
- **指标：** logits 的平均绝对误差；遗憾值，即选出的 token 在 fp32 参照下比最大的 logit 低多少，argmax 一致时为 0。

| | prefill 平均误差 | decode 平均误差 | 最大遗憾值（prefill / decode） |
|---|---|---|---|
| 我们的模型，gather 注意力 | 0.037–0.049 | 0.032–0.039 | 0.19 / 0.077 |
| 我们的模型，flex 注意力 | 0.037–0.049 | 0.035–0.043 | 0.19 / 0.085 |
| HF 的 bf16 模型 | 0.036–0.049 | 0.032–0.042 | 0.19 / 0.10 |

- **误差和 HF 自己的 bf16 同一水平。** 单个位置的最大误差偶尔超过 1，HF 的 bf16 也一样（最大 1.66），所以看平均误差和遗憾值。
- **贪心生成不是逐 token 一致：** 和 HF 的 bf16 相比，分歧出现在第 1–33 个 token；flex 注意力下有一条 prompt 的 64 个 token 完全相同。每个分歧处，HF 自己 top-2 的 logits 差距都是 0 或 0.125，也就是两个候选在 bf16 下打平了（logit 在 16–32 之间时，bf16 能表示的最小间隔是 0.125）。两边的 kernel 不同，打平时就会选不同的 token。

**GPT-2 和 nanoGPT 对比：**

- **做法：** 4 条 prompt，用 `<|endoftext|>` 加验证集文档的开头，长度 17、120、517、896，生成 128 个 token。参照是 nanoGPT 自己的代码，和它生成时一样开 bf16 autocast。
- **gather 注意力（调用的 SDPA 内核和 nanoGPT 相同）：**
  - 每条 prompt 单独跑时，4 条的 prefill logits 都和 nanoGPT 逐位相同，贪心生成的 128 个 token 也完全相同。
  - 4 条打包在一起跑时，3 条仍然逐位相同；17 个 token 的那条有微小的差别（最大 0.0625），在一个 top-2 差距为 0 的位置分岔了。
- **flex 注意力：** 注意力的 kernel 和 nanoGPT 不同，不再逐位相同，但误差和 nanoGPT 自己的 bf16 同一水平：相对 fp32 的平均误差是 0.0081–0.0095，nanoGPT 是 0.0082–0.0094。打包时 4 条的 128 个 token 全部和 nanoGPT 相同，单独跑时 4 条中有 3 条相同。

**批不变性：** 上面 17 个 token 那条单独跑时和 nanoGPT 逐位相同，和别的序列打包以后就不同了。打包后矩阵乘法从 17 行变成 1550 行，cuBLAS 选了不同的 kernel，累加顺序随之改变。在连续批处理的引擎里，同一条请求的输出会随同一批里的其他请求变化；这在贪心解码下表现为偶尔在打平的位置分岔。

**引擎层面的测试**（都在 [tests/](tests/) 里，全部通过）：

- [tests/test_engine.py](tests/test_engine.py)：在每一种 KV 分配、准入和批处理策略下（包括用很小的池强制抢占和重算），把每条生成结果整段输入参照模型，检查每一步生成的 token 都是参照下的 argmax，最多只差 bf16 的噪声；还检查了录 CUDA graph 和不录图生成的 token 逐个相同。
- [tests/test_attention.py](tests/test_attention.py)：两种注意力实现在各种长度（包括 1、16、17、128、129 这些边界）和 batch 大小下，都和 fp32 参照比较；同时统计形状不断变化时编译了多少个图。
- [tests/test_sampler.py](tests/test_sampler.py)：temperature、top-k、top-p 和它们的组合，在已知分布上各采样 10 万次，频率和理论值一致。

### E1：基线

- **两种静态批处理：**
  - **HF：** transformers 的 generate，按负载顺序每 24 条一批，prompt 左侧补齐，每批一直生成到最长的请求结束。batch 32 会在生成到一半时显存不够：KV cache 每步用 torch.cat 变长，显存碎片太多。所以用 24，这是能跑完 W1 的最大 batch。
  - **我们的引擎：** `kv=contiguous`，`batching=static`。按负载顺序装满 32 个区域，一批全部结束后再装下一批。
- **运行条件：** 2026-10-06，代码版本 `192d2b2`。先跑我们的引擎（预热后跑 3 次），再跑 HF（预热后跑 1 次）。HF 跑一次要 15 分钟，它只作为参照，和我们的差距是 11 倍，单次运行的波动不影响结论，所以只跑了 1 次。

| | HF（batch 24） | 我们的引擎 |
|---|---|---|
| 输出吞吐 | 160 token/s | 1760 token/s（3 次：1747、1760、1762） |
| 总耗时 | 914.9 s | 83.0 s |
| 生成的步数 | 10,777 | 15 次 prefill，7,994 次 decode |
| 平均每步的耗时 | 84.9 ms（含 prefill） | decode 9.7 ms |
| 算出的 token 中有用的比例 | 58.3% | 100% |
| decode batch（平均 / 最多） | 24 / 24（最后一批 16） | 18.8 / 32 |
| KV 利用率 | — | 0.521 |
| 池的分配比例 | — | 0.588 |
| 池里 token 的比例 | — | 0.262 |
| TTFT p50 / p99 | — | 36.3 s / 72.8 s |
| TPOT p50 / p99 | — | 10.2 ms / 11.7 ms |
| 显存峰值（已分配） | 6.77 GiB | 8.61 GiB |

- **HF 的步数和有用比例：** 按负载和 batch 大小算出。每批的步数是这批里最长的输出长度，11 批共 10,777 步；每一步每一行都算一个 token，共 250,544 个，其中 146,005 个是请求真正要的。

结果：

- **我们的引擎比 HF 快 11.0 倍。** 下面几个因素都有贡献，但没有逐项拆开测：
  - **HF 有一半的计算是浪费的：** 每批都要生成到最长的请求结束，已经结束的行还在继续算，算出的 token 里只有 58.3% 有用。我们的引擎里，结束的序列马上退出 decode。
  - **每一步慢得多：** HF 平均每步 84.9 ms，我们每个 decode 步 9.7 ms。HF 在 Python 里逐个提交 kernel，KV cache 每步用 torch.cat 整块重新拷贝一遍，GQA 还要把 KV 头复制一份；我们的 decode 录成了 CUDA graph，注意力直接读分页池。
  - **batch 24 对 32。**
- **静态批处理本身的问题**（E2、E3 要改进的地方）：
  - **每批的耗时由最长的请求决定：** 8 批的 decode 步数之和（7,994）正好等于每批最长的输出长度之和减去 8（每条请求的第一个 token 由 prefill 生成）。一批里的请求陆续结束，batch 越来越小，平均只有 18.8 条，最多 32 条。
  - **KV 预留得多、用得少：** 每条序列都占 2048 个槽位，平均只用了 52.1%；一批里结束的序列空出的区域，在这批结束前也不会给新请求用，池平均只分配出去 58.8%。两者合起来，池里平均只有 26.2% 的槽位存着 token。
- **延迟：** 所有请求在 t = 0 到达，要排队等前面的批次结束，TTFT 的中位数是 36.3 秒。TPOT 的中位数是 10.2 ms。
- **录图的开销：** 20 个 decode 图录了 5.0 秒，多占约 0.2 GiB 显存。

**结论：** 起点是 1760 输出 token/s。主要的浪费来自按最大长度预留 KV 和静态批处理的尾部。E2 只改第一点：KV 改成分页，仍然是静态批处理。

## 3. 工程记录与设计决策

### 3.1 用 FlexAttention 做分页注意力

- **decode：** 不用 `create_block_mask`（单独检查时每次要 4–8 ms），而是直接从 block table 构造 BlockMask：每条序列除了最后一页都标成完整的块，不用 mask；最后一页标成部分块，mask 是 `kv_idx % 16 < 最后一页的长度`。KV 池按 `[1, KV 头数, 槽位数, head_dim]` 的视图传进去，不用拷贝。
- **prefill：** 打平的多条序列一次处理。块级的结构直接由每条序列的起止位置算出：两个块落在同一条序列里就是完整块，跨了序列边界就是部分块，由 mask 判断 token 属于哪条序列。
- **编译：**
  - prefill 用动态形状编译，token 数每步都不同，也不用重新编译。
  - Inductor 只有在 batch 维是静态的时候，才会用 flex 的 decoding kernel；否则退回通用的 kernel，而通用 kernel 不支持 16 的 KV 块。所以 decode 按 batch 大小的桶分别编译，桶和 CUDA graph 的桶一致。
  - prefill 少于 128 个 token 时，Inductor 也会选 decoding kernel，而它处理不了 q、k、v 这种从投影输出切出来的非连续视图，所以 prefill 强制用通用 kernel。
- **编译次数超过上限时直接报错：** dynamo 对同一个函数默认只缓存 8 个编译版本，超过以后会改用 eager 实现。flex 的 eager 实现不看块级的结构，decode 的 mask 又依赖块级结构来排除其他序列的页，结果是错的：单独检查时，decode 的误差从 0.006 变成 0.69。现在把上限调到 64，并让超过上限时报错。

### 3.2 decode 的 CUDA graph

- **补齐：** batch 补到桶的大小。补出来的行是一个"假序列"：只有位置 0 的一个 token，KV 写进池外面单独留出的一个块，注意力也只读这个块。这样补齐不需要任何特殊处理。
- **静态输入：** token、位置、slot_mapping、block table 和上下文长度都放在预先分配的张量里，每步回放前原地写入。flex 的 BlockMask 每个桶只建一次，它的几个索引张量在图里原地重算。
- **只把前向录到 hidden states：** logits 在图外面、只对真实的行计算。如果把 logits 也录进去，20 个桶各自的输出要占约 666 MB（各桶大小之和 2191 行 × 151,936 × 2 字节）；只录 hidden states 只要约 4.5 MB。
- **和不录图完全一致：** 两条路径补齐的方式相同，kernel 相同，tests/test_engine.py 检查了生成的 token 逐个相同。

### 3.3 给 PyTorch 的显存分配器设上限

- **问题：** 张量的形状每步都在变时，缓存分配器会不断申请新的块、把旧的块缓存起来，最后占满整块 GPU。之后某个不经过 PyTorch 分配器的 CUDA 分配就会失败，报 "CUDA error: out of memory"，CUDA 上下文也随之不可用。单独检查时，HF 的 generate 在 batch 64 下生成到第 226 步就这样崩溃了；用一个每步换一种长度、只调用 SDPA 的小脚本复现，崩溃前分配器保留了 11.15 GiB，实际只用了 0.04 GiB。
- **解决：** 启动时用 `torch.cuda.set_per_process_memory_fraction` 把分配器的上限设为当时空闲显存减 0.5 GiB（[minivllm/utils.py](minivllm/utils.py)）。到了上限，分配器会先释放缓存再重试。引擎和 HF 的对照脚本都这样设置。

### 3.4 不录图时，decode 被 CPU 提交 kernel 拖住

单独检查（Qwen3，上下文约 1100 个 token，flex 注意力）：

| | batch 1 | batch 35 |
|---|---|---|
| 不录图 | 27.0 ms | 25.6 ms |
| 录 CUDA graph | 6.9 ms | 14.3 ms |

- 不录图时，每一步的 kernel 都要由 CPU 逐个提交，batch 1 和 batch 35 的耗时几乎一样；用 profiler 看，batch 35 时 GPU 只有不到一半的时间在算。这时吞吐几乎和 batch 大小成正比，会夸大分页和连续批处理的收益，所以 E1–E3 都在录图的引擎上测（见第 1 节）。
- E5 会在最终的引擎上正式比较录图和不录图。

## 4. 实验列表

| 编号 | 改动 | 对照 | 负载 | 状态 |
|---|---|---|---|---|
| E0 | 正确性：Qwen3 对比 HF，GPT-2 对比 nanoGPT | — | 4 条真实文本 | 已完成，见第 2 节 |
| E1 | 基线：HF 的静态批处理；我们的 `kv=contiguous`、`batching=static` | — | W1 | 已完成：1760 对 160 输出 token/s，见第 2 节 |
| E2 | KV 改成分页（`kv=paged`，`admission=reserve`），仍是静态批处理 | E1 里我们的配置 | W1 | 下一步 |
| E3 | 连续批处理；之后再试乐观准入加抢占 | E2 | W1、W2 | 待做 |
| E4 | 注意力的消融：gather、FlexAttention、自己用 Triton 写的分页注意力 | FlexAttention | W1；单步 decode | 待做 |
| E5 | CUDA graph 的消融：关掉录图 | 录图 | W1；batch 1 | 待做 |
| E6 | 前缀缓存 | 不开 | W3 | 待做 |
| E7 | 分块 prefill | 不分块 | W2 | 待做 |
| E8 | 投机解码（GPT-2）：n-gram、MTP-1、DSpark 式起草，置信度调度 | 不用投机解码 | W4 | 待做 |
