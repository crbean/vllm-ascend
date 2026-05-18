# Qwen3.5-27B vllm-ascend 适配进度

## 常驻区

### 模型信息
- HF 模型路径: /home/cb/glm5.1/vllm-custom/Qwen3.5-27B
- architectures: Qwen3_5ForConditionalGeneration
- 架构类型: 混合架构 VLM（linear_attention + full_attention 交替层 + ViT 视觉编码器 + MTP）
- 参数量: 27B

### 运行环境
- NPU 型号: 8x Ascend 910B3 (Atlas 800I A2)
- CANN 版本: cann-9.0.0-beta.2
- vLLM 版本: 0.20.2rc1.dev57 (commit: c7aa186d67b6f051680831418e957c67f34ba7a2)
- vLLM 源码路径: /home/cb/glm5.1/vllm-custom/vllm
- vllm-ascend 源码路径: /home/cb/glm5.1/vllm-custom/vllm-ascend
- transformers 源码路径: /home/cb/glm5.1/vllm-custom/transformers
- torch-npu 版本: 2.9.0
- torch 版本: 2.9.0+cpu
- 部署卡数: 8

### 适配评估
- 适配难度: L2（小幅修改）
- vLLM 支持状态: 已支持（qwen3_5.py, qwen3_5_mtp.py，继承 Qwen3Next + Qwen3VL）
- vllm-ascend 覆盖度: 大部分已覆盖（GDN、GQA、mRoPE、RMSNorm、ViT 均有 NPU 实现），仅需少量 model_type 判断条件扩展

### 环境注意事项
- LD_PRELOAD 包含 libproxychains，推理时需 unset
- torch 为 +cpu wheel，启动推理需 OMP_NUM_THREADS=1 KMP_INIT_AT_FORK=FALSE
- 代理地址: http://127.0.0.1:9912

### 进度概览

| 阶段 | 名称 | 状态 | 开始时间 | 完成时间 |
|------|------|------|---------|---------|
| 0 | 模型分析与适配评估 | 已完成 | 2026-05-14 | 2026-05-15 |
| 1 | vLLM 模型实现 | 跳过（已支持） | | |
| 2 | vllm-ascend 后端适配 | 已完成（代码已包含适配） | 2026-05-15 | 2026-05-15 |
| 3 | 精度对齐与验证 | 已完成（功能验证通过） | 2026-05-15 | 2026-05-15 |
| 4 | 性能优化 | 已完成（ACL Graph 2.2x 提升） | 2026-05-15 | 2026-05-18 |
| 5 | 测试与 PR 准备 | 跳过 | | |

---

## 工作区

### 阶段 0：模型分析与适配评估（进行中）

#### 模型架构详细分析

##### 整体拓扑
```
Qwen3_5ForConditionalGeneration
├── visual: Qwen3_VisionTransformer (ViT)
│   ├── patch_embed: Conv3d(3, 1152, kernel=(2,16,16))
│   ├── pos_embed: Embedding(2304, 1152)
│   ├── blocks: 27 x VisionBlock (LayerNorm + MHA + GELU MLP)
│   └── merger: PatchMerger (1152*4 -> 5120)
├── language_model: Qwen3_5ForCausalLM
│   ├── model: Qwen3_5Model
│   │   ├── embed_tokens: VocabParallelEmbedding(248320, 5120)
│   │   ├── layers: 64 x Qwen3_5DecoderLayer
│   │   │   ├── linear_attention 层 (48 层): GatedDeltaNetAttention
│   │   │   ├── full_attention 层 (16 层): Qwen3NextAttention (GQA + mRoPE + q_norm/k_norm + attn_output_gate)
│   │   │   └── mlp: Qwen2MoeMLP (SiLU SwiGLU, 5120 -> 17408 -> 5120)
│   │   └── norm: GemmaRMSNorm(5120)
│   └── lm_head: ParallelLMHead(5120, 248320)
└── MTP: Qwen3_5MultiTokenPredictor (1 层)
```

##### 层类型分布 (64 层)
- 每 4 层一个 cycle: [linear, linear, linear, full]
- 共 16 个 full_attention 层 (索引 3,7,11,...,63)
- 共 48 个 linear_attention 层 (其余所有层)

##### Full Attention 层参数 (标准 GQA)
- num_attention_heads: 24
- num_key_value_heads: 4 (GQA ratio = 6)
- head_dim: 256
- Q 投影输出维度: 24 * 256 * 2 = 12288 (包含 gate 分支)
- K/V 投影: 4 * 256 = 1024
- Q/K 有 RMSNorm (q_norm, k_norm)
- attn_output_gate: sigmoid 门控 (Q 投影输出一分为二，一半 Q 一半 gate)

##### Linear Attention 层参数 (Gated Delta Net)
- linear_num_key_heads: 16
- linear_num_value_heads: 48 (GQA ratio = 3)
- linear_key_head_dim: 128
- linear_value_head_dim: 128
- linear_conv_kernel_dim: 4 (causal conv1d)
- 输入投影: in_proj_qkv (key_dim*2 + value_dim = 4096+6144=10240), in_proj_z (6144), in_proj_b (48), in_proj_a (48)
- 状态维度: recurrent_state = [batch, num_v_heads, head_k_dim, head_v_dim]
- mamba_ssm_dtype: float32

##### 通用参数
- hidden_size: 5120
- intermediate_size: 17408
- vocab_size: 248320
- num_hidden_layers: 64
- rms_norm_eps: 1e-6
- Norm 类型: GemmaRMSNorm (weight 中心化: (1+w)*x, 初始化为 0)
- 激活函数: SiLU
- max_position_embeddings: 262144

##### RoPE 参数
- 类型: mRoPE (multimodal Rotary Position Embedding, 3D 交织)
- mrope_section: [11, 11, 10] (T/H/W 维度分配)
- partial_rotary_factor: 0.25 (仅 25% 的 head_dim 参与 RoPE)
- rope_theta: 10000000
- 实际 RoPE dim: 256 * 0.25 = 64 per head

##### ViT 参数
- depth: 27
- hidden_size: 1152
- num_heads: 16
- head_dim: 72
- intermediate_size: 4304
- patch_size: 16
- spatial_merge_size: 2
- temporal_patch_size: 2
- 激活函数: GELU (tanh variant)
- Norm: LayerNorm

#### vLLM 支持状态

##### 注册信息
- registry.py 已注册: 是
- 实现文件: `vllm/model_executor/models/qwen3_5.py`
- MTP 实现: `vllm/model_executor/models/qwen3_5_mtp.py`
- 注册架构:
  - `Qwen3_5ForConditionalGeneration` -> `qwen3_5.py`
  - `Qwen3_5MoeForConditionalGeneration` -> `qwen3_5.py`
  - `Qwen3_5MTP` -> `qwen3_5_mtp.py`
  - `Qwen3_5MoeMTP` -> `qwen3_5_mtp.py`

##### vLLM 实现覆盖度
vLLM 的 Qwen3.5 实现继承自 Qwen3Next 和 Qwen3VL:
- `Qwen3_5DecoderLayer` 继承 `Qwen3NextDecoderLayer`
- `Qwen3_5Model` 继承 `Qwen3NextModel`
- `Qwen3_5ForConditionalGeneration` 继承 `Qwen3VLForConditionalGeneration` + `IsHybrid`
- Linear Attention: 使用 `GatedDeltaNetAttention` (来自 vLLM 核心)
- Full Attention: 使用 `Qwen3NextAttention` (包含 q_norm/k_norm + attn_output_gate)
- ViT: 使用 `Qwen3_VisionTransformer` (与 Qwen3VL 共享)
- MLP: 使用 `Qwen2MoeMLP` (SwiGLU)
- RoPE: 使用 `MRotaryEmbedding` (mRoPE, 3D 交织)
- MTP: 使用 `Qwen3_5MultiTokenPredictor`
- 接口: `HasInnerState`, `SupportsEagle3`, `SupportsLoRA`, `SupportsPP`, `IsHybrid`

##### 结论: vLLM 支持完整，无需新建模型实现（阶段 1 跳过）

#### vllm-ascend 覆盖度评估

##### 逐组件检查

| 组件 | 检查结果 | 说明 |
|------|---------|------|
| **GDN (Linear Attention)** | 已支持 | `vllm_ascend/ops/gdn.py` 有 `AscendGatedDeltaNetAttention`，通过 CustomOp 机制注册。使用 `npu_recurrent_gated_delta_rule` (Decode) + `chunk_gated_delta_rule` (Prefill) + `npu_causal_conv1d_custom`。`check_gdn_layer` 会正确检测 Qwen3.5 的 `linear_attention` 层类型 |
| **Full Attention (GQA)** | 已支持 | `vllm_ascend/attention/attention_v1.py` 标准 GQA 路径，使用 `npu_fused_infer_attention_score`。Qwen3NextAttention 使用标准 QKVParallelLinear + Attention 流程 |
| **mRoPE** | 已支持 | `vllm_ascend/ops/rotary_embedding.py` 有 `AscendMRotaryEmbedding`，通过 CustomOp 注册 |
| **RMSNorm (GemmaRMSNorm)** | 已支持 | `vllm_ascend/ops/layernorm.py` 有 `AscendGemmaRMSNorm`，通过 CustomOp 注册 |
| **RMSNormGated** | 已支持 | `vllm_ascend/ops/layernorm.py` 有 `AscendRMSNormGated`，用于 GDN 的 gated norm |
| **SiLU 激活函数** | 已支持 | `vllm_ascend/ops/activation.py` 有 `AscendSiluAndMul` |
| **Conv3d (ViT)** | 已支持 | `vllm_ascend/ops/conv.py` 有 `AscendConv3dLayer` |
| **MM Encoder Attention (ViT)** | 已支持 | `vllm_ascend/ops/mm_encoder_attention.py` 有 `AscendMMEncoderAttention` |
| **Linear 投影** | 已支持 | `AscendColumnParallelLinear`, `AscendRowParallelLinear`, `AscendQKVParallelLinear` 等 |
| **Embedding / LMHead** | 已支持 | `AscendVocabParallelEmbedding`, `AscendParallelLMHead`, `AscendLogitsProcessor` |
| **GDN 状态管理** | 已支持 | `model_runner_v1.py` 有完整的 `_has_gdn` 检测和 `gdn_query_start_loc` 管理 |
| **Fused QKVZ gating** | 已支持 | `vllm_ascend/ops/triton/fla/fused_qkvzba_split_reshape_cat` 和 `fused_gdn_gating_patch` |

##### 可能需要适配的点

| 组件 | 风险 | 说明 |
|------|------|------|
| **pcp_utils hybrid detection** | 低 | `pcp_utils.py` 仅检查 `model_type == "qwen3_next"`，未包含 `"qwen3_5"`。若使用 PCP (Pipeline Context Parallel)，可能需扩展判断条件 |
| **recompute_scheduler hybrid detection** | 低 | `recompute_scheduler.py` 检查 `"qwen3_next" in model_type`，`"qwen3_5_text"` 不匹配此条件。若使用 streaming KV + recomputation，需扩展 |
| **quantization modelslim_config** | 低 | 无 `"qwen3_5"` 的量化配置项（modelslim 是华为量化工具，当前任务不涉及量化） |
| **spec_decode eagle_proposer** | 低 | Eagle 投机解码的模型列表中无 Qwen3.5，但 Qwen3.5 支持 Eagle3 接口。若需使用投机解码需添加 |
| **Vision encoder deepstack** | 低 | `patch_qwen3vl.py` 为 Qwen3VL 添加了 deepstack `_get_deepstack_input_embeds`，Qwen3.5 继承 Qwen3VLForConditionalGeneration，此 patch 可能需确认覆盖 |

##### 特化模型代码
- **是否需要**: 可能不需要
- Qwen3.5 在 vllm-ascend 中通过 CustomOp 机制自动替换算子（GDN、RMSNorm、Linear、Attention 等），无需编写 `vllm_ascend/models/` 下的特化子类
- 现有 `AscendGatedDeltaNetAttention` 已针对 Qwen3Next (相同 GDN 架构) 实现了 NPU 优化路径
- 需要在实际运行中验证 patch 机制是否完整覆盖 Qwen3.5 的所有组件

#### 适配难度评级

**整体难度: L2 (小幅修改)**

**理由**:
1. vLLM 已有完整的 Qwen3.5 实现，无需新建模型代码
2. vllm-ascend 已有 GDN (linear attention) 的完整 NPU 优化实现
3. Full Attention、mRoPE、RMSNorm、MLP 等标准组件均已有 NPU 优化
4. ViT 编码器的关键算子 (Conv3d、MMEncoderAttention) 已有 NPU 适配
5. 主要工作在于:(a) 确认各 patch 路径对 Qwen3.5 的覆盖完整性；(b) 处理少数 model_type 判断条件的扩展；(c) 功能验证和精度对齐

#### 适配工作项

| 阶段 | 工作项 | 修改文件 | 难度 | 预估工时 |
|------|--------|---------|------|---------|
| ascend 后端 | 扩展 pcp_utils hybrid 检测条件 | `vllm_ascend/worker/pcp_utils.py` | 低 | 0.5h |
| ascend 后端 | 扩展 recompute_scheduler hybrid 检测 | `vllm_ascend/core/recompute_scheduler.py` | 低 | 0.5h |
| ascend 后端 | 确认 deepstack patch 对 Qwen3.5 的覆盖 | `vllm_ascend/patch/worker/patch_qwen3vl.py` | 低 | 1h |
| ascend 后端 | 如需投机解码，扩展 eagle_proposer 模型列表 | `vllm_ascend/spec_decode/eagle_proposer.py` | 低 | 0.5h |
| 功能验证 | 首次启动验证 (text-only, eager mode) | 无 | 中 | 2h |
| 功能验证 | 多模态 (image/video) 输入验证 | 无 | 中 | 2h |
| 精度对齐 | logits 对齐验证 (HF transformers vs vllm-ascend) | 无 | 中 | 4h |
| 精度对齐 | 逐层精度排查 (如有偏差) | 无 | 中 | 4h |
| 性能优化 | 图模式适配 (ge_graph/npugraph_ex) | 待定 | 中 | 4-8h |
| 测试验证 | 端到端 benchmark 测试 | 无 | 低 | 2h |

**总预估工时: 2-4 天** (不含性能优化深度迭代)

#### 风险点

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| GDN 算子在 Qwen3.5 参数下精度不达标 | 中 | 高 | Qwen3.5 的 GDN 参数 (head_dim=128, num_heads=16/48) 与 Qwen3Next 不同，需验证 npu_recurrent_gated_delta_rule 和 chunk_gated_delta_rule 在这些参数下的精度 |
| mRoPE partial_rotary_factor=0.25 在 NPU 上行为异常 | 低 | 高 | MRotaryEmbedding 已适配但 partial_rotary 需确认 cos/sin 构造正确 |
| Qwen3_5RMSNorm (GemmaRMSNorm) weight 初始化为 0 的差异 | 低 | 中 | GemmaRMSNorm 使用 (1+w)*x，vllm-ascend 已有 AscendGemmaRMSNorm |
| ViT encoder 在 NPU 上的兼容性 | 低 | 中 | Conv3d 和 MMEncoderAttention 已有 NPU 实现，但需验证 ViT 全流程 |
| 64 层 (48 GDN + 16 GQA) 的混合 attention 对 model_runner 的压力 | 低 | 中 | model_runner_v1 已支持 hybrid 模式，但需验证大量 GDN 层的状态管理 |

### 阶段 2：vllm-ascend 后端适配分析

#### 模型 model_type 确认
- `config.json` model_type: `qwen3_5`
- `text_config` model_type: `qwen3_5_text`
- Qwen3.5 继承关系: `Qwen3_5ForConditionalGeneration` 继承 `Qwen3VLForConditionalGeneration` + `IsHybrid`
- `Qwen3_5DecoderLayer` 继承 `Qwen3NextDecoderLayer`，使用 `Qwen3NextAttention` (full attention) 和 `GatedDeltaNetAttention` (linear attention)
- 因此 Qwen3.5 的所有底层组件与 Qwen3Next 完全相同，vllm-ascend 的 CustomOp 替换机制对 Qwen3.5 同样生效

#### 逐组件适配分析

##### 1. GDN (Linear Attention) - AscendGatedDeltaNetAttention
- **状态: 已完全支持，无需修改**
- `vllm_ascend/ops/gdn.py` 中 `AscendGatedDeltaNetAttention` 通过 CustomOp 机制注册到 `GatedDeltaNetAttention`
- `vllm_ascend/utils.py:719` 中注册映射: `"GatedDeltaNetAttention": AscendGatedDeltaNetAttention`
- Qwen3.5 的 GDN 层直接使用 `GatedDeltaNetAttention` 类（与 Qwen3Next 相同的类），CustomOp 自动替换为 Ascend 版本
- GDN 参数兼容性确认:
  - Qwen3.5: linear_num_key_heads=16, linear_num_value_heads=48, head_dim=128
  - Qwen3Next: 类似参数结构
  - `npu_recurrent_gated_delta_rule` (Decode) 和 `chunk_gated_delta_rule` (Prefill) 无参数限制
- `patch_gdn_attn.py` 的 GDN metadata builder 补丁已通过 `GDNAttentionMetadataBuilder` 全局生效，覆盖所有使用 GDN 的模型

##### 2. Full Attention (GQA) - 标准 GQA FA 路径
- **状态: 已支持，但需确认 Qwen3NextAttention 的 patch 覆盖**
- `vllm_ascend/attention/attention_v1.py` 标准 GQA 路径已支持
- `Qwen3NextAttention` 包含 q_norm/k_norm + RoPE + Attention + O 投影
- **关键 patch 覆盖确认**: `patch_qwen3vl.py` 中 `forward_with_split_qkv_rmsnorm_mrope` 仅 patch 到 `Qwen3Attention` 和 `Qwen3MoeAttention`，**未 patch `Qwen3NextAttention`**
- 但这不是问题：Qwen3NextAttention 的 forward 使用标准 q_norm/k_norm 分离计算 + rotary_emb(positions, q, k) + attn(q, k, v) 流程，其中:
  - `q_norm`/`k_norm` 是 `Qwen3NextRMSNorm` -> 映射到 `AscendGemmaRMSNorm`（CustomOp 自动替换）
  - `rotary_emb` 是 `MRotaryEmbedding` -> 映射到 `AscendMRotaryEmbedding`（CustomOp 自动替换）
  - `attn` 是 `Attention` -> 使用 `attention_v1.py` 的 FA 路径
- `forward_with_split_qkv_rmsnorm_mrope` 是性能优化（融合 QKV+Norm+RoPE），Qwen3NextAttention 即使不走融合路径，功能也完整

##### 3. mRoPE - AscendMRotaryEmbedding
- **状态: 已支持，需确认 partial_rotary_factor 处理**
- `vllm_ascend/ops/rotary_embedding.py:472` 中 `AscendMRotaryEmbedding` 已实现
- Qwen3.5 的 `partial_rotary_factor=0.25`，实际 RoPE dim = 256 * 0.25 = 64
- `set_cos_and_sin` (rotary_embedding.py:78-86) 已正确处理:
  ```python
  if hasattr(model_config.hf_text_config, "partial_rotary_factor"):
      rope_dim = int(rope_dim * model_config.hf_text_config.partial_rotary_factor)
  ```
- `AscendMRotaryEmbedding.forward_oot` 中 `mrope_section=[16, 24, 24]` 判断:
  - Qwen3.5 的 `mrope_section=[11, 11, 10]`，不等于 `[16, 24, 24]`
  - 代码走 `super().forward_oot(positions, query, key)` -> 回退到 vLLM 原始 mRoPE 实现
  - 或走 `forward_triton` 路径（如果 `HAS_TRITON and positions.ndim == 2 and self.mrope_interleaved`）
- **结论: partial_rotary_factor=0.25 和 mrope_section=[11,11,10] 都已正确处理，无需修改**

##### 4. GemmaRMSNorm (Qwen3NextRMSNorm)
- **状态: 已支持**
- `vllm_ascend/ops/layernorm.py:90` 中 `AscendGemmaRMSNorm` 已实现 `(1+w)*x` 变换
- Qwen3NextRMSNorm 在 vLLM 中继承 GemmaRMSNorm，CustomOp 机制自动替换

##### 5. RMSNormGated (GDN gated norm)
- **状态: 已支持**
- `vllm_ascend/ops/layernorm.py:160` 中 `AscendRMSNormGated` 已实现

##### 6. ViT (Vision Encoder)
- **状态: 已支持，deepstack patch 覆盖已确认**
- `patch_qwen3vl.py` 中:
  - `Qwen3VLForConditionalGeneration._get_deepstack_input_embeds` 已被 tensor_parallel_wrap 包装
  - `Qwen3_VisionTransformer.fast_pos_embed_interpolate` 已被替换为 NPU 实现
- Qwen3.5 继承 `Qwen3VLForConditionalGeneration`，因此 deepstack patch 自动生效
- `Qwen3Attention.forward` 和 `Qwen3MoeAttention.forward` 的 fusion patch 仅用于 Qwen3/Qwen3Moe 模型，Qwen3.5 使用 `Qwen3NextAttention` 不受影响

##### 7. MTP (Multi-Token Prediction)
- **状态: 部分覆盖，需确认**
- `patch_qwen3_next_mtp.py` 替换了 `vllm.v1.worker.utils.bind_kv_cache`，这是一个全局 patch，对所有模型生效
- Qwen3.5 的 MTP 使用 `Qwen3_5MultiTokenPredictor` 和 `Qwen3_5MTP`（在 qwen3_5_mtp.py 中定义）
- vLLM 注册: `Qwen3_5MTP` -> `qwen3_5_mtp.py`
- MTP 内部使用 `Qwen3_5DecoderLayer`（包含 GDN 和 Full Attention），CustomOp 替换机制同样生效
- **不需要额外修改**

##### 8. Attention Backend
- **状态: 已支持**
- Qwen3.5 是 hybrid 模型 (GDN + GQA)，vllm-ascend 的 model_runner_v1 已支持 hybrid attention
- GDN 层使用 `GDNAttentionMetadataBuilder`（已被 patch_gdn_attn.py 增强）
- Full Attention 层使用标准 GQA FA backend
- model_runner_v1 中的 `_has_gdn` 检测使用 `check_gdn_layer` 遍历模型层结构，不依赖 model_type 字符串，因此自动覆盖 Qwen3.5

#### 需修改的文件清单

##### 修改项 1: pcp_utils.py - hybrid 检测条件扩展
- **文件**: `vllm_ascend/worker/pcp_utils.py`
- **行号**: 130
- **当前代码**:
  ```python
  self.pcp_use_hybrid_attn = self.vllm_config.model_config.hf_config.model_type == "qwen3_next"
  ```
- **需改为**:
  ```python
  self.pcp_use_hybrid_attn = self.vllm_config.model_config.hf_config.model_type in ("qwen3_next", "qwen3_5")
  ```
- **影响**: PCP (Prefill Context Parallelism) 场景下 hybrid attention 的正确处理
- **风险等级**: 低。当前不使用 PCP 时不会触发，但为了功能完整性必须修改
- **相关上下文**:
  - pcp_use_hybrid_attn 控制以下行为:
    - `update_tokens_for_pcp` 中的 linear attention 层的 token 分配逻辑（第 632-767 行）
    - `get_logits_indices` 中的 hybrid attention logits 索引（第 769-789 行）
    - `get_padded_slot_mapping` 中的 padding 策略（第 791-811 行）
    - `get_restore_hidden_states` 中的 allgather + restore 逻辑（第 813-841 行）
    - `generate_pcp_metadata` 中的 metadata 构建（第 1029-1272 行）

##### 修改项 2: recompute_scheduler.py - hybrid 检测条件扩展
- **文件**: `vllm_ascend/core/recompute_scheduler.py`
- **行号**: 123-125
- **当前代码**:
  ```python
  self.is_hybrid_model = (
      "qwen3_next" in self.vllm_config.model_config.hf_text_config.model_type
  )
  ```
- **需改为**:
  ```python
  self.is_hybrid_model = (
      "qwen3_next" in self.vllm_config.model_config.hf_text_config.model_type
      or "qwen3_5" in self.vllm_config.model_config.hf_text_config.model_type
  )
  ```
- **影响**: KV Transfer (KV Producer/Consumer) 场景下 hybrid 模型的 prompt token 处理
- **风险等级**: 低。仅在 KV Transfer 场景下（分布式推理）使用
- **相关上下文**:
  - is_hybrid_model 控制第 146 行: `if self.is_kv_producer and self.is_hybrid_model and request.num_tokens > 1:`
  - 该条件会从 prompt 中 pop 最后一个 token（hybrid 模型的特殊处理）
- **注意**: 使用 `"qwen3_5" in model_type` 而非 `==` 判断，与现有 `"qwen3_next" in model_type` 风格一致，可同时覆盖 `qwen3_5` 和 `qwen3_5_text`

##### 无需修改的组件汇总

| 组件 | 文件 | 原因 |
|------|------|------|
| GDN (Linear Attention) | `vllm_ascend/ops/gdn.py` | CustomOp 机制自动替换，与 model_type 无关 |
| GDN metadata builder | `vllm_ascend/patch/worker/patch_gdn_attn.py` | 全局 patch GDNAttentionMetadataBuilder，不检查 model_type |
| Full Attention (GQA) | `vllm_ascend/attention/attention_v1.py` | 标准 GQA FA 路径，不检查 model_type |
| mRoPE | `vllm_ascend/ops/rotary_embedding.py` | AscendMRotaryEmbedding 通过 CustomOp 注册，partial_rotary_factor 已处理 |
| GemmaRMSNorm | `vllm_ascend/ops/layernorm.py` | AscendGemmaRMSNorm 通过 CustomOp 注册 |
| RMSNormGated | `vllm_ascend/ops/layernorm.py` | AscendRMSNormGated 通过 CustomOp 注册 |
| SiLU activation | `vllm_ascend/ops/activation.py` | AscendSiluAndMul 通过 CustomOp 注册 |
| Conv3d (ViT) | `vllm_ascend/ops/conv.py` | AscendConv3dLayer 通过 CustomOp 注册 |
| MM Encoder Attention | `vllm_ascend/ops/mm_encoder_attention.py` | AscendMMEncoderAttention 通过 CustomOp 注册 |
| Linear 投影 | 各 ops 文件 | AscendColumnParallelLinear 等通过 CustomOp 注册 |
| Embedding / LMHead | 各 ops 文件 | AscendVocabParallelEmbedding 等通过 CustomOp 注册 |
| ViT deepstack | `vllm_ascend/patch/worker/patch_qwen3vl.py` | Qwen3.5 继承 Qwen3VLForConditionalGeneration，patch 自动生效 |
| MTP bind_kv_cache | `vllm_ascend/patch/worker/patch_qwen3_next_mtp.py` | 全局替换 bind_kv_cache，对所有模型生效 |
| model_runner GDN 检测 | `vllm_ascend/worker/model_runner_v1.py` | check_gdn_layer 遍历模型层结构，不依赖 model_type |
| quantization config | `vllm_ascend/quantization/modelslim_config.py` | 当前不涉及量化，非阻塞项 |
| eagle proposer | `vllm_ascend/spec_decode/` | 当前不涉及投机解码，非阻塞项 |

#### 是否需要 AscendC 新算子
- **不需要**
- Qwen3.5 的所有组件均有对应的 NPU 实现:
  - GDN: npu_recurrent_gated_delta_rule (Decode) + chunk_gated_delta_rule (Prefill) + npu_causal_conv1d_custom
  - GQA FA: npu_fused_infer_attention_score
  - mRoPE: triton_mrope 或 npu_mrope
  - GemmaRMSNorm: npu_gemma_rms_norm / npu_add_rms_norm
  - 其他标准算子: torch_npu 内置 API

#### 特化模型代码需求
- **不需要**
- Qwen3.5 在 vllm-ascend 中通过 CustomOp 机制自动替换所有组件，无需在 `vllm_ascend/models/` 下创建特化子类
- 所有替换通过 `vllm_ascend/utils.py:719` 的注册映射自动完成

#### 适配方案总结

**整体修改量: 2 个文件, 2 处改动, 每处改动约 1-2 行**

| # | 文件 | 行号 | 改动内容 | 触发条件 |
|---|------|------|---------|---------|
| 1 | `vllm_ascend/worker/pcp_utils.py` | 130 | hybrid 检测条件添加 `"qwen3_5"` | 使用 PCP 时 |
| 2 | `vllm_ascend/core/recompute_scheduler.py` | 123-125 | hybrid 检测条件添加 `"qwen3_5"` | 使用 KV Transfer 时 |

**首次运行验证步骤**:
1. 修改上述 2 个文件
2. 使用 eager 模式 + text-only 输入启动:
   ```bash
   env -u LD_PRELOAD -u PROXYCHAINS_CONF_FILE \
       OMP_NUM_THREADS=1 KMP_INIT_AT_FORK=FALSE \
       vllm serve /home/cb/glm5.1/vllm-custom/Qwen3.5-27B \
       --tensor-parallel-size 8 \
       --max-model-len 2048 \
       --gpu-memory-utilization 0.8 \
       --enforce-eager \
       --trust-remote-code \
       --distributed-executor-backend mp
   ```
3. 发送验证请求确认输出合理
4. 若 text-only 通过，再测试 multimodal 输入

### 阶段 3：NPU 功能验证（已完成）

#### 环境发现
- **关键问题**: `VLLM_WORKER_MULTIPROC_METHOD=spawn` 必须设置
  - 原因: torch 2.9.0+cpu 的 OpenMP 在 fork 子进程中调用 `set_num_threads` 时线程池状态无效
  - 错误: `pool INTERNAL ASSERT FAILED at ParallelOpenMP.cpp:64: Invalid thread pool!`
  - 解决: 使用 spawn 代替 fork 启动 worker 进程
- **不需要** `OMP_NUM_THREADS=1` 和 `KMP_INIT_AT_FORK=FALSE`（spawn 模式下无此问题）

#### 验证命令
```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
env -u LD_PRELOAD -u PROXYCHAINS_CONF_FILE \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
python3 -c "
from vllm import LLM, SamplingParams
llm = LLM(
    model='/home/cb/glm5.1/vllm-custom/Qwen3.5-27B',
    tensor_parallel_size=4,
    trust_remote_code=True,
    max_model_len=4096,
    enforce_eager=True,
)
outputs = llm.generate(['What is the capital of France?', ...], SamplingParams(max_tokens=128, temperature=0))
"
```

#### 验证结果
| 测试项 | 结果 | 说明 |
|--------|------|------|
| 模型加载 | PASS | TP=4, eager mode, 4 workers 正常初始化 |
| "What is the capital of France?" | PASS | 输出 "Paris" + 详细解释，语义正确 |
| "Write a haiku about programming." | PASS | 生成合理俳句 |
| "2 + 2 = " | PASS | 正确回答 4 |
| 无 NaN/Inf | PASS | 输出无异常值 |
| 无运行时错误 | PASS | 无 aicore timeout / HCCL 错误 / OOM |
| HCCL 通信 | PASS | 4 rank 分布式通信正常 |

#### 关键环境变量清单
```bash
env -u LD_PRELOAD -u PROXYCHAINS_CONF_FILE VLLM_WORKER_MULTIPROC_METHOD=spawn
```

### 阶段 4：性能优化（进行中）

#### 4.1 性能基线数据

##### Throughput 基线 (eager mode, TP=4, max_model_len=4096)

| 场景 | input_len | output_len | batch | total_toks/s | out_toks/s | decode_lat(ms) | peak_HBM(MB) | total_time(s) |
|------|-----------|------------|-------|-------------|-----------|---------------|-------------|--------------|
| tp4_short_1 | 128 | 128 | 1 | 9.74 | 3.91 | 256.0 | 60019 | 32.8 |
| tp4_short_4 | 128 | 128 | 4 | 31.88 | 12.79 | 78.2 | 60019 | 40.0 |
| tp4_short_8 | 128 | 128 | 8 | 65.41 | 20.49 | 48.8 | 60100 | 34.0 |
| tp4_medium_1 | 512 | 256 | 1 | 13.39 | 4.14 | 241.4 | 60017 | 61.8 |
| tp4_medium_4 | 512 | 256 | 4 | 43.29 | 9.05 | 110.4 | 60098 | 66.7 |
| tp4_long_1 | 2048 | 512 | 1 | 18.06 | 4.16 | 240.4 | 60117 | 123.1 |

##### 显存分析 (per NPU, TP=4)
- 总 HBM: 65536 MB
- 模型权重: 12873 MB (12.87 GB) per NPU
- KV Cache: ~41500 MB (41.5 GB) per NPU — 1,196,032 tokens
- 峰值激活: ~1190 MB (1.19 GB)
- 非torch内存: ~550 MB (0.55 GB)
- NPU Graph: 0 MB (eager mode)
- 剩余: ~6 GB
- KV cache 并发度: 292x (4096 tokens/request)

##### 关键发现
1. **解码延迟极高**: batch=1 时 ~240-256 ms/token，远高于 4x 910B3 应有的水平（预期 <30ms）
2. **Batch 扩展性良好**: batch=4 延迟降至 78-110ms，batch=8 降至 49ms
3. **输出吞吐低**: batch=1 仅 3.9-4.2 tokens/s
4. **显存利用率合理**: 模型权重+KV cache 占用 ~55GB/65.5GB
5. **优化空间**: 图模式(torch.compile/aclgraph) + 融合算子 + SuperKernel 预期可将解码延迟降低 5-10x

##### TP 扩展性 (eager mode, short_1 场景)
| TP | out_toks/s | decode_lat(ms) | HBM(MB) | 备注 |
|----|-----------|---------------|---------|------|
| 4 | 3.91 | 256.0 | 60019 | 基线 |
| 8 | 2.43 | 412.3 | 60010 | 更慢！通信开销 > 计算收益 |

#### 4.2 性能瓶颈分析

##### 瓶颈根因
当前 eager mode 下 decode 延迟 240-256ms/token（理论最优 ~11ms），**21x 差距**主要来自：

1. **Kernel Launch 开销（最大瓶颈）**: 64 层每层 20+ 个小算子，每次 launch 需 50-200μs CPU 开销，总计 ~10-15ms/layer
2. **无算子融合**: RMSNorm+残差、gate+SiLU+mul、QKV 投影等均为独立 kernel
3. **无图模式**: 每个 token 重复一次完整的 Python 解释 → op dispatch 流程
4. **通信开销**: TP all-reduce 在 eager mode 下无法与计算 overlap

##### 已有 CustomOp 注册（已生效）
| 组件 | CustomOp | 替换方式 |
|------|----------|---------|
| GDN (48层) | AscendGatedDeltaNetAttention | npu_recurrent_gated_delta_rule + chunk_gated_delta_rule |
| RMSNorm | AscendGemmaRMSNorm | npu_gemma_rms_norm |
| RMSNorm+残差 | AscendRMSNormGated | npu_add_rms_norm |
| SiLU+Gate | AscendSiluAndMul | npu_silu_and_mul |
| mRoPE | AscendMRotaryEmbedding | CustomOp |
| FA (16层) | 标准 GQA | npu_fused_infer_attention_score |
| Conv3d (ViT) | AscendConv3dLayer | CustomOp |

##### 优化方向评估（按收益排序）

| 优先级 | 优化方向 | 预期收益 | 可行性 | 依赖 |
|--------|---------|---------|--------|------|
| **P0** | ACL Graph (PIECEWISE) | 解码延迟降低 3-5x | **立即可用** | 去掉 --enforce-eager |
| P1 | torch.compile (inductor) | 额外 1.5-2x | 需安装 torchair | torchair 库 |
| P2 | 融合算子增强 | 额外 10-20% | 中等 | 分析 fusion pass |
| P3 | 权重预取 | decode 5-10% | 低 | npu_prefetch API 已有 |
| P4 | SuperKernel | decode 10-20% | 不可用 | 当前环境无 SuperKernel |

##### P0: ACL Graph 实施方案
- **操作**: 去掉 `--enforce-eager`，使用 vLLM 默认编译模式
- vllm-ascend 默认使用 `CUDAGraphMode.PIECEWISE` (ACL Graph)
- ACL Graph 将每层的计算图缓存，消除 Python 解释和 kernel launch 开销
- **预期**: batch=1 decode 延迟降至 ~50-80ms/token，吞吐提升 3-5x
- **风险**: 混合架构 (GDN+GQA) 可能触发 graph break，需验证
- **命令**:
  ```bash
  env -u LD_PRELOAD -u PROXYCHAINS_CONF_FILE VLLM_WORKER_MULTIPROC_METHOD=spawn \
  vllm serve /home/cb/glm5.1/vllm-custom/Qwen3.5-27B \
    --tensor-parallel-size 4 --max-model-len 4096 \
    --gpu-memory-utilization 0.92 --trust-remote-code
  ```

#### 4.3 优化实施

##### P0: ACL Graph (PIECEWISE) — 已完成

启用 ACL Graph 模式（去掉 --enforce-eager），使用 PIECEWISE cudagraph:
- cudagraph_mode: PIECEWISE
- splitting_ops: 包含 gdn_attention_core + unified_attention_with_output + 其他 15 个
- cudagraph_capture_sizes: [1, 16, 48, 88, 120, 152, 192, 224, 256]
- NPU graph memory: ~0.1 GiB (vs 0 GiB eager)

**Eager vs ACL Graph 完整对比 (TP=4)**:

| 场景 | Eager out/s | ACL out/s | Eager lat(ms) | ACL lat(ms) | 提升 |
|------|------------|-----------|--------------|------------|------|
| short_1 (b=1) | 3.91 | **8.64** | 256.0 | **115.7** | **2.21x** |
| short_4 (b=4) | 12.79 | **30.81** | 78.2 | **32.5** | **2.41x** |
| short_8 (b=8) | 20.49 | **54.86** | 48.8 | **18.2** | **2.68x** |
| medium_1 (b=1) | 4.14 | **7.31** | 241.4 | **136.9** | **1.76x** |
| medium_4 (b=4) | 9.05 | **18.07** | 110.4 | **55.3** | **2.00x** |
| long_1 (b=1) | 4.16 | **8.87** | 240.4 | **112.8** | **2.13x** |

**关键发现**:
1. ACL Graph (PIECEWISE) 在所有场景下均有 **1.8-2.7x** 性能提升
2. batch=8 时 decode 延迟降至 **18.2ms**，接近理论最优 (~11ms)
3. batch=1 时延迟从 240-256ms 降至 **113-137ms**，仍有优化空间
4. 图编译耗时: ~78s (torch.compile) + ~18s (ACL Graph capture)，首次启动慢
5. ACL Graph 内存开销: 仅 0.1 GiB
6. 长输入场景 (medium_1, 512→256) 提升仅 1.76x，可能因 prefill 占比较高

**性能瓶颈分析**:
- batch=1 decode 延迟 113ms，仍比理论最优 (~11ms) 慢 10x
- PIECEWISE 模式 64 层产生 64+ 个 graph segment，段间有 Python 调度开销
- 通信 (HCCL all-reduce) 在每个 graph segment 后仍需 CPU 参与

**后续优化方向**:
- 尝试 FULL_DECODE_ONLY 模式（消除 graph split 开销）
- 安装 torchair 启用 npugraph_ex（更高效的图执行）
- 设置 HCCL_OP_EXPANSION_MODE=AIV 优化通信性能
