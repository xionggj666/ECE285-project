# Controllable Tarot Card Generation Pipeline

一个面向 **AIGC 算法工程** 的塔罗牌生成项目，目标是在统一风格下生成整套塔罗牌，并尽可能保证语义元素完整、布局合理、符号数量可控。

本项目采用**结构化生成流程**，将塔罗牌生成拆分为多个阶段：语义规格生成（spec）、布局控制（layout）、扩散渲染（diffusion / ControlNet）、LoRA 风格适配，以及符号级组合与后处理。近期还补充了 **vLLM 语义规划** 与 **推理 benchmark**，使项目更接近实际 AIGC 算法工程场景。

---

## 1. 项目目标

相比直接使用 prompt 生成图片，本项目希望解决以下问题：

- **整套卡牌风格一致**：同一副牌在色调、构图、绘画风格上保持统一
- **语义结构更可控**：不同牌面的主体、背景、符号元素尽量符合塔罗语义
- **数量控制更稳定**：对于 Minor Arcana（如 *Four of Pentacles*），尽量保证符号数量正确
- **流程工程化**：支持模块化推理、benchmark、LoRA/ControlNet 对比、vLLM 语义规划接入

---

## 2. 项目结构

```text
data/
├── images/                  # 原始或整理后的图像数据
├── layouts/                 # 旧版布局/控制图
├── layouts_new/             # 新版布局/控制图
├── tarot-images.json        # 图像元数据
├── tarot_specs.json         # 塔罗牌规则/spec 数据
└── train.jsonl              # 训练/微调相关数据

loras/
└── ...                      # LoRA 权重目录

scripts/
├── serve_vllm.sh            # 本地/远程 vLLM 服务启动脚本
├── benchmark_tarot_infer.py # 推理 benchmark 脚本
├── pipeline.py              # 生成流程主逻辑/模块封装
├── prepare_images.py        # 数据预处理与图像准备
└── tarot_controlnet_generate.py  # 基于 ControlNet 的生成脚本
```

---

## 3. 方法概览

项目当前使用的核心思路如下：

### 3.1 语义规格生成（Spec Generation）
输入全局 deck prompt（例如 `impressionist tarot deck, dark blue and gold, antique mystical atmosphere`）以及具体牌名（例如 `The Fool`、`The Hermit`、`Four of Pentacles`），生成结构化卡牌规格，包括：

- 卡牌名称
- 主体人物/物体
- 背景描述
- 符号元素
- 情绪/氛围
- 色彩提示
- 数量提示（count hint）

最初使用**规则式 spec 生成**，后续补充了**vLLM 驱动的 spec 生成**，用于提升语义规划灵活性。

### 3.2 布局控制（Layout Guidance）
根据卡牌 spec 或预定义模板生成控制布局，用于约束主体位置、符号分布和整体构图。  
目前项目中 `data/layouts/` 与 `data/layouts_new/` 存放了相关布局资源。

### 3.3 扩散渲染（Diffusion Rendering）
使用 Stable Diffusion / Diffusers 作为主渲染器，根据 deck prompt + card prompt 生成塔罗牌画面。  
在此基础上可接入：

- **ControlNet**：增强布局与轮廓控制
- **LoRA**：增强整体风格一致性

### 3.4 符号级组合（Symbol Composition）
对于 Minor Arcana，项目尝试将 canonical symbol prototype 与 slot/copy-place 机制结合，提升重复符号（如金币、圣杯、权杖、宝剑）数量与位置控制能力。

---

## 4. 已完成的工程化扩展

除了基础生成流程外，项目还补充了以下工程化能力：

### 4.1 推理 benchmark
已对以下配置进行了单张推理 benchmark：

- `base`
- `LoRA-only`
- `ControlNet + LoRA`

记录指标包括：

- 单张生成耗时
- 峰值 GPU 显存
- 配置差异对推理开销的影响

### 4.2 LoRA / ControlNet 推理对比
通过相同卡牌、相同 prompt、不同配置的对比，分析：

- 风格增强是否有效
- ControlNet 是否提升布局约束
- 推理开销是否显著增加

### 4.3 vLLM 语义规划接入
项目已补充 vLLM 远程推理 demo，用于通过指令微调开源模型生成结构化 tarot spec。  
目标是将原有 rule-based spec parser 替换为可切换的 **vLLM semantic planner**，从而实现：

```text
用户 deck prompt + card name
-> vLLM 生成 spec
-> layout / rendering pipeline
-> final tarot card
```

---

## 5. 运行流程

### 5.1 数据准备
根据需要预处理图像或布局资源：

```bash
python scripts/prepare_images.py
```

### 5.2 启动 vLLM（可选）
如果要使用 vLLM 生成 spec，可先启动服务：

```bash
bash scripts/serve_vllm.sh
```

或者使用远程 RunPod / vLLM endpoint。

### 5.3 生成塔罗牌
使用主生成脚本进行单张或整套生成，例如：

```bash
python scripts/tarot_controlnet_generate.py \
  --prompt "impressionist tarot deck, dark blue and gold, antique mystical atmosphere" \
  --card "The Fool" \
  --layout_dir data/layouts \
  --lora_path loras/your_lora \
  --controlnet_path lllyasviel/sd-controlnet-scribble
```

> 实际参数请根据你的本地代码版本和模型路径调整。

### 5.4 运行推理 benchmark
```bash
python scripts/benchmark_tarot_infer.py
```

---

## 6. 实验设置

项目中常见的对比设置包括：

### Base
- 不使用 LoRA
- 不使用 ControlNet

### LoRA-only
- 使用 LoRA
- 不使用 ControlNet

### ControlNet + LoRA
- 同时使用 LoRA 与 ControlNet

这三类设置可用于分析生成质量、风格一致性、布局控制能力和推理开销之间的关系。



