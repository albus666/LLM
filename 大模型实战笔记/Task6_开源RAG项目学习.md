# 开源RAG项目学习

> [toc]

本次学习的是两个大模型RAG范例。分别是 ***个人知识库助手*** 和 ***人情世故大模型***，具体内容如下：

## 一、个人知识库助手

### 1. 项目背景介绍

在数据量激增的当代社会，有效管理和检索信息成为了一项关键技能。为了应对这一挑战，本项目应运而生，旨在构建一个基于 Langchain 的个人知识库助手。该助手通过高效的信息管理系统和强大的检索功能，为用户提供了一个可靠的信息获取平台。

本项目的核心目标是充分发挥大型语言模型在处理自然语言查询方面的优势，同时针对用户需求进行定制化开发，以实现对复杂信息的智能理解和精确回应。在项目开发过程中，团队深入分析了大型语言模型的潜力与局限，特别是其在生成幻觉信息方面的倾向。为了解决这一问题，项目集成了 RAG 技术，这是一种结合检索和生成的方法，能够在生成回答之前先从大量数据中检索相关信息，从而显著提高了回答的准确性和可靠性。

通过 RAG 技术的引入，本项目不仅提升了对信息的检索精度，还有效抑制了 Langchain 可能产生的误导性信息。这种结合检索和生成的方法确保了智能助手在提供信息时的准确性和权威性，使其成为用户在面对海量数据时的得力助手。

### 2. 目标与意义

本项目致力于开发一个高效、智能的个人知识库系统，旨在优化用户在信息洪流中的知识获取流程。该系统通过集成 Langchain 和自然语言处理技术，实现了对分散数据源的快速访问和整合，使用户能够通过直观的自然语言交互高效地检索和利用信息。

项目的核心价值体现在以下几个方面：

1.**优化信息检索效率**：利用基于 Langchain 的框架，系统能够在生成回答前先从广泛的数据集中检索到相关信息，从而加速信息的定位和提取过程。

2.**强化知识组织与管理**：支持用户构建个性化的知识库，通过结构化存储和分类，促进知识的积累和有效管理，进而提升用户对专业知识的掌握和运用。

3.**辅助决策制定**：通过精确的信息提供和分析，系统增强了用户在复杂情境下的决策能力，尤其是在需要迅速做出判断和反应的场合。

4.**个性化信息服务**：系统允许用户根据自己的特定需求定制知识库，实现个性化的信息检索和服务，确保用户能够获得最相关和最有价值的知识。

5.**技术创新示范**：项目展示了 RAG 技术在解决 Langchain 幻觉问题方面的优势，通过结合检索和生成的方式，提高了信息的准确性和可靠性，为智能信息管理领域的技术创新提供了新的思路。

6.**推广智能助理应用**：通过用户友好的界面设计和便捷的部署选项，项目使得智能助理技术更加易于理解和使用，推动了该技术在更广泛领域的应用和普及

### 3. 技术实现

- **CPU**: Intel 5代处理器（云CPU方面，建议选择 2 核以上的云CPU服务）
- **内存（RAM）**: 至少 4 GB
- **操作系统**：Windows、macOS、Linux均可

**克隆储存库**

```shell
git clone https://github.com/logan-zou/Chat_with_Datawhale_langchain.git
cd Chat_with_Datawhale_langchainCopy to clipboardErrorCopied
```

**创建 Conda 环境并安装依赖项**

- python>=3.9
- pytorch>=2.0.0

```shell
# 创建 Conda 环境
conda create -n llm-universe python==3.9.0
# 激活 Conda 环境
conda activate llm-universe
# 安装依赖项
pip install -r requirements.txtCopy to clipboardErrorCopied
```

**项目运行**

- 启动服务为本地 API

```shell
# Linux 系统
cd project/serve
uvicorn api:app --reload Copy to clipboardErrorCopied
```

```shell
# Windows 系统
cd project/serve
python api.pyCopy to clipboardErrorCopied
```

- 运行项目

```shell
cd llm-universe/project/serve
python run_gradio.py -model_name='chatglm_std' -embedding_model='m3e' - db_path='../../data_base/knowledge_db' -persist_path='../../data_base/vector_db'
```

## 二、人情世故大模型

### 1. 项目背景介绍

在中国，餐桌敬酒不仅仅是简单的举杯祝酒，它更是一种深刻的社交艺术，蕴含着丰富的文化传统和细腻的人情世故。在各种宴会、聚餐场合中，如何恰当地进行敬酒，不仅能够展现出主人的热情与礼貌，还能够加深与宾客之间的感情，促进双方的关系更加和谐。但对于许多人来说，餐桌敬酒的繁琐礼节和难以把握的度，往往让人感到既苦恼又头疼。

别急别急，人情世故小助手*Tianji*（天机）已上线，帮助我们解决一切餐桌敬酒的难题。从准备酒言到举杯祝福，从轮次安排到回敬策略，它将为我们提供一系列的指南和建议，帮助我们轻松应对各种场合，展现我们的风采和智慧。让我们一起走进人情世故小助手天机的世界吧！

天机是 SocialAI（来事儿AI）制作的一款免费使用、非商业用途的人工智能系统。我们可以利用它进行涉及传统人情世故的任务，如：如何敬酒、如何说好话、如何会来事儿等，以提升您的情商和"核心竞争能力"。

来事儿AI构建并开源了常见的大模型应用范例，涉及prompt、Agent、知识库、模型训练等多种技术。

### 2.目标与意义

在人工智能的发展历程中，我们一直在探索如何让机器更加智能，如何使它们不仅仅能够理解复杂的数据和逻辑，更能够理解人类的情感、文化乃至人情世故。这一追求不仅仅是技术的突破，更是对人类智慧的一种致敬。

天机团队旨在探索大模型与人情世故法则结合的多种技术路线，构建AI服务于生活的智能应用。这是通向通用人工智能（AGI）的关键一步，是实现机器与人类更深层次交流的必由之路。

*我们坚信，只有人情世故才是未来AI的核心技术，只有会来事儿的AI才有机会走向AGI，让我们携手见证通用人工智能的来临。 —— "天机不可泄漏。"*

### 3. 技术实现

```shell
克隆仓库：git clone https://github.com/SocialAI-tianji/Tianji.git
创建虚拟环境：conda create -n TJ python=3.11
激活环境：conda activate TJ
安装环境依赖：pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

在项目内创建.env文件，填写你的大模型秘钥

```env
OPENAI_API_KEY=
OPENAI_API_BASE=
ZHIPUAI_API_KEY=
BAIDU_API_KEY=
OPENAI_API_MODEL=
HF_HOME='./cache/'
HF_ENDPOINT = 'https://hf-mirror.com'
HF_TOKEN=
```
