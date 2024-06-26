# 构建RAG应用

> ## 目录
>
> [toc]

## 一、LLM接入LangChain

LangChain 为基于 LLM 开发自定义应用提供了高效的开发框架，便于开发者迅速地激发 LLM 的强大能力，搭建 LLM 应用。LangChain 也同样支持多种大模型，内置了 OpenAI、LLAMA 等大模型的调用接口。但是，LangChain 并没有内置所有大模型，它通过允许用户自定义 LLM 类型，来提供强大的可扩展性。

### 使用LangChain调用百度文言一心

**注：以下代码需要将我们封装的代码[wenxin_llm.py](./wenxin_llm.py)下载到本 Notebook 的同级目录下，才可以直接使用。因为新版 LangChain 可以直接调用文心千帆 API，我们更推荐使用下一部分的代码来调用文心一言模型**

```python
# 需要下载源码
from wenxin_llm import Wenxin_LLM
```

我们希望像调用 ChatGPT 那样直接将秘钥存储在 .env 文件中，并将其加载到环境变量，从而隐藏秘钥的具体细节，保证安全性。因此，我们需要在 .env 文件中配置 `QIANFAN_AK` 和 `QIANFAN_SK`，并使用以下代码加载：

```python
from dotenv import find_dotenv, load_dotenv
import os

# 读取本地/项目的环境变量。

# find_dotenv()寻找并定位.env文件的路径
# load_dotenv()读取该.env文件，并将其中的环境变量加载到当前的运行环境中
# 如果你设置的是全局的环境变量，这行代码则没有任何作用。
_ = load_dotenv(find_dotenv())

# 获取环境变量 API_KEY
wenxin_api_key = os.environ["QIANFAN_AK"]
wenxin_secret_key = os.environ["QIANFAN_SK"]

llm = Wenxin_LLM(api_key=wenxin_api_key, secret_key=wenxin_secret_key, system="你是一个助手！")
llm.invoke("你好，请你自我介绍一下！")

## 或者使用
#llm(prompt="你好，请你自我介绍一下！")
```

## 二、构建检索问答链

在 `搭建数据库` 章节，我们已经介绍了如何根据自己的本地知识文档，搭建一个向量知识库。 在接下来的内容里，我们将使用搭建好的向量数据库，对 query 查询问题进行召回，并将召回结果和 query 结合起来构建 prompt，输入到大模型中进行问答。  

### 1. 加载向量数据库

首先，我们加载在前一章已经构建的向量数据库。注意，此处你需要使用和构建时相同的 Emedding。

```python
import sys
sys.path.append("../C3 搭建知识库") # 将父目录放入系统路径中

# 使用智谱 Embedding API，注意，需要将上一章实现的封装代码下载到本地
from zhipuai_embedding import ZhipuAIEmbeddings

from langchain.vectorstores.chroma import Chroma

# 从环境变量中加载你的 API_KEY
from dotenv import load_dotenv, find_dotenv
import os

_ = load_dotenv(find_dotenv())    # read local .env file
zhipuai_api_key = os.environ['ZHIPUAI_API_KEY']

# 加载向量数据库，其中包含了 ../../data_base/knowledge_db 下多个文档的 Embedding
# 定义 Embeddings
embedding = ZhipuAIEmbeddings()

# 向量数据库持久化路径
persist_directory = '../C3 搭建知识库/data_base/vector_db/chroma'

# 加载数据库
vectordb = Chroma(
    persist_directory=persist_directory,  # 允许我们将persist_directory目录保存到磁盘上
    embedding_function=embedding
)
print(f"向量库中存储的数量：{vectordb._collection.count()}")
```

我们可以测试一下加载的向量数据库，使用一个问题 query 进行向量检索。如下代码会在向量数据库中根据相似性进行检索，返回前 k 个最相似的文档。

> [!NOTE]
>
> 使用相似性搜索前，请确保已安装 OpenAI 开源的快速分词工具 tiktoken 包：`pip install tiktoken`

```py
question = "什么是prompt engineering?"
docs = vectordb.similarity_search(question,k=3)
print(f"检索到的内容数：{len(docs)}")

# 打印检索到的内容
for i, doc in enumerate(docs):
    print(f"检索到的第{i}个内容: \n {doc.page_content}", end="\n-----------------------------------------------------\n")
```

### 2. 创建一个LLM

在先前的笔记中已经实现，这里就不再过多赘述

### 3. 构建检索问答链

```python
from langchain.prompts import PromptTemplate

template = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答
案。最多使用三句话。尽量使答案简明扼要。总是在回答的最后说“谢谢你的提问！”。
{context}
问题: {question}
"""

QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context","question"],
                                 template=template)

# 再创建一个基于模板的检索链：
from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(llm,
                                       retriever=vectordb.as_retriever(),
                                       return_source_documents=True,
                                       chain_type_kwargs={"prompt":QA_CHAIN_PROMPT})
```

创建检索 QA 链的方法 RetrievalQA.from_chain_type() 有如下参数：
+ llm：指定使用的 LLM

+ 指定 chain type : RetrievalQA.from_chain_type(chain_type="map_reduce")，也可以利用load_qa_chain()方法指定chain type。

+ 自定义 prompt ：通过在RetrievalQA.from_chain_type()方法中，指定chain_type_kwargs参数，而该参数：chain_type_kwargs = {"prompt": PROMPT}

+ 返回源文档：通过RetrievalQA.from_chain_type()方法中指定：return_source_documents=True参数；也可以使用RetrievalQAWithSourceChain()方法，返回源文档的引用（坐标或者叫主键、索引）

### 4. 检索问答链测试效果

我们发现 LLM 对于一些近几年的知识以及非常识性的专业问题，回答的并不是很好。而加上我们的本地知识，就可以帮助 LLM 做出更好的回答。另外，也有助于缓解大模型的“幻觉”问题。

### 5. 添加历史对话的记忆功能

现在我们已经实现了通过上传本地知识文档，然后将他们保存到向量知识库，通过将查询问题与向量知识库的召回结果进行结合输入到 LLM 中，我们就得到了一个相比于直接让 LLM 回答要好得多的结果。在与语言模型交互时，有一个关键问题 - **它们并不记得你之前的交流内容**。这在我们构建一些应用程序（如聊天机器人）的时候，带来了很大的挑战，使得对话似乎缺乏真正的连续性。

+ #### 记忆

  在本节中我们将介绍 LangChain 中的储存模块，即如何将先前的对话嵌入到语言模型中的，使其具有连续对话的能力。我们将使用 `ConversationBufferMemory` ，它保存聊天消息历史记录的列表，这些历史记录将在回答问题时与问题一起传递给聊天机器人，从而将它们添加到上下文中。

  ```python
  from langchain.memory import ConversationBufferMemory
  
  memory = ConversationBufferMemory(
      memory_key="chat_history",  # 与 prompt 的输入变量保持一致。
      return_messages=True  # 将以消息列表的形式返回聊天记录，而不是单个字符串
  )
  ```
  
+ #### 对话检索链

  对话检索链（ConversationalRetrievalChain）在检索 QA 链的基础上，增加了处理对话历史的能力。
  
  它的工作流程是:
  1. 将之前的对话与新问题合并生成一个完整的查询语句。
  2. 在向量数据库中搜索该查询的相关文档。
  3. 获取结果后,存储所有答案到对话记忆区。
  4. 用户可在 UI 中查看完整的对话流程。
  
  
  
  这种链式方式将新问题放在之前对话的语境中进行检索，可以处理依赖历史信息的查询。并保留所有信
  
  息在对话记忆中，方便追踪。



## 三、部署知识库助手

### 1. streamlit 简介

Streamlit 是一种快速便捷的方法，可以直接在 ***Python 中通过友好的 Web 界面演示机器学习模型***。我们可以**使用它为生成式人工智能应用程序构建用户界面**。在构建了机器学习模型后，如果你想构建一个 demo 给其他人看，也许是为了获得反馈并推动系统的改进，或者只是因为你觉得这个系统很酷，所以想演示一下：Streamlit 可以让您通过 Python 接口程序快速实现这一目标，而无需编写任何前端、网页或 JavaScript 代码。


`Streamlit` 是一个用于快速创建数据应用程序的开源 Python 库。它的设计目标是让数据科学家能够轻松地将数据分析和机器学习模型转化为具有交互性的 Web 应用程序，而无需深入了解 Web 开发。和常规 Web 框架，如 Flask/Django 的不同之处在于，它不需要你去编写任何客户端代码（HTML/CSS/JS），只需要编写普通的 Python 模块，就可以在很短的时间内创建美观并具备高度交互性的界面，从而快速生成数据分析或者机器学习的结果；另一方面，和那些只能通过拖拽生成的工具也不同的是，你仍然具有对代码的完整控制权。

Streamlit 提供了一组简单而强大的基础模块，用于构建数据应用程序：

- st.write()：这是最基本的模块之一，用于在应用程序中呈现文本、图像、表格等内容。

- st.title()、st.header()、st.subheader()：这些模块用于添加标题、子标题和分组标题，以组织应用程序的布局。

- st.text()、st.markdown()：用于添加文本内容，支持 Markdown 语法。

- st.image()：用于添加图像到应用程序中。

- st.dataframe()：用于呈现 Pandas 数据框。

- st.table()：用于呈现简单的数据表格。

- st.pyplot()、st.altair_chart()、st.plotly_chart()：用于呈现 Matplotlib、Altair 或 Plotly 绘制的图表。

- st.selectbox()、st.multiselect()、st.slider()、st.text_input()：用于添加交互式小部件，允许用户在应用程序中进行选择、输入或滑动操作。

- st.button()、st.checkbox()、st.radio()：用于添加按钮、复选框和单选按钮，以触发特定的操作。

这些基础模块使得通过 Streamlit 能够轻松地构建交互式数据应用程序，并且在使用时可以根据需要进行组合和定制，更多内容请查看[官方文档](https://docs.streamlit.io/get-started)

### 2. 构建应用程序

```python
# 首先，创建一个新的 Python 文件并将其保存 streamlit_app.py在工作目录的根目录中
# 导入必要的 Python 库
import streamlit as st
from langchain_openai import ChatOpenAI

# 创建应用程序的标题 st.title
st.title('🦜🔗 动手学大模型应用开发')

# 添加一个文本输入框，供用户输入其 OpenAI API 密钥
openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')

# 定义一个函数，使用用户密钥对 OpenAI API 进行身份验证、发送提示并获取 AI 生成的响应。该函数接受用户的提示作为参数，并使用 st.info 来在蓝色框中显示 AI 生成的响应
def generate_response(input_text):
    llm = ChatOpenAI(temperature=0.7, openai_api_key=openai_api_key)
    st.info(llm(input_text))

# 最后，使用 st.form() 创建一个文本框（st.text_area()）供用户输入。当用户单击`Submit`时，`generate-response()`将使用用户的输入作为参数来调用该函数
with st.form('my_form'):
    text = st.text_area('Enter text:', 'What are the three key pieces of advice for learning how to code?')
    submitted = st.form_submit_button('Submit')
    if not openai_api_key.startswith('sk-'):
        st.warning('Please enter your OpenAI API key!', icon='⚠')
    if submitted and openai_api_key.startswith('sk-'):
        generate_response(text)

# 保存当前的文件 streamlit_app.py
# 返回计算机的终端以运行该应用程序
streamlit run streamlit_app.py
```

