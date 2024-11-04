<h2 align="center">HanLP: Han Language Processing</h2>

<<<<<<< HEAD
 [English](https://github.com/hankcs/HanLP/tree/master) | [1.x版](https://github.com/hankcs/HanLP/tree/1.x) | [论坛](https://bbs.hankcs.com/) | [docker](https://github.com/WalterInSH/hanlp-jupyter-docker)

面向生产环境的多语种自然语言处理工具包，基于 TensorFlow 2.0，目标是普及落地最前沿的NLP技术。HanLP具备功能完善、性能高效、架构清晰、语料时新、可自定义的特点。内部算法经过工业界和学术界考验，配套书籍[《自然语言处理入门》](http://nlp.hankcs.com/book.php)已经出版。目前，基于深度学习的HanLP 2.0正处于alpha测试阶段，未来将实现知识图谱、问答系统、自动摘要、文本语义相似度、指代消解、三元组抽取、实体链接等功能。欢迎加入[蝴蝶效应](https://bbs.hankcs.com/)参与讨论，或者反馈bug和功能请求到[issue区](https://github.com/hankcs/HanLP/issues)。Java用户请使用[1.x分支](https://github.com/hankcs/HanLP/tree/1.x) ，经典稳定，永久维护。RESTful API正在开发中，2.0正式版将支持包括Java、Python在内的开发语言。

 ## 安装
=======
<div align="center">
    <a href="https://github.com/hankcs/HanLP/actions/workflows/unit-tests.yml">
       <img alt="Unit Tests" src="https://github.com/hankcs/hanlp/actions/workflows/unit-tests.yml/badge.svg?branch=master">
    </a>
    <a href="https://pypi.org/project/hanlp/">
        <img alt="PyPI Version" src="https://img.shields.io/pypi/v/hanlp?color=blue">
    </a>
    <a href="https://pypi.org/project/hanlp/">
        <img alt="Python Versions" src="https://img.shields.io/pypi/pyversions/hanlp?colorB=blue">
    </a>
    <a href="https://pepy.tech/project/hanlp">
        <img alt="Downloads" src="https://static.pepy.tech/badge/hanlp">
    </a>
    <a href="https://mybinder.org/v2/gh/hankcs/HanLP/doc-zh?filepath=plugins%2Fhanlp_demo%2Fhanlp_demo%2Fzh%2Ftutorial.ipynb">
        <img alt="在线运行" src="https://mybinder.org/badge_logo.svg">
    </a>
</div>
<h4 align="center">
    <a href="https://github.com/hankcs/HanLP/tree/master">English</a> |
    <a href="https://github.com/hankcs/HanLP/tree/doc-ja">日本語</a> |
    <a href="https://hanlp.hankcs.com/docs/">文档</a> |
    <a href="https://bbs.hankcs.com/t/topic/3940">论文</a> |
    <a href="https://bbs.hankcs.com/">论坛</a> |
    <a href="https://github.com/wangedison/hanlp-jupyterlab-docker">docker</a> |
    <a href="https://mybinder.org/v2/gh/hankcs/HanLP/doc-zh?filepath=plugins%2Fhanlp_demo%2Fhanlp_demo%2Fzh%2Ftutorial.ipynb">▶️在线运行</a>
</h4>



面向生产环境的多语种自然语言处理工具包，基于PyTorch和TensorFlow 2.x双引擎，目标是普及落地最前沿的NLP技术。HanLP具备功能完善、精度准确、性能高效、语料时新、架构清晰、可自定义的特点。

[![demo](https://raw.githubusercontent.com/hankcs/OpenCC-to-HanLP/img/demo.gif)](https://mybinder.org/v2/gh/hankcs/HanLP/doc-zh?filepath=plugins%2Fhanlp_demo%2Fhanlp_demo%2Fzh%2Ftutorial.ipynb)

借助世界上最大的多语种语料库，HanLP2.1支持包括简繁中英日俄法德在内的[130种语言](https://hanlp.hankcs.com/docs/api/hanlp/pretrained/mtl.html#hanlp.pretrained.mtl.UD_ONTONOTES_TOK_POS_LEM_FEA_NER_SRL_DEP_SDP_CON_MMINILMV2L6)上的10种联合任务以及多种单任务。HanLP预训练了十几种任务上的数十个模型并且正在持续迭代语料库与模型：

<div align="center">

| 功能                                                         | RESTful                                                      | 多任务                                                       | 单任务                                                       | 模型                                                         | 标注标准                                                     |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| [分词](https://hanlp.hankcs.com/demos/tok.html)              | [教程](https://github.com/hankcs/HanLP/blob/doc-zh/plugins/hanlp_demo/hanlp_demo/zh/tok_restful.ipynb) | [教程](https://github.com/hankcs/HanLP/blob/doc-zh/plugins/hanlp_demo/hanlp_demo/zh/tok_mtl.ipynb) | [教程](https://github.com/hankcs/HanLP/blob/doc-zh/plugins/hanlp_demo/hanlp_demo/zh/tok_stl.ipynb) | [tok](https://hanlp.hankcs.com/docs/api/hanlp/pretrained/tok.html) | [粗分](https://hanlp.hankcs.com/docs/annotations/tok/msr.html)、[细分](https://hanlp.hankcs.com/docs/annotations/tok/ctb.html) |
| [词性标注](https://hanlp.hankcs.com/demos/pos.html)          | [教程](https://github.com/hankcs/HanLP/blob/doc-zh/plugins/hanlp_demo/hanlp_demo/zh/pos_restful.ipynb) | [教程](https://github.com/hankcs/HanLP/blob/doc-zh/plugins/hanlp_demo/hanlp_demo/zh/pos_mtl.ipynb) | [教程](https://github.com/hankcs/HanLP/blob/doc-zh/plugins/hanlp_demo/hanlp_demo/zh/pos_stl.ipynb) | [pos](https://hanlp.hankcs.com/docs/api/hanlp/pretrained/pos.html) | [CTB](https://hanlp.hankcs.com/docs/annotations/pos/ctb.html)、[PKU](https://hanlp.hankcs.com/docs/annotations/pos/pku.html)、[863](https://hanlp.hankcs.com/docs/annotations/pos/863.html) |
| [命名实体识别](https://hanlp.hankcs.com/demos/ner.html)      | [教程](https://github.com/hankcs/HanLP/blob/doc-zh/plugins/hanlp_demo/hanlp_demo/zh/ner_restful.ipynb) | [教程](https://github.com/hankcs/HanLP/blob/doc-zh/plugins/hanlp_demo/hanlp_demo/zh/ner_mtl.ipynb) | [教程](https://github.com/hankcs/HanLP/blob/doc-zh/plugins/hanlp_demo/hanlp_demo/zh/ner_stl.ipynb) | [ner](https://hanlp.hankcs.com/docs/api/hanlp/pretrained/ner.html) | [PKU](https://hanlp.hankcs.com/docs/annotations/ner/pku.html)、[MSRA](https://hanlp.hankcs.com/docs/annotations/ner/msra.html)、[OntoNotes](https://hanlp.hankcs.com/docs/annotations/ner/ontonotes.html) |
| [依存句法分析](https://hanlp.hankcs.com/demos/dep.html)      | [教程](https://github.com/hankcs/HanLP/blob/doc-zh/plugins/hanlp_demo/hanlp_demo/zh/dep_restful.ipynb) | [教程](https://github.com/hankcs/HanLP/blob/doc-zh/plugins/hanlp_demo/hanlp_demo/zh/dep_mtl.ipynb) | [教程](https://github.com/hankcs/HanLP/blob/doc-zh/plugins/hanlp_demo/hanlp_demo/zh/dep_stl.ipynb) | [dep](https://hanlp.hankcs.com/docs/api/hanlp/pretrained/dep.html) | [SD](https://hanlp.hankcs.com/docs/annotations/dep/sd_zh.html)、[UD](https://hanlp.hankcs.com/docs/annotations/dep/ud.html#chinese)、[PMT](https://hanlp.hankcs.com/docs/annotations/dep/pmt.html) |
| [成分句法分析](https://hanlp.hankcs.com/demos/con.html)      | [教程](https://github.com/hankcs/HanLP/blob/doc-zh/plugins/hanlp_demo/hanlp_demo/zh/con_restful.ipynb) | [教程](https://github.com/hankcs/HanLP/blob/doc-zh/plugins/hanlp_demo/hanlp_demo/zh/con_mtl.ipynb) | [教程](https://github.com/hankcs/HanLP/blob/doc-zh/plugins/hanlp_demo/hanlp_demo/zh/con_stl.ipynb) | [con](https://hanlp.hankcs.com/docs/api/hanlp/pretrained/constituency.html) | [Chinese Tree Bank](https://hanlp.hankcs.com/docs/annotations/constituency/ctb.html) |
| [语义依存分析](https://hanlp.hankcs.com/demos/sdp.html)      | [教程](https://github.com/hankcs/HanLP/blob/doc-zh/plugins/hanlp_demo/hanlp_demo/zh/sdp_restful.ipynb) | [教程](https://github.com/hankcs/HanLP/blob/doc-zh/plugins/hanlp_demo/hanlp_demo/zh/sdp_mtl.ipynb) | [教程](https://github.com/hankcs/HanLP/blob/doc-zh/plugins/hanlp_demo/hanlp_demo/zh/sdp_stl.ipynb) | [sdp](https://hanlp.hankcs.com/docs/api/hanlp/pretrained/sdp.html) | [CSDP](https://hanlp.hankcs.com/docs/annotations/sdp/semeval16.html#) |
| [语义角色标注](https://hanlp.hankcs.com/demos/srl.html)      | [教程](https://github.com/hankcs/HanLP/blob/doc-zh/plugins/hanlp_demo/hanlp_demo/zh/srl_restful.ipynb) | [教程](https://github.com/hankcs/HanLP/blob/doc-zh/plugins/hanlp_demo/hanlp_demo/zh/srl_mtl.ipynb) | [教程](https://github.com/hankcs/HanLP/blob/doc-zh/plugins/hanlp_demo/hanlp_demo/zh/srl_stl.ipynb) | [srl](https://hanlp.hankcs.com/docs/api/hanlp/pretrained/srl.html) | [Chinese Proposition Bank](https://hanlp.hankcs.com/docs/annotations/srl/cpb.html) |
| [抽象意义表示](https://hanlp.hankcs.com/demos/amr.html)      | [教程](https://github.com/hankcs/HanLP/blob/doc-zh/plugins/hanlp_demo/hanlp_demo/zh/amr_restful.ipynb) | 暂无                                                         | [教程](https://github.com/hankcs/HanLP/blob/doc-zh/plugins/hanlp_demo/hanlp_demo/zh/amr_stl.ipynb) | [amr](https://hanlp.hankcs.com/docs/api/hanlp/pretrained/amr.html) | [CAMR](https://www.hankcs.com/nlp/corpus/introduction-to-chinese-abstract-meaning-representation.html) |
| [指代消解](https://hanlp.hankcs.com/demos/cor.html)          | [教程](https://github.com/hankcs/HanLP/blob/doc-zh/plugins/hanlp_demo/hanlp_demo/zh/cor_restful.ipynb) | 暂无                                                         | 暂无                                                         | 暂无                                                         | OntoNotes                                                    |
| [语义文本相似度](https://hanlp.hankcs.com/demos/sts.html)    | [教程](https://github.com/hankcs/HanLP/blob/doc-zh/plugins/hanlp_demo/hanlp_demo/zh/sts_restful.ipynb) | 暂无                                                         | [教程](https://github.com/hankcs/HanLP/blob/doc-zh/plugins/hanlp_demo/hanlp_demo/zh/sts_stl.ipynb) | [sts](https://hanlp.hankcs.com/docs/api/hanlp/pretrained/sts.html) | 暂无                                                         |
| [文本风格转换](https://hanlp.hankcs.com/demos/tst.html)      | [教程](https://github.com/hankcs/HanLP/blob/doc-zh/plugins/hanlp_demo/hanlp_demo/zh/tst_restful.ipynb) | 暂无                                                         | 暂无                                                         | 暂无                                                         | 暂无                                                         |
| [关键词短语提取](https://hanlp.hankcs.com/demos/keyphrase.html) | [教程](https://github.com/hankcs/HanLP/blob/doc-zh/plugins/hanlp_demo/hanlp_demo/zh/keyphrase_restful.ipynb) | 暂无                                                         | 暂无                                                         | 暂无                                                         | 暂无                                                         |
| [抽取式自动摘要](https://hanlp.hankcs.com/demos/exsum.html)  | [教程](https://github.com/hankcs/HanLP/blob/doc-zh/plugins/hanlp_demo/hanlp_demo/zh/extractive_summarization_restful.ipynb) | 暂无                                                         | 暂无                                                         | 暂无                                                         | 暂无                                                         |
| [生成式自动摘要](https://hanlp.hankcs.com/demos/absum.html)  | [教程](https://github.com/hankcs/HanLP/blob/doc-zh/plugins/hanlp_demo/hanlp_demo/zh/abstractive_summarization_restful.ipynb) | 暂无                                                         | 暂无                                                         | 暂无                                                         | 暂无                                                         |
| [文本语法纠错](https://hanlp.hankcs.com/demos/gec.html)      | [教程](https://github.com/hankcs/HanLP/blob/doc-zh/plugins/hanlp_demo/hanlp_demo/zh/gec_restful.ipynb) | 暂无                                                         | 暂无                                                         | 暂无                                                         | 暂无                                                         |
| [文本分类](https://hanlp.hankcs.com/demos/classification.html) | [教程](https://github.com/hankcs/HanLP/blob/doc-zh/plugins/hanlp_demo/hanlp_demo/zh/classification_restful.ipynb) | 暂无                                                         | 暂无                                                         | 暂无                                                         | 暂无                                                         |
| [情感分析](https://hanlp.hankcs.com/demos/sentiment.html)    | [教程](https://github.com/hankcs/HanLP/blob/doc-zh/plugins/hanlp_demo/hanlp_demo/zh/sentiment_restful.ipynb) | 暂无                                                         | 暂无                                                         | 暂无                                                         | `[-1,+1]`                                                    |
| [语种检测](https://hanlp.hankcs.com/demos/classification.html) | [教程](https://github.com/hankcs/HanLP/blob/doc-zh/plugins/hanlp_demo/hanlp_demo/zh/lid_restful.ipynb) | 暂无                                                         | [教程](https://github.com/hankcs/HanLP/blob/doc-zh/plugins/hanlp_demo/hanlp_demo/zh/lid_stl.ipynb) | 暂无                                                         | [ISO 639-1编码](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes) |

</div>

- 词干提取、词法语法特征提取请参考[英文教程](https://hanlp.hankcs.com/docs/tutorial.html)；[词向量](https://hanlp.hankcs.com/docs/api/hanlp/pretrained/word2vec.html)和[完形填空](https://hanlp.hankcs.com/docs/api/hanlp/pretrained/mlm.html)请参考相应文档。
- 简繁转换、拼音、新词发现、文本聚类请参考[1.x教程](https://github.com/hankcs/HanLP/tree/1.x)。

量体裁衣，HanLP提供**RESTful**和**native**两种API，分别面向轻量级和海量级两种场景。无论何种API何种语言，HanLP接口在语义上保持一致，在代码上坚持开源。如果您在研究中使用了HanLP，请引用我们的[EMNLP论文](https://aclanthology.org/2021.emnlp-main.451/)。

### 轻量级RESTful API

仅数KB，适合敏捷开发、移动APP等场景。简单易用，无需GPU配环境，秒速安装。语料更多、模型更大、精度更高，**强烈推荐**。服务器GPU算力有限，匿名用户配额较少，[建议申请**免费公益**API秘钥`auth`](https://bbs.hanlp.com/t/hanlp2-1-restful-api/53)。

#### Python

```shell
pip install hanlp_restful
```

创建客户端，填入服务器地址和秘钥：

```python
from hanlp_restful import HanLPClient
HanLP = HanLPClient('https://www.hanlp.com/api', auth=None, language='zh') # auth不填则匿名，zh中文，mul多语种
```

#### Golang

安装 `go get -u github.com/hankcs/gohanlp@main` ，创建客户端，填入服务器地址和秘钥：

```go
HanLP := hanlp.HanLPClient(hanlp.WithAuth(""),hanlp.WithLanguage("zh")) // auth不填则匿名，zh中文，mul多语种
```

#### Java

在`pom.xml`中添加依赖：

```xml
<dependency>
    <groupId>com.hankcs.hanlp.restful</groupId>
    <artifactId>hanlp-restful</artifactId>
    <version>0.0.12</version>
</dependency>
```

创建客户端，填入服务器地址和秘钥：

```java
HanLPClient HanLP = new HanLPClient("https://www.hanlp.com/api", null, "zh"); // auth不填则匿名，zh中文，mul多语种
```

#### 快速上手

无论何种开发语言，调用`parse`接口，传入一篇文章，得到HanLP精准的分析结果。

```java
HanLP.parse("2021年HanLPv2.1为生产环境带来次世代最先进的多语种NLP技术。阿婆主来到北京立方庭参观自然语义科技公司。")
```

更多功能包括语义相似度、风格转换、指代消解等，请参考[文档](https://hanlp.hankcs.com/docs/api/restful.html)和[测试用例](https://github.com/hankcs/HanLP/blob/master/plugins/hanlp_restful/tests/test_client.py)。

### 海量级native API

依赖PyTorch、TensorFlow等深度学习技术，适合**专业**NLP工程师、研究者以及本地海量数据场景。要求Python 3.6至3.10，支持Windows，推荐*nix。可以在CPU上运行，推荐GPU/TPU。安装PyTorch版：
>>>>>>> 8328b7731f100fa2b4947f0faf85724cfff865a1

```bash
pip install hanlp
```

<<<<<<< HEAD
要求Python 3.6以上，支持Windows，可以在CPU上运行，推荐GPU/TPU。

## 快速上手

### 分词（中文分词、中文斷詞、英文分词、任意语种分词）

作为终端用户，第一步需要从磁盘或网络加载预训练模型。比如，此处用两行代码加载一个名为 `PKU_NAME_MERGED_SIX_MONTHS_CONVSEG` 的分词模型。

```python
>>> import hanlp
>>> tokenizer = hanlp.load('PKU_NAME_MERGED_SIX_MONTHS_CONVSEG')
```

HanLP 会自动将 `PKU_NAME_MERGED_SIX_MONTHS_CONVSEG` 解析为一个URL，然后自动下载并解压。由于巨大的用户量，万一下载失败请使用[国内镜像](https://bbs.hankcs.com/t/topic/1992)或参考提示手动下载。 

一旦模型下载完毕，即可将`tokenizer`当成一个函数调用：

```python
>>> tokenizer('商品和服务')
['商品', '和', '服务']
```

如果你要处理英文，一个基于规则的普通函数应该足够了。

```python
>>> tokenizer = hanlp.utils.rules.tokenize_english
>>> tokenizer("Don't go gentle into that good night.")
['Do', "n't", 'go', 'gentle', 'into', 'that', 'good', 'night', '.']
```

#### 并行

好消息，你可以运行得更快。在深度学习的时代，批处理通常带来`batch_size`的加速比。你可以并行切分多个句子，代价是消耗更多GPU显存。

```python
>>> tokenizer(['萨哈夫说，伊拉克将同联合国销毁伊拉克大规模杀伤性武器特别委员会继续保持合作。',
               '上海华安工业（集团）公司董事长谭旭光和秘书张晚霞来到美国纽约现代艺术博物馆参观。',
               'HanLP支援臺灣正體、香港繁體，具有新詞辨識能力的中文斷詞系統'])
[['萨哈夫', '说', '，', '伊拉克', '将', '同', '联合国', '销毁', '伊拉克', '大', '规模', '杀伤性', '武器', '特别', '委员会', '继续', '保持', '合作', '。'], 
 ['上海', '华安', '工业', '（', '集团', '）', '公司', '董事长', '谭旭光', '和', '秘书', '张晚霞', '来到', '美国', '纽约', '现代', '艺术', '博物馆', '参观', '。'], 
 ['HanLP', '支援', '臺灣', '正體', '、', '香港', '繁體', '，', '具有', '新詞', '辨識', '能力', '的', '中文', '斷詞', '系統']]
```

就是如此简单，你现在已经能够将HanLP提供的最新的深度学习模型应用到你的研究和工作中了。下面是一些小技巧：

- 打印 `hanlp.pretrained.ALL` 来列出HanLP中的所有预训练模型。比如，`CTB6_CONVSEG`是在CTB6上训练的分词模型。

- 参考[demo](https://github.com/hankcs/HanLP/blob/master/tests/demo/zh/demo_cws_trie.py)挂载用户词典，或嵌入正则表达式来应对你的业务逻辑。

- 使用 `hanlp.pretrained.*` 来分门别类地浏览预训练模型，你还可以通过变量来加载模型。

  ```python
  >>> hanlp.pretrained.cws.PKU_NAME_MERGED_SIX_MONTHS_CONVSEG
  'https://file.hankcs.com/hanlp/cws/pku98_6m_conv_ngram_20200110_134736.zip'
  ```

### 词性标注

词性标注器的输入是单词，输出是每个单词的词性标签。

```python
>>> tagger = hanlp.load(hanlp.pretrained.pos.PTB_POS_RNN_FASTTEXT_EN)
>>> tagger([['I', 'banked', '2', 'dollars', 'in', 'a', 'bank', '.'],
            ['Is', 'this', 'the', 'future', 'of', 'chamber', 'music', '?']])
[['PRP', 'VBD', 'CD', 'NNS', 'IN', 'DT', 'NN', '.'], 
 ['VBZ', 'DT', 'DT', 'NN', 'IN', 'NN', 'NN', '.']]
```

词性标注同样支持多语种，取决于你加载的是哪个模型（注意变量名后面的`ZH`）。

```python
>>> tagger = hanlp.load(hanlp.pretrained.pos.CTB5_POS_RNN_FASTTEXT_ZH)
>>> tagger(['我', '的', '希望', '是', '希望', '和平'])
['PN', 'DEG', 'NN', 'VC', 'VV', 'NN']
```

注意到句子中两个 `希望`的词性各不相同，第一个是名词而第二个是动词。关于词性标签，请参考[《自然语言处理入门》](http://nlp.hankcs.com/book.php)第七章，或等待正式文档。这个标注器使用了fasttext[^fasttext] 作为嵌入层，所以免疫于OOV。

### 命名实体识别

命名实体识别模块的输入是单词列表，输出是命名实体的边界和类别。

```python
>>> recognizer = hanlp.load(hanlp.pretrained.ner.CONLL03_NER_BERT_BASE_UNCASED_EN)
>>> recognizer(["President", "Obama", "is", "speaking", "at", "the", "White", "House"])
[('Obama', 'PER', 1, 2), ('White House', 'LOC', 6, 8)]
```

中文命名实体识别是字符级模型，所以不要忘了用 `list`将字符串转换为字符列表。至于输出，格式为 `(entity, type, begin, end)`。

```python
>>> recognizer = hanlp.load(hanlp.pretrained.ner.MSRA_NER_BERT_BASE_ZH)
>>> recognizer([list('上海华安工业（集团）公司董事长谭旭光和秘书张晚霞来到美国纽约现代艺术博物馆参观。'),
                list('萨哈夫说，伊拉克将同联合国销毁伊拉克大规模杀伤性武器特别委员会继续保持合作。')])
[[('上海华安工业（集团）公司', 'NT', 0, 12), ('谭旭光', 'NR', 15, 18), ('张晚霞', 'NR', 21, 24), ('美国', 'NS', 26, 28), ('纽约现代艺术博物馆', 'NS', 28, 37)], 
 [('萨哈夫', 'NR', 0, 3), ('伊拉克', 'NS', 5, 8), ('联合国销毁伊拉克大规模杀伤性武器特别委员会', 'NT', 10, 31)]]
```

这里的 `MSRA_NER_BERT_BASE_ZH` 是基于 BERT[^bert]的最准确的模型，你可以浏览该模型的评测指标：

```bash
$ cat ~/.hanlp/ner/ner_bert_base_msra_20200104_185735/test.log 
20-01-04 18:55:02 INFO Evaluation results for test.tsv - loss: 1.4949 - f1: 0.9522 - speed: 113.37 sample/sec 
processed 177342 tokens with 5268 phrases; found: 5316 phrases; correct: 5039.
accuracy:  99.37%; precision:  94.79%; recall:  95.65%; FB1:  95.22
               NR: precision:  96.39%; recall:  97.83%; FB1:  97.10  1357
               NS: precision:  96.70%; recall:  95.79%; FB1:  96.24  2610
               NT: precision:  89.47%; recall:  93.13%; FB1:  91.27  1349
```

### 依存句法分析

句法分析是NLP的核心任务，在许多硬派的学者和面试官看来，不懂句法分析的人称不上NLP研究者或工程师。然而通过HanLP，只需两行代码即可完成句法分析。

```python
>>> syntactic_parser = hanlp.load(hanlp.pretrained.dep.PTB_BIAFFINE_DEP_EN)
>>> print(syntactic_parser([('Is', 'VBZ'), ('this', 'DT'), ('the', 'DT'), ('future', 'NN'), ('of', 'IN'), ('chamber', 'NN'), ('music', 'NN'), ('?', '.')]))
1	Is	_	VBZ	_	_	4	cop	_	_
2	this	_	DT	_	_	4	nsubj	_	_
3	the	_	DT	_	_	4	det	_	_
4	future	_	NN	_	_	0	root	_	_
5	of	_	IN	_	_	4	prep	_	_
6	chamber	_	NN	_	_	7	nn	_	_
7	music	_	NN	_	_	5	pobj	_	_
8	?	_	.	_	_	4	punct	_	_
```

句法分析器的输入是单词列表及词性列表，输出是 CoNLL-X 格式[^conllx]的句法树，用户可通过 `CoNLLSentence` 类来操作句法树。一个中文例子:

```python
>>> syntactic_parser = hanlp.load(hanlp.pretrained.dep.CTB7_BIAFFINE_DEP_ZH)
>>> print(syntactic_parser([('蜡烛', 'NN'), ('两', 'CD'), ('头', 'NN'), ('烧', 'VV')]))
1	蜡烛	_	NN	_	_	4	nsubj	_	_
2	两	_	CD	_	_	3	nummod	_	_
3	头	_	NN	_	_	4	dep	_	_
4	烧	_	VV	_	_	0	root	_	_
```

关于句法标签，请参考[《自然语言处理入门》](http://nlp.hankcs.com/book.php)第11章，或等待正式文档。

### 语义依存分析

语义分析结果为一个有向无环图，称为语义依存图（Semantic Dependency Graph）。图中的节点为单词，边为语义依存弧，边上的标签为语义关系。

```python
>>> semantic_parser = hanlp.load(hanlp.pretrained.sdp.SEMEVAL15_PAS_BIAFFINE_EN)
>>> print(semantic_parser([('Is', 'VBZ'), ('this', 'DT'), ('the', 'DT'), ('future', 'NN'), ('of', 'IN'), ('chamber', 'NN'), ('music', 'NN'), ('?', '.')]))
1	Is	_	VBZ	_	_	0	ROOT	_	_
2	this	_	DT	_	_	1	verb_ARG1	_	_
3	the	_	DT	_	_	0	ROOT	_	_
4	future	_	NN	_	_	1	verb_ARG2	_	_
4	future	_	NN	_	_	3	det_ARG1	_	_
4	future	_	NN	_	_	5	prep_ARG1	_	_
5	of	_	IN	_	_	0	ROOT	_	_
6	chamber	_	NN	_	_	0	ROOT	_	_
7	music	_	NN	_	_	5	prep_ARG2	_	_
7	music	_	NN	_	_	6	noun_ARG1	_	_
8	?	_	.	_	_	0	ROOT	_	_
```

HanLP实现了最先进的biaffine[^biaffine] 模型，支持任意语种的语义依存分析：

```python
>>> semantic_parser = hanlp.load(hanlp.pretrained.sdp.SEMEVAL16_NEWS_BIAFFINE_ZH)
>>> print(semantic_parser([('蜡烛', 'NN'), ('两', 'CD'), ('头', 'NN'), ('烧', 'VV')]))
1	蜡烛	_	NN	_	_	3	Poss	_	_
1	蜡烛	_	NN	_	_	4	Pat	_	_
2	两	_	CD	_	_	3	Quan	_	_
3	头	_	NN	_	_	4	Loc	_	_
4	烧	_	VV	_	_	0	Root	_	_
```

输出依然是 `CoNLLSentence` 格式，只不过这次是一个图，图中任意节点可以有零个或任意多个中心词，比如 `蜡烛` 有两个中心词 (ID 3 和 4)。语义依存关系可参考《[中文语义依存分析语料库](https://www.hankcs.com/nlp/sdp-corpus.html)》，或等待正式文档。

### 流水线

既然句法和语义分析依赖于词性标注，而词性标注又依赖于分词。如果有一种类似于计算图的机制自动将这些模块串联起来就好了。HanLP设计的流水线可以灵活地将多个组件（统计模型或规则系统）组装起来：

```python
pipeline = hanlp.pipeline() \
=======
- HanLP每次发布都通过了Linux、macOS和Windows上Python3.6至3.10的[单元测试](https://github.com/hankcs/HanLP/actions?query=branch%3Amaster)，不存在安装问题。

HanLP发布的模型分为多任务和单任务两种，多任务速度快省显存，单任务精度高更灵活。

#### 多任务模型

HanLP的工作流程为加载模型然后将其当作函数调用，例如下列联合多任务模型：

```python
import hanlp
HanLP = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH) # 世界最大中文语料库
HanLP(['2021年HanLPv2.1为生产环境带来次世代最先进的多语种NLP技术。', '阿婆主来到北京立方庭参观自然语义科技公司。'])
```

Native API的输入单位为句子，需使用[多语种分句模型](https://github.com/hankcs/HanLP/blob/master/plugins/hanlp_demo/hanlp_demo/sent_split.py)或[基于规则的分句函数](https://github.com/hankcs/HanLP/blob/master/hanlp/utils/rules.py#L19)先行分句。RESTful和native两种API的语义设计完全一致，用户可以无缝互换。简洁的接口也支持灵活的参数，常用的技巧有：

- 灵活的`tasks`任务调度，任务越少，速度越快，详见[教程](https://mybinder.org/v2/gh/hankcs/HanLP/doc-zh?filepath=plugins%2Fhanlp_demo%2Fhanlp_demo%2Fzh%2Ftutorial.ipynb)。在内存有限的场景下，用户还可以[删除不需要的任务](https://github.com/hankcs/HanLP/blob/master/plugins/hanlp_demo/hanlp_demo/zh/demo_del_tasks.py)达到模型瘦身的效果。
- 高效的trie树自定义词典，以及强制、合并、校正3种规则，请参考[demo](https://github.com/hankcs/HanLP/blob/doc-zh/plugins/hanlp_demo/hanlp_demo/zh/tok_mtl.ipynb)和[文档](https://hanlp.hankcs.com/docs/api/hanlp/components/tokenizers/transformer.html)。规则系统的效果将无缝应用到后续统计模型，从而快速适应新领域。

#### 单任务模型

根据我们的[最新研究](https://aclanthology.org/2021.emnlp-main.451)，多任务学习的优势在于速度和显存，然而精度往往不如单任务模型。所以，HanLP预训练了许多单任务模型并设计了优雅的[流水线模式](https://hanlp.hankcs.com/docs/api/hanlp/components/pipeline.html#hanlp.components.pipeline.Pipeline)将其组装起来。

```python
import hanlp
HanLP = hanlp.pipeline() \
>>>>>>> 8328b7731f100fa2b4947f0faf85724cfff865a1
    .append(hanlp.utils.rules.split_sentence, output_key='sentences') \
    .append(hanlp.load('FINE_ELECTRA_SMALL_ZH'), output_key='tok') \
    .append(hanlp.load('CTB9_POS_ELECTRA_SMALL'), output_key='pos') \
    .append(hanlp.load('MSRA_NER_ELECTRA_SMALL_ZH'), output_key='ner', input_key='tok') \
    .append(hanlp.load('CTB9_DEP_ELECTRA_SMALL', conll=0), output_key='dep', input_key='tok')\
    .append(hanlp.load('CTB9_CON_ELECTRA_SMALL'), output_key='con', input_key='tok')
HanLP('2021年HanLPv2.1为生产环境带来次世代最先进的多语种NLP技术。阿婆主来到北京立方庭参观自然语义科技公司。')
```

<<<<<<< HEAD
注意流水线的第一级管道是一个普通的Python函数 `split_sentence`，用来将文本拆分为句子。而`input_key`和`output_key`指定了这些管道的连接方式，你可以将这条流水线打印出来观察它的结构：
=======
更多功能，请参考[demo](https://github.com/hankcs/HanLP/tree/doc-zh/plugins/hanlp_demo/hanlp_demo/zh)和[文档](https://hanlp.hankcs.com/docs/api/hanlp/pretrained/index.html)了解更多模型与用法。
>>>>>>> 8328b7731f100fa2b4947f0faf85724cfff865a1

### 输出格式

<<<<<<< HEAD
这次，就像你在日常工作中最常见的场景一样，我们一次性输入一整篇文章 `text`：
=======
无论何种API何种开发语言何种自然语言，HanLP的输出统一为`json`格式兼容`dict`的[`Document`](https://hanlp.hankcs.com/docs/api/common/document.html):
>>>>>>> 8328b7731f100fa2b4947f0faf85724cfff865a1

```json
{
  "tok/fine": [
    ["2021年", "HanLPv2.1", "为", "生产", "环境", "带来", "次", "世代", "最", "先进", "的", "多", "语种", "NLP", "技术", "。"],
    ["阿婆主", "来到", "北京", "立方庭", "参观", "自然", "语义", "科技", "公司", "。"]
  ],
  "tok/coarse": [
    ["2021年", "HanLPv2.1", "为", "生产", "环境", "带来", "次世代", "最", "先进", "的", "多语种", "NLP", "技术", "。"],
    ["阿婆主", "来到", "北京立方庭", "参观", "自然语义科技公司", "。"]
  ],
  "pos/ctb": [
    ["NT", "NR", "P", "NN", "NN", "VV", "JJ", "NN", "AD", "JJ", "DEG", "CD", "NN", "NR", "NN", "PU"],
    ["NN", "VV", "NR", "NR", "VV", "NN", "NN", "NN", "NN", "PU"]
  ],
  "pos/pku": [
    ["t", "nx", "p", "vn", "n", "v", "b", "n", "d", "a", "u", "a", "n", "nx", "n", "w"],
    ["n", "v", "ns", "ns", "v", "n", "n", "n", "n", "w"]
  ],
  "pos/863": [
    ["nt", "w", "p", "v", "n", "v", "a", "nt", "d", "a", "u", "a", "n", "ws", "n", "w"],
    ["n", "v", "ns", "n", "v", "n", "n", "n", "n", "w"]
  ],
  "ner/pku": [
    [],
    [["北京立方庭", "ns", 2, 4], ["自然语义科技公司", "nt", 5, 9]]
  ],
  "ner/msra": [
    [["2021年", "DATE", 0, 1], ["HanLPv2.1", "ORGANIZATION", 1, 2]],
    [["北京", "LOCATION", 2, 3], ["立方庭", "LOCATION", 3, 4], ["自然语义科技公司", "ORGANIZATION", 5, 9]]
  ],
  "ner/ontonotes": [
    [["2021年", "DATE", 0, 1], ["HanLPv2.1", "ORG", 1, 2]],
    [["北京立方庭", "FAC", 2, 4], ["自然语义科技公司", "ORG", 5, 9]]
  ],
  "srl": [
    [[["2021年", "ARGM-TMP", 0, 1], ["HanLPv2.1", "ARG0", 1, 2], ["为生产环境", "ARG2", 2, 5], ["带来", "PRED", 5, 6], ["次世代最先进的多语种NLP技术", "ARG1", 6, 15]], [["最", "ARGM-ADV", 8, 9], ["先进", "PRED", 9, 10], ["技术", "ARG0", 14, 15]]],
    [[["阿婆主", "ARG0", 0, 1], ["来到", "PRED", 1, 2], ["北京立方庭", "ARG1", 2, 4]], [["阿婆主", "ARG0", 0, 1], ["参观", "PRED", 4, 5], ["自然语义科技公司", "ARG1", 5, 9]]]
  ],
  "dep": [
    [[6, "tmod"], [6, "nsubj"], [6, "prep"], [5, "nn"], [3, "pobj"], [0, "root"], [8, "amod"], [15, "nn"], [10, "advmod"], [15, "rcmod"], [10, "assm"], [13, "nummod"], [15, "nn"], [15, "nn"], [6, "dobj"], [6, "punct"]],
    [[2, "nsubj"], [0, "root"], [4, "nn"], [2, "dobj"], [2, "conj"], [9, "nn"], [9, "nn"], [9, "nn"], [5, "dobj"], [2, "punct"]]
  ],
  "sdp": [
    [[[6, "Time"]], [[6, "Exp"]], [[5, "mPrep"]], [[5, "Desc"]], [[6, "Datv"]], [[13, "dDesc"]], [[0, "Root"], [8, "Desc"], [13, "Desc"]], [[15, "Time"]], [[10, "mDegr"]], [[15, "Desc"]], [[10, "mAux"]], [[8, "Quan"], [13, "Quan"]], [[15, "Desc"]], [[15, "Nmod"]], [[6, "Pat"]], [[6, "mPunc"]]],
    [[[2, "Agt"], [5, "Agt"]], [[0, "Root"]], [[4, "Loc"]], [[2, "Lfin"]], [[2, "ePurp"]], [[8, "Nmod"]], [[9, "Nmod"]], [[9, "Nmod"]], [[5, "Datv"]], [[5, "mPunc"]]]
  ],
  "con": [
    ["TOP", [["IP", [["NP", [["NT", ["2021年"]]]], ["NP", [["NR", ["HanLPv2.1"]]]], ["VP", [["PP", [["P", ["为"]], ["NP", [["NN", ["生产"]], ["NN", ["环境"]]]]]], ["VP", [["VV", ["带来"]], ["NP", [["ADJP", [["NP", [["ADJP", [["JJ", ["次"]]]], ["NP", [["NN", ["世代"]]]]]], ["ADVP", [["AD", ["最"]]]], ["VP", [["JJ", ["先进"]]]]]], ["DEG", ["的"]], ["NP", [["QP", [["CD", ["多"]]]], ["NP", [["NN", ["语种"]]]]]], ["NP", [["NR", ["NLP"]], ["NN", ["技术"]]]]]]]]]], ["PU", ["。"]]]]]],
    ["TOP", [["IP", [["NP", [["NN", ["阿婆主"]]]], ["VP", [["VP", [["VV", ["来到"]], ["NP", [["NR", ["北京"]], ["NR", ["立方庭"]]]]]], ["VP", [["VV", ["参观"]], ["NP", [["NN", ["自然"]], ["NN", ["语义"]], ["NN", ["科技"]], ["NN", ["公司"]]]]]]]], ["PU", ["。"]]]]]]
  ]
}
```

<<<<<<< HEAD
中文处理和英文一模一样，事实上，HanLP2.0认为所有人类语言都是统一的符号系统：
=======
特别地，Python RESTful和native API支持基于等宽字体的[可视化](https://hanlp.hankcs.com/docs/tutorial.html#visualization)，能够直接将语言学结构在控制台内可视化出来：
>>>>>>> 8328b7731f100fa2b4947f0faf85724cfff865a1

```python
HanLP(['2021年HanLPv2.1为生产环境带来次世代最先进的多语种NLP技术。', '阿婆主来到北京立方庭参观自然语义科技公司。']).pretty_print()

Dep Tree    	Token    	Relati	PoS	Tok      	NER Type        	Tok      	SRL PA1     	Tok      	SRL PA2     	Tok      	PoS    3       4       5       6       7       8       9 
────────────	─────────	──────	───	─────────	────────────────	─────────	────────────	─────────	────────────	─────────	─────────────────────────────────────────────────────────
 ┌─────────►	2021年    	tmod  	NT 	2021年    	───►DATE        	2021年    	───►ARGM-TMP	2021年    	            	2021年    	NT ───────────────────────────────────────────►NP ───┐   
 │┌────────►	HanLPv2.1	nsubj 	NR 	HanLPv2.1	───►ORGANIZATION	HanLPv2.1	───►ARG0    	HanLPv2.1	            	HanLPv2.1	NR ───────────────────────────────────────────►NP────┤   
 ││┌─►┌─────	为        	prep  	P  	为        	                	为        	◄─┐         	为        	            	为        	P ───────────┐                                       │   
 │││  │  ┌─►	生产       	nn    	NN 	生产       	                	生产       	  ├►ARG2    	生产       	            	生产       	NN ──┐       ├────────────────────────►PP ───┐       │   
 │││  └─►└──	环境       	pobj  	NN 	环境       	                	环境       	◄─┘         	环境       	            	环境       	NN ──┴►NP ───┘                               │       │   
┌┼┴┴────────	带来       	root  	VV 	带来       	                	带来       	╟──►PRED    	带来       	            	带来       	VV ──────────────────────────────────┐       │       │   
││       ┌─►	次        	amod  	JJ 	次        	                	次        	◄─┐         	次        	            	次        	JJ ───►ADJP──┐                       │       ├►VP────┤   
││  ┌───►└──	世代       	nn    	NN 	世代       	                	世代       	  │         	世代       	            	世代       	NN ───►NP ───┴►NP ───┐               │       │       │   
││  │    ┌─►	最        	advmod	AD 	最        	                	最        	  │         	最        	───►ARGM-ADV	最        	AD ───────────►ADVP──┼►ADJP──┐       ├►VP ───┘       ├►IP
││  │┌──►├──	先进       	rcmod 	JJ 	先进       	                	先进       	  │         	先进       	╟──►PRED    	先进       	JJ ───────────►VP ───┘       │       │               │   
││  ││   └─►	的        	assm  	DEG	的        	                	的        	  ├►ARG1    	的        	            	的        	DEG──────────────────────────┤       │               │   
││  ││   ┌─►	多        	nummod	CD 	多        	                	多        	  │         	多        	            	多        	CD ───►QP ───┐               ├►NP ───┘               │   
││  ││┌─►└──	语种       	nn    	NN 	语种       	                	语种       	  │         	语种       	            	语种       	NN ───►NP ───┴────────►NP────┤                       │   
││  │││  ┌─►	NLP      	nn    	NR 	NLP      	                	NLP      	  │         	NLP      	            	NLP      	NR ──┐                       │                       │   
│└─►└┴┴──┴──	技术       	dobj  	NN 	技术       	                	技术       	◄─┘         	技术       	───►ARG0    	技术       	NN ──┴────────────────►NP ───┘                       │   
└──────────►	。        	punct 	PU 	。        	                	。        	            	。        	            	。        	PU ──────────────────────────────────────────────────┘   

Dep Tree    	Tok	Relat	Po	Tok	NER Type        	Tok	SRL PA1 	Tok	SRL PA2 	Tok	Po    3       4       5       6 
────────────	───	─────	──	───	────────────────	───	────────	───	────────	───	────────────────────────────────
         ┌─►	阿婆主	nsubj	NN	阿婆主	                	阿婆主	───►ARG0	阿婆主	───►ARG0	阿婆主	NN───────────────────►NP ───┐   
┌┬────┬──┴──	来到 	root 	VV	来到 	                	来到 	╟──►PRED	来到 	        	来到 	VV──────────┐               │   
││    │  ┌─►	北京 	nn   	NR	北京 	───►LOCATION    	北京 	◄─┐     	北京 	        	北京 	NR──┐       ├►VP ───┐       │   
││    └─►└──	立方庭	dobj 	NR	立方庭	───►LOCATION    	立方庭	◄─┴►ARG1	立方庭	        	立方庭	NR──┴►NP ───┘       │       │   
│└─►┌───────	参观 	conj 	VV	参观 	                	参观 	        	参观 	╟──►PRED	参观 	VV──────────┐       ├►VP────┤   
│   │  ┌───►	自然 	nn   	NN	自然 	◄─┐             	自然 	        	自然 	◄─┐     	自然 	NN──┐       │       │       ├►IP
│   │  │┌──►	语义 	nn   	NN	语义 	  │             	语义 	        	语义 	  │     	语义 	NN  │       ├►VP ───┘       │   
│   │  ││┌─►	科技 	nn   	NN	科技 	  ├►ORGANIZATION	科技 	        	科技 	  ├►ARG1	科技 	NN  ├►NP ───┘               │   
│   └─►└┴┴──	公司 	dobj 	NN	公司 	◄─┘             	公司 	        	公司 	◄─┘     	公司 	NN──┘                       │   
└──────────►	。  	punct	PU	。  	                	。  	        	。  	        	。  	PU──────────────────────────┘   
```

<<<<<<< HEAD
输出为一个json化的 `dict`，大部分用户应当很熟悉。

- 请发挥你的想象力和创造力，在流水线中加入更多预处理和后处理管道（包括词典、正则等）。记住，任意普通的Python函数都可以作为一级管道。
- 使用 `pipeline.save('zh.json')` 将流水线序列化并部署到生产服务器。

## 训练你自己的模型

写深度学习模型一点都不难，难的是复现较高的准确率。下列代码展示了如何在MSR语料库上训练一个 97% F1 的中文分词模型。
=======
关于标注集含义，请参考[《语言学标注规范》](https://hanlp.hankcs.com/docs/annotations/index.html)及[《格式规范》](https://hanlp.hankcs.com/docs/data_format.html)。我们购买、标注或采用了世界上量级最大、种类最多的语料库用于联合多语种多任务学习，所以HanLP的标注集也是覆盖面最广的。

## 训练你自己的领域模型

写深度学习模型一点都不难，难的是复现较高的准确率。下列[代码](https://github.com/hankcs/HanLP/blob/master/plugins/hanlp_demo/hanlp_demo/zh/train_sota_bert_pku.py)展示了如何在sighan2005 PKU语料库上花6分钟训练一个超越学术界state-of-the-art的中文分词模型。
>>>>>>> 8328b7731f100fa2b4947f0faf85724cfff865a1

```python
tokenizer = TransformerTaggingTokenizer()
save_dir = 'data/model/cws/sighan2005_pku_bert_base_96.73'
tokenizer.fit(
    SIGHAN2005_PKU_TRAIN_ALL,
    SIGHAN2005_PKU_TEST,  # Conventionally, no devset is used. See Tian et al. (2020).
    save_dir,
    'bert-base-chinese',
    max_seq_len=300,
    char_level=True,
    hard_constraint=True,
    sampler_builder=SortingSamplerBuilder(batch_size=32),
    epochs=3,
    adam_epsilon=1e-6,
    warmup_steps=0.1,
    weight_decay=0.01,
    word_dropout=0.1,
    seed=1660853059,
)
tokenizer.evaluate(SIGHAN2005_PKU_TEST, save_dir)
```

<<<<<<< HEAD
训练和评测日志如下所示。
=======
其中，由于指定了随机数种子，结果一定是`96.73`。不同于那些虚假宣传的学术论文或商业项目，HanLP保证所有结果可复现。如果你有任何质疑，我们将当作最高优先级的致命性bug第一时间排查问题。
>>>>>>> 8328b7731f100fa2b4947f0faf85724cfff865a1

请参考[demo](https://github.com/hankcs/HanLP/tree/master/plugins/hanlp_demo/hanlp_demo/zh/train)了解更多训练脚本。

<<<<<<< HEAD
类似地，你可以训练一个情感分析模型来判断酒店评论的情感极性。
=======
## 性能
>>>>>>> 8328b7731f100fa2b4947f0faf85724cfff865a1

<table><thead><tr><th rowspan="2">lang</th><th rowspan="2">corpora</th><th rowspan="2">model</th><th colspan="2">tok</th><th colspan="4">pos</th><th colspan="3">ner</th><th rowspan="2">dep</th><th rowspan="2">con</th><th rowspan="2">srl</th><th colspan="4">sdp</th><th rowspan="2">lem</th><th rowspan="2">fea</th><th rowspan="2">amr</th></tr><tr><th>fine</th><th>coarse</th><th>ctb</th><th>pku</th><th>863</th><th>ud</th><th>pku</th><th>msra</th><th>ontonotes</th><th>SemEval16</th><th>DM</th><th>PAS</th><th>PSD</th></tr></thead><tbody><tr><td rowspan="2">mul</td><td rowspan="2">UD2.7<br>OntoNotes5</td><td>small</td><td>98.62</td><td>-</td><td>-</td><td>-</td><td>-</td><td>93.23</td><td>-</td><td>-</td><td>74.42</td><td>79.10</td><td>76.85</td><td>70.63</td><td>-</td><td>91.19</td><td>93.67</td><td>85.34</td><td>87.71</td><td>84.51</td><td>-</td></tr><tr><td>base</td><td>98.97</td><td>-</td><td>-</td><td>-</td><td>-</td><td>90.32</td><td>-</td><td>-</td><td>80.32</td><td>78.74</td><td>71.23</td><td>73.63</td><td>-</td><td>92.60</td><td>96.04</td><td>81.19</td><td>85.08</td><td>82.13</td><td>-</td></tr><tr><td rowspan="5">zh</td><td rowspan="2">open</td><td>small</td><td>97.25</td><td>-</td><td>96.66</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>95.00</td><td>84.57</td><td>87.62</td><td>73.40</td><td>84.57</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>base</td><td>97.50</td><td>-</td><td>97.07</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>96.04</td><td>87.11</td><td>89.84</td><td>77.78</td><td>87.11</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td rowspan="3">close</td><td>small</td><td>96.70</td><td>95.93</td><td>96.87</td><td>97.56</td><td>95.05</td><td>-</td><td>96.22</td><td>95.74</td><td>76.79</td><td>84.44</td><td>88.13</td><td>75.81</td><td>74.28</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>base</td><td>97.52</td><td>96.44</td><td>96.99</td><td>97.59</td><td>95.29</td><td>-</td><td>96.48</td><td>95.72</td><td>77.77</td><td>85.29</td><td>88.57</td><td>76.52</td><td>73.76</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>ernie</td><td>96.95</td><td>97.29</td><td>96.76</td><td>97.64</td><td>95.22</td><td>-</td><td>97.31</td><td>96.47</td><td>77.95</td><td>85.67</td><td>89.17</td><td>78.51</td><td>74.10</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr></tbody></table>

<<<<<<< HEAD
由于语料库一般领域相关，且BERT模型体积较大，HanLP不准备发布那么多预训练文本分类模型。

欲了解更多训练脚本，请参考 [`tests/train`](https://github.com/hankcs/HanLP/tree/master/tests/train)。更多的使用案例可以在 [`tests/demo`](https://github.com/hankcs/HanLP/tree/master/tests/demo)中找到。文档，RESTful API都在开发中。

## 引用

如果你在研究中使用了HanLP，请按如下格式引用：
=======
- 根据我们的[最新研究](https://aclanthology.org/2021.emnlp-main.451)，单任务学习的性能往往优于多任务学习。在乎精度甚于速度的话，建议使用[单任务模型](https://hanlp.hankcs.com/docs/api/hanlp/pretrained/index.html)。

HanLP采用的数据预处理与拆分比例与流行方法未必相同，比如HanLP采用了[完整版的MSRA命名实体识别语料](https://bbs.hankcs.com/t/topic/3033)，而非大众使用的阉割版；HanLP使用了语法覆盖更广的[Stanford Dependencies标准](https://hanlp.hankcs.com/docs/annotations/dep/sd_zh.html)，而非学术界沿用的Zhang and Clark (2008)标准；HanLP提出了[均匀分割CTB的方法](https://bbs.hankcs.com/t/topic/3024)，而不采用学术界不均匀且遗漏了51个黄金文件的方法。HanLP开源了[一整套语料预处理脚本与相应语料库](https://github.com/hankcs/HanLP/blob/master/plugins/hanlp_demo/hanlp_demo/zh/train/open_small.py)，力图推动中文NLP的透明化。

总之，HanLP只做我们认为正确、先进的事情，而不一定是流行、权威的事情。

## 引用
>>>>>>> 8328b7731f100fa2b4947f0faf85724cfff865a1

如果你在研究中使用了HanLP，请按如下格式引用：

```bibtex
@inproceedings{he-choi-2021-stem,
    title = "The Stem Cell Hypothesis: Dilemma behind Multi-Task Learning with Transformer Encoders",
    author = "He, Han and Choi, Jinho D.",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.451",
    pages = "5555--5577",
    abstract = "Multi-task learning with transformer encoders (MTL) has emerged as a powerful technique to improve performance on closely-related tasks for both accuracy and efficiency while a question still remains whether or not it would perform as well on tasks that are distinct in nature. We first present MTL results on five NLP tasks, POS, NER, DEP, CON, and SRL, and depict its deficiency over single-task learning. We then conduct an extensive pruning analysis to show that a certain set of attention heads get claimed by most tasks during MTL, who interfere with one another to fine-tune those heads for their own objectives. Based on this finding, we propose the Stem Cell Hypothesis to reveal the existence of attention heads naturally talented for many tasks that cannot be jointly trained to create adequate embeddings for all of those tasks. Finally, we design novel parameter-free probes to justify our hypothesis and demonstrate how attention heads are transformed across the five tasks during MTL through label analysis.",
}
```

## License

<<<<<<< HEAD
HanLP 的授权协议为 **Apache License 2.0**，可免费用做商业用途。请在产品说明中附加HanLP的链接和授权协议。HanLP受版权法保护，侵权必究。
=======
### 源代码

HanLP源代码的授权协议为 **Apache License 2.0**，可免费用做商业用途。请在产品说明中附加HanLP的链接和授权协议。HanLP受版权法保护，侵权必究。
>>>>>>> 8328b7731f100fa2b4947f0faf85724cfff865a1

##### 自然语义（青岛）科技有限公司

HanLP从v1.7版起独立运作，由自然语义（青岛）科技有限公司作为项目主体，主导后续版本的开发，并拥有后续版本的版权。

##### 大快搜索

HanLP v1.3~v1.65版由大快搜索主导开发，继续完全开源，大快搜索拥有相关版权。

##### 上海林原公司

HanLP 早期得到了上海林原公司的大力支持，并拥有1.28及前序版本的版权，相关版本也曾在上海林原公司网站发布。
<<<<<<< HEAD
=======

### 预训练模型

机器学习模型的授权在法律上没有定论，但本着尊重开源语料库原始授权的精神，如不特别说明，HanLP的多语种模型授权沿用[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)，中文模型授权为仅供研究与教学使用。
>>>>>>> 8328b7731f100fa2b4947f0faf85724cfff865a1

## References

https://hanlp.hankcs.com/docs/references.html

