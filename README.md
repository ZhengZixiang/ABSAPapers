# ABSAPapers
Worth-reading papers and related awesome resources on aspect-based sentiment analysis (ABSA). This repository mainly focused on aspect-term sentiment classification (ATSC). ABSA task contains five fine-grained subtasks:
- Aspect Term Sentiment Classification (ATSC)
- Aspect Term Extraction (ATE)
- Aspect Category Sentiment Classification (ACSC)
- Aspect Category Detection (ACD)
- Opniton Term Extraction (OTE)

Suggestions about adding papers, repositories and other resource are welcomed!

值得一读的方面级情感分析论文与相关资源集合。这里主要关注方面词（aspect-term）的情感分类。具体来说，方面级情感分析包括方面词情感分类、方面词抽取、方面类目情感分类、方面类目抽取、观点词抽取五个子任务。

欢迎新增论文、代码仓库与其他资源等建议！

> Update to COLING 2020! We will add a score table of representative and latest ABSA models like [NLP-progress](http://nlpprogress.com/english/sentiment_analysis.html) in the near future, so stay tuned!

> 新增COLING 2020论文！近期将参考[NLP-progress](http://nlpprogress.com/english/sentiment_analysis.html)的形式增加一个数据集分值表，敬请期待！

## Paper
- **Effective LSTMs for Target-Dependent Sentiment Classification**. *Duyu Tang, Bing Qin, Xiaocheng Feng, Ting Liu*. (COLING 2016) [[paper]](https://www.aclweb.org/anthology/C16-1311)[[code]](https://drive.google.com/drive/folders/17RF8MZs456ov9MDiUYZp0SCGL6LvBQl6) - ***TD-LSTM TC-LSTM***
- **Attention-based LSTM for Aspect-level Sentiment Classification**. *Yequan Wang, Minlie Huang, Xiaoyan Zhu, Li Zhao*. (EMNLP 2016) [[paper]](https://aclweb.org/anthology/D16-1058) - ***ATAE-LSTM***
- **A Hierarchical Model of Reviews for Aspect-based Sentiment Analysis**. *Sebastian Ruder, Parsa Ghaffari, John G. Breslin*. (EMNLP 2016) [[paper]](https://arxiv.org/pdf/1609.02745.pdf) - ***H-LSTM***
- **Aspect Level Sentiment Classification with Deep Memory Network**. *Duyu Tang, Bing Qin, Ting Liu*. (EMNLP 2016) [[paper]](https://www.aclweb.org/anthology/D16-1021)[[code]](https://drive.google.com/drive/folders/1Hc886aivHmIzwlawapzbpRdTfPoTyi1U) - ***MemNet***
- **Interactive Attention Networks for Aspect-Level Sentiment Classification**. *Dehong Ma, Sujian Li, Xiaodong Zhang, Houfeng Wang*. (IJCAI 2017) [[paper]](https://www.ijcai.org/proceedings/2017/0568.pdf) - ***IAN***
- **Recurrent Attention Network on Memory for Aspect Sentiment Analysis**. *Peng Chen, Zhongqian Sun, Lidong Bing, Wei Yang*. (EMNLP 2017) [[paper]](https://www.aclweb.org/anthology/D17-1047)[[unofficial code]](https://github.com/lpq29743/RAM) - ***RAM***
- **Document-Level Multi-Aspect Sentiment Classification as Machine Comprehension**. *Yichun Yin, Yangqiu Song, Ming Zhang*. (EMNLP 2017) [[paper]](http://www.cse.ust.hk/~yqsong/papers/2017-EMNLP-AspectClassification.pdf)
- **Attention Modeling for Targeted Sentiment**. *Jiangming Liu, Yue Zhang*. (EACL 2017) [[paper]](https://www.aclweb.org/anthology/E17-2091/) - ***BiLSTM-ATT-G***
- **Aspect-level Sentiment Classification with HEAT (HiErarchical ATtention) Network**. *Jiajun Cheng, Shenglin Zhao, Jiani Zhang, Irwin King, Xin Zhang, Hui Wang*. (CIKM 2017) [[paper]](https://dl.acm.org/citation.cfm?doid=3132847.3133037)
- **Aspect Based Sentiment Analysis with Gated Convolutional Networks**. *Wei Xue, Tao Li*. (ACL 2018) [[paper]](https://www.aclweb.org/anthology/P18-1234)[[code]](https://github.com/wxue004cs/GCAE) - ***GCAE***
- **Target-Sensitive Memory Networks for Aspect Sentiment Classification**. *Shuai Wang, Sahisnu Mazumder, Bing Liu, Mianwei Zhou, Yi Chang*. (ACL 2018) [[paper]](https://www.aclweb.org/anthology/P18-1088/) - ***TMN***
- **Transformation Networks for Target-Oriented Sentiment Classification**. *Xin Li, Lidong Bing, Wai Lam, Bei Shi*. (ACL 2018) [[paper]](https://aclweb.org/anthology/P18-1087)[[code]](https://github.com/lixin4ever/TNet) - ***TNet***
- **Exploiting Document Knowledge for Aspect-level Sentiment Classification**. *Ruidan He, Wee Sun Lee, Hwee Tou Ng, Daniel Dahlmeier*. (ACL 2018) [[paper]](https://arxiv.org/abs/1806.04346)[[code]](https://github.com/ruidan/Aspect-level-sentiment)
- **Learning to Attend via Word-Aspect Associative Fusion for Aspect-Based Sentiment Analysis**. *Yi Tay, Luu Anh Tuan, Siu Cheung Hui*. (AAAI 2018) [[paper]](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16570/16162) - ***AF-LSTM***
- **Targeted Aspect-Based Sentiment Analysis via Embedding Commonsense Knowledge into an Attentive LSTM**. *Yukun Ma, Haiyun Peng, Erik Cambria*. (AAAI 2018) [[paper]](https://sentic.net/sentic-lstm.pdf)[[code]](https://github.com/SenticNet/sentic-lstm) - ***Sentic LSTM***
- **A Position-aware Bidirectional Attention Network for Aspect-level Sentiment Analysis**. *Shuqin Gu, Lipeng Zhang, Yuexian Hou, Yin Song*. (COLING 2018) [[paper]](https://aclweb.org/anthology/C18-1066/)[[code]](https://github.com/hiyouga/PBAN-PyTorch) - ***PBAN***
- **Enhanced Aspect Level Sentiment Classification with Auxiliary Memory**. *Peisong Zhu, Tieyun Qian*. (COLING 2018) [[paper]](https://www.aclweb.org/anthology/C18-1092/) - ***DAuM***
- **Document-level Multi-aspect Sentiment Classification by Jointly Modeling Users, Aspects, and Overall Ratings**. *Junjie Li, Haitong Yang, Chengqing Zong*. (COLING 2018) [[paper]](https://www.aclweb.org/anthology/C18-1079/) - ***HUARN***
- **Effective Attention Modeling for Aspect-Level Sentiment Classification**. *Ruidan He, Wee Sun Lee, Hwee Tou Ng, Daniel Dahlmeier*. (COLING 2018) [[paper]](https://www.aclweb.org/anthology/C18-1096/)
- **Modeling Inter-Aspect Dependencies for Aspect-Based Sentiment Analysis**. *Devamanyu Hazarika, Soujanya Poria, Prateek Vij, Gangeshwar Krishnamurthy, Erik Cambria, Roger Zimmermann*. (NAACL 2018) [[paper]](https://www.aclweb.org/anthology/N18-2043/)[[unofficial code]](https://github.com/xgy221/lstm-inter-aspect)
- **Recurrent Entity Networks with Delayed Memory Update for Targeted Aspect-Based Sentiment Analysis**. *Fei Liu, Trevor Cohn, Timothy Baldwin*. (NAACL 2018) [[paper]](https://www.aclweb.org/anthology/N18-2045/) [[code]](https://github.com/liufly/delayed-memory-update-entnet)
- **Content Attention Model for Aspect Based Sentiment Analysis**. *Qiao Liu, Haibin Zhang, Yifu Zeng, Ziqi Huang, Zufeng Wu*. (WWW 2018) [[paper]](https://dl.acm.org/citation.cfm?doid=3178876.3186001)[[code]](https://github.com/uestcnlp/Cabasc) - ***Cabasc***
- **Aspect Level Sentiment Classification with Attention-over-Attention Neural Networks**. *Binxuan Huang, Yanglan Ou, Kathleen M. Carley*. (SBP-BRiMS 2018) [[paper]](https://arxiv.org/pdf/1804.06536.pdf) - ***AOA***
- **Aspect Sentiment Classification with both Word-level and Clause-level Attention Networks**. *Jingjing Wang, Jie Li, Shoushan Li, Yangyang Kang, Min Zhang, Luo Si, Guodong Zhou*. (IJCAI 2018) [[paper]](https://www.ijcai.org/proceedings/2018/0617)
- **IARM: Inter-Aspect Relation Modeling with Memory Networks in Aspect-Based Sentiment Analysis**. *Navonil Majumder, Soujanya Poria, Alexander F. Gelbukh, Md. Shad Akhtar, Erik Cambria, Asif Ekbal*. (EMNLP 2018) [[paper]](https://aclweb.org/anthology/D18-1377/)[[code]](https://github.com/SenticNet/IARM)
- **Multi-grained Attention Network for Aspect-Level Sentiment Classification**. *Feifan Fan, Yansong Feng, Dongyan Zhao*. (EMNLP 2018) [[paper]](https://aclweb.org/anthology/D18-1380) - ***MGAN***
- **Parameterized Convolutional Neural Networks for Aspect Level Sentiment Classification**. *Binxuan Huang, Kathleen M. Carley*. (EMNLP 2018) [[paper]](https://aclweb.org/anthology/D18-1136/) - ***PCNN***
- **Left-Center-Right Separated Neural Network for Aspect-based Sentiment Analysis with Rotatory Attention**. *Shiliang Zheng, Rui Xia*. (CoRR 2018) [[paper]](https://arxiv.org/abs/1802.00892) - ***LCR-Rot***
- **Syntax-Aware Aspect-Level Sentiment Classification with Proximity-Weighted Convolution Network**. *Chen Zhang, Qiuchi Li, Dawei Song*. (SIGIR 2019) [[paper]](https://arxiv.org/abs/1909.10171)[[code]](https://github.com/GeneZC/PWCN) - ***PWCN***
- **Aspect-based Sentiment Classification with Aspect-specific Graph Convolutional Networks**. *Chen Zhang, Qiuchi Li, Dawei Song*. (EMNLP 2019) [[paper]](https://arxiv.org/abs/1909.03477)[[code]](https://github.com/GeneZC/ASGCN) - ***ASGCN***
- **Aspect-Level Sentiment Analysis Via Convolution over Dependency Tree**. *Kai Sun, Richong Zhang, Samuel Mensah, Yongyi Mao, Xudong Liu*. (EMNLP 2019) [[paper]](https://www.aclweb.org/anthology/D19-1569/)[[code]](https://github.com/sunkaikai/CDT_ABSA) - ***CDT-ABSA***
- **A Novel Aspect-Guided Deep Transition Model for Aspect Based Sentiment Analysis**. *Yunlong Liang, Fandong Meng, Jinchao Zhang, Jinan Xu, Yufeng Chen, Jie Zhou*. (EMNLP 2019) [[paper]](https://arxiv.org/pdf/1909.00324.pdf)[[code]](https://github.com/XL2248/AGDT) - ***AGDT***
- **CAN: Constrained Attention Networks for Multi-Aspect Sentiment**. *Mengting Hu, Shiwan Zhao, Li Zhang, Keke Cai, Zhong Su, Renhong Cheng, Xiaowei Shen*. (EMNLP 2019) [[paper]](https://arxiv.org/pdf/1812.10735.pdf) - ***CAN***
- **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**. *Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova*. (NAACL 2019) [[paper]](https://www.aclweb.org/anthology/N19-1423/) - ***BERT-SPC***
- **BERT Post-Training for Review Reading Comprehension and Aspect-based Sentiment Analysis**. *Hu Xu, Bing Liu, Lei Shu, Philip S. Yu*.  (NAACL 2019) [[paper]](https://www.aclweb.org/anthology/N19-1242)[[code]](https://github.com/howardhsu/BERT-for-RRC-ABSA) - ***BERT-PT***
- **Utilizing BERT for Aspect-Based Sentiment Analysis via Constructing Auxiliary Sentence**. *Chi Sun, Luyao Huang, Xipeng Qiu*. (NAACL 2019) [[paper]](https://www.aclweb.org/anthology/N19-1035/)[[code]](https://github.com/HSLCY/ABSA-BERT-pair)
- **Attentional Encoder Network for Targeted Sentiment Classification**. *Youwei Song, Jiahai Wang, Tao Jiang, Zhiyue Liu, Yanghui Rao*. (CoRR 2019) [[paper]](https://arxiv.org/pdf/1902.09314.pdf)[[code]](https://github.com/songyouwei/ABSA-PyTorch/blob/master/models/aen.py) - ***AEN-BERT***
- **LCF: A Local Context Focus Mechanism for Aspect-Based Sentiment Classification**. *Biqing Zeng, Heng Yang 2, Ruyang Xu, Wu Zhou, Xuli Han* (Applied Sciences 2019) [[paper]](https://www.mdpi.com/2076-3417/9/16/3389/pdf)[[code]](https://github.com/yangheng95/LCF-ABSA) - ***LCF-BERT***
- **A Multi-task Learning Model for Chinese-oriented Aspect Polarity Classification and Aspect Term Extraction**. *Heng Yang, Biqing Zeng, Jianhao Yang, Youwei Song, Ruyang Xu*. (CoRR 2019) [[paper]](https://arxiv.org/abs/1912.07976)[[code]](https://github.com/yangheng95/LCF-ATEPC) - ***LCF-ATEPC***
- **Target-Dependent Sentiment Classification With BERT**. *Zhengjie Gao, Ao Feng, Xinyu Song, Xi Wu*. (IEEE Access Volumn 7 2019) [[paper]](https://ieeexplore.ieee.org/document/8864964)[[code]](https://github.com/Xiang-Pan/ABSA-PyTorch/blob/master/models/td_bert.py) - ***TD-BERT***
- **Adapt or Get Left Behind: Domain Adaptation through BERT Language Model Finetuning for Aspect-Target Sentiment Classification**. *Alexander Rietzler, Sebastian Stabinger, Paul Opitz, Stefan Engl*. (LREC 2020) [[paper]](https://arxiv.org/abs/1908.11860)[[code]](https://github.com/deepopinion/domain-adapted-atsc) - ***BERT-ADA***
- **Convolution over Hierarchical Syntactic and Lexical Graphs for Aspect Level Sentiment Analysis**. *Mi Zhang, Tieyun Qian*. (EMNLP 2020) [[paper]](https://www.aclweb.org/anthology/2020.emnlp-main.286/)
- **Modelling Context and Syntactical Features for Aspect-based Sentiment Analysis**. *Minh Hieu Phan, Philip O. Ogunbona*. (ACL 2020) [[paper]](https://www.aclweb.org/anthology/2020.acl-main.293/)[[code]](https://github.com/StevePhan101/LCFS-BERT) - ***LCFS-BERT***
- **Constituency Lattice Encoding for Aspect Term Extraction**. *Yunyi Yang, Kun Li, Xiaojun Quan, Weizhou Shen, Qinliang Su*. (COLING 2020) [[paper]](https://www.aclweb.org/anthology/2020.coling-main.73/)
- **Attention Transfer Network for Aspect-level Sentiment Classification**. *Fei Zhao, Zhen Wu, Xinyu Dai*. (COLING 2020) [[paper]](https://www.aclweb.org/anthology/2020.coling-main.70/)[[code]](https://github.com/1429904852/ATN) - ***ATN***
- **Jointly Learning Aspect-Focused and Inter-Aspect Relations with Graph Convolutional Networks for Aspect Sentiment Analysis**. *Bin Liang, Rongdi Yin, Lin Gui, Jiachen Du, Ruifeng Xu*. [[paper]](https://www.aclweb.org/anthology/2020.coling-main.13/) - ***InterGCN***
- **Syntax-Aware Graph Attention Network for Aspect-Level Sentiment Classification**. *Lianzhe Huang, Xin Sun, Sujian Li, Linhao Zhang, Houfeng Wang*. (COLING 2020) [[paper]](https://www.aclweb.org/anthology/2020.coling-main.69/) - ***SAGAT***
- **Constituency Lattice Encoding for Aspect Term Extraction**. *Yunyi Yang, Kun Li, Xiaojun Quan, Weizhou Shen, Qinliang Su*. (COLING 2020) [[paper]](https://www.aclweb.org/anthology/2020.coling-main.73/)

### Multi-task Learning & End-to-End
Combining two or more ABSA's subtasks in one framework to produce results is an intutively effective way for industrial application. There are three patterns of multi-task learning: pipeline, joint and end-to-end model. For pipeline pattern, the framework complete subtasks in more than one step, using the result of last step to guide the next step's output, which might lead to error propogation problem. Joint model processes the data with shared layers to extract universal semantic features for all subtasks. Then model outputs results of different tasks through task-specific layers. End-to-end model complete tasks like sequence labeling.

#### Aspect Extraction & Sentiment Classification
- **MTNA: A Neural Multi-task Model for Aspect Category Classification and Aspect Term Extraction On Restaurant Reviews**. *Wei Xue, Wubai Zhou, Tao Li, Qing Wang*. (IJCNLP 2017) [[paper]](https://www.aclweb.org/anthology/I17-2026/) - ***MTNA***
- **Exploiting Coarse-to-Fine Task Transfer for Aspect-Level Sentiment Classification**. *Zheng Li, Ying Wei, Yu Zhang, Xiang Zhang, Xin Li*. (AAAI 2019) [[paper]](https://www.aaai.org/ojs/index.php/AAAI/article/view/4332) - ***MGAN***
- **A Unified Model for Opinion Target Extraction and Target Sentiment Prediction**. *Xin Li, Lidong Bing, Piji Li, Wai Lam*. (AAAI 2019) [[paper]](https://aaai.org/ojs/index.php/AAAI/article/view/4643)[[code]](https://github.com/lixin4ever/E2E-TBSA) - ***UNIFIED E2E-TBSA***
- **An Interactive Multi-Task Learning Network for End-to-End Aspect-Based Sentiment Analysis**. *Ruidan He, Wee Sun Lee, Hwee Tou Ng, Daniel Dahlmeier*. (ACL 2019) [[paper]](https://arxiv.org/abs/1906.06906)[[code]](https://github.com/ruidan/IMN-E2E-ABSA) - ***IMN-E2E-ABSA***
- **DOER: Dual Cross-Shared RNN for Aspect Term-Polarity Co-Extraction**. *Huaishao Luo, Tianrui Li, Bing Liu, Junbo Zhang*. (ACL 2019) [[paper]](https://www.aclweb.org/anthology/P19-1056/)[[code]](https://github.com/ArrowLuo/DOER)
- **Learning Explicit and Implicit Structures for Targeted Sentiment Analysis**. *Hao Li, Wei Lu*. (EMNLP 2019) [[paper]](https://www.aclweb.org/anthology/D19-1550/)[[code]](https://github.com/leodotnet/ei) - ***EI***
- **Exploiting BERT for End-to-End Aspect-based Sentiment Analysis**. *Xin Li, Lidong Bing, Wenxuan Zhang, Wai Lam*. (EMNLP 2019) [[paper]](https://www.aclweb.org/anthology/D19-5505/)[[code]](https://github.com/lixin4ever/BERT-E2E-ABSA) - ***BERT-E2E-ABSA***
- **Transferable End-to-End Aspect-based Sentiment Analysis with Selective Adversarial Learning**. *Zheng Li, Xin Li, Ying Wei, Lidong Bing, Yu Zhang, Qiang Yang*. (EMNLP 2019) [[paper]](https://www.aclweb.org/anthology/D19-1466/)[[code]](https://github.com/hsqmlzno1/Transferable-E2E-ABSA) - ***Transferable-E2E-ABSA***
- **Knowing What, How and Why: A Near Complete Solution for Aspect-based Sentiment Analysis**. *Haiyun Peng, Lu Xu, Lidong Bing, Fei Huang, Wei Lu, Luo Si*. (AAAI 2020) [[paper]](https://arxiv.org/abs/1911.01616)[[data]](https://github.com/xuuuluuu/SemEval-Triplet-data) - ***ASTE***
- **Open-Domain Targeted Sentiment Analysis via Span-Based Extraction and Classification**. *Minghao Hu, Yuxing Peng, Zhen Huang, Dongsheng Li, Yiwei Lv*. (ACL 2019) [[paper]](https://arxiv.org/abs/1906.03820)[[code]](https://github.com/huminghao16/SpanABSA) - ***SpanABSA***
- **Relation-Aware Collaborative Learning for Unified Aspect-Based Sentiment Analysis**. *Zhuang Chen, Tieyun Qian*. (ACL 2020) [[paper]](https://www.aclweb.org/anthology/2020.acl-main.340/)[[code]](https://github.com/NLPWM-WHU/RACL) - ***RACL***
- **Label Correction Model for Aspect-based Sentiment Analysis**. *Qianlong Wang, Jiangtao Ren*. (COLING 2020) [[paper]](https://www.aclweb.org/anthology/2020.coling-main.71/)
- **Aspect-Category based Sentiment Analysis with Hierarchical Graph Convolutional Network**. *Hongjie Cai, Yaofeng Tu, Xiangsheng Zhou, Jianfei Yu, Rui Xia*. (COLING 2020) [[paper]](https://www.aclweb.org/anthology/2020.coling-main.72/)
- **Joint Aspect Extraction and Sentiment Analysis with Directional Graph Convolutional Networks**. *Guimin Chen, Yuanhe Tian, Yan Song*. (COLING 2020) [[paper]](https://www.aclweb.org/anthology/2020.coling-main.24/) - ***D-GCN***

#### Aspect-Opinion Pair Extraction
- **Target-oriented Opinion Words Extraction with Target-fused Neural Sequence Labeling**. *Zhifang Fan, Zhen Wu, Xin-Yu Dai, Shujian Huang, Jiajun Chen*. (NAACL 2019) [[paper]](https://www.aclweb.org/anthology/N19-1259/)[[data]](https://github.com/NJUNLP/TOWE) - ***TOWE***
- **Latent Opinions Transfer Network for Target-Oriented Opinion Words Extraction**. *Latent Opinions Transfer Network for Target-Oriented Opinion Words Extraction*. (AAAI 2020) [[paper]](https://arxiv.org/abs/2001.01989)[[code]](https://github.com/NJUNLP/TOWE)
- **Synchronous Double-channel Recurrent Network for Aspect-Opinion Pair Extraction**. *Shaowei Chen, Jie Liu, Yu Wang, Wenzheng Zhang, Ziming Chi*. (ACL 2020) [[paper]](https://www.aclweb.org/anthology/2020.acl-main.582/)[[code]](https://github.com/NKU-IIPLab/SDRN) - ***AOPE SDRN***
- **SpanMlt: A Span-based Multi-Task Learning Framework for Pair-wise Aspect and Opinion Terms Extraction**. *He Zhao, Longtao Huang, Rong Zhang, Quan Lu, Hui Xue*. (ACL 2020) [[paper]](https://www.aclweb.org/anthology/2020.acl-main.296/)
- **Syntactically Aware Cross-Domain Aspect and Opinion Terms Extraction**. *Oren Pereg, Daniel Korat, Moshe Wasserblat*. (COLING 2020) [[paper]](https://www.aclweb.org/anthology/2020.coling-main.158/)

#### Emotion-Cause Pair Extraction
- **Emotion-Cause Pair Extraction: A New Task to Emotion Analysis in Texts**. *Rui Xia, Zixiang Ding*. (ACL 2019) [[paper]](https://arxiv.org/abs/1906.01267)[[code]](https://github.com/NUSTM/ECPE)
- **ECPE-2D: Emotion-Cause Pair Extraction based on Joint Two-Dimensional Representation, Interaction and Prediction**. *Zixiang Ding, Rui Xia, Jianfei Yu*. (ACL 2020) [[paper]](https://www.aclweb.org/anthology/2020.acl-main.288/)[[code]](https://github.com/NUSTM/ECPE-2D)
- **Effective Inter-Clause Modeling for End-to-End Emotion-Cause Pair Extraction**. *Penghui Wei, Jiahao Zhao, Wenji Mao*. (ACL 2020) [[paper]](https://www.aclweb.org/anthology/2020.acl-main.289/)[[code]](https://github.com/Determined22/Rank-Emotion-Cause)
- **Transition-based Directed Graph Construction for Emotion-Cause Pair Extraction**. *Chuang Fan, Chaofa Yuan, Jiachen Du, Lin Gui, Min Yang, Ruifeng Xu*. (ACL 2020) [[paper]](https://www.aclweb.org/anthology/2020.acl-main.342/)[[code]](https://github.com/HLT-HITSZ/TransECPE)

## Dataset
- **SemEval-2014 Task 4: Aspect Based Sentiment Analysis**. *Maria Pontiki, Dimitris Galanis, John Pavlopoulos, Harris Papageorgiou, Ion Androutsopoulos, Suresh Manandhar*. [[paper]](https://www.aclweb.org/anthology/S14-2004/)[[preprocessed data 1]](https://github.com/songyouwei/ABSAPyTorch/tree/master/datasets/semeval14)[[preprocessed data 2]](https://github.com/howardhsu/BERT-for-RRC-ABSA) - ***Restaurants14 & Laptop14***
- **Tasty Burgers, Soggy Fries: Probing Aspect Robustness in Aspect-Based Sentiment Analysis**. *Xiaoyu Xing, Zhijing Jin, Di Jin, Bingning Wang, Qi Zhang, Xuanjing Huang*. (EMNLP 2020) [[paper]](https://arxiv.org/pdf/2009.07964) [[data]](https://github.com/zhijing-jin/ARTS_TestSet) -**ARTS** (Challenge set for SemEval14)
- **Adaptive Recursive Neural Network for Target-dependent Twitter Sentiment Classification**. *Li Dong, Furu Wei, Chuanqi Tan, Duyu Tang, Ming Zhou, Ke Xu*. (ACL 2014) [[paper]](https://www.aclweb.org/anthology/P14-2009)[[preprocessed data]](https://github.com/songyouwei/ABSA-PyTorch/tree/master/datasets/acl-14-short-data) - ***Twitter for ATSC***
- **Open Domain Targeted Sentiment**. *Margaret Mitchell, Jacqui Aguilar, Theresa Wilson, Benjamin Van Durme*. (EMNLP 2013) [[paper]](https://www.aclweb.org/anthology/D13-1171/)[[data]](http://www.m-mitchell.com/code/index.html)
- **SemEval-2015 Task 12: Aspect Based Sentiment Analysis**. *Maria Pontiki, Dimitris Galanis, Haris Papageorgiou, Suresh Manandhar, Ion Androutsopoulos*. [[paper]](https://www.aclweb.org/anthology/S15-2082/)[[data]](http://alt.qcri.org/semeval2015/task12/)
- **SemEval-2016 Task 5: Aspect Based Sentiment Analysis**. *Maria Pontiki, Dimitris Galanis, Haris Papageorgiou, Ion Androutsopoulos, Suresh Manandhar, Mohammad Al-Smadi, Mahmoud Al-Ayyoub, Yanyan Zhao, Bing Qin, Orphée De Clercq, Véronique Hoste, Marianna Apidianaki, Xavier Tannier, Natalia V. Loukachevitch, Evgeniy V. Kotelnikov, Núria Bel, Salud María Jiménez Zafra, Gülsen Eryigit*. [[paper]](https://www.aclweb.org/anthology/S16-1002/)[[data]](http://alt.qcri.org/semeval2016/task5/)
- **SentiHood: Targeted Aspect Based Sentiment Analysis Dataset for Urban Neighbourhoods**. *Marzieh Saeidi, Guillaume Bouchard, Maria Liakata, Sebastian Riedel*. (COLING 2016) [[paper]](https://www.aclweb.org/anthology/C16-1146)[[data]](https://github.com/uclmr/jack/tree/master/data/sentihood) -  ***LSTM-LOC***
- **Target-oriented Opinion Words Extraction with Target-fused Neural Sequence Labeling**. *Zhifang Fan, Zhen Wu, Xin-Yu Dai, Shujian Huang, Jiajun Chen*. (NAACL 2019) [[paper]](https://www.aclweb.org/anthology/N19-1259/)[[data]](https://github.com/NJUNLP/TOWE) - ***TOWE***
- **A Challenge Dataset and Effective Models for Aspect-Based Sentiment Analysis**. *Qingnan Jiang, Lei Chen, Ruifeng Xu, Xiang Ao, Min Yang*. (EMNLP 2019) [[paper]](https://www.aclweb.org/anthology/D19-1654/)[[data]](https://github.com/siat-nlp/MAMS-for-ABSA) - ***MAMS***
### Normal Sentiment Analysis Dataset (Coarse-grained)
- **Emotion Corpus Construction Based on Selection from Hashtags**. *Minglei Li, Yunfei Long, Qin Lu, Wenjie Li*. (LREC 2016) [[paper]](http://www.lrec-conf.org/proceedings/lrec2016/summaries/515.html)[[data]](https://github.com/CLUEbenchmark/CLUEmotionAnalysis2020)

## Survey & Review & Tutorial
- **Sentiment Analysis and Opinion Mining**. *Bing Liu*. (AAAI 2011 Tutorial) [[slide]](https://www.seas.upenn.edu/~cis630/Sentiment-Analysis-tutorial-AAAI-2011.pdf)
- **Deep Learning for Aspect-Based Sentiment Analysis: A Comparative Review**. *Hai Ha Do, P. W. C. Prasad, Angelika Maag, Abeer Alsadoon*. (ESWA 2019) [[paper]](https://doi.org/10.1016/j.eswa.2018.10.003)
- **Deep Learning for Aspect-Level Sentiment Classification: Survey, Vision, and Challenges**. *Jie Zhou, Jimmy Xiangji Huang, Qin Chen, Qinmin Vivian Hu, Tingting Wang, Liang He*. (IEEE Access 2019) [[paper]](https://ieeexplore.ieee.org/document/8726353)
- **Issues and Challenges of Aspect-based Sentiment Analysis: A Comprehensive Survey**. *Ambreen Nazir, Yuan Rao, Lianwei Wu, Ling Sun*. (IEEE-TAC 2020) [[paper]](https://ieeexplore.ieee.org/abstract/document/8976252)

## Others
- **SKEP: Sentiment Knowledge Enhanced Pre-training for Sentiment Analysis**. *Hao Tian, Can Gao, Xinyan Xiao, Hao Liu, Bolei He, Hua Wu, Haifeng Wang, Feng Wu*. (ACL 2020) [[paper]](https://arxiv.org/abs/2005.05635)
- **PoD: Positional Dependency-Based Word Embedding for Aspect Term Extraction**. *Yichun Yin, Chenguang Wang, Ming Zhang*. (COLING 2020) [[paper]](https://www.aclweb.org/anthology/2020.coling-main.150/)
- **Understanding Pre-trained BERT for Aspect-based Sentiment Analysis**. *Hu Xu, Lei Shu, Philip Yu, Bing Liu*. (COLING 2020) [[paper]](https://www.aclweb.org/anthology/2020.coling-main.21/)

## Repositories/Resources
- [songyouwei / ABSA-PyTorch](https://github.com/songyouwei/ABSA-PyTorch) - Aspect Based Sentiment Analysis, PyTorch Implementations. 基于方面的情感分析，使用PyTorch实现
- [AlexYangLi / ABSA_Keras](https://github.com/AlexYangLi/ABSA_Keras) - Keras Implementation of Aspect based Sentiment Analysis
- [NUSTM / ABSC](https://github.com/NUSTM/ABSC) - aspect-based sentiment classification
- [jiangqn / Aspect-Based-Sentiment-Analysis](https://github.com/jiangqn/Aspect-Based-Sentiment-Analysis) - A paper list for aspect based sentiment analysis.
- [haiker2011 / awesome-nlp-sentiment-analysis](https://github.com/haiker2011/awesome-nlp-sentiment-analysis) - 收集NLP领域相关的数据集、论文、开源实现，尤其是情感分析、情绪原因识别、评价对象和评价词抽取方面

## Posts
### Chinese
- [阿里巴巴 / 细粒度情感分析任务（ABSA）的最新进展](https://mp.weixin.qq.com/s/Jzra95XfjNtDDTNDMD8Lkw)
- [Kaiyuan Gao / 基于特定实体的文本情感分类总结 [PART I]](https://blog.csdn.net/Kaiyuan_sjtu/article/details/89788314)[[PART II]](https://blog.csdn.net/Kaiyuan_sjtu/article/details/89811824)[[PART III]](https://blog.csdn.net/Kaiyuan_sjtu/article/details/89850685)
- [NJU / 如何理解用户评论中的细粒度情感？面向目标的观点词抽取](https://mp.weixin.qq.com/s/zz_9YpaPn5lYzhaNKxylJA)
- [平安寿险PAI / 细粒度情感分析在保险行业的应用](https://zhuanlan.zhihu.com/p/151216832)
