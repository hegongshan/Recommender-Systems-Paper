## Recommender Systems (RS)

* [1.Collaborative Filtering based Recommendation](#collaborative-filtering)
  * [1.1 Neighborhood-based CF](#neighborhood-based)
  * [1.2 Model-based CF](#model-based)
    * [1.2.1 Matrix Factorization](#matrix-factorization)
    * [1.2.2 Distance-based CF](#distance-based-cf)
      * [1.2.2.1 Euclidean Embedding](#euclidean-embedding)
      * [1.2.2.2 Metric Learning](#metric-learning)
* [2.Content-based Recommendation](#content-based)
  * [2.1 Review-based Recommendation](#review-based-recommendation)
* [3.Knowledge Graph based  Recommendation](#knowledge-graph)
* [4.Hybrid Recommendation](#hybrid-recommendation)
* [5.Deep Learning based  Recommendation](#deep-learning)
  * [5.1 Multi-layer Perceptron (MLP)](#multi-layer-perceptron-mlp)
  * [5.2 Autoencoders (AE)](#autoencoders-ae)
  * [5.3 Convolutional Neural Networks (CNNs)](#convolutional-neural-networks-cnns)
* [6.Click-Through Rate (CTR) Prediction](#click-through-rate-ctr-prediction)
* [7.Learning to Rank (LTR)](#learning-to-rank-ltr)
  * [7.1 Pairwise LTR](#pairwise-ltr)
  * [7.2 Listwise LTR](#listwise-ltr)
* [8.Graph-based Recommendation](#graph-based-recommendation)
  * [8.1 Heterogeneous Information Networks](#heterogeneous-information-networks-hin)
  * [8.2 Graph Neural Networks](#graph-neural-networks-gnns)
* [9.Social Recommendation](#social-recommendation)
* [10.Cross-domain Recommendation](#cross-domain-recommendation)
* [11.Group Recommendation](#group-recommendation)
* [12.Cold Start Recommendation](#cold-start-recommendation)
* [13.Point-of-Interest (POI) Recommendation](#point-of-interest-poi-recommendation)
* [14.Context-aware Recommendation](#context-aware-recommendation)
* [15.Sequential Recommendation](#sequential-recommendation)
* [16.Explainable Recommendation](#explainable-recommendation)
* [17.Conversational Recommendation](#conversationalinteractive-recommendation)
* [18.Others](#others)
  * [18.1 Evaluation Metrics](#evaluation-metrics)
  * [18.2 Network Embedding](#network-embedding)
  * [18.3 Survey on Recommender Systems](#survey-on-recommender-systems)

### Collaborative Filtering

* [1992 CACM] **Using collaborative filtering to weave an information Tapestry**. [[PDF](http://citeseer.ist.psu.edu/viewdoc/download;jsessionid=00569AFBC79FB7CC64911E5C977EAC7D?doi=10.1.1.104.3739&rep=rep1&type=pdf)]

> Collaborative Filtering (CF) was firstly proposed.

#### Neighborhood-based

* [1994 CSCW] UserCF: **GroupLens: An Open Architecture for Collaborative Filtering of Netnews**. [[PDF](http://brettb.net/project/papers/1994%20GroupLens%20an%20open%20architecture%20for%20collaborative%20filtering%20of%20netnews.pdf)]
* [1995 CHI] **Social information filtering: algorithms for automating “word of mouth”**. [[PDF](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.100.5571&rep=rep1&type=pdf)]
* [2001 WWW] ItemCF: **Item-Based Collaborative Filtering Recommendation Algorithms**. [[PDF](https://dl.acm.org/doi/pdf/10.1145/371920.372071)]
* [2003] **Amazon.com recommendations: item-to-item collaborative filtering**. [[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1167344)]
* [2004 TOIS] ItemKNN: **Item-Based Top-N Recommendation Algorithms**. [[PDF](http://glaros.dtc.umn.edu/gkhome/fetch/papers/itemrsTOIS04.pdf)]
* [2005] Slope One: **Slope One Predictors for Online Rating-Based Collaborative Filtering**. [[PDF](https://arxiv.org/pdf/cs/0702144.pdf)]
* [2011 ICDM] **Slim: Sparse linear methods for top-n recommender systems**. [[PDF](http://glaros.dtc.umn.edu/gkhome/fetch/papers/SLIM2011icdm.pdf)]
* [2013 KDD] **FISM: Factored Item Similarity Models for Top-N Recommender Systems**. [[PDF](http://glaros.dtc.umn.edu/gkhome/fetch/papers/fismkdd13.pdf)]
* [2014 PAKDD] **HOSLIM: Higher-Order Sparse LInear Method for Top-N Recommender Systems**. [[PDF](https://www-users.cs.umn.edu/~chri2951/hoslim14pakdd.pdf)]
* [2016 RecSys] GLSLIM: **Local Item-Item Models for Top-N Recommendation**. [[PDF](https://www-users.cs.umn.edu/~chri2951/recsy368-christakopoulouA.pdf)]

#### Model-based

* [1999 IJCAI] **Latent Class Models for Collaborative Filtering**. [[PDF](https://www.ijcai.org/Proceedings/99-2/Papers/005.pdf)]
* [2004 TOIS] pLSA: **Latent Semantic Models for Collaborative Filtering**. [[PDF](https://cs.brynmawr.edu/Courses/cs380/fall2006/p89-hofmann.pdf)]

##### Matrix Factorization

* [1998 ICML] SVD: **Learning Collaborative Information Filters**. [[PDF](https://www.ics.uci.edu/~pazzani/Publications/MLC98.pdf)]
* [2004 NIPS] MMMF: **Maximum-Margin Matrix Factorization**. [[PDF](https://papers.nips.cc/paper/2655-maximum-margin-matrix-factorization.pdf)]
* [2005 ICML] **Fast Maximum Margin Matrix Factorization for Collaborative Prediction**. [[PDF](https://icml.cc/Conferences/2005/proceedings/papers/090_FastMaxmimum_RennieSrebro.pdf)]
* [2006] FunkSVD (Latent Factor Model, LFM): **Netflix Update: Try This at Home**. http://sifter.org/~simon/journal/20061211.html
* [2007 KDD] NSVD: **Improving regularized singular value decomposition for collaborative filtering**. [[PDF](http://arek-paterek.com/ap_kdd.pdf)]
* [2007 NIPS] PMF: **Probabilistic matrix factorization**. [[PDF](https://papers.nips.cc/paper/3208-probabilistic-matrix-factorization.pdf)]
* [2008 ICML] BPMF: **Bayesian Probabilistic Matrix Factorization using Markov Chain Monte Carlo**. [[PDF](https://icml.cc/Conferences/2008/papers/600.pdf)]
* [2008 KDD] SVD++: **Factorization meets the neighborhood: a multifaceted collaborative filtering model**. [[PDF](https://dl.acm.org/doi/pdf/10.1145/1401890.1401944)]
* [2008 ICDM] **One-Class Collaborative Filtering**. [[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4781145)]
* [2008 ICDM] WRMF: **Collaborative Filtering for Implicit Feedback Datasets**. [[PDF](http://yifanhu.net/PUB/cf.pdf)]
* [2009 KDD] timeSVD: **Collaborative filtering with temporal dynamics**. [[PDF](http://cseweb.ucsd.edu/classes/fa17/cse291-b/reading/p447-koren.pdf)]
* [2009] BiasedMF: **Matrix Factorization Techniques for Recommender Systems**. [[PDF](http://www.columbia.edu/~jwp2128/Teaching/E4903/papers/ieeecomputer.pdf)]
* [2010] PureSVD: **Performance of Recommender Algorithms on Top-N Recommendation Tasks**. [[PDF](https://tooob.com/api/objs/read/noteid/28575258/)]
* [2011 arXiv] **Feature-Based Matrix Factorization**. [[PDF](https://arxiv.org/pdf/1109.2271.pdf)]
* [2012 JMLR] **SVDFeature: A Toolkit for Feature-based Collaborative Filtering**. [[PDF](https://jmlr.org/papers/volume13/chen12a/chen12a.pdf)]
* [2014 NIPS] Logistic MF: **Logistic Matrix Factorization for Implicit Feedback Data**. [[PDF](http://stanford.edu/~rezab/nips2014workshop/submits/logmat.pdf)] [[Codes](https://github.com/MrChrisJohnson/logistic-mf)]
* [2016 SIGIR] eALS: **Fast Matrix Factorization for Online Recommendation with Implicit Feedback**. [[PDF](http://staff.ustc.edu.cn/~hexn/papers/sigir16-eals-cm.pdf)] [[Codes](https://github.com/hexiangnan/sigir16-eals/)]
* [2020 RecSys] **Neural Collaborative Filtering vs. Matrix Factorization Revisited**. [[PDF](https://arxiv.org/pdf/2005.09683.pdf)] [[Codes](https://github.com/google-research/google-research/tree/master/dot_vs_learned_similarity)]

##### Distance-based CF

###### Euclidean Embedding

* [2010 RecSys] EE: **Collaborative Filtering via Euclidean Embedding**. [[PDF](https://dollar.biz.uiowa.edu/~nstreet/research/recsys10_cfmds.pdf)]
* [2012 APWeb] TEE: **Collaborative Filtering via Temporal Euclidean Embedding**. [[PDF](http://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=36EA6B84B716E282C059541BD0C0AEC9?doi=10.1.1.302.9485&rep=rep1&type=pdf)]

###### Metric Learning

* [2017 WWW] CML: **Collaborative Metric Learning**. [[PDF](https://vision.cornell.edu/se3/wp-content/uploads/2017/03/WWW-fp0554-hsiehA.pdf)]
* [2018 WWW] LRML: **Latent relational metric learning via memory-based attention for collaborative ranking**. [[PDF](https://arxiv.org/pdf/1707.05176.pdf)]
* [2018 arXiv] MetricF: **Metric Factorization: Recommendation beyond Matrix Factorization**. [[PDF](https://arxiv.org/pdf/1802.04606.pdf)]
* [2018 ICDM] TransCF: **Collaborative Translational Metric Learning**. [[PDF](https://arxiv.org/pdf/1906.01637.pdf)]
* [2020 AAAI] SML: **Symmetric Metric Learning with Adaptive Margin for Recommendation**. [[PDF](https://aaai.org/ojs/index.php/AAAI/article/view/5894/5750)]
* [2020 KDD] PMLAM: **Probabilistic Metric Learning with Adaptive Margin for Top-K Recommendation**. [[PDF](https://dl.acm.org/doi/10.1145/3394486.3403147)]

### Content-based

* [2007] **Content-based Recommendation Systems**. [[PDF](https://cs.fit.edu/~pkc/apweb/related/pazzani07aw.pdf)]

#### Review-based Recommendation

* [2017 WSDM] DeepCoNN: **Joint Deep Modeling of Users and Items Using Reviews for Recommendation**. [[PDF](https://arxiv.org/pdf/1701.04783.pdf)]

### Knowledge Graph

* [2016] **Collaborative Knowledge Base Embedding for Recommender Systems**. [[PDF](https://dl.acm.org/doi/pdf/10.1145/2939672.2939673)]
* [2019 WWW] KGCN: **Knowledge Graph Convolutional Networks for Recommender Systems**. [[PDF](https://arxiv.org/pdf/1904.12575.pdf)] [[Codes](https://github.com/hwwang55/KGCN)]
* [2019 KDD] KGNN-LS: **Knowledge-aware Graph Neural Networks with Label Smoothness Regularization for Recommender Systems**. [[PDF](https://arxiv.org/pdf/1905.04413.pdf)] [[Codes](https://github.com/hwwang55/KGNN-LS)]
* [2019 KDD] **KGAT: Knowledge Graph Attention Network for Recommendation**. [[PDF](https://arxiv.org/pdf/1905.07854.pdf)]

### Hybrid Recommendation

* [2017 CIKM] JRL: **Joint Representation Learning for Top-N Recommendation with Heterogeneous Information Sources**. [[PDF](http://shichuan.org/hin/topic/Embedding/2017.%20CIKM%20Joint%20Representation%20Learning%20for%20Top%20N%20Recommendation%20with%20Heterogeneous%20Information%20Sources.pdf)]

### Deep Learning

* [2007] RBM: **Restricted Boltzmann Machines for Collaborative Filtering**. [[PDF](http://www.cs.toronto.edu/~amnih/papers/rbmcf.pdf)]
* [2018 SIGIR] CMN: **Collaborative Memory Network for Recommendation Systems**. [[PDF](https://arxiv.org/pdf/1804.10862.pdf)]

#### Multi-layer Perceptron (MLP)

* [2015 arXiv] NNMF: **Neural Network Matrix Factorization**. [[PDF](https://arxiv.org/pdf/1511.06443.pdf)]
* [2016] **Deep Neural Networks for YouTube Recommendations**. [[PDF](https://dl.acm.org/doi/pdf/10.1145/2959100.2959190)]
* [2016 RecSys] **Wide & Deep Learning for Recommender Systems**. [[PDF](https://arxiv.org/pdf/1606.07792.pdf)]
* [2017 WWW] NCF: **Neural Collaborative Filtering**. [[PDF](http://staff.ustc.edu.cn/~hexn/papers/www17-ncf.pdf)] [[Codes](https://github.com/hexiangnan/neural_collaborative_filtering)]
* [2017 IJCAI] DMF: **Deep Matrix Factorization Models for Recommender Systems**. [[PDF](https://www.ijcai.org/Proceedings/2017/0447.pdf)]
* [2017 CIKM] NNCF: **A Neural Collaborative Filtering Model with Interaction-based Neighborhood**. [[PDF](https://www.researchgate.net/profile/Ting_Bai/publication/320885772_A_Neural_Collaborative_Filtering_Model_with_Interaction-based_Neighborhood/links/5af0d300a6fdcc24364aca37/A-Neural-Collaborative-Filtering-Model-with-Interaction-based-Neighborhood.pdf)]
* [2018 IJCAI] **DELF: A Dual-Embedding based Deep Latent Factor Model for Recommendation**. [[PDF](https://www.ijcai.org/Proceedings/2018/0462.pdf)]
* [2018 TKDE] **NAIS: Neural Attentive Item Similarity Model for Recommendation**. [[PDF](https://arxiv.org/pdf/1809.07053.pdf)]
* [2019 AAAI] **DeepCF: A Unified Framework of Representation Learning and Matching Function Learning in Recommender System**. [[PDF](https://arxiv.org/pdf/1901.04704.pdf)] [[Codes](https://github.com/familyld/DeepCF)]
* [2019 TOIS] DeepICF: **Deep Item-based Collaborative Filtering for Top-N Recommendation**. [[PDF](https://arxiv.org/pdf/1811.04392.pdf)] [[Codes](https://github.com/linzh92/DeepICF)]
* [2019 TOIS] J-NCF: **Joint Neural Collaborative Filtering for Recommender Systems**. [[PDF](https://arxiv.org/pdf/1907.03459.pdf)]

#### Autoencoders (AE)

* [2015] **AutoRec: Autoencoders Meet Collaborative Filtering**. [[PDF](http://users.cecs.anu.edu.au/~akmenon/papers/autorec/autorec-paper.pdf)]
* [2015] CDL: **Collaborative Deep Learning for Recommender Systems**. [[PDF](https://arxiv.org/pdf/1409.2944.pdf)]
* [2016 WSDM] CDAE: **Collaborative Denoising Auto-Encoders for Top-N Recommender Systems**. [[PDF](http://www.alicezheng.org/papers/wsdm16-cdae.pdf)]
* [2017 KDD] CVAE: **Collaborative Variational Autoencoder for Recommender Systems**. [[PDF](http://eelxpeng.github.io/assets/paper/Collaborative_Variational_Autoencoder.pdf)]
* [2018 WWW] Mult-VAE: **Variational Autoencoders for Collaborative Filtering**. [[PDF](https://arxiv.org/pdf/1802.05814.pdf)] [[Codes](https://github.com/dawenl/vae_cf)]

#### Convolutional Neural Networks (CNNs)

* [2018 IJCAI] ONCF: **Outer Product-based Neural Collaborative Filtering**. [[PDF](https://www.ijcai.org/Proceedings/2018/0308.pdf)]

### Click-Through Rate (CTR) Prediction

* [2016 RecSys] FFM: **Field-aware Factorization Machines for CTR Prediction**. [[PDF](https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf)]

* [2017 IJCAI] **DeepFM: A Factorization-Machine based Neural Network for CTR Prediction**. [[PDF](https://www.ijcai.org/Proceedings/2017/0239.pdf)]

* [2018 KDD] DIN: **Deep Interest Network for Click-Through Rate Prediction**. [[PDF](https://arxiv.org/pdf/1706.06978.pdf)]
* [2019 AAAI] DIEN: **Deep Interest Evolution Network for Click-Through Rate Prediction**. [[PDF](https://arxiv.org/pdf/1809.03672.pdf)]
* [2019 IJCAI] DSIN: **Deep Session Interest Network for Click-Through Rate Prediction**. [[PDF](https://www.ijcai.org/Proceedings/2019/0319.pdf)]

### Learning to Rank (LTR)

#### Pairwise LTR

* [2008] **EigenRank: A Ranking-Oriented Approach to Collaborative Filtering**. [[PDF](http://www.cse.ust.hk/faculty/qyang/Docs/2008/SIGIR297-liu.pdf)]
* [2009 UAI] **BPR: Bayesian Personalized Ranking from Implicit Feedback**. [[PDF](https://arxiv.org/pdf/1205.2618.pdf)]
* [2012] **Collaborative Ranking**. [[PDF](https://dl.acm.org/doi/10.1145/2124295.2124314)]
* [2012 JMLR] RankSGD: **Collaborative Filtering Ensemble for Ranking**. [[PDF](http://proceedings.mlr.press/v18/jahrer12b/jahrer12b.pdf)]
* [2012 RecSys] RankALS: **Alternating Least Squares for Personalized Ranking**. [[PDF](https://www.researchgate.net/profile/Gabor_Takacs3/publication/254464370_Alternating_least_squares_for_personalized_ranking/links/5444216d0cf2e6f0c0fb9cdc.pdf)] 
* [2013 SDM] **CoFiSet: Collaborative Filtering via Learning Pairwise Preferences over Item-sets**. [[PDF](https://epubs.siam.org/doi/pdf/10.1137/1.9781611972832.20)]
* [2014 WSDM] **Improving Pairwise Learning for Item Recommendation from Implicit Feedback**. [[PDF](https://www.uni-konstanz.de/mmsp/pubsys/publishedFiles/ReFr14.pdf)]
* [2014] LCR: **Local Collaborative Ranking**. [[PDF](http://bengio.abracadoudou.com/cv/publications/pdf/lee_2014_www.pdf)]
* [2014] **VSRank: A Novel Framework for Ranking-Based Collaborative Filtering**. [[PDF](https://www.researchgate.net/profile/Shuaiqiang_Wang/publication/281834407_VSRank_A_Novel_Framework_for_Ranking-Based_Collaborative_Filtering/links/55fa691408aec948c4a734a8.pdf)]
* [2017 KDD] **Large-scale Collaborative Ranking in Near-Linear Time**. [[PDF](http://www.stat.ucdavis.edu/~chohsieh/rf/KDD_Collaborative_Ranking.pdf)]
* [2018] **CPLR: Collaborative pairwise learning to rank for personalized recommendation**. [[PDF](https://www.sciencedirect.com/science/article/pii/S0950705118300819)]

#### Listwise LTR

* [2007] **COFI RANK - Maximum Margin Matrix Factorization for Collaborative Ranking**. [[PDF](http://papers.nips.cc/paper/3359-cofi-rank-maximum-margin-matrix-factorization-for-collaborative-ranking.pdf)] [[Codes](https://github.com/markusweimer/cofirank)]
* [2010] ListRankMF: **List-wise Learning to Rank with Matrix Factorization for Collaborative Filtering**. [[PDF](http://dmirlab.tudelft.nl/sites/default/files/List-wise%20learning%20to%20rank%20with%20matrix%20factorization%20for%20collaborative%20filtering_RecSys2010.pdf)]
* [2012] **CLiMF: Learning to Maximize Reciprocal Rank with Collaborative Less-is-More Filtering**. [[PDF](https://www.researchgate.net/profile/Alexandros_Karatzoglou/publication/241276529_CLiMF_Learning_to_Maximize_Reciprocal_Rank_with_Collaborative_Less-is-More_Filtering/links/0deec51f752a9714bd000000/CLiMF-Learning-to-Maximize-Reciprocal-Rank-with-Collaborative-Less-is-More-Filtering.pdf)]
* [2012] **TFMAP: Optimizing MAP for Top-N Context-aware Recommendation**. [[PDF](http://dmirlab.tudelft.nl/sites/default/files/SIGIR2012-TFMAP-shi.pdf)]
* [2015] **Collaborative Ranking with a Push at the Top**. [[PDF](https://www-users.cs.umn.edu/~baner029/papers/15/collab-ranking.pdf)]
* [2015] ListCF: **Listwise Collaborative Filtering**. [[PDF](http://users.jyu.fi/~swang/publications/SIGIR15.pdf)]
* [2016] **Ranking-Oriented Collaborative Filtering: A Listwise Approach**. [[PDF]()]

* [2018] **SQL-Rank: A Listwise Approach to Collaborative Ranking**. [[PDF](http://proceedings.mlr.press/v80/wu18c/wu18c.pdf)]

#### Setwise LTR

* [2020 AAAI] **SetRank: A Setwise Bayesian Approach for Collaborative Ranking from Implicit Feedback**. [[PDF](https://arxiv.org/pdf/2002.09841.pdf)]

### Graph-based Recommendation

* [2003 TKDE] Personalized PageRank: **Topic-Sensitive PageRank: A Context-Sensitive Ranking Algorithm for Web Search**. [[PDF](https://ieeexplore.ieee.org/document/1208999)]
* [2007 IJCAI] **ItemRank: A Random-Walk Based Scoring Algorithm for Recommender Engines**. [[PDF](https://www.ijcai.org/Proceedings/07/Papers/444.pdf)]
* [2012 CIKM] **PathRank: A Novel Node Ranking Measure on a Heterogeneous Graph for Recommender Systems**. [[PDF](https://dl.acm.org/doi/pdf/10.1145/2396761.2398488)]
* [2013 ESWA] **PathRank: Ranking nodes on a heterogeneous graph for flexible hybrid recommender systems**. [[PDF](https://www.sciencedirect.com/science/article/pii/S0957417412009657)]
* [2015 NIPS] **Collaborative Filtering with Graph Information: Consistency and Scalable Methods**. [[PDF](https://papers.nips.cc/paper/5938-collaborative-filtering-with-graph-information-consistency-and-scalable-methods.pdf)]

#### Heterogeneous Information Networks (HIN)

* [2013 RecSys] **Recommendation in heterogeneous information networks with implicit user feedback**. [[PDF](https://dl.acm.org/doi/10.1145/2507157.2507230)]

* [2014 WWW] **Random Walks in Recommender Systems: Exact Computation and Simulations**. [[PDF](https://nms.kcl.ac.uk/colin.cooper/papers/recommender-rw.pdf)]
* [2014 WSDM] HeteRec: **Personalized Entity Recommendation: A Heterogeneous Information Network Approach**.  [[PDF](http://hanj.cs.illinois.edu/pdf/wsdm14_xyu.pdf)]
* [2016 TKDE] HeteRS: **A General Recommendation Model for Heterogeneous Networks**. [[PDF](https://ieeexplore.ieee.org/document/7546911)]
* [2017 KDD] FMG: **Meta-Graph Based Recommendation Fusion over Heterogeneous Information Networks**. [[PDF](https://www.researchgate.net/publication/317523407_Meta-Graph_Based_Recommendation_Fusion_over_Heterogeneous_Information_Networks)] [[Codes](https://github.com/HKUST-KnowComp/FMG)]
* [2018 WSDM] HeteLearn: **Recommendation in Heterogeneous Information Networks Based on Generalized Random Walk Model and Bayesian Personalized Ranking**. [[PDF](https://dl.acm.org/doi/pdf/10.1145/3159652.3159715)]
* [2018 TKDE] HERec: **Heterogeneous information network embedding for recommendation**. [[PDF](https://arxiv.org/pdf/1711.10730.pdf)]
* [2019 IJCAI] HueRec: **Unified Embedding Model over Heterogeneous Information Network for Personalized Recommendation**. [[PDF](https://www.ijcai.org/Proceedings/2019/0529.pdf)]

#### Graph Neural Networks (GNNs)

**Rating Prediction**

* [2018 KDD] GCMC: **Graph Convolutional Matrix Completion**. [[PDF](https://www.kdd.org/kdd2018/files/deep-learning-day/DLDay18_paper_32.pdf)]
* [2019 IJCAI] **STAR-GCN: Stacked and Reconstructed Graph Convolutional Networks for Recommender Systems**. [[PDF](https://www.ijcai.org/Proceedings/2019/0592.pdf)] [[Codes](https://github.com/jennyzhang0215/STAR-GCN)]
* [2020 ICLR] IGMC: **Inductive Matrix Completion Based on Graph Neural Networks**. [[PDF](https://openreview.net/pdf?id=ByxxgCEYDS)] [[Codes](https://github.com/muhanzhang/IGMC)]

**Top-N Recommendation**

* [2018 KDD] PinSage: **Graph Convolutional Neural Networks for Web-Scale Recommender Systems**. [[PDF](https://arxiv.org/pdf/1806.01973.pdf)]
* [2018 RecSys] SpectralCF: **Spectral Collaborative Filtering**. [[PDF](https://arxiv.org/pdf/1808.10523.pdf)] [[Codes](https://github.com/lzheng21/SpectralCF)]
* [2019 SIGIR] NGCF: **Neural Graph Collaborative Filtering**. [[PDF](http://staff.ustc.edu.cn/~hexn/papers/sigir19-NGCF.pdf)]
* [2019 ICDM] Multi-GCCF: **Multi-Graph Convolution Collaborative Filtering**. [[PDF](https://arxiv.org/pdf/2001.00267.pdf)]
* [2020 AAAI] LR-GCCF: **Revisiting Graph Based Collaborative Filtering: A Linear Residual Graph Convolutional Network Approach**. [[PDF](https://aaai.org/ojs/index.php/AAAI/article/view/5330/5186)] [[Codes](https://github.com/newlei/LR-GCCF)]
* [2020 SIGIR] **LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation**. [[PDF](http://staff.ustc.edu.cn/~hexn/papers/sigir20-LightGCN.pdf)] [[Codes](https://github.com/kuandeng/LightGCN)]
* [2021 WWW] IMP-GCN: **Interest-aware Message-Passing GCN for Recommendation**. [[PDF](https://arxiv.org/pdf/2102.10044.pdf)] [[Codes](https://github.com/liufancs/IMP_GCN)]

#####  Disentangled GNN

* [2020 AAAI] MCCF: **Multi-Component Graph Convolutional Collaborative Filtering**. [[PDF](https://arxiv.org/pdf/1911.10699.pdf)]

* [2020 SIGIR] DGCF: **Disentangled Graph Collaborative Filtering**. [[PDF](https://arxiv.org/pdf/2007.01764.pdf)] [[Codes](https://github.com/xiangwang1223/disentangled_graph_collaborative_filtering)]

### Social Recommendation

* [2008] **SoRec: Social Recommendation Using Probabilistic Matrix Factorization**. [[PDF](http://www.cse.cuhk.edu.hk/irwin.king.new/_media/people/bo_xu/paper_cikm08_sorec_hao.pdf)]
* [2009] RSTE: **Learning to Recommend with Social Trust Ensemble**. [[PDF](http://www.cse.cuhk.edu.hk/~king/PUB/SIGIR2009-p203.pdf)]
* [2010] SocialMF: **A Matrix Factorization Technique with Trust Propagation for Recommendation in Social Networks**. [[PDF](http://web.cs.ucla.edu/~yzsun/classes/2014Spring_CS7280/Papers/Recommendation/p135-jamali.pdf)]
* [2011] SoReg: **Recommender systems with social regularization**. [[PDF](https://dl.acm.org/doi/10.1145/1935826.1935877)]
* [2013 IJCAI] LOCABAL: **Exploiting Local and Global Social Context for Recommendation**. [[PDF](https://www.ijcai.org/Proceedings/13/Papers/399.pdf)]
* [2013 IJCAI] TrustMF: **Social Collaborative Filtering by Trust**. [[PDF](https://www.ijcai.org/Proceedings/13/Papers/404.pdf)]
* [2015] **TrustSVD: Collaborative Filtering with Both the Explicit and Implicit Influence of User Trust and of Item Ratings**. [[PDF](https://guoguibing.github.io/papers/guo2015trustsvd.pdf)]
* [2018 AAAI] SERec: **Collaborative Filtering with Social Exposure: A Modular Approach to Social Recommendation**. [[PDF](https://arxiv.org/pdf/1711.11458.pdf)]
* [2019 WWW] GraphRec: **Graph Neural Networks for Social Recommendation**. [[PDF](https://arxiv.org/pdf/1902.07243.pdf)]
* [2019 SIGIR] DiffNet: **A neural influence diffusion model for social recommendation**. [[PDF](https://arxiv.org/pdf/1904.10322.pdf)]
* [2021 TKDE] **DiffNet++: A Neural Influence and Interest Diffusion Network for Social Recommendation**. [[PDF](https://arxiv.org/pdf/2002.00844.pdf)]

### Cross-domain Recommendation

* [2008] **If You Like the Devil Wears Prada the Book, Will You also Enjoy the Devil Wears Prada the Movie? A Study of Cross-Domain Recommendations**. [[PDF](https://link.springer.com/content/pdf/10.1007/s00354-008-0041-0.pdf)]
* [2011 RecSys] **A Generic Semantic-based Framework for Cross-domain Recommendation**. [[PDF](http://ir.ii.uam.es/hetrec2011/res/papers/hetrec2011_paper04.pdf)]

### Group Recommendation

* [2009 VLDB] **Group Recommendation: Semantics and Efficiency**. [[PDF](https://www.researchgate.net/profile/Sihem_Amer-Yahia/publication/234829365_Group_Recommendation_Semantics_and_Efficiency/links/54f9c4e10cf25371374ff92e/Group-Recommendation-Semantics-and-Efficiency.pdf)]
* [2018 SIGIR] AGREE: **Attentive Group Recommendation**. [[PDF](http://staff.ustc.edu.cn/~hexn/papers/sigir18-groupRS.pdf)] [[Codes](https://github.com/LianHaiMiao/Attentive-Group-Recommendation)] [[Slides](http://staff.ustc.edu.cn/~hexn/slides/sigir18-group-recsys.pdf)]

### Cold Start Recommendation

* [2002 SIGIR] **Methods and Metrics for Cold-Start Recommendations**. [[PDF](https://dl.acm.org/doi/10.1145/564376.564421)]
* [2008 ICUIMC] **Addressing Cold-Start Problem in Recommendation Systems**. [[PDF](https://dl.acm.org/doi/10.1145/1352793.1352837)]
* [2009 RecSys] **Pairwise Preference Regression for Cold-start Recommendation**. [[PDF](https://dl.acm.org/doi/10.1145/1639714.1639720)]
* [2011 SIGIR] **Functional Matrix Factorizations for Cold-Start Recommendation**. [[PDF](http://nyc.lti.cs.cmu.edu/classes/11-741/s17/Papers/SIGIR11fMF.pdf)]
* [2014 RecSys] **Social Collaborative Filtering for Cold-start Recommendations**. [[PDF](https://dariusb.bitbucket.io/papers/SedhainEtAl-RecSys2014.pdf)]
* [2014 RecSys] **Item Cold-Start Recommendations: Learning Local Collective Embeddings**. [[PDF](https://web.media.mit.edu/~msaveski/assets/publications/2014_item_cold_start/paper.pdf)]

### Context-aware Recommendation

* [2010 RecSys] **Multiverse Recommendation: N-dimensional Tensor Factorization for Context-aware Collaborative Filtering**. [[PDF](https://xamat.github.io/pubs/karatzoglu-recsys-2010.pdf)]
* [2011 RecSys] **Matrix Factorization Techniques for Context Aware Recommendation**. [[PDF](https://www.researchgate.net/profile/Bernd_Ludwig/publication/221140971_Matrix_factorization_techniques_for_context_aware_recommendation/links/0deec52b992aa0ec52000000/Matrix-factorization-techniques-for-context-aware-recommendation.pdf)]
* [2016 RecSys] **Convolutional Matrix Factorization for Document Context-Aware Recommendation**. [[PDF](https://dl.acm.org/doi/10.1145/2959100.2959165)] [[Codes](https://github.com/cartopy/ConvMF)]

### Point-of-Interest (POI) Recommendation

* [2011 SIGIR] **Exploiting Geographical Influence for Collaborative Point-of-Interest Recommendation**. [[PDF](http://www.cse.cuhk.edu.hk/irwin.king.new/_media/presentations/p325.pdf)]
* [2013 SIGIR] **Time-aware Point-of-interest Recommendation**. [[PDF](https://personal.ntu.edu.sg/axsun/paper/sun_sigir13quan.pdf)] [[Datasets](https://personal.ntu.edu.sg/gaocong/data/poidata.zip)]
* [2013 KDD] **Learning Geographical Preferences for Point-of-Interest Recommendation**. [[PDF](http://binbenliu.github.io/papers/poi_kdd13.pdf)] [[Slides](http://www.cse.cuhk.edu.hk/irwin.king.new/_media/presentations/bin_liu07oct2013.pdf)]
* [2013 CIKM] **Personalized Point-of-Interest Recommendation by Mining Users’ Preference Transition**. [[PDF](http://www.ntulily.org/wp-content/uploads/conference/Personalized_Point-of-Interest_Recommendation_by_Mining_Users_Preference_Transition_accepted.pdf)]
* [2014 KDD] **GeoMF: Joint Geographical Modeling and Matrix Factorization for Point-of-Interest Recommendation**. [[PDF](http://staff.ustc.edu.cn/~cheneh/paper_pdf/2014/Defu-Lian-KDD.pdf)]
* [2014 CIKM] **Graph-based Point-of-interest Recommendation with Geographical and Temporal Influences**. [[PDF](https://personal.ntu.edu.sg/axsun/paper/cikm14-yuan.pdf)] [[Datasets](https://personal.ntu.edu.sg/gaocong/data/poidata.zip)]
* [2015 SIGIR] **Rank-GeoFM: A Ranking based Geographical Factorization Method for Point of Interest Recommendation**. [[PDF](https://personal.ntu.edu.sg/gaocong/papers/SIGIR2015_ID246.pdf)] [[Datasets](https://personal.ntu.edu.sg/gaocong/data/poidata.zip)]
* [2016 KDD] **Point-of-Interest Recommendations: Learning Potential Check-ins from Friends**. [[PDF](https://www.kdd.org/kdd2016/papers/files/rfp0448-liA.pdf)]

### Sequential Recommendation

aka. **Next-item Recommendation** or **Next-basket Recommendation**

* [2017 RecSys] TransRec: **Translation-based Recommendation**. [[PDF](https://arxiv.org/pdf/1707.02410.pdf)]
* [2018 WSDM] **Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding**. [[PDF](https://arxiv.org/pdf/1809.07426.pdf)]
* [2018 ICDM] SASRec: **Self-Attentive Sequential Recommendation**. [[PDF](https://arxiv.org/pdf/1808.09781.pdf)]

### Explainable Recommendation

* [2014 SIGIR] EFM: **Explicit Factor Models for Explainable Recommendation based on Phrase-level Sentiment Analysis**. [[PDF](http://yongfeng.me/attach/efm-zhang.pdf)]

### Conversational/Interactive Recommendation

* [2013 CIKM] **Interactive Collaborative Filtering**. [[PDF](http://www0.cs.ucl.ac.uk/staff/w.zhang/papers/icf-cikm.pdf)]
* [2018 SIGIR] **Conversational Recommender System**. [[PDF](https://arxiv.org/pdf/1806.03277.pdf)]

### Debias

* [2021 WSDM] **Denoising Implicit Feedback for Recommendation**. [[PDF](https://arxiv.org/pdf/2006.04153.pdf)]

### Others

#### Evaluation Metrics

* [2004 TOIS] **Evaluating Collaborative Filtering Recommender Systems**. [[PDF](https://grouplens.org/site-content/uploads/evaluating-TOIS-20041.pdf)]
* [2020 KDD <strong style='color:red'>Best  paper</strong>] **On Sampled Metrics for Item Recommendation**. [[PDF](https://dl.acm.org/doi/pdf/10.1145/3394486.3403226)]
* [2020 KDD] **On Sampling Top-K Recommendation Evaluation**. [[PDF](https://dl.acm.org/doi/10.1145/3394486.3403262)]
* [2021 AAAI] **On Estimating Recommendation Evaluation Metrics under Sampling**. [[PDF](https://arxiv.org/pdf/2103.01474.pdf)]

#### Network Embedding

* [2014] **DeepWalk: Online Learning of Social Representations**. [[PDF](https://arxiv.org/pdf/1403.6652.pdf)]
* [2016] **Item2Vec: Neural Item Embedding for Collaborative Filtering**. [[PDF](https://arxiv.org/pdf/1603.04259.pdf)]
* [2016] **node2vec: Scalable Feature Learning for Networks**. [[PDF](https://arxiv.org/pdf/1607.00653.pdf)]
* [2017 KDD] **metapath2vec: Scalable Representation Learning for Heterogeneous Networks**. [[PDF](https://www3.nd.edu/~dial/publications/dong2017metapath2vec.pdf)]
* [2018] **BiNE: Bipartite Network Embedding**. [[PDF](https://www.comp.nus.edu.sg/~xiangnan/papers/sigir18-bipartiteNE.pdf)]
* [2020] **A Survey on Heterogeneous Graph Embedding: Methods, Techniques, Applications and Sources**. [[PDF](https://arxiv.org/pdf/2011.14867.pdf)]

#### Survey on Recommender Systems

* [2002] ***Hybrid Recommender Systems*: Survey and Experiments**. [[PDF](http://gamejam.cti.depaul.edu/~rburke/pubs/burke-umuai02.pdf)]
* [2005 TKDE] **Toward the Next Generation of Recommender Systems: A Survey of the State-of-the-Art and Possible Extensions**. [[PDF](http://pages.stern.nyu.edu/~atuzhili/pdf/TKDE-Paper-as-Printed.pdf)]
* [2009] **A Survey of *Collaborative Filtering* Techniques**. [[PDF](https://downloads.hindawi.com/archive/2009/421425.pdf)]
* [2012] ***Context-aware Recommender Systems* for Learning: a Survey and Future Challenges**. [[PDF](https://www.researchgate.net/profile/Hendrik_Drachsler/publication/234057252_Context-Aware_Recommender_Systems_for_Learning_A_Survey_and_Future_Challenges/links/09e4150eacc85f2864000000.pdf)]
* [2012] ***Cross-domain recommender systems*: A survey of the State of the Art**. [[PDF](https://www.researchgate.net/profile/Francesco_Ricci5/publication/267227272_Cross-domain_recommender_systems_A_survey_of_the_State_of_the_Art/links/5469d3d00cf20dedafd10b29.pdf)]
* [2013] **A survey of collaborative filtering based *social recommender systems***. [[PDF](https://romisatriawahono.net/lecture/rm/survey/information%20retrieval/Yang%20-%20Social%20Recommender%20Systems%20-%202014.pdf)]
* [2013 KBS] **Recommender systems survey**. [[PDF](https://romisatriawahono.net/lecture/rm/survey/information%20retrieval/Bobadilla%20-%20Recommender%20Systems%20-%202013.pdf)]
* [2014] **Collaborative filtering beyond the user-item matrix: A survey of the state of the art and future challenges**. [[PDF](https://dl.acm.org/doi/pdf/10.1145/2556270)]
* [2014] ***Group Recommendations*: Survey and Perspectives**. [[PDF](https://www.researchgate.net/profile/Michal_Kompan/publication/275544963_Group_Recommendations_Survey_and_Perspectives/links/553f56ce0cf24c6a05d2023b.pdf)]
* [2014] ***Active Learning* in Collaborative Filtering Recommender Systems**. [[PDF](http://www.inf.unibz.it/~fricci/papers/al-ec-web-2014.pdf)]
* [2015] **Survey on Learning-to-Rank Based Recommendation Algorithms**. [[PDF](http://jos.org.cn/jos/ch/reader/create_pdf.aspx?file_no=4948&journal_id=jos)]
* [2016] **A Survey of *Point-of-interest Recommendation* in Location-based Social Networks**. [[PDF](https://arxiv.org/pdf/1607.00647.pdf)]
* [2018] ***Explainable Recommendation*: A Survey and New Perspectives**. [[PDF](https://arxiv.org/pdf/1804.11192.pdf)] 
* [2018] ***Deep Learning based Recommender System*: A Survey and New Perspectives**. [[PDF](https://arxiv.org/pdf/1707.07435.pdf)]
* [2019] **A Survey on *Session-based* Recommender Systems**. [[PDF](https://arxiv.org/pdf/1902.04864.pdf)]
* [2019] **Research Commentary on Recommendations with *Side Information*: A Survey and Research Directions**. [[PDF](https://arxiv.org/pdf/1909.12807.pdf)]
* [2020] **A Survey on *Knowledge Graph*-Based Recommender Systems**. [[PDF](https://arxiv.org/pdf/2003.00911.pdf)]
* [2020] **Survey of *Privacy-Preserving* Collaborative Filtering**. [[PDF](https://arxiv.org/pdf/2003.08343.pdf)]
* [2020] **Deep Learning on *Knowledge Graph* for Recommender System: A Survey**. [[PDF](https://arxiv.org/pdf/2004.00387.pdf)]
* [2020] **A Survey on *Conversational* Recommender Systems**. [[PDF](https://arxiv.org/pdf/2004.00646.pdf)]
* [2020] ***Graph Neural Networks* in Recommender Systems: A Survey**. [[PDF](https://arxiv.org/pdf/2011.02260.pdf)]
* [2021 TKDE] ***Bias and Debias* in Recommender System: A Survey and Future Directions**. [[PDF](https://arxiv.org/pdf/2010.03240.pdf)]

