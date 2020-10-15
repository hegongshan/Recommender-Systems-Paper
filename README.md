## Recommender Systems (RS)

* [1.Collaborative Filtering based Recommender Systems](#collaborative-filtering)
  * [1.1 Neighborhood-based CF](#neighborhood-based)
  * [1.2 Model-based CF](#model-based)
* [2.Content-based Recommender Systems](#content-based)
  * [2.1 Review-based Recommendation](#review-based-recommendation)
* [3.Knowledge Graph based  Recommender Systems](#knowledge-graph)
* [4.Deep Learning based  Recommender Systems](#deep-learning)
  * [4.1 Multilayer Perceptron](#multilayer-perceptron-mlp)
  * [4.2 Autoencoders](#autoencoders-ae)
* [5.Click-Through Rate (CTR) Prediction](#click-through-rate-ctr-prediction)
* [6.Learning to Rank (LTR)](#learning-to-rank-ltr)
  * [6.1 Pairwise LTR](#pairwise-ltr)
  * [6.2 Listwise LTR](#listwise-ltr)
* [7.Graph-based Recommendation](#graph-based-recommendation)
  * [7.1 Graph Neural Network](#graph-neural-network-gnn)
* [8.Social Recommendation](#social-recommendation)
* [9.Cold Start Problem](#cold-start-problem)
* [10.Point-of-Interest (POI) Recommendation](#point-of-interest-poi-recommendation)
* [11.Sequential Recommendation](#sequential-recommendation)
* [12.Conversational Recommendation](#conversationalinteractive-recommendation)
* [13.Others](#others)
  * [13.1 Evaluation Metrics](#evaluation-metrics)
  * [13.2 Embedding](#embedding)

### Collaborative Filtering

#### Neighborhood-based

* [2001] ItemCF: **Item-Based Collaborative Filtering Recommendation Algorithms**. [[PDF](https://dl.acm.org/doi/pdf/10.1145/371920.372071)]
* [2003] **Amazon.com recommendations: item-to-item collaborative filtering**. [[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1167344)]
* [2004] **Item-Based Top-N Recommendation Algorithms**. [[PDF](http://glaros.dtc.umn.edu/gkhome/fetch/papers/itemrsTOIS04.pdf)]
* [2005] Slope One: **Slope One Predictors for Online Rating-Based Collaborative Filtering**. [[PDF](https://arxiv.org/pdf/cs/0702144.pdf)]

#### Model-based

* [1998] SVD: **Learning Collaborative Information Filters**. [[PDF](https://www.ics.uci.edu/~pazzani/Publications/MLC98.pdf)]
* [2006] FunkSVD (Latent Factor Model, LFM): **Netflix Update: Try This at Home**. http://sifter.org/~simon/journal/20061211.html
* [2007] PMF: **Probabilistic matrix factorization**. [[PDF](https://papers.nips.cc/paper/3208-probabilistic-matrix-factorization.pdf)]
* [2007] NSVD: **Improving regularized singular value decomposition for collaborative filtering**. [[PDF](http://arek-paterek.com/ap_kdd.pdf)]
* [2008] **One-Class Collaborative Filtering**. [[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4781145)]
* [2008] WRMF: **Collaborative Filtering for Implicit Feedback Datasets**. [[PDF](http://yifanhu.net/PUB/cf.pdf)]
* [2008] SVD++: **Factorization meets the neighborhood: a multifaceted collaborative filtering model**. [[PDF](https://dl.acm.org/doi/pdf/10.1145/1401890.1401944)]
* [2009] timeSVD: **Collaborative filtering with temporal dynamics**. [[PDF](http://cseweb.ucsd.edu/classes/fa17/cse291-b/reading/p447-koren.pdf)]
* [2009] **Matrix Factorization Techniques for Recommender Systems**. [[PDF](http://www.columbia.edu/~jwp2128/Teaching/E4903/papers/ieeecomputer.pdf)]
* [2010] PureSVD: **Performance of Recommender Algorithms on Top-N Recommendation Tasks**. [[PDF](https://tooob.com/api/objs/read/noteid/28575258/)]
* [2011] **Slim: Sparse linear methods for top-n recommender systems**. [[PDF](http://glaros.dtc.umn.edu/gkhome/fetch/papers/SLIM2011icdm.pdf)]
* [2013] **FISM: Factored Item Similarity Models for Top-N Recommender Systems**. [[PDF](http://glaros.dtc.umn.edu/gkhome/fetch/papers/fismkdd13.pdf)]
* [2014] Logistic MF: **Logistic Matrix Factorization for Implicit Feedback Data**. [[PDF](http://stanford.edu/~rezab/nips2014workshop/submits/logmat.pdf)] [[Codes](https://github.com/MrChrisJohnson/logistic-mf)]
* [2016] eALS: **Fast Matrix Factorization for Online Recommendation with Implicit Feedback**. [[PDF](http://staff.ustc.edu.cn/~hexn/papers/sigir16-eals-cm.pdf)] [[Codes](https://github.com/hexiangnan/sigir16-eals/)]
* [2020] **Neural Collaborative Filtering vs. Matrix Factorization Revisited**. [[PDF](https://arxiv.org/pdf/2005.09683.pdf)] [[Codes](https://github.com/google-research/google-research/tree/master/dot_vs_learned_similarity)]

### Content-based

* [2007] **Content-based Recommendation Systems**. [[PDF](https://cs.fit.edu/~pkc/apweb/related/pazzani07aw.pdf)]

#### Review-based Recommendation

* [2017 WSDM] DeepCoNN: **Joint Deep Modeling of Users and Items Using Reviews for Recommendation**. [[PDF](https://arxiv.org/pdf/1701.04783.pdf)]

### Knowledge Graph

* [2016] **Collaborative Knowledge Base Embedding for Recommender Systems**. [[PDF](https://dl.acm.org/doi/pdf/10.1145/2939672.2939673)]

### Deep Learning

* [2007] RBM: **Restricted Boltzmann Machines for Collaborative Filtering**. [[PDF](http://www.cs.toronto.edu/~amnih/papers/rbmcf.pdf)]

#### Multilayer Perceptron (MLP)

* [2016] **Deep Neural Networks for YouTube Recommendations**. [[PDF](https://dl.acm.org/doi/pdf/10.1145/2959100.2959190)]
* [2016] **Wide & Deep Learning for Recommender Systems**. [[PDF](https://arxiv.org/pdf/1606.07792.pdf)]
* [2017 WWW] NCF: **Neural Collaborative Filtering**. [[PDF](http://staff.ustc.edu.cn/~hexn/papers/www17-ncf.pdf)] [[Codes](https://github.com/hexiangnan/neural_collaborative_filtering)]
* [2019 AAAI] **DeepCF: A Unified Framework of Representation Learning and Matching Function Learning in Recommender System**. [[PDF](https://arxiv.org/pdf/1901.04704.pdf)] [[Codes](https://github.com/familyld/DeepCF)]
* [2019 TOIS] J-NCF: **Joint Neural Collaborative Filtering for Recommender Systems**. [[PDF](https://arxiv.org/pdf/1907.03459.pdf)]

#### Autoencoders (AE)

* [2015] **AutoRec: Autoencoders Meet Collaborative Filtering**. [[PDF](http://users.cecs.anu.edu.au/~akmenon/papers/autorec/autorec-paper.pdf)]
* [2015] CDL: **Collaborative Deep Learning for Recommender Systems**. [[PDF](https://arxiv.org/pdf/1409.2944.pdf)]
* [2016] CDAE: **Collaborative Denoising Auto-Encoders for Top-N Recommender Systems**. [[PDF](http://www.alicezheng.org/papers/wsdm16-cdae.pdf)]
* [2017 KDD] CVAE: **Collaborative Variational Autoencoder for Recommender Systems**. [[PDF](http://eelxpeng.github.io/assets/paper/Collaborative_Variational_Autoencoder.pdf)]
* [2018 WWW] Mult-VAE: **Variational Autoencoders for Collaborative Filtering**. [[PDF](https://arxiv.org/pdf/1802.05814.pdf)] [[Codes](https://github.com/dawenl/vae_cf)]

### Click-Through Rate (CTR) Prediction

* [2016 RecSys] FFM: **Field-aware Factorization Machines for CTR Prediction**. [[PDF](https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf)]

* [2017 IJCAI] **DeepFM: A Factorization-Machine based Neural Network for CTR Prediction**. [[PDF](https://www.ijcai.org/Proceedings/2017/0239.pdf)]

* [2018 KDD] DIN: **Deep Interest Network for Click-Through Rate Prediction**. [[PDF](https://arxiv.org/pdf/1706.06978.pdf)]
* [2019 AAAI] DIEN: **Deep Interest Evolution Network for Click-Through Rate Prediction**. [[PDF](https://arxiv.org/pdf/1809.03672.pdf)]
* [2019 IJCAI] DSIN: **Deep Session Interest Network for Click-Through Rate Prediction**. [[PDF](https://www.ijcai.org/Proceedings/2019/0319.pdf)]

### Learning to Rank (LTR)

#### Pairwise LTR

* [2008] **EigenRank: A Ranking-Oriented Approach to Collaborative Filtering**. [[PDF](http://www.cse.ust.hk/faculty/qyang/Docs/2008/SIGIR297-liu.pdf)]
* [2009] **BPR: Bayesian Personalized Ranking from Implicit Feedback**. [[PDF](https://arxiv.org/pdf/1205.2618.pdf)]
* [2012] **Collaborative Ranking**. [[PDF](https://dl.acm.org/doi/10.1145/2124295.2124314)]
* [2012 JMLR] RankSGD: **Collaborative Filtering Ensemble for Ranking**. [[PDF](http://proceedings.mlr.press/v18/jahrer12b/jahrer12b.pdf)]
* [2012 RecSys] RankALS: **Alternating Least Squares for Personalized Ranking**. [[PDF](https://www.researchgate.net/profile/Gabor_Takacs3/publication/254464370_Alternating_least_squares_for_personalized_ranking/links/5444216d0cf2e6f0c0fb9cdc.pdf)] 
* [2014] LCR: **Local Collaborative Ranking**. [[PDF](http://bengio.abracadoudou.com/cv/publications/pdf/lee_2014_www.pdf)]
* [2014] **VSRank: A Novel Framework for Ranking-Based Collaborative Filtering**. [[PDF](https://www.researchgate.net/profile/Shuaiqiang_Wang/publication/281834407_VSRank_A_Novel_Framework_for_Ranking-Based_Collaborative_Filtering/links/55fa691408aec948c4a734a8.pdf)]
* [2018] **CPLR: Collaborative pairwise learning to rank for personalized recommendation**. [[PDF](https://www.sciencedirect.com/science/article/pii/S0950705118300819)]

#### Listwise LTR

* [2007] **COFI RANK - Maximum Margin Matrix Factorization for Collaborative Ranking**. [[PDF](http://papers.nips.cc/paper/3359-cofi-rank-maximum-margin-matrix-factorization-for-collaborative-ranking.pdf)] [[Codes](https://github.com/markusweimer/cofirank)]
* [2010] ListRankMF: **List-wise Learning to Rank with Matrix Factorization for Collaborative Filtering**. [[PDF](http://dmirlab.tudelft.nl/sites/default/files/List-wise%20learning%20to%20rank%20with%20matrix%20factorization%20for%20collaborative%20filtering_RecSys2010.pdf)]
* [2012] **CLiMF: Learning to Maximize Reciprocal Rank with Collaborative Less-is-More Filtering**. [[PDF](https://www.researchgate.net/profile/Alexandros_Karatzoglou/publication/241276529_CLiMF_Learning_to_Maximize_Reciprocal_Rank_with_Collaborative_Less-is-More_Filtering/links/0deec51f752a9714bd000000/CLiMF-Learning-to-Maximize-Reciprocal-Rank-with-Collaborative-Less-is-More-Filtering.pdf)]
* [2012] **TFMAP: Optimizing MAP for Top-N Context-aware Recommendation**. [[PDF](http://dmirlab.tudelft.nl/sites/default/files/SIGIR2012-TFMAP-shi.pdf)]
* [2015] **Collaborative Ranking with a Push at the Top**. [[PDF](https://www-users.cs.umn.edu/~baner029/papers/15/collab-ranking.pdf)]
* [2015] ListCF: **Listwise Collaborative Filtering**. [[PDF](http://users.jyu.fi/~swang/publications/SIGIR15.pdf)]
* [2016] **Ranking-Oriented Collaborative Filtering: A Listwise Approach**. [[PDF]()]

### Graph-based Recommendation

* [2003 TKDE] **Topic-Sensitive PageRank: A Context-Sensitive Ranking Algorithm for Web Search**. [[PDF](https://ieeexplore.ieee.org/document/1208999)]
* [2007 IJCAI] **ItemRank: A Random-Walk Based Scoring Algorithm for Recommender Engines**. [[PDF](https://www.ijcai.org/Proceedings/07/Papers/444.pdf)]
* [2012 CIKM] **PathRank: A Novel Node Ranking Measure on a Heterogeneous Graph for Recommender Systems**. [[PDF](https://dl.acm.org/doi/pdf/10.1145/2396761.2398488)]
* [2013 ESWA] **PathRank: Ranking nodes on a heterogeneous graph for flexible hybrid recommender systems**. [[PDF](https://www.sciencedirect.com/science/article/pii/S0957417412009657)]
* [2014 WWW] **Random Walks in Recommender Systems: Exact Computation and Simulations**. [[PDF](https://nms.kcl.ac.uk/colin.cooper/papers/recommender-rw.pdf)]
* [2014 WSDM] HeteRec: **Personalized Entity Recommendation: A Heterogeneous Information Network Approach**.  [[PDF](http://hanj.cs.illinois.edu/pdf/wsdm14_xyu.pdf)]
* [2016 TKDE] HeteRS: **A General Recommendation Model for Heterogeneous Networks**. [[PDF](https://ieeexplore.ieee.org/document/7546911)]
* [2017 KDD] FMG: **Meta-Graph Based Recommendation Fusion over Heterogeneous Information Networks**. [[PDF](https://www.researchgate.net/publication/317523407_Meta-Graph_Based_Recommendation_Fusion_over_Heterogeneous_Information_Networks)] [[Codes](https://github.com/HKUST-KnowComp/FMG)]
* [2018 WSDM] HeteLearn: **Recommendation in Heterogeneous Information Networks Based on Generalized Random Walk Model and Bayesian Personalized Ranking**. [[PDF](https://dl.acm.org/doi/pdf/10.1145/3159652.3159715)]
* [2019 IJCAI] HueRec: **Unified Embedding Model over Heterogeneous Information Network for Personalized Recommendation**. [[PDF](https://www.ijcai.org/Proceedings/2019/0529.pdf)]

#### Graph Neural Network (GNN)

* [2018] **Graph Convolutional Neural Networks for Web-Scale Recommender Systems**. [[PDF](https://arxiv.org/pdf/1806.01973.pdf)]

* [2020] **LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation**. [[PDF](http://staff.ustc.edu.cn/~hexn/papers/sigir20-LightGCN.pdf)]

### Social Recommendation

* [2008] **SoRec: Social Recommendation Using Probabilistic Matrix Factorization**. [[PDF](http://www.cse.cuhk.edu.hk/irwin.king.new/_media/people/bo_xu/paper_cikm08_sorec_hao.pdf)]
* [2009] RSTE: **Learning to Recommend with Social Trust Ensemble**. [[PDF](http://www.cse.cuhk.edu.hk/~king/PUB/SIGIR2009-p203.pdf)]
* [2010] SociaMF: **A Matrix Factorization Technique with Trust Propagation for Recommendation in Social Networks**. [[PDF](http://web.cs.ucla.edu/~yzsun/classes/2014Spring_CS7280/Papers/Recommendation/p135-jamali.pdf)]
* [2011] SoReg: **Recommender systems with social regularization**. [[PDF](https://dl.acm.org/doi/10.1145/1935826.1935877)]
* [2013] TrustMF: **Social Collaborative Filtering by Trust**. [[PDF](https://www.ijcai.org/Proceedings/13/Papers/404.pdf)]
* [2015] **TrustSVD: Collaborative Filtering with Both the Explicit and Implicit Influence of User Trust and of Item Ratings**. [[PDF](https://guoguibing.github.io/papers/guo2015trustsvd.pdf)]
* [2018 AAAI] SERec: **Collaborative Filtering with Social Exposure: A Modular Approach to Social Recommendation**. [[PDF](https://arxiv.org/pdf/1711.11458.pdf)]

### Cold Start Problem

* [2002 SIGIR] **Methods and Metrics for Cold-Start Recommendations**. [[PDF](https://dl.acm.org/doi/10.1145/564376.564421)]

* [2008 ICUIMC] **Addressing Cold-Start Problem in Recommendation Systems**. [[PDF](https://dl.acm.org/doi/10.1145/1352793.1352837)]
* [2009 RecSys] **Pairwise Preference Regression for Cold-start Recommendation**. [[PDF](https://dl.acm.org/doi/10.1145/1639714.1639720)]

* [2011 SIGIR] **Functional Matrix Factorizations for Cold-Start Recommendation**. [[PDF](http://nyc.lti.cs.cmu.edu/classes/11-741/s17/Papers/SIGIR11fMF.pdf)]

* [2014 RecSys] **Social Collaborative Filtering for Cold-start Recommendations**. [[PDF](https://dariusb.bitbucket.io/papers/SedhainEtAl-RecSys2014.pdf)]
* [2014 RecSys] **Item Cold-Start Recommendations: Learning Local Collective Embeddings**. [[PDF](https://web.media.mit.edu/~msaveski/assets/publications/2014_item_cold_start/paper.pdf)]

### Point-of-Interest (POI) Recommendation

* [2011 SIGIR] **Exploiting Geographical Influence for Collaborative Point-of-Interest Recommendation**. [[PDF](http://www.cse.cuhk.edu.hk/irwin.king.new/_media/presentations/p325.pdf)]
* [2013 KDD] **Learning Geographical Preferences for Point-of-Interest Recommendation**. [[PDF](http://dnslab.jnu.ac.kr/classes/old_courses/2015s_das/[KDD_2013]%20Learning%C3%A5%20geographical%20preferences%20for%20point-of-interest%20recommendation.pdf)] [[Slides](http://www.cse.cuhk.edu.hk/irwin.king.new/_media/presentations/bin_liu07oct2013.pdf)]
* [2013 CIKM] **Personalized Point-of-Interest Recommendation by Mining Users’ Preference Transition**. [[PDF](http://www.ntulily.org/wp-content/uploads/conference/Personalized_Point-of-Interest_Recommendation_by_Mining_Users_Preference_Transition_accepted.pdf)]
* [2014 KDD] **GeoMF: Joint Geographical Modeling and Matrix Factorization for Point-of-Interest Recommendation**. [[PDF](http://staff.ustc.edu.cn/~cheneh/paper_pdf/2014/Defu-Lian-KDD.pdf)]
* [2014 CIKM] **Graph-based Point-of-interest Recommendation with Geographical and Temporal Influences**. [[PDF](https://www.ntu.edu.sg/home/axsun/paper/cikm14-yuan.pdf)]
* [2015 SIGIR] **Rank-GeoFM: A Ranking based Geographical Factorization Method for Point of Interest Recommendation**. [[PDF](https://www.ntu.edu.sg/home/gaocong/papers/SIGIR2015_ID246.pdf)]
* [2016 KDD] **Point-of-Interest Recommendations: Learning Potential Check-ins from Friends**. [[PDF](https://www.kdd.org/kdd2016/papers/files/rfp0448-liA.pdf)]

### Sequential Recommendation

* [2018 WSDM] **Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding**. [[PDF](https://arxiv.org/pdf/1809.07426.pdf)]

### Conversational/Interactive Recommendation

* [2013 CIKM] **Interactive Collaborative Filtering**. [[PDF](http://www0.cs.ucl.ac.uk/staff/w.zhang/papers/icf-cikm.pdf)]
* [2018 SIGIR] **Conversational Recommender System**. [[PDF](https://arxiv.org/pdf/1806.03277.pdf)]

### Others

#### Evaluation Metrics

* [2020] **On Sampled Metrics for Item Recommendation**. [[PDF](https://dl.acm.org/doi/pdf/10.1145/3394486.3403226)]

#### Embedding

* [2014] **DeepWalk: Online Learning of Social Representations**. [[PDF](https://arxiv.org/pdf/1403.6652.pdf)]

* [2016] **Item2Vec: Neural Item Embedding for Collaborative Filtering**. [[PDF](https://arxiv.org/pdf/1603.04259.pdf)]

* [2016] **node2vec: Scalable Feature Learning for Networks**. [[PDF](https://arxiv.org/pdf/1607.00653.pdf)]

* [2018] **BiNE: Bipartite Network Embedding**. [[PDF](https://www.comp.nus.edu.sg/~xiangnan/papers/sigir18-bipartiteNE.pdf)]