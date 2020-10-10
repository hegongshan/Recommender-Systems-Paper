## Recommender Systems (RS)

* [1.Collaborative Filtering based Recommender Systems](#collaborative-filtering)
  * [1.1 Neighborhood-based CF](#neighborhood-based)
  * [1.2 Model-based CF](#model-based)
* [2.Content-based Recommender Systems](#content-based)
* [3.Knowledge Graph based  Recommender Systems](#knowledge-graph)
* [4.Deep Learning based  Recommender Systems](#deep-learning)
  * [4.1 Multilayer Perceptron](#multilayer-perceptron-mlp)
  * [4.2 Autoencoders](#autoencoders-ae)
* [5.Click-Through Rate (CTR) Prediction](#click-through-rate-ctr-prediction)
* [6.Learning to Rank (LTR)](#learning-to-rank-ltr)
  * [6.1 Pairwise LTR](#pairwise-ltr)
  * [6.2 Listwise LTR](#listwise-ltr)
* [7.Graph-based RS](#graph-based-rs)
  * [7.1 Graph Neural Network](#graph-neural-network-gnn)
* [8.Social Recommendation](#social-recommendation)
* [9.Others](#others)
  * [9.1 Evaluation Metrics](#evaluation-metrics)
  * [9.2 Embedding](#embedding)

### Collaborative Filtering

#### Neighborhood-based

* [2001] ItemCF: **Item-Based Collaborative Filtering Recommendation Algorithms**. [[PDF](https://dl.acm.org/doi/pdf/10.1145/371920.372071)]
* [2003] **Amazon.com recommendations: item-to-item collaborative filtering**. [[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1167344)]
* [2004] **Item-Based Top-N Recommendation Algorithms**. [[PDF](http://glaros.dtc.umn.edu/gkhome/fetch/papers/itemrsTOIS04.pdf)]
* [2005] Slope One: **Slope One Predictors for Online Rating-Based Collaborative Filtering**. [[PDF](https://arxiv.org/pdf/cs/0702144.pdf)]
* [2011] **Slim: Sparse linear methods for top-n recommender systems**. [[PDF](http://glaros.dtc.umn.edu/gkhome/fetch/papers/SLIM2011icdm.pdf)]

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
* [2013] **FISM: Factored Item Similarity Models for Top-N Recommender Systems**. [[PDF](http://glaros.dtc.umn.edu/gkhome/fetch/papers/fismkdd13.pdf)]
* [2014] Logistic MF: **Logistic Matrix Factorization for Implicit Feedback Data**. [[PDF](http://stanford.edu/~rezab/nips2014workshop/submits/logmat.pdf)] [[Codes](https://github.com/MrChrisJohnson/logistic-mf)]
* [2016] eALS: **Fast Matrix Factorization for Online Recommendation with Implicit Feedback**. [[PDF](http://staff.ustc.edu.cn/~hexn/papers/sigir16-eals-cm.pdf)] [[Codes](https://github.com/hexiangnan/sigir16-eals/)]
* [2020] **Neural Collaborative Filtering vs. Matrix Factorization Revisited**. [[PDF](https://arxiv.org/pdf/2005.09683.pdf)] [[Codes](https://github.com/google-research/google-research/tree/master/dot_vs_learned_similarity)]

### Content-based

* [2007] **Content-based Recommendation Systems**. [[PDF](https://cs.fit.edu/~pkc/apweb/related/pazzani07aw.pdf)]

### Knowledge Graph

* [2016] **Collaborative Knowledge Base Embedding for Recommender Systems**. [[PDF](https://dl.acm.org/doi/pdf/10.1145/2939672.2939673)]

### Deep Learning

* [2007] RBM: **Restricted Boltzmann Machines for Collaborative Filtering**. [[PDF](http://www.cs.toronto.edu/~amnih/papers/rbmcf.pdf)]
* [2015] CDL: **Collaborative Deep Learning for Recommender Systems**. [[PDF](https://arxiv.org/pdf/1409.2944.pdf)]

#### Multilayer Perceptron (MLP)

* [2016] **Deep Neural Networks for YouTube Recommendations**. [[PDF](https://dl.acm.org/doi/pdf/10.1145/2959100.2959190)]
* [2016] **Wide & Deep Learning for Recommender Systems**. [[PDF](https://arxiv.org/pdf/1606.07792.pdf)]
* [2017] NCF: **Neural Collaborative Filtering**. [[PDF](http://staff.ustc.edu.cn/~hexn/papers/www17-ncf.pdf)] [[Codes](https://github.com/hexiangnan/neural_collaborative_filtering)]
* [2019] **DeepCF: A Unified Framework of Representation Learning and Matching Function Learning in Recommender System**. [[PDF](https://arxiv.org/pdf/1901.04704.pdf)] [[Codes](https://github.com/familyld/DeepCF)]

#### Autoencoders (AE)

* [2015] **AutoRec: Autoencoders Meet Collaborative Filtering**. [[PDF](http://users.cecs.anu.edu.au/~akmenon/papers/autorec/autorec-paper.pdf)]
* [2016] CDAE: **Collaborative Denoising Auto-Encoders for Top-N Recommender Systems**. [[PDF](http://www.alicezheng.org/papers/wsdm16-cdae.pdf)]
* [2018] VAE: **Variational Autoencoders for Collaborative Filtering**. [[PDF](https://arxiv.org/pdf/1802.05814.pdf)] [[Codes](https://github.com/dawenl/vae_cf)]

### Click-Through Rate (CTR) Prediction

* [2018] DIN: **Deep Interest Network for Click-Through Rate Prediction**. [[PDF](https://arxiv.org/pdf/1706.06978.pdf)]

### Learning to Rank (LTR)

#### Pairwise LTR

* [2008] **EigenRank: A Ranking-Oriented Approach to Collaborative Filtering**. [[PDF](http://www.cse.ust.hk/faculty/qyang/Docs/2008/SIGIR297-liu.pdf)]

* [2009] **BPR: Bayesian Personalized Ranking from Implicit Feedback**. [[PDF](https://arxiv.org/pdf/1205.2618.pdf)]
* [2014] **VSRank: A Novel Framework for Ranking-Based Collaborative Filtering**. [[PDF](https://www.researchgate.net/profile/Shuaiqiang_Wang/publication/281834407_VSRank_A_Novel_Framework_for_Ranking-Based_Collaborative_Filtering/links/55fa691408aec948c4a734a8.pdf)]
* [2018] **CPLR: Collaborative pairwise learning to rank for personalized recommendation**. [[PDF](https://www.sciencedirect.com/science/article/pii/S0950705118300819)]

#### Listwise LTR

* [2007] **COFI RANK - Maximum Margin Matrix Factorization for Collaborative Ranking**. [[PDF](http://papers.nips.cc/paper/3359-cofi-rank-maximum-margin-matrix-factorization-for-collaborative-ranking.pdf)] [[Codes](https://github.com/markusweimer/cofirank)]
* [2010] ListRankMF: **List-wise Learning to Rank with Matrix Factorization for Collaborative Filtering**. [[PDF](http://dmirlab.tudelft.nl/sites/default/files/List-wise%20learning%20to%20rank%20with%20matrix%20factorization%20for%20collaborative%20filtering_RecSys2010.pdf)]
* [2012] **CLiMF: Learning to Maximize Reciprocal Rank with Collaborative Less-is-More Filtering**. [[PDF](https://www.researchgate.net/profile/Alexandros_Karatzoglou/publication/241276529_CLiMF_Learning_to_Maximize_Reciprocal_Rank_with_Collaborative_Less-is-More_Filtering/links/0deec51f752a9714bd000000/CLiMF-Learning-to-Maximize-Reciprocal-Rank-with-Collaborative-Less-is-More-Filtering.pdf)]
* [2012] **TFMAP: Optimizing MAP for Top-N Context-aware Recommendation**. [[PDF](http://dmirlab.tudelft.nl/sites/default/files/SIGIR2012-TFMAP-shi.pdf)]
* [2015] **Collaborative Ranking with a Push at the Top**. [[PDF](https://www-users.cs.umn.edu/~baner029/papers/15/collab-ranking.pdf)]
* [2015] ListCF: **Listwise Collaborative Filtering**. [[PDF](http://users.jyu.fi/~swang/publications/SIGIR15.pdf)]

### Graph-based RS

* [2014] **Random Walks in Recommender Systems: Exact Computation and Simulations**. [[PDF](https://nms.kcl.ac.uk/colin.cooper/papers/recommender-rw.pdf)]

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

### Others

#### Evaluation Metrics

* [2020] **On Sampled Metrics for Item Recommendation**. [[PDF](https://dl.acm.org/doi/pdf/10.1145/3394486.3403226)]

#### Embedding

* [2014] **DeepWalk: Online Learning of Social Representations**. [[PDF](https://arxiv.org/pdf/1403.6652.pdf)]

* [2016] **Item2Vec: Neural Item Embedding for Collaborative Filtering**. [[PDF](https://arxiv.org/pdf/1603.04259.pdf)]

* [2016] **node2vec: Scalable Feature Learning for Networks**. [[PDF](https://arxiv.org/pdf/1607.00653.pdf)]

* [2018] **BiNE: Bipartite Network Embedding**. [[PDF](https://www.comp.nus.edu.sg/~xiangnan/papers/sigir18-bipartiteNE.pdf)]