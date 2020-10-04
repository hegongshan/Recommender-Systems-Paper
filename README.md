## Recommender Systems (RS)

* [1.Collaborative Filtering based Recommender Systems](#collaborative-filtering)
  * [1.1 Neighborhood-based CF](#neighborhood-based)
  * [1.2 Model-based CF](#model-based)
* [2.Content-based Recommender Systems](#content-based)
* [3.Knowledge Graph based  Recommender Systems](#knowledge-graph)
* [4.Deep Learning based  Recommender Systems](#deep-learning)
  * [4.1 Multilayer Perceptron](#multilayer-perceptron-mlp)
  * [4.2 Autoencoders](#autoencoders-ae)
  * [4.3 Graph Convolutional Network](#graph-convolutional-network-gcn)
* [5.Click-Through Rate (CTR) Prediction](#click-through-rate-ctr-prediction)
* [6.Others](#others)
  * [6.1 Ranking](#ranking)
  * [6.2 Evaluation Metrics](#evaluation-metrics)
  * [6.3 Embedding](#embedding)

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
* [2008] **One-Class Collaborative Filtering**. [[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4781145)]
* [2008] WRMF: **Collaborative Filtering for Implicit Feedback Datasets**. [[PDF](http://yifanhu.net/PUB/cf.pdf)]
* [2008] SVD++: **Factorization meets the neighborhood: a multifaceted collaborative filtering model**. [[PDF](https://dl.acm.org/doi/pdf/10.1145/1401890.1401944)]
* [2009] BiasSVD: **Matrix Factorization Techniques for Recommender Systems**. [[PDF](http://www.columbia.edu/~jwp2128/Teaching/E4903/papers/ieeecomputer.pdf)]
* [2010] timeSVD: **Collaborative filtering with temporal dynamics**. [[PDF](https://dl.acm.org/doi/pdf/10.1145/1721654.1721677)]
* [2013] **FISM: Factored Item Similarity Models for Top-N Recommender Systems**. [[PDF](http://glaros.dtc.umn.edu/gkhome/fetch/papers/fismkdd13.pdf)]
* [2014] Logistic MF: **Logistic Matrix Factorization for Implicit Feedback Data**. [[PDF](http://stanford.edu/~rezab/nips2014workshop/submits/logmat.pdf)] [[Codes](https://github.com/MrChrisJohnson/logistic-mf)]
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

#### Autoencoders (AE)

* [2015] **AutoRec: Autoencoders Meet Collaborative Filtering**. [[PDF](http://users.cecs.anu.edu.au/~akmenon/papers/autorec/autorec-paper.pdf)]
* [2016] CDAE: **Collaborative Denoising Auto-Encoders for Top-N Recommender Systems**. [[PDF](http://www.alicezheng.org/papers/wsdm16-cdae.pdf)]
* [2018] VAE: **Variational Autoencoders for Collaborative Filtering**. [[PDF](https://arxiv.org/pdf/1802.05814.pdf)] [[Codes](https://github.com/dawenl/vae_cf)]

#### Graph Convolutional Network (GCN)

* [2020] **LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation**. [[PDF](http://staff.ustc.edu.cn/~hexn/papers/sigir20-LightGCN.pdf)]

### Click-Through Rate (CTR) Prediction

* [2017] DIN: **Deep Interest Network for Click-Through Rate Prediction**. [[PDF](https://arxiv.org/pdf/1706.06978.pdf)]

### Others

#### Ranking

* [2009] **BPR: Bayesian Personalized Ranking from Implicit Feedback**. [[PDF](https://arxiv.org/pdf/1205.2618.pdf)]

#### Evaluation Metrics

* [2020] **On Sampled Metrics for Item Recommendation**. [[PDF](https://dl.acm.org/doi/pdf/10.1145/3394486.3403226)]

#### Embedding

* [2014] **DeepWalk: Online Learning of Social Representations**. [[PDF](https://arxiv.org/pdf/1403.6652.pdf)]

* [2016] **Item2Vec: Neural Item Embedding for Collaborative Filtering**. [[PDF](https://arxiv.org/pdf/1603.04259.pdf)]

* [2016] **node2vec: Scalable Feature Learning for Networks**. [[PDF](https://arxiv.org/pdf/1607.00653.pdf)]

* [2018] **BiNE: Bipartite Network Embedding**. [[PDF](https://www.comp.nus.edu.sg/~xiangnan/papers/sigir18-bipartiteNE.pdf)]