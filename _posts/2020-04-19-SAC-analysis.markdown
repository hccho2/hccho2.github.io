---
layout: post
comments: true  # boolean type, the global switch for posts comments.
shortname: 'https-hccho2-github-io' 
title:  "Soft Actor Critic 분석"
date:   2020-04-19 15:53:34 +0900
---

========


# ACER를 분석해 보자!!!

> * Soft Actor Critic은 Policy Gradient 방식인 Actor Critic 모델의 변형이면서도 Replay Buffer로 train하는 off policy 모델이다. PG이면서도 off policy가 가능한 것은 Soft Q-Learning을 기반으로 하기 때문이다. 
> * SAC는 Continuous Action Space를 대상으로 하는 모델이다. SAC를 Discrete Action Space에 적용할 수 있도록 수정한 모델도 있다.
> * 모델의 성능은 PPO보다 못하다.


-----

## 개요

* 2018년 1월, [Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, Sergey Levine. Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290){:target="_blank"}
* 2018년 12월 [Tuomas Haarnoja, Aurick Zhou, Kristian Hartikainen, George Tucker, Sehoon Ha, Jie Tan, Vikash Kumar, Henry Zhu, Abhishek Gupta, Pieter Abbeel, Sergey Levine. Soft Actor-Critic Algorithms and Applications](https://arxiv.org/abs/1812.05905){:target="_blank"}: Value Network을 없애고, Q-Network과 Policy Network으로 구성. Policy Nework에 사용하는 temperature hyperparameter $$\alpha$$를 train하는 식이 추가되었다.
* SAC for Discrete Action Space(2019년 10월): [Soft Actor-Critic for Discrete Action Settings](https://arxiv.org/abs/1910.07207){:target="_blank"}
* Soft Q-Learning(2017년 2월): [Reinforcement Learning with Deep Energy-Based Policies](https://arxiv.org/abs/1702.08165){:target="_blank"}


PPO            |  SAC
:-------------------------:|:-------------------------:
![]({{ '/assets/images/pendulum-ppo.gif' | relative_url }}){: style="width: 70%;" class="center"}  |  ![]({{ '/assets/images/pendulum-sac.gif' | relative_url }}){: style="width: 70%;" class="center"}

*PPO가 SAC보다 안정적이다. SAC모델 결과가 좀 더 불안정해 보이는 이유는, SAC 모델이 가능한 random action을 추구하면서도 임무(task)를 성공시키는 것이 목적이기 때문이다.

## Policy & Entropy

* Policy Gradient의 일반적인 이야기를 해보자. Policy Gradient의 Objective Function은 다음과 같은 형태로 주어진다.

$$\max\ [r\log p] \quad \quad \text{or}  \quad \quad \max\ [A \log p]$$

* Policy 최적화는 높은 reward $r$을 주는 확룰 $p$를 더 크게 하거나 높은 advantage $$A$$를 주는 확률 $$p$$를 더 크게 하려고 최적화를 수행한다. SAC에서는 Objective Function 

$$\max\ \big[Q + \alpha (-\log p)\big] \quad \quad \text{or}  \quad \quad \min\ \big[\alpha \log p - Q\big] $$

로 주어진다. 이 식은 2가지로 해석할 수 있다.
```
	** 해석1: target 확률을 $$\frac{1}{\alpha}\exp(Q)$$(또는 softmax $$Q$$)로 보고, 확률 $$p$$를 생성한 distribution $$\pi$$와 $$\frac{1}{\alpha}\exp Q$$와의 KL-Divergence를 최소화 한다. 즉 $$D_{KL}\big(\pi \ \Vert\ \frac{1}{\alpha}\exp Q\big) \approx \alpha \log p - Q$$
	** 해석2: $Q$값과 entorpy $$(-\log p)$$의 weighted sum을 최대화한다. entropy의 weight가 $$\alpha$$이다. $$Q$$를 advantage로 대체하는 것도 가능하다.
```
* SAC는 보통의 Policy Gradient와 다른 Soft Q-Learning을 사용하기 때문에, Replay Buffer를 사용하는 Off Policy 모델이다. Soft Q-Learning은 $$\exp Q$$ (또는 softmax $$Q$$) 값을 policy 확률로 보는 방법이다.















## Reference

* [딥러닝 정리 자료](https://drive.google.com/open?id=16olGwVvk_smtgopmuUtouOf1ad1RGpIf){:target="_blank"}
* <https://bair.berkeley.edu/blog/2017/10/06/soft-q-learning/>{:target="_blank"}

