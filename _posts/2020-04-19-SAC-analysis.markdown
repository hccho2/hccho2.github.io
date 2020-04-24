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
![]({{ '/assets/images/pendulum-ppo.gif' | relative_url }}){: style="width: 50%;" class="center"}  |  ![]({{ '/assets/images/pendulum-sac.gif' | relative_url }}){: style="width: 50%;" class="center"}

*PPO가 SAC보다 안정적이다. SAC모델 결과가 좀 더 불안정해 보이는 이유는, SAC 모델이 가능한 random action을 추구하면서도 임무(task)를 성공시키는 것이 목적이기 때문이다.


## Reference

* [딥러닝 정리 자료](https://drive.google.com/open?id=16olGwVvk_smtgopmuUtouOf1ad1RGpIf){:target="_blank"}
* <https://bair.berkeley.edu/blog/2017/10/06/soft-q-learning/>{:target="_blank"}

