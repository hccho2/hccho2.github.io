---
layout: post
comments: true  # boolean type, the global switch for posts comments.
shortname: 'https-hccho2-github-io' 
title:  "Soft Actor Critic 분석"
date:   2020-04-19 15:53:34 +0900
---

========


# ACER를 분석해 보자!!!

> Soft Actor Critic은 Policy Gradient 방식인 Actor Critic 모델의 변형이면서도 Replay Buffer로 train하는 off policy 모델이다. PG이면서도 off policy가 가능한 것은 Soft Q-Learning을 기반으로 하기 때문이다. 
SAC는 Continuous Action Space를 대상으로 하는 모델이다. SAC를 discrete Action Space에 적용할 수 수정한 모델도 있다.

-----

## 개요

* [Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, Sergey Levine. Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290)
* [Tuomas Haarnoja, Aurick Zhou, Kristian Hartikainen, George Tucker, Sehoon Ha, Jie Tan, Vikash Kumar, Henry Zhu, Abhishek Gupta, Pieter Abbeel, Sergey Levine. Soft Actor-Critic Algorithms and Applications](https://arxiv.org/abs/1812.05905)
* SAC for Discrete Action Space: [Soft Actor-Critic for Discrete Action Settings](https://arxiv.org/abs/1910.07207)

## Reference

* [딥러닝 정리 자료](https://drive.google.com/open?id=16olGwVvk_smtgopmuUtouOf1ad1RGpIf){:target="_blank"}
* <https://bair.berkeley.edu/blog/2017/10/06/soft-q-learning/>{:target="_blank"}

