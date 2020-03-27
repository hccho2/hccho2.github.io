---
layout: post
title:  "ACER 뽀개기"
date:   2020-03-26 14:53:34 +0900
categories: jekyll update
---
### ACER를 분석해 보자!!!

> Policy Gradient는 기본적으로 on policy algorithm이다. PPO, ACER 등은 off policy train이 가능할 수 있는 모델을 제시하고 있다. 여기서는 ACER를 모델과 구현에 대해서 살펴보고자 한다. 

<!--more-->

* [Sample Efficient Actor-Critic with Experience Replay](https://arxiv.org/abs/1611.01224)
* 구현 코드는 OpenAI의 [baselines](https://github.com/openai/baselines/tree/master/baselines/acer)을 참고하면 된다. 이 구현은 논문에 아주 충실하게 구현되어 있다.