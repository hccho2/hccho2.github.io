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
* 구현 코드는 OpenAI의 [baselines](https://github.com/openai/baselines/tree/master/baselines/acer)을 참고하면 된다. 이 구현은 논문에 아주 충실하다.
* Actor-Critic Model에서는 action 확률과 value를 예측하는 방식인데, ACER에서는 action-value function $$Q(x_t,a_t)$$를 예측하고, 그 기대값인 $$V(x_t)$$도 사용한다.
* Off Policy algorithm에서 advantage 계산에 사용되는 $Q$값을 예측하기 위해서 lambda return 식을 제안하고 있다.


$$
\begin{eqnarray*}
x_t &=& \text { state}, \ \a_t = \text{ action} \\
\pi(a_t|x_t) &=& \text {모델이 추정한 action 확률}, \ \ \mu(a_{t}|x_{t}) = \text{ old policy 확률}\\
Q(x_t,a_t) &=& \text{모델이 추정한 action에 대한 $Q$ 값} \\
V(x_t) &=& \E [Q(x_t, a_t)] =  \sum_{a_t} \pi(a_t|x_t) Q(x_t,a_t)
\end{eqnarray*}
$$