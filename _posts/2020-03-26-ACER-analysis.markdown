---
layout: post
title:  "ACER 뽀개기"
date:   2020-03-27 20:53:34 +0900
tag: Reinforcement-Learning, Off-Policy-RL
---

# ACER를 분석해 보자!!!

> Policy Gradient는 기본적으로 on policy algorithm이다. PPO, ACER 등은 off policy train이 가능할 수 있는 모델을 제시하고 있다. 여기서는 ACER를 모델과 구현에 대해서 살펴보고자 한다. 


* [Sample Efficient Actor-Critic with Experience Replay](https://arxiv.org/abs/1611.01224)
* 구현 코드는 OpenAI의 [baselines](https://github.com/openai/baselines/tree/master/baselines/acer)을 참고하면 된다. 이 구현은 논문에 아주 충실하다.
* Actor-Critic Model에서는 action 확률과 value를 예측하는 방식인데, ACER에서는 action-value function $$Q(x_t,a_t)$$를 예측하고, 그 기대값인 $$V(x_t)$$도 사용한다.



$$
\begin{eqnarray*}
x_t &=& \text { state}, \ \ a_t = \text{ action} \\
\pi(a_t|x_t) &=& \text {모델이 추정한 action 확률}, \ \ \mu(a_{t}|x_{t}) = \text{ old policy 확률}\\
Q(x_t,a_t) &=& \text{모델이 추정한 action에 대한 $Q$ 값} \\
V(x_t) &=& \mathbb{E} [Q(x_t, a_t)] =  \sum_{a_t} \pi(a_t|x_t) Q(x_t,a_t)
\end{eqnarray*}
$$

-----

## Retrace

* 다른 Off Policy algorithm에서 advantage 계산에 사용되는 $$Q$$값을 예측하기 위해서 lambda return 식을 제안하고 있다. (see T. Degris, M. White, and R. S. Sutton. Off-policy actor-critic. In ICML, pp. 457–464, 2012.)

$$ R^{\lambda}_{t} = r_t + (1-\lambda)\gamma V(x_{t+1}) + \lambda \gamma \rho_{t+1} R^{\lambda}_{t+1}$$ 

* This estimator requires that we know how to choose $$\lambda$$ ahead of time to trade off bias and variance. Moreover, when using small values of $$\lambda$$ to reduce variance, occasional large importance weights can still cause instability.  
* 그래서 이 논문에서는 Retrace algorithm( see R. Munos, T. Stepleton, A. Harutyunyan, and M. G. Bellemare. Safe and efficient off-policy reinforcement learning. arXiv preprint arXiv:1606.02647, 2016.)을 사용하고자 한다.
* Given a trajectory(episode path) generated under the behavior policy $$\mu$$(old policy), the Retrace estimator can be expressed recursively as follows($$\lambda = 1$$}:

$$Q^{ret}(x_t, a_t) = r_t + \gamma  \bar{\rho}_{t+1} \Big[ Q^{ret}(x_{t+1}, a_{t+1}) -   Q(x_{t+1}, a_{t+1})\Big] + \gamma V(x_{t+1}),$$

where $$\bar{\rho}_{t}$$ is the truncated importance weight,  $$\bar{\rho}_{t} = \min \{c, \rho_t \}$$ with $$\rho_{t} = \frac{\pi (a_t | x_t)}{ \mu (a_t | x_t)}$$.

Retrace is an off-policy, return-based algorithm which has low variance and is proven to converge (in the tabular case) to the value function of the target policy for any behavior policy.
* 좀 더 구체적으로, Retrace를 계산해 보자. trajectory $$(r_t,D_t,\rho_t, V_t^{\pi}, Q_t^{\pi})_{t=1,\cdots,T}$$와 $$V_{T+1}^{\pi}$$이 주어져 있다고 하자. 
다음 계산은 batch 단위로 이루어지며, 각 time step에서 action $$a_t$$는 여러 action이 아니고, tratjectory에 있는 action이다. 그리고 $$\rho_t$$에 사용되는 old policy는 trajectory 생성에 사용된 확률이다. 
$$Q_t^{\pi}, V_t^{\pi}$$에는 모델의 추정치(즉, new policy에 의한 추정)이다. 즉 trainable variable이 포함되어 있다.  OpenAI baselines 구현에는 $$c=1$$이 사용되었다.

$$
\begin{eqnarray*}
z_{T+1} &=& V_{T+1}^{\pi} \\
Q^{ret}(x_T, a_T) &=& r_T + \gamma z_{T+1} (1-D_T)  \ \ \ \leftarrow \text{$a_T$는 trajectory에 있는 action이다.} \\
z_{T} &=& \bar{\rho}_{T}\Big[ Q^{ret}(x_T, a_T) - Q^{\pi}(x_T, a_T) \Big] + V_T^{\pi} \\
&\vdots& \\
Q^{ret}(x_t, a_t) &=& r_t + \gamma z_{t+1} (1-D_t) \\
z_t &=& \bar{\rho}_{t} \Big[ Q^{ret}(x_t, a_t) - Q^{\pi}(x_t, a_t) \Big] + V_t^{\pi}\\
&\vdots&
\end{eqnarray*}$$

이렇게 계산된 $$\{ Q^{ret}(x_t, a_t) \}$$의 gradient는 backpropagation에 사용하지 않는다.

-----

## Policy Gradient
















