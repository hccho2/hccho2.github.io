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

*그림: PPO가 SAC보다 안정적이다. SAC모델 결과가 좀 더 불안정해 보이는 이유는, SAC 모델이 가능한 random action을 추구하면서도 임무(task)를 성공시키는 것이 목적이기 때문이다.

## Policy & Entropy

* Policy Gradient의 일반적인 이야기를 해보자. Policy Gradient의 Objective Function은 다음과 같은 형태로 주어진다.

$$\max\ [r\log p] \quad \quad \text{or}  \quad \quad \max\ [A \log p]$$

* Policy 최적화는 높은 reward $$r$$을 주는 확룰 $$p$$를 더 크게 하거나 높은 advantage $$A$$를 주는 확률 $$p$$를 더 크게 하려고 최적화를 수행한다. SAC에서는 Objective Function 

$$\max\ \big[Q + \alpha (-\log p)\big] \quad \quad \text{or}  \quad \quad \min\ \big[\alpha \log p - Q\big] $$

로 주어진다. 
* 이 식은 2가지로 해석할 수 있다.
	* 해석1: target 확률을 $$\frac{1}{\alpha}\exp(Q)$$(또는 softmax $$Q$$)로 보고, 확률 $$p$$를 생성한 distribution $$\pi$$와 $$\frac{1}{\alpha}\exp Q$$와의 KL-Divergence를 최소화 한다. 즉 $$D_{KL}\big(\pi \ \Vert\ \frac{1}{\alpha}\exp Q\big) \approx \alpha \log p - Q$$
	* 해석2: $Q$값과 entorpy $$(-\log p)$$의 weighted sum을 최대화한다. entropy의 weight가 $$\alpha$$이다. $$Q$$를 advantage로 대체하는 것도 가능하다.

* SAC는 보통의 Policy Gradient와 다른 Soft Q-Learning을 사용하기 때문에, Replay Buffer를 사용하는 Off Policy 모델이다. Soft Q-Learning은 $$\exp Q$$ (또는 softmax $$Q$$) 값을 policy 확률로 보는 방법이다.

![]({{ '/assets/images/sac-unimoda-multimodall-policy.png' | relative_url }}){: style="width: 100%;" class="center"}

*그림: A multimodal $$Q$$-function: 1. 왼쪽 그림과 같은 $$Q$$-function이 주어져 있을 때, policy를 정규분포로 예측한다면, 오른꼭의 작은 mode에 대한 action 확률이 높아질 수 없다. 2. policy를 $$\exp Q(s_t,a_t)$$에 비례하게 만든다면, 상황 변화에 더 잘 대응할 수 있는 모델을 만들 수 있다.

* SAC모델은 일반적인 sate value function을 대신하여, \underline{soft state value function}을 다음과 같이 정의한다. 일반적인 state value function에 etropy term을 추가하여 exploration이 가능하도록 했다.

$$\begin{eqnarray}
H(p) &=& \mathbb{E}_{x\sim p} \Big[ -\log p(x) \Big], \nonumber\\
 V(s_t) &=&  \mathbb{E}_{a_t\sim\pi} \big[ Q(s_t,a_t) \big] +\alpha H \big(\pi(\cdot | s_t)\big)  \nonumber\\
				&=&  \mathbb{E}_{a_t\sim\pi} \big[ Q(s_t,a_t) -\alpha \log \pi(a_t \vert s_t) \big] \ \  \text{for } \alpha > 0, \label{eq43}\\
Q(s_t,a_t) &=&  \mathbb{E}_{s_{t+1}\sim p, a_{t+1} \sim \pi} \Bigg[ r(s_t, a_t, s_{t+1}) + \gamma \Big( Q(s_{t+1}, a_{t+1}) + \alpha H(\pi(\cdot | s_{t+1}))  \Big) \Bigg]\nonumber \\
&=&  \mathbb{E}_{s_{t+1}\sim p} \Bigg[ r(s_t, a_t, s_{t+1}) + \gamma V(s_{t+1}) \Bigg] \nonumber
\end{eqnarray}$$

* entropy regularization coefficient(or temperature coefficient) $$\alpha$$는 상수로 고정하는 경우도 있고, training을 통해 update하는 경우도 있다.
* 이론적으로 위와 같이 soft value function을 정의해도 optimal policy로 수렴한다.

* Policy Network($$\pi_\phi$$), Value Network($$V_\psi, V_{\bar{\psi}}$$), Q-Network($$Q_{\theta}$$) network이 필요하다. Value값은 $$Q$$값으로 부터 구할 수도 있지만, SAC에서는 모델의 안정성을 위해 별도의 Value Network을 두었다. $$V_{\bar{\psi}}$$는 Value target Network으로 train 대상이 되지는 않고, $$V_\psi$$로 부터 exponentially moving average로 update된다. 두번째 SAC 논문에서는 Value Network 없이 Q-Network만으로 구현하고 있다. continuous action space이기 때문에, Q-Network의 입력은 state와 action의 concat이 된다.
* SAC는 Replay Buffer를 사용하는 off policy 모델이다. Replay Buffer $$\mathcal{D} = \{ (s_t,a_t,r_t,d_n,s_{t+1}) \}$$로 구성된다. 
* Value Loss: Value Network $$V_\psi$$ training.

$$\begin{eqnarray}
J_V(\psi) = \mathbb{E}_{s_t \sim \mathcal{D}}\Bigg[{\frac{1}{2}\bigg(V_\psi(s_t) - \mathbb{E}_{\bar{a}_t\sim\pi_\phi}\big[{Q_\theta(s_t, \bar{a}_t) - \alpha\log \pi_\phi(\bar{a}_t|s_t)}\big] \bigg)^2} \Bigg] \label{eq41}\tag{1}
\end{eqnarray}$$

![]({{ '/assets/images/sac_1.png' | relative_url }}){: style="width: 70%;" class="center"}

*그림: Value Loss: $$\bar{a}_t$$는 replay buffer에서의 action이 아니고, policy network에서 새롭게 생성한 action이다. 참고로 $$s_t$$ 대신 $$s_{t+1}$$를 이용해서 value loss를 구할 수도 있다.

* 식($$\ref{eq41}$$)에서 기대값을 직접 계산하기 어렵기 때문에 policy network $$\pi_\phi$$에서 sampling을 통해 action $$\bar{a}_t$$를 뽑아서 계산하면 된다.

$$  \mathbb{E}_{\bar{a}_t\sim\pi_\phi}\big[Q_\theta(s_t, \bar{a}_t) - \alpha\log \pi_\phi(\bar{a}_t|s_t)\big] \ \rightarrow \ Q_\theta(s_t, \bar{a}_t) - \alpha\log \pi_\phi(\bar{a}_t|s_t) $$

* Q-Function Loss: Q-Network($$Q_{\theta}$$) training.
$$
\begin{eqnarray}
J_Q(\theta)  &=& \mathbb{E}_{(s_t,a_t)\sim\mathcal{D}} \Bigg[ \frac{1}{2} \bigg( Q_\theta(s_t,a_t) - \underbrace{\hat{Q}(s_t,a_t)}_{\text{stop gradient}} \bigg)^2 \Bigg]  \ \ \text{with}  \nonumber \\
\hat{Q}(s_t,a_t) &:=& r(s_t,a_t) + \gamma \mathbb{E}_{s_{t+1}\sim p} \Big [ V_{\bar{\psi}} (s_{t+1})  \Big] \ \ \leftarrow {p: \text{transition probability.}} \label{eq42}\tag{2}
\end{eqnarray}$$

* next state $$s_{t+1}$$은 $$s_{t+1}\sim p$$로 표시되어 있지만, 구현에서는 replay buffer에 있는 $$s_{t+1}$$이다. $$\hat{Q}(s_t,a_t)$$의 gradient는 계산되지 않아야 한다. 
* 식($$\ref{eq41}$$)에서 Value Network의 target에 Q-Network값이 사용되고, 식(\ref{eq42})에서 Q-Newtork의 target에 Value Network이 사용된다. 이런 순환 참조 구조를 벗어나기 위해, Value Network의 copy인 Value target Network $$V_{\bar{\psi}}$$를 만드는 것이다. 
또 다른 측면으로, 식(\ref{eq42})에서 Value Network $$V_{\psi}$$ 대신 target Network $$V_{\bar{\psi}}$$를 사용하는 것은 DQN에서 main network, target network으로 2개 사용하는 것과 같은 원리이다.
* target value Network $$V_{\bar{\psi}}$$ update 시키는 방식 2가지.
	* 주기적으로(periodically) update.
	* Polyak averaging: exponentially weighted averaging

* Policy Loss: expected KL-Divergence 

$$J_{\pi}(\phi)  = \mathbb{E}_{s_t \sim \mathcal{D}} \Bigg [ D_{KL} \bigg( \pi_{\phi} (\cdot | s_t)\ \Vert\ \frac{\exp(\frac{1}{\alpha}Q_\theta(s_t, \cdot))}{Z_\theta(s_t)}  \bigg) \Bigg] $$ 

* 보통의 Policy Gradient 모델이 Off Policy 방법을 적용하기 위해, Old Policy로 생성된 Reaplay Buffer의 data로 New Policy를 update하기 위한 방법이 있어야 한다. TRPO, PPO에서는 probability ratio를 이용한 important sampling 기법을 사용한다. 
SAC에서는 $$\exp Q(s_t,\cdot)$$를 normalization($$Z_\theta(s_t)$$)해서 확률분포로 간주한다. 그래서 new policy와 $$\exp Q(s_t, \cdot)$$의 KL-Divergence가 최소화 되도록 한다. 실제 계산에서는 normalization term인 $$Z_\theta(s_t)$$는 무시해도 된다.

$$\begin{eqnarray}
J_{\pi}(\phi)  &=&  \mathbb{E}_{s_t \sim \mathcal{D}} \Bigg [ \mathbb{E}_{\bar{a}_t\sim \pi_{\phi}} \Big[ \alpha \log \pi_{\phi} ( \bar{a}_t | s_t)   - Q_\theta(s_t, \bar{a}_t)  \Big]  \Bigg] \label{eq44} \\
&\approx& \mathbb{E}_{s_t \sim \mathcal{D}} \Big [  \alpha \log \pi_{\phi} ( \bar{a}_t | s_t)   - Q_\theta(s_t, \bar{a}_t)  \Big] \label{eq49} 
\end{eqnarray}$$

* 이 식에서도 기대값 계산은 Value Loss와 동일하게 sampling을 통해 처리한다. 그런데, sampling된 $$\bar{a}_t$$에 관한 backpropagation의 미분이 필요하기 때문에 Policy Network의 distribution에서 reparameterization trick이 필요하다.




## Reference

* [딥러닝 정리 자료](https://drive.google.com/open?id=16olGwVvk_smtgopmuUtouOf1ad1RGpIf){:target="_blank"}
* <https://bair.berkeley.edu/blog/2017/10/06/soft-q-learning/>{:target="_blank"}
