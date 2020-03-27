---
layout: post
title:  "ACER 뽀개기"
date:   2020-03-26 14:53:34 +0900
categories: jekyll update
---
*** ACER를 분석해 보자!!!

$$D_{KL}(p \| q) = \int_x p(x) \log \frac{p(x)}{q(x)} dx$$

이 식의 의미

$$
\begin{eqnarray}
\widehat{g}_t^{acer}  &=& \bar{\rho}_{t}  \nabla_{\theta} \log \pi_{\theta}(a_t| x_t) [Q^{ret}(x_t, a_t) - V_{\theta_v}(x_t)]  \\
& & + \underset{a \sim \pi}{\mathbb{E}} \left( \left[\frac{\rho_{t}(a) - c}{\rho_{t}(a)} \right]_+ \hspace{-3mm}
\nabla_{\theta}  \log \pi_{\theta}(a| x_t)[Q_{\theta_v}(x_t, a)- V_{\theta_v}(x_t)] \right)
\end{eqnarray}$$

행렬
$$
\begin{pmatrix}
 1 & a_1 & a_1^2 & \cdots & a_1^n \\
 1 & a_2 & a_2^2 & \cdots & a_2^n \\
 \vdots  & \vdots& \vdots & \ddots & \vdots \\
 1 & a_m & a_m^2 & \cdots & a_m^n    
 \end{pmatrix}
$$

$$
\begin{eqnarray}
f(x) &=& \int_0^\infty g(t,x) dt \\
&=& z(x)
\end{eqnarray}
$$


$$\begin{eqnarray} 
y &=& x^4 + 4      \nonumber \\
&=& (x^2+2)^2 -4x^2 \nonumber \\
&\le&(x^2+2)^2    \nonumber
\end{eqnarray}$$
