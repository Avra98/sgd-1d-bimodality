# 1-D SGD · iterate shape beyond stochastic stability

Interactive browser lab for the scalar cocycle

\[
X_{t+1}=(1-\eta h_t)X_t+\alpha X_t^3-\beta X_t^5+\sigma\xi_t
\]

with random curvature \(h_t\) (Gaussian / Uniform / Bernoulli / χ²). Knobs for \(\eta\), \(\mu\), \(v\), \(\alpha\), \(\beta\), \(\sigma\), and seed; live \(\eta_{MS}\), \(\eta_L\), \(\hat\eta_H\).

**Live demo:** https://avra98.github.io/sgd-1d-bimodality/

## 2-D sharp–tangent lab (`sgd-2d-bimodality.html`)

Eq. 37 of the draft with sampled curvature on both directions:

\[
z_{t+1}=z_t-\eta[(h_t+2\beta y_t)z_t+\tfrac{c_4}{6}z_t^3+c_6 z_t^5]+\sigma_z\xi_t,\qquad
y_{t+1}=y_t-\eta[g_t y_t+\beta z_t^2]+\sigma_y\xi_t'
\]

Sharp curvature \(h_t\sim(\mu_z,v_z)\), tangent curvature \(g_t\sim(\mu_y,v_y)\), each from its own
law (Gaussian / uniform / two-point / χ² / rare-spike / deterministic). Live diagnostics: bare and
effective Lyapunov exponents \(\hat\chi,\hat\chi_{\rm eff}\), tangent offset \(\bar y\) vs the predictor
fixed point \(-(\beta/\mu_y)\mathbb{E}[z^2]\), predicted lobe centers \(\pm\nu_\star\), phase-plane cloud,
z/y marginals, and an η-strip. Presets cover the spectral-condition matrix
\((\eta\mu_z\gtrless 2,\ \eta\mu_y<2,\ v_z,v_y,\beta,c_4\gtrless 0)\).
