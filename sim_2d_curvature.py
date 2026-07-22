"""2-D extension of the sampled-curvature bimodality model (eq. 37 + random h).

State (z, y):
    L_t = h_t/2 z^2 + c4/24 z^4 + (z^6 confinement) + lam_y/2 y^2 + beta z^2 y
    z' = z - eta * [ (h_t + 2 beta y) z + (c4/6) z^3 + 2 z^5 ]
    y' = y - eta * [ lam_y y + beta z^2 ]
h_t i.i.d. Gaussian, mean a=1, sd s.  Per-step update clipped at 0.8 (as in the
1-D multiplicative note).  beta=0 recovers the 1-D model exactly.

For each (eta, s, beta) we report:
    M0        mass at |z|<0.05  (onset dial)
    ybar      stationary tangent mean  (self-stabilization shift)
    Ez2       oscillation energy
    chi_bare  E ln|1 - eta h|                       (1-D prediction)
    chi_eff   E ln|1 - eta (h + 2 beta y_t)|  measured along the trajectory
              (the self-consistent Lyapunov exponent)
"""
import numpy as np

R, T, BURN = 512, 30000, 10000
CLIP = 0.8
C4 = -6.0          # cubic coeff c4/6 = -1, matches f' = h x - x^3 + 2x^5
LAM_Y = 0.5


def run(eta, s, beta, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 0.1, R)
    y = np.zeros(R)
    m0_hits = 0
    n_obs = 0
    ybar_acc = 0.0
    ez2_acc = 0.0
    chi_eff_acc = 0.0
    zs_sample = []
    for t in range(T):
        h = rng.normal(1.0, s, R)
        h_eff = h + 2 * beta * y
        gz = h_eff * z + (C4 / 6.0) * z**3 + 2.0 * z**5
        gy = LAM_Y * y + beta * z**2
        z = z - np.clip(eta * gz, -CLIP, CLIP)
        y = y - np.clip(eta * gy, -CLIP, CLIP)
        if t >= BURN:
            n_obs += 1
            m0_hits += np.mean(np.abs(z) < 0.05)
            ybar_acc += y.mean()
            ez2_acc += np.mean(z**2)
            chi_eff_acc += np.mean(np.log(np.abs(1 - eta * h_eff) + 1e-300))
            if t % 200 == 0:
                zs_sample.append(z.copy())
    hh = rng.normal(1.0, s, 2_000_000)
    chi_bare = np.mean(np.log(np.abs(1 - eta * hh) + 1e-300))
    return dict(
        M0=m0_hits / n_obs,
        ybar=ybar_acc / n_obs,
        Ez2=ez2_acc / n_obs,
        chi_bare=chi_bare,
        chi_eff=chi_eff_acc / n_obs,
        z=np.concatenate(zs_sample),
    )


if __name__ == "__main__":
    print(f"{'eta':>5} {'s':>4} {'beta':>4} | {'chi_bare':>8} {'chi_eff':>8} "
          f"{'M0':>6} {'ybar':>7} {'Ez2':>7}")
    for s in (0.30, 0.90):
        for beta in (0.0, 0.5):
            for eta in (1.7, 1.8, 1.9, 2.0, 2.05, 2.1, 2.2):
                r = run(eta, s, beta)
                print(f"{eta:5.2f} {s:4.2f} {beta:4.1f} | "
                      f"{r['chi_bare']:8.3f} {r['chi_eff']:8.3f} "
                      f"{r['M0']:6.3f} {r['ybar']:7.3f} {r['Ez2']:7.4f}")
            print()
