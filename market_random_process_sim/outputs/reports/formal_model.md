# Formal Model

## A. Agent Sets

At each day `t`, buyers and sellers are represented as finite sets:

```text
B_t = {1, ..., N_b}
S_t = {1, ..., N_s}
```

## B. Order Submission Rule

Each buyer submits `(p_i^b, q_i^b)` and each seller submits `(p_j^s, q_j^s)`.

```text
p_i^b = P_{t-1} exp(epsilon_i^b)
p_j^s = P_{t-1} exp(epsilon_j^s)
epsilon_i^b, epsilon_j^s ~ N(0, sigma^2)
q_i^b, q_j^s ~ Uniform(1, q_max)
```

## C. Budget and Inventory Constraints

Buyer cash constraint:

```text
q_i^b p_i^b <= W_i
```

Seller inventory constraint:

```text
q_j^s <= H_j
```

The ablation study switches these constraints on and off to identify their separate effects.

## D. Aggregate Demand and Supply

```text
D_t(p) = sum_i q_i^b 1(p_i^b >= p)
S_t(p) = sum_j q_j^s 1(p_j^s <= p)
```

## E. Matched Volume and Clearing Rule

```text
V_t(p) = min(D_t(p), S_t(p))
I_t(p) = |D_t(p) - S_t(p)|
```

The clearing price first maximizes `V_t(p)`, then minimizes `I_t(p)`, then uses the median candidate price if multiple prices remain.

## F. Price Evolution

```text
P_t = p_t*
r_t = log(P_t) - log(P_{t-1})
x_t = log(P_t / P_0)
```

## G. Mean Reversion Test

```text
r_{t+1} = alpha + beta x_t + epsilon_t
```

`beta < 0` indicates weak mean reversion.

## H. Volatility and Extreme Movement

```text
sigma_r = Std(r_t)
E = max_t |r_t|
```

## I. Curve Roughness

```text
D_norm(p) = D(p) / max(D(p))
S_norm(p) = S(p) / max(S(p))
R_D = mean(|Delta D_norm|)
R_S = mean(|Delta S_norm|)
R = (R_D + R_S) / 2
```

Lower roughness means smoother aggregate order curves.

## J. Random Process Interpretation

- Unlimited wealth: the order-auction mechanism produces a random-walk-like process.
- Wealth constraints: feasible order flow changes with price level, producing boundedness and weak mean reversion.
- Market depth: deeper markets produce smoother curves, lower volatility, and fewer extreme jumps.
