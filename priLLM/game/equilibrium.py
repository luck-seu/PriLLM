import numpy as np
import cvxpy as cp


def solve_weighted_linear_congestion_eq(p, c, d, D,
                                        w_p=1.0, w_d=1.0, w_q=1.0, w0=0.0,
                                        solver="OSQP", **solver_opts):
    """
    Nash equilibrium with *weighted* linear cost:
        C_ij = w0 + w_p * p_j + w_d * d_ij + w_q * (q_j / c_j)

    Parameters
    ----------
    p : (m,)  prices of service providers
    c : (m,)  capacities (positive)
    d : (n,m) distance/latency matrix
    D : (n,)  demand of each user
    w_p, w_d, w_q, w0 : float
        Weights for price, distance, congestion, and constant bias.
    solver : str, optional
        CVXPY solver name (default "OSQP").  Others: "GUROBI", "CPLEX", …
    **solver_opts : dict
        Extra options forwarded to prob.solve(...).

    Returns
    -------
    x_star : (n,m) ndarray
        Equilibrium allocation.
    q_star : (m,)  ndarray
        Total load per provider.
    total_cost : float
        Sum_i Sum_j x_ij * C_ij  (includes w0 term).
    status : str
        Solver status.
    """

    # ---------- input arrays ----------
    p = np.asarray(p, dtype=float).ravel()
    c = np.asarray(c, dtype=float).ravel()
    d = np.asarray(d, dtype=float)
    D = np.asarray(D, dtype=float).ravel()

    n, m = d.shape
    # print(n, m)
    assert p.shape == (m,) and c.shape == (m,), f"p{p.shape}, c{c.shape} must be length m{m}"
    assert D.shape == (n,), "D must be length n"
    assert (c > 0).all(), "capacities must be positive"
    assert (D >= 0).all(), "demands must be non‑negative"

    # ---------- variables ----------
    x = cp.Variable((n, m))          # flow allocation
    q = cp.sum(x, axis=0)            # provider loads

    # ---------- objective ----------
    obj = (
        cp.sum(cp.multiply(w_p * p, q)) +
        0.5 * w_q * cp.sum(cp.multiply(1 / c, cp.square(q))) +
        w_d * cp.sum(cp.multiply(d, x))
    )

    obj += 0.5 * w_q * cp.sum(
    cp.multiply(1 / c,          # (m,)
                cp.sum(cp.square(x), axis=0)  # ∑_i x_ij^2  →  shape (m,)
               )
    )

    # ---------- constraints ----------
    constraints = [
        cp.sum(x, axis=1) == D,      # fulfill each user's demand
        x >= 0
    ]

    # ---------- solve ----------
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(solver=solver, **solver_opts)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"QP not solved to optimality (status: {prob.status})")

    # ---------- post‑compute total cost (incl. w0) ----------
    # x_star = x.value
    # q_star = q.value
    # C_ij = w0 + w_p * p + w_d * d + w_q * (q_star / c)  # broadcast
    # total_cost = np.sum(x_star * C_ij)

    # ---------- post-processing ----------
    x_star = x.value
    q_star = q.value
    C_ij   = (w0 + w_p * p + w_d * d + w_q * (q_star / c))  # broadcast
    cost_by_user = np.sum(x_star * C_ij, axis=1)            # (n,)
    total_cost   = float(cost_by_user.sum())

    return x_star, q_star, total_cost, prob.status

def cal_rev_solve(price_list, user_num, provider_num, config_json):
    eq_strategy, q_star, total_cost, solve_status = solve_weighted_linear_congestion_eq(p=price_list,
                                                        c=config_json["provider_cap"][:provider_num],
                                                        d=np.array(config_json["dist"])[:user_num, :provider_num],
                                                        D=config_json["user_demand"][:user_num],
                                                        w_p=config_json["wp"],
                                                        w_d=config_json["wd"],
                                                        w_q=config_json["wqf"],
                                                        w0=0.0)
    # formatted_str = np.array_str(eq_strategy, precision=2, suppress_small=True)
    # print(f"eq_strategy: {formatted_str}")
    # print(f"solve_status: {solve_status}")

    provider_rev = [0.0] * provider_num
    for user_i_strategy in eq_strategy:
        for provider_i in range(provider_num):
            provider_rev[provider_i] += user_i_strategy[provider_i] * (price_list[provider_i]- config_json["cost"][provider_i])
            # provider_rev[provider_i] += user_i_strategy[provider_i] * price_list[provider_i]

            
    return provider_rev

# --- 以下是反向求解函数 ---
def find_price_for_flow(x_star, c, d,
                        w_p=0.6, w_d=0.3, w_q=0.1, w0=0.0,
                        solver="OSQP", **solver_opts):
    """
    Given a target equilibrium flow x_star, find a feasible price vector p.
    This is the inverse of solve_weighted_linear_congestion_eq.

    Parameters
    ----------
    x_star : (n,m) ndarray
        The target equilibrium flow allocation.
    c : (m,) ndarray
        Capacities of service providers (positive).
    d : (n,m) ndarray
        Distance/latency matrix.
    w_p, w_d, w_q, w0 : float
        Weights for price, distance, congestion, and constant bias.
    solver : str, optional
        CVXPY solver name (default "OSQP").
    **solver_opts : dict
        Extra options forwarded to prob.solve(...).

    Returns
    -------
    p_found : (m,) ndarray or None
        A feasible price vector that could produce x_star. Returns None if no
        such non-negative price vector exists.
    lambda_found : (n,) ndarray or None
        The corresponding equilibrium costs for each user.
    status : str
        Solver status.
    """
    # ---------- Basic checks and pre-computation ----------
    x_star = np.asarray(x_star, dtype=float)
    c = np.asarray(c, dtype=float).ravel()
    d = np.asarray(d, dtype=float)

    n, m = x_star.shape
    assert c.shape == (m,), "c must be length m"
    assert d.shape == (n, m), "d must have shape (n, m)"
    assert w_p > 0, "Price weight w_p must be positive to have an effect."

    # Calculate the total load q* from the given flow x*
    q_star = np.sum(x_star, axis=0)

    # ---------- CVXPY Problem Setup ----------
    # Define the variables we want to find
    p = cp.Variable(m, name="prices")
    lambda_i = cp.Variable(n, name="user_equilibrium_costs")

    constraints = []

    # Known part of the cost (everything except the price component)
    # C_ij = w0 + w_p * p_j + known_cost_ij
    known_cost = w0 + w_d * d + w_q * (q_star / c) # Uses broadcasting

    # Add equilibrium constraints for each user and provider
    for i in range(n):
        for j in range(m):
            cost_ij = w_p * p[j] + known_cost[i, j]

            # Use a small tolerance for floating point comparisons
            if x_star[i, j] > 1e-6:
                # If flow is used, cost must equal the user's equilibrium cost
                constraints.append(cost_ij == lambda_i[i])
            else:
                # If flow is not used, cost must be greater than or equal
                constraints.append(cost_ij >= lambda_i[i]+ 0.5)

    # Add a practical constraint that prices cannot be negative
    constraints.append(p >= 0)

    # ---------- Objective Function ----------
    # We want a feasible solution with the smallest possible prices.
    # This makes the solution unique and economically sensible.
    objective = cp.Minimize(cp.sum(p))

    # ---------- Solve the LP ----------
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=solver, **solver_opts)

    if prob.status in ("optimal", "optimal_inaccurate"):
        return p.value, lambda_i.value, prob.status
    else:
        # If the problem is infeasible, no such non-negative price vector exists
        return None, None, prob.status


def find_w0_for_flow(x_star, c, d, p,
                     w_p=1.0, w_d=1.0, w_q=1.0,
                     solver="OSQP", **solver_opts):
    """
    给定一个目标均衡流量 x_star 和一个价格向量 p，寻找一个可行的 w0 向量。
    w0 是一个长度为 m 的向量，代表每个服务提供商的内在质量/偏好。

    This is the inverse problem to find a feasible quality/bias vector w0.

    Parameters
    ----------
    x_star : (n,m) ndarray
        目标均衡流量分配。
    c : (m,) ndarray
        服务提供商的容量（正数）。
    d : (n,m) ndarray
        距离/延迟矩阵。
    p : (m,) ndarray
        给定的价格向量。
    w_p, w_d, w_q : float
        价格、距离和拥塞的权重。
    solver : str, optional
        CVXPY 求解器名称 (默认为 "OSQP")。
    **solver_opts : dict
        传递给 prob.solve(...) 的额外选项。

    Returns
    -------
    w0_found : (m,) ndarray or None
        一个可行的 w0 向量，可以产生 x_star。如果不存在这样的非负 w0 向量，则返回 None。
    lambda_found : (n,) ndarray or None
        每个用户对应的均衡成本。
    status : str
        求解器状态。
    """
    # ---------- 基本检查和预计算 ----------
    x_star = np.asarray(x_star, dtype=float)
    c = np.asarray(c, dtype=float).ravel()
    d = np.asarray(d, dtype=float)
    p = np.asarray(p, dtype=float).ravel()

    n, m = x_star.shape
    assert c.shape == (m,), "c 必须是长度为 m 的向量"
    assert d.shape == (n, m), "d 的形状必须是 (n, m)"
    assert p.shape == (m,), "p 必须是长度为 m 的向量"

    # 根据给定的流量 x* 计算总负载 q*
    q_star = np.sum(x_star, axis=0)

    # ---------- CVXPY 问题设置 ----------
    # 定义我们想要寻找的变量
    w0 = cp.Variable(m, name="quality_bias_vector")
    lambda_i = cp.Variable(n, name="user_equilibrium_costs")

    constraints = []

    # 成本的已知部分 (除了 w0 之外的所有部分)
    # C_ij = w0_j + known_cost_ij
    known_cost = w_p * p + w_d * d + w_q * (q_star / c) # 利用广播机制

    # 为每个用户和服务提供商添加均衡约束
    for i in range(n):
        for j in range(m):
            # 成本函数现在包含向量 w0 的一个元素
            cost_ij = w0[j] + known_cost[i, j]

            # 使用一个小的容差进行浮点数比较
            if x_star[i, j] > 1e-6:
                # 如果流量被使用，成本必须等于用户的均衡成本
                constraints.append(cost_ij == lambda_i[i])
            else:
                # 如果流量未使用，成本必须大于或等于均衡成本
                # 增加一个小的正数epsilon可以提高数值稳定性
                constraints.append(cost_ij >= lambda_i[i] + 1e-7)

    # 添加一个实际约束，即“质量”因子不能为负
    constraints.append(w0 >= 0)

    # ---------- 目标函数 ----------
    # 我们想要一个可行的解，并且这个解的 w0 总和最小。
    # 这使得解唯一且在经济学上具有意义（寻找最小的必要质量/偏好）。
    objective = cp.Minimize(cp.sum(w0))

    # ---------- 求解 LP ----------
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=solver, **solver_opts)

    if prob.status in ("optimal", "optimal_inaccurate"):
        return w0.value, lambda_i.value, prob.status
    else:
        # 如果问题不可行，则不存在这样的非负 w0 向量
        return None, None, prob.status





# --------------- quick demo -----------------
# if __name__ == "__main__":
#     np.set_printoptions(suppress=True)
#     # small toy instance
#     p = [1.2, 1.5, 1.1]
#     c = [50., 40., 60.]
#     d = np.random.rand(5, 3) * 0.2
#     D = [1.0, 0.8, 0.6, 1.2, 0.7]

#     # example weights
#     w_p, w_d, w_q, w0 = 0.6, 0.2, 0.2, 0.0

#     x_eq, q_eq, tot_cost, status = solve_weighted_linear_congestion_eq(
#         p, c, d, D, w_p, w_d, w_q, w0)
#     print("Status :", status)
#     print("Equilibrium x*\n", x_eq)
#     print("Loads q*\n", q_eq)
#     print("Total cost :", tot_cost)
