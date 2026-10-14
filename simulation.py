#А.1. Задание параметров модели и выбор ближайшего события

import numpy as np
from math import gamma


def make_params(  # функция создания словаря с параметрами моделирования
    s,
    b,
    N=None,
    lambda1=0.02,
    lambda2=0.01,
    d2=1.0,
    d3=1.0,
    d4=0.5,
    d5=2.0,
    c2=100.0,
    c3=200.0,
    c4=50.0,
    c5=100.0,
    tie_break="big",
    failure_dist="exp",
    beta1=2.0,
    beta2=2.0,
):
    if s <= 0 or b <= 0:
        raise ValueError("Параметры s и b должны быть положительными.")
    if lambda1 < 0 or lambda2 < 0:
        raise ValueError("Интенсивности отказов должны быть неотрицательными.")
    if d2 < 0 or d3 < 0 or d4 < 0 or d5 < 0:
        raise ValueError("Длительности работ должны быть неотрицательными.")
    if tie_break not in ["big", "small"]:
        raise ValueError("tie_break должен быть 'big' или 'small'.")
    if N is not None:
        if N <= 0 or int(N) != N:
            raise ValueError("N должно быть положительным целым числом.")
        N = int(N)
        if abs(b - N * s) > 1e-9:
            raise ValueError("Для режима со слоями должно выполняться b = N*s.")

    return {
        "s": s,
        "b": b,
        "N": N,
        "lambda1": lambda1,
        "lambda2": lambda2,
        "d2": d2,
        "d3": d3,
        "d4": d4,
        "d5": d5,
        "c2": c2,
        "c3": c3,
        "c4": c4,
        "c5": c5,
        "tie_break": tie_break,
        "failure_dist": failure_dist,
        "beta1": beta1,
        "beta2": beta2,
    }

def exp_sample(rng, rate): # функция генерации экспоненциальной величины с интенсивностью rate
    if rate < 0:
        raise ValueError("Интенсивность должна быть неотрицательной.")
    if rate == 0:
        return np.inf
    return rng.exponential(1.0 / rate)

def weibull_residual_sample(rng, age, lambda_ref, beta):
    if lambda_ref <= 0:
        return np.inf

    mean_time = 1.0 / lambda_ref
    scale = mean_time / gamma(1.0 + 1.0 / beta)

    z = rng.exponential(1.0)
    total_life = scale * ((age / scale) ** beta + z) ** (1.0 / beta)

    return total_life - age

def failure_sample(rng, age, lambda_ref, dist="exp", beta=2.0):
    if dist == "exp":
        return exp_sample(rng, lambda_ref)

    if dist == "weibull":
        return weibull_residual_sample(rng, age, lambda_ref, beta)

    raise ValueError("dist должен быть 'exp' или 'weibull'.")

def choose_next_event(x1, x2, eta_s, eta_b, tie_break="big", eps=1e-12): # функция выбора следующего события
    events = [
        ("fail1", x1),
        ("fail2", x2),
        ("small_pm", eta_s),
        ("big_pm", eta_b),
    ]
    if tie_break == "big":
        priority = {"big_pm": 0, "small_pm": 1, "fail1": 2, "fail2": 3}
    else:
        priority = {"small_pm": 0, "big_pm": 1, "fail1": 2, "fail2": 3}

    t_min = min(t for _, t in events)
    near = [(name, t) for name, t in events if abs(t - t_min) <= eps]
    near.sort(key=lambda pair: priority[pair[0]])

    return near[0][0], t_min

#А.2. Класс имитационной модели

class Simulator:
    def __init__(self, params, seed=42):
        self.params = params
        self.rng = np.random.default_rng(seed)
        self.use_layers = (params["N"] is not None)
        self.reset()

    def reset(self):
        if self.use_layers:
            N = self.params["N"]
            self.layer_stats = {
                "A_k": np.zeros(N, dtype=np.int64),
                "work_u": [[] for _ in range(N)],
                "rep2_u": [[] for _ in range(N)],
                "rep3_u": [[] for _ in range(N)],
                "small_count": np.zeros(max(N - 1, 0), dtype=np.int64),
                "big_count": 0,
                }
        self.total_kept = 0
        self.kept_time = 0.0
        self.kept_work_time = 0.0
        self.kept_cost = 0.0
        self.current_state = 1
        self.current_u = 0.0
        self.current_v = 0.0
        self.current_k = 0

    def _run_step_layers(self):
        dt_total = 0.0
        dt_work = 0.0
        dcost = 0.0
        p = self.params
        if self.current_state == 1:
            k = self.current_k
            u = self.current_u
            s = p["s"]
            v = u + k * s
            eta_s = s - u
            eta_b = p["b"] - v

            if eta_s < 0:
                eta_s = 0.0
            if eta_b < 0:
                eta_b = 0.0

            x1 = failure_sample(
   self.rng,
   age=u,
   lambda_ref=p["lambda1"],
   dist=p["failure_dist"],
   beta=p["beta1"],
            )

            x2 = failure_sample(
   self.rng,
   age=v,
   lambda_ref=p["lambda2"],
   dist=p["failure_dist"],
   beta=p["beta2"],
             )
            event, dt = choose_next_event(x1, x2, eta_s, eta_b, p["tie_break"])
            dt_total += dt
            dt_work += dt

            if event == "fail1":
                dcost += p["c2"]
                next_state = 2
            elif event == "fail2":
                dcost += p["c3"]
                next_state = 3
            elif event == "small_pm":
                dcost += p["c4"]
                next_state = 4
            else:
                dcost += p["c5"]
                next_state = 5
            self.current_u = u + dt
            self.current_state = next_state

        elif self.current_state == 2:
            dt_total += p["d2"]
            self.current_state = 1
        elif self.current_state == 3:
            dt_total += p["d3"]
            self.current_state = 1
        elif self.current_state == 4:
            dt_total += p["d4"]
            self.current_k += 1
            self.current_u = 0.0
            self.current_state = 1
        elif self.current_state == 5:
            dt_total += p["d5"]
            self.current_k = 0
            self.current_u = 0.0
            self.current_state = 1

        return dt_total, dt_work, dcost

    def _run_step_general(self):
        dt_total = 0.0
        dt_work = 0.0
        dcost = 0.0
        p = self.params

        if self.current_state == 1:
            u = self.current_u
            v = self.current_v
            eta_s = p["s"] - u
            eta_b = p["b"] - v

            if eta_s < 0:
                eta_s = 0.0
            if eta_b < 0:
                eta_b = 0.0

            x1 = failure_sample(
   self.rng,
   age=u,
   lambda_ref=p["lambda1"],
   dist=p["failure_dist"],
   beta=p["beta1"],
            )

            x2 = failure_sample(
   self.rng,
   age=v,
   lambda_ref=p["lambda2"],
   dist=p["failure_dist"],
   beta=p["beta2"],
             )
            event, dt = choose_next_event(x1, x2, eta_s, eta_b, p["tie_break"])
            dt_total += dt
            dt_work += dt

            if event == "fail1":
                dcost += p["c2"]
                next_state = 2
            elif event == "fail2":
                dcost += p["c3"]
                next_state = 3
            elif event == "small_pm":
                dcost += p["c4"]
                next_state = 4
            else:
                dcost += p["c5"]
                next_state = 5

            self.current_u = u + dt
            self.current_v = v + dt
            self.current_state = next_state

        elif self.current_state == 2:
            dt_total += p["d2"]
            self.current_state = 1
        elif self.current_state == 3:
            dt_total += p["d3"]
            self.current_state = 1
        elif self.current_state == 4:
            dt_total += p["d4"]
            self.current_u = 0.0
            self.current_state = 1
        elif self.current_state == 5:
            dt_total += p["d5"]
            self.current_u = 0.0
            self.current_v = 0.0
            self.current_state = 1

        return dt_total, dt_work, dcost

    def _collect_stat_layers(self):
        if self.current_state == 1:
            k = self.current_k
            u = self.current_u
            if abs(u) <= 1e-15:
                self.layer_stats["A_k"][k] += 1
            else:
                self.layer_stats["work_u"][k].append(u)

        elif self.current_state == 2:
            self.layer_stats["rep2_u"][self.current_k].append(self.current_u)
        elif self.current_state == 3:
            self.layer_stats["rep3_u"][self.current_k].append(self.current_u)
        elif self.current_state == 4:
            if self.current_k < len(self.layer_stats["small_count"]):
                self.layer_stats["small_count"][self.current_k] += 1
        elif self.current_state == 5:
            self.layer_stats["big_count"] += 1

    def run(self, n_steps, burn_in=20000):
        self.reset()

        for step in range(n_steps):
            if self.use_layers:
                dt_total, dt_work, dcost = self._run_step_layers()
            else:
                dt_total, dt_work, dcost = self._run_step_general()

            if step >= burn_in:
                self.total_kept += 1
                self.kept_time += dt_total
                self.kept_work_time += dt_work
                self.kept_cost += dcost

                if self.use_layers:
                    self._collect_stat_layers()

    def get_results(self, bins=50):
        if self.use_layers:
            return self._get_results_layers(bins)
        return self._get_results_general()

    def _get_results_layers(self, bins):
        total = self.total_kept
        stats = self.layer_stats
        N = self.params["N"]
        s = self.params["s"]

        if total == 0:
            raise RuntimeError("Нет собранной статистики.")

        A_k_emp = stats["A_k"] / total

        work_densities = []
        rep2_densities = []
        rep3_densities = []

        for k in range(N):
            data_work = np.array(stats["work_u"][k], dtype=float)
            if len(data_work) == 0:
                work_densities.append(None)
            else:
                hist, edges = np.histogram(data_work, bins=bins, range=(0.0, s), density=False)
                width = edges[1] - edges[0]
                dens = hist / (total * width)
                work_densities.append((edges, dens))

            data_rep2 = np.array(stats["rep2_u"][k], dtype=float)
            if len(data_rep2) == 0:
                rep2_densities.append(None)
            else:
                hist, edges = np.histogram(data_rep2, bins=bins, range=(0.0, s), density=False)
                width = edges[1] - edges[0]
                dens = hist / (total * width)
                rep2_densities.append((edges, dens))

            data_rep3 = np.array(stats["rep3_u"][k], dtype=float)
            if len(data_rep3) == 0:
                rep3_densities.append(None)
            else:
                hist, edges = np.histogram(data_rep3, bins=bins, range=(0.0, s), density=False)
                width = edges[1] - edges[0]
                dens = hist / (total * width)
                rep3_densities.append((edges, dens))

        P4_k_emp = stats["small_count"] / total if N > 1 else np.array([], dtype=float)
        P5_emp = stats["big_count"] / total

        lam1 = self.params["lambda1"]
        lam2 = self.params["lambda2"]
        lam = lam1 + lam2

        A0_theory = 1.0 / (2.0 * N * (1.0 + lam * s))
        A_k_theory = np.full(N, A0_theory)

        g1_theory = A0_theory * lam
        g2_theory = A0_theory * lam1
        g3_theory = A0_theory * lam2

        P4_theory = A0_theory
        P5_theory = A0_theory

        K_g_emp = self.kept_work_time / self.kept_time
        C_emp = self.kept_cost / self.kept_time

        d2 = self.params["d2"]
        d3 = self.params["d3"]
        d4 = self.params["d4"]
        d5 = self.params["d5"]
        c2 = self.params["c2"]
        c3 = self.params["c3"]
        c4 = self.params["c4"]
        c5 = self.params["c5"]

        E_Tc = N * s + N * lam1 * s * d2 + N * lam2 * s * d3 + (N - 1) * d4 + d5
        K_g_theory = (N * s) / E_Tc

        E_Cc = N * c2 * lam1 * s + N * c3 * lam2 * s + (N - 1) * c4 + c5
        C_theory = E_Cc / E_Tc

        return {
            "mode": "layers",
            "total": total,
            "A_k_emp": A_k_emp,
            "A_k_theory": A_k_theory,
            "work_densities": work_densities,
            "rep2_densities": rep2_densities,
            "rep3_densities": rep3_densities,
            "g1_theory": g1_theory,
            "g2_theory": g2_theory,
            "g3_theory": g3_theory,
            "P4_k_emp": P4_k_emp,
            "P4_theory": P4_theory,
            "P5_emp": P5_emp,
            "P5_theory": P5_theory,
            "K_g_emp": K_g_emp,
            "K_g_theory": K_g_theory,
            "C_emp": C_emp,
            "C_theory": C_theory,
            "kept_time": self.kept_time,
            "kept_work_time": self.kept_work_time,
            "kept_cost": self.kept_cost,
            "failure_dist": self.params["failure_dist"],
        }

    def _get_results_general(self):
        total = self.total_kept
        if total == 0:
            raise RuntimeError("Нет собранной статистики.")

        K_g_emp = self.kept_work_time / self.kept_time
        C_emp = self.kept_cost / self.kept_time

        return {
            "mode": "general",
            "total": total,
            "K_g_emp": K_g_emp,
            "C_emp": C_emp,
            "kept_time": self.kept_time,
            "kept_work_time": self.kept_work_time,
            "kept_cost": self.kept_cost,
            "failure_dist": self.params["failure_dist"],
        }
