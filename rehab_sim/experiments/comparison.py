"""Five-method, three-patient Phase 10 comparison runner.

The experiment operates on the existing MuJoCo/Gymnasium environment. Every
method returns the same normalized action ``[dDxy,dDtheta,dKa,dlambda_v]``;
none of the comparison policies can issue motor torque or current commands.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rehab_sim.experiments.config import VALID_METHODS, Phase10Config
from rehab_sim.rl.config import SACExperimentConfig, load_sac_config
from rehab_sim.rl.vector_env import environment_class, make_normalized_env

FloatArray = NDArray[np.float64]
Observation = dict[str, np.ndarray]
ActionPolicy = Callable[[Observation], FloatArray]


@dataclass(frozen=True)
class EpisodeMetrics:
    """One reproducible episode result in SI-compatible summary fields."""

    method: str
    patient_profile: str
    task_name: str
    seed: int
    episode: int
    reward: float
    success: bool
    unsafe: bool
    length: int
    mean_tracking_error: float
    max_interaction_force: float
    mean_interaction_force: float
    patient_active_power_mean: float
    patient_active_work_ratio_proxy: float
    robot_assistance_energy: float
    parameter_change_total: float
    parameter_oscillation_rate: float
    final_fatigue: float

    def as_dict(self) -> dict[str, str | int | float | bool]:
        """Return a CSV/JSON-friendly row."""

        return {
            "method": self.method,
            "patient_profile": self.patient_profile,
            "task_name": self.task_name,
            "seed": self.seed,
            "episode": self.episode,
            "reward": self.reward,
            "success": self.success,
            "unsafe": self.unsafe,
            "length": self.length,
            "mean_tracking_error": self.mean_tracking_error,
            "max_interaction_force": self.max_interaction_force,
            "mean_interaction_force": self.mean_interaction_force,
            "patient_active_power_mean": self.patient_active_power_mean,
            "patient_active_work_ratio_proxy": self.patient_active_work_ratio_proxy,
            "robot_assistance_energy": self.robot_assistance_energy,
            "parameter_change_total": self.parameter_change_total,
            "parameter_oscillation_rate": self.parameter_oscillation_rate,
            "final_fatigue": self.final_fatigue,
        }


class _EpisodeAccumulator:
    """Accumulate raw Gymnasium info without changing the environment."""

    def __init__(self) -> None:
        self.reward = 0.0
        self.length = 0
        self.error_sum = 0.0
        self.force_sum = 0.0
        self.max_force = 0.0
        self.power_sum = 0.0
        self.assistance_sum = 0.0
        self.parameter_change_total = 0.0
        self.parameter_deltas: list[FloatArray] = []
        self.previous_parameters: FloatArray | None = None
        self.success = False
        self.unsafe = False

    def add(self, reward: float, info: Mapping[str, Any]) -> None:
        self.reward += float(reward)
        self.length += 1
        error = float(info.get("tracking_error_norm", 0.0))
        force = float(info.get("interaction_force_norm", 0.0))
        self.error_sum += error
        self.force_sum += force
        self.max_force = max(self.max_force, force)
        self.power_sum += float(info.get("human_power_w", 0.0))
        components = info.get("reward_components", {})
        if isinstance(components, Mapping):
            self.assistance_sum += float(components.get("robot_assistance_energy", 0.0))
        self.parameter_change_total += float(info.get("parameter_change_norm", 0.0))
        parameters_value = info.get("admittance_parameters")
        if parameters_value is not None:
            parameters = np.asarray(parameters_value, dtype=np.float64)
            if parameters.shape == (5,):
                if self.previous_parameters is not None:
                    self.parameter_deltas.append(parameters - self.previous_parameters)
                self.previous_parameters = parameters.copy()
        self.success = bool(info.get("is_success", self.success))
        self.unsafe = bool(info.get("unsafe_reason") is not None or self.unsafe)

    def finish(self, fatigue: float) -> tuple[float, float]:
        signs = 0
        for previous, current in zip(
            self.parameter_deltas, self.parameter_deltas[1:], strict=False
        ):
            previous_sign = np.sign(previous)
            current_sign = np.sign(current)
            signs += int(
                np.any(
                    (previous_sign != 0.0) & (current_sign != 0.0) & (previous_sign != current_sign)
                )
            )
        transitions = max(1, len(self.parameter_deltas) - 1)
        oscillation = signs / transitions
        active_mean = self.power_sum / max(1, self.length)
        assistance_mean = self.assistance_sum / max(1, self.length)
        ratio = active_mean / max(active_mean + assistance_mean, 1.0e-12)
        return float(oscillation), float(ratio)


def _clip_action(values: list[float]) -> FloatArray:
    return np.clip(np.asarray(values, dtype=np.float64), -1.0, 1.0)


def _rule_action(observation: Observation, parameters: Mapping[str, float]) -> FloatArray:
    error = observation["tracking_error"]
    force_norm = float(np.linalg.norm(observation["interaction_force"][:2]))
    error_norm = float(np.linalg.norm(error[:2]))
    speed_norm = float(np.linalg.norm(observation["end_effector_velocity"]))
    force_delta = force_norm - parameters["force_reference_n"]
    return _clip_action(
        [
            parameters["damping_gain"] * force_delta,
            parameters["theta_gain"] * abs(float(error[2])),
            parameters["assist_gain"] * error_norm,
            -parameters["velocity_gain"] * speed_norm,
        ]
    )


def _ramp(value: float, low: float, high: float) -> float:
    if high <= low:
        raise ValueError("fuzzy membership high threshold must exceed low threshold")
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def _fuzzy_action(observation: Observation, parameters: Mapping[str, float]) -> FloatArray:
    error = observation["tracking_error"]
    force_norm = float(np.linalg.norm(observation["interaction_force"][:2]))
    error_norm = float(np.linalg.norm(error[:2]))
    speed_norm = float(np.linalg.norm(observation["end_effector_velocity"]))
    force_high = _ramp(force_norm, parameters["force_low_n"], parameters["force_high_n"])
    error_high = _ramp(error_norm, parameters["error_low_m"], parameters["error_high_m"])
    speed_high = _ramp(speed_norm, parameters["speed_low_mps"], parameters["speed_high_mps"])
    damping = (
        parameters["damping_force_weight"] * force_high
        + parameters["damping_speed_weight"] * speed_high
    )
    assist = parameters["assist_error_weight"] * error_high * (1.0 - 0.5 * force_high)
    velocity = -parameters["velocity_speed_weight"] * speed_high
    return _clip_action([damping, 0.15 * abs(float(error[2])), assist, velocity])


def classical_policy(
    method: str, policy_parameters: Mapping[str, Mapping[str, float]]
) -> ActionPolicy:
    """Return a deterministic fixed/rule/fuzzy action policy."""

    if method == "fixed_admittance":
        return lambda _observation: np.zeros(4, dtype=np.float64)
    if method == "rule_adaptive":
        parameters = policy_parameters[method]
        return lambda observation: _rule_action(observation, parameters)
    if method == "fuzzy_control":
        parameters = policy_parameters[method]
        return lambda observation: _fuzzy_action(observation, parameters)
    raise ValueError(f"{method} is not a classical comparison method")


def _environment_fatigue(environment: Any) -> float:
    patient = getattr(environment, "_patient", None)
    return float(getattr(patient, "fatigue", 0.0))


def evaluate_classical_episode(
    *,
    method: str,
    task_name: str,
    patient_profile: str,
    seed: int,
    episode: int,
    policy_parameters: Mapping[str, Mapping[str, float]],
) -> EpisodeMetrics:
    """Evaluate one classical method on one deterministic episode."""

    environment_class_type = environment_class(task_name)
    environment = environment_class_type(patient_profile=patient_profile, seed=seed)
    policy = classical_policy(method, policy_parameters)
    accumulator = _EpisodeAccumulator()
    try:
        observation, _ = environment.reset(seed=seed)
        while True:
            observation, reward, terminated, truncated, info = environment.step(policy(observation))
            accumulator.add(float(reward), info)
            if terminated or truncated:
                break
        fatigue = _environment_fatigue(environment)
    finally:
        environment.close()
    oscillation, active_ratio = accumulator.finish(fatigue)
    return EpisodeMetrics(
        method=method,
        patient_profile=patient_profile,
        task_name=task_name,
        seed=seed,
        episode=episode,
        reward=accumulator.reward,
        success=accumulator.success,
        unsafe=accumulator.unsafe,
        length=accumulator.length,
        mean_tracking_error=accumulator.error_sum / max(1, accumulator.length),
        max_interaction_force=accumulator.max_force,
        mean_interaction_force=accumulator.force_sum / max(1, accumulator.length),
        patient_active_power_mean=accumulator.power_sum / max(1, accumulator.length),
        patient_active_work_ratio_proxy=active_ratio,
        robot_assistance_energy=accumulator.assistance_sum,
        parameter_change_total=accumulator.parameter_change_total,
        parameter_oscillation_rate=oscillation,
        final_fatigue=fatigue,
    )


@dataclass
class _RLArtifact:
    """Trained model and normalization state retained during one comparison."""

    method: str
    model: Any
    training_env: Any
    model_path: Path
    normalization_path: Path

    def close(self) -> None:
        self.training_env.close()


def train_rl_artifact(
    *,
    method: str,
    sac_config: SACExperimentConfig,
    training_patient: str,
    task_name: str,
    seed: int,
    timesteps: int,
    device: str,
    output_dir: Path,
) -> _RLArtifact:
    """Train a small reproducible SAC/PPO artifact for the comparison."""

    if method not in ("sac", "ppo"):
        raise ValueError("train_rl_artifact only supports sac and ppo")
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.utils import set_random_seed

    train_config = replace(
        sac_config,
        task_name=task_name,
        patient_profile=training_patient,
        n_envs=1,
        device=device,
    )
    set_random_seed(seed)
    training_env = make_normalized_env(train_config, seed=seed, n_envs=1, training=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{method}_seed_{seed:04d}"
    normalization_path = output_dir / f"{method}_seed_{seed:04d}_vecnormalize.pkl"
    if method == "sac":
        model = SAC(
            "MultiInputPolicy",
            training_env,
            learning_rate=train_config.learning_rate,
            buffer_size=max(train_config.buffer_size, max(64, timesteps)),
            learning_starts=min(train_config.learning_starts, max(1, timesteps // 4)),
            batch_size=min(train_config.batch_size, max(2, timesteps // 4)),
            tau=train_config.tau,
            gamma=train_config.gamma,
            train_freq=train_config.train_frequency,
            gradient_steps=train_config.gradient_steps,
            ent_coef=train_config.ent_coef,
            policy_kwargs={"net_arch": list(train_config.policy_net_arch)},
            device=device,
            seed=seed,
            verbose=0,
        )
    else:
        n_steps = max(2, min(128, max(2, timesteps)))
        model = PPO(
            "MultiInputPolicy",
            training_env,
            learning_rate=train_config.learning_rate,
            n_steps=n_steps,
            batch_size=min(64, n_steps),
            n_epochs=2,
            gamma=train_config.gamma,
            policy_kwargs={"net_arch": list(train_config.policy_net_arch)},
            device=device,
            seed=seed,
            verbose=0,
        )
    if timesteps > 0:
        model.learn(total_timesteps=timesteps, progress_bar=False)
    model.save(str(model_path))
    training_env.save(str(normalization_path))
    return _RLArtifact(method, model, training_env, model_path, normalization_path)


def _evaluate_rl_episode(
    *,
    artifact: _RLArtifact,
    sac_config: SACExperimentConfig,
    task_name: str,
    patient_profile: str,
    seed: int,
    episode: int,
) -> EpisodeMetrics:
    """Evaluate one deterministic SB3 episode with training normalization."""

    import copy

    eval_config = replace(
        sac_config,
        task_name=task_name,
        patient_profile=patient_profile,
        n_envs=1,
    )
    evaluation_env = make_normalized_env(eval_config, seed=seed, n_envs=1, training=False)
    evaluation_env.obs_rms = copy.deepcopy(artifact.training_env.obs_rms)
    evaluation_env.ret_rms = copy.deepcopy(artifact.training_env.ret_rms)
    evaluation_env.training = False
    evaluation_env.norm_reward = False
    accumulator = _EpisodeAccumulator()
    try:
        base_environment = evaluation_env.venv.envs[0]
        base_environment.reset(seed=seed)
        observation = evaluation_env.reset()
        while True:
            action, _ = artifact.model.predict(observation, deterministic=True)
            observation, rewards, dones, infos = evaluation_env.step(action)
            info = dict(infos[0])
            accumulator.add(float(rewards[0]), info)
            if bool(dones[0]):
                break
        fatigue = _environment_fatigue(base_environment.unwrapped)
    finally:
        evaluation_env.close()
    oscillation, active_ratio = accumulator.finish(fatigue)
    return EpisodeMetrics(
        method=artifact.method,
        patient_profile=patient_profile,
        task_name=task_name,
        seed=seed,
        episode=episode,
        reward=accumulator.reward,
        success=accumulator.success,
        unsafe=accumulator.unsafe,
        length=accumulator.length,
        mean_tracking_error=accumulator.error_sum / max(1, accumulator.length),
        max_interaction_force=accumulator.max_force,
        mean_interaction_force=accumulator.force_sum / max(1, accumulator.length),
        patient_active_power_mean=accumulator.power_sum / max(1, accumulator.length),
        patient_active_work_ratio_proxy=active_ratio,
        robot_assistance_energy=accumulator.assistance_sum,
        parameter_change_total=accumulator.parameter_change_total,
        parameter_oscillation_rate=oscillation,
        final_fatigue=fatigue,
    )


def _config_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted((root / "configs").glob("*.yaml")):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _git_commit(root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def _metric_values(values: list[EpisodeMetrics], name: str) -> np.ndarray:
    """Extract one numeric metric from a group of episode rows."""

    return np.asarray([float(getattr(item, name)) for item in values], dtype=np.float64)


def aggregate_rows(rows: list[EpisodeMetrics]) -> list[dict[str, str | int | float]]:
    """Aggregate episode rows by method and patient profile."""

    groups: defaultdict[tuple[str, str], list[EpisodeMetrics]] = defaultdict(list)
    for row in rows:
        groups[(row.method, row.patient_profile)].append(row)
    summary: list[dict[str, str | int | float]] = []
    for (method, patient), values in groups.items():
        reward = _metric_values(values, "reward")
        success = _metric_values(values, "success")
        unsafe = _metric_values(values, "unsafe")
        summary.append(
            {
                "method": method,
                "patient_profile": patient,
                "episodes": len(values),
                "reward_mean": float(np.mean(reward)),
                "reward_std": float(np.std(reward)),
                "success_rate": float(np.mean(success)),
                "unsafe_rate": float(np.mean(unsafe)),
                "mean_tracking_error": float(
                    np.mean(_metric_values(values, "mean_tracking_error"))
                ),
                "max_interaction_force_mean": float(
                    np.mean(_metric_values(values, "max_interaction_force"))
                ),
                "mean_interaction_force": float(
                    np.mean(_metric_values(values, "mean_interaction_force"))
                ),
                "patient_active_power_mean": float(
                    np.mean(_metric_values(values, "patient_active_power_mean"))
                ),
                "patient_active_work_ratio_proxy": float(
                    np.mean(_metric_values(values, "patient_active_work_ratio_proxy"))
                ),
                "robot_assistance_energy": float(
                    np.mean(_metric_values(values, "robot_assistance_energy"))
                ),
                "parameter_change_total_mean": float(
                    np.mean(_metric_values(values, "parameter_change_total"))
                ),
                "parameter_oscillation_rate": float(
                    np.mean(_metric_values(values, "parameter_oscillation_rate"))
                ),
                "final_fatigue_mean": float(np.mean(_metric_values(values, "final_fatigue"))),
            }
        )
    return sorted(summary, key=lambda item: (str(item["method"]), str(item["patient_profile"])))


def write_experiment_outputs(
    *,
    rows: list[EpisodeMetrics],
    output_dir: Path,
    metadata: Mapping[str, Any],
    plot_dpi: int,
) -> list[dict[str, str | int | float]]:
    """Write raw data, aggregate tables, JSON metadata, charts and report."""

    output_dir.mkdir(parents=True, exist_ok=True)
    row_dicts = [row.as_dict() for row in rows]
    episode_path = output_dir / "episode_metrics.csv"
    with episode_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row_dicts[0]))
        writer.writeheader()
        writer.writerows(row_dicts)
    summary = aggregate_rows(rows)
    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    payload = {"metadata": dict(metadata), "conditions": summary}
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_plots(summary, output_dir, plot_dpi)
    _write_report(summary, metadata, output_dir / "phase10_report.md")
    return summary


def _write_plots(summary: list[dict[str, str | int | float]], output_dir: Path, dpi: int) -> None:
    """Create deterministic PNG plots from the aggregate table."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = list(dict.fromkeys(str(item["method"]) for item in summary))
    patients = list(dict.fromkeys(str(item["patient_profile"]) for item in summary))
    lookup = {(str(item["method"]), str(item["patient_profile"])): item for item in summary}
    x = np.arange(len(methods), dtype=np.float64)
    width = 0.8 / max(1, len(patients))

    fig, axis = plt.subplots(figsize=(10, 5))
    for index, patient in enumerate(patients):
        values = [float(lookup[(method, patient)]["success_rate"]) for method in methods]
        axis.bar(x + (index - (len(patients) - 1) / 2) * width, values, width, label=patient)
    axis.set_ylabel("Success rate")
    axis.set_ylim(0.0, 1.0)
    axis.set_title("Phase 10 success-rate comparison")
    axis.set_xticks(x, methods, rotation=20, ha="right")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "success_rate.png", dpi=dpi)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for index, patient in enumerate(patients):
        error_values = [
            float(lookup[(method, patient)]["mean_tracking_error"]) for method in methods
        ]
        force_values = [
            float(lookup[(method, patient)]["max_interaction_force_mean"]) for method in methods
        ]
        axes[0].bar(
            x + (index - (len(patients) - 1) / 2) * width,
            error_values,
            width,
            label=patient,
        )
        axes[1].bar(
            x + (index - (len(patients) - 1) / 2) * width,
            force_values,
            width,
            label=patient,
        )
    axes[0].set_title("Mean tracking error")
    axes[0].set_ylabel("m/rad norm")
    axes[1].set_title("Mean peak interaction force")
    axes[1].set_ylabel("N")
    for axis in axes:
        axis.set_xticks(x, methods, rotation=20, ha="right")
        axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "tracking_force_comparison.png", dpi=dpi)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 5))
    for patient in patients:
        values = [
            float(lookup[(method, patient)]["parameter_oscillation_rate"]) for method in methods
        ]
        axis.plot(methods, values, marker="o", label=patient)
    axis.set_title("Parameter oscillation rate")
    axis.set_ylabel("sign-transition ratio")
    axis.tick_params(axis="x", rotation=20)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "parameter_stability.png", dpi=dpi)
    plt.close(fig)


def _write_report(
    summary: list[dict[str, str | int | float]], metadata: Mapping[str, Any], path: Path
) -> None:
    lines = [
        "# Phase 10 系统实验报告",
        "",
        "本报告由 `scripts/run_phase10_experiments.py` 自动生成。"
        "所有策略输出均为低频导纳参数增量，未输出电机力矩或电流。",
        "",
        "## 可复现元数据",
        "",
        f"- Git commit：`{metadata.get('git_commit', 'unknown')}`",
        f"- 配置哈希：`{metadata.get('config_sha256', 'unknown')}`",
        f"- 随机种子：`{metadata.get('seeds', [])}`",
        f"- 任务：`{metadata.get('task_name', 'unknown')}`",
        "",
        "## 方法与患者条件汇总",
        "",
        "| 方法 | 患者 | episodes | reward mean ± std | success | mean error | "
        "peak force | oscillation |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary:
        method = str(item["method"])
        patient_profile = str(item["patient_profile"])
        episodes = int(item["episodes"])
        reward_mean = float(item["reward_mean"])
        reward_std = float(item["reward_std"])
        success_rate = float(item["success_rate"])
        mean_tracking_error = float(item["mean_tracking_error"])
        max_force = float(item["max_interaction_force_mean"])
        oscillation = float(item["parameter_oscillation_rate"])
        lines.append(
            f"| {method} | {patient_profile} | {episodes} | {reward_mean:.3f} ± "
            f"{reward_std:.3f} | {success_rate:.3f} | {mean_tracking_error:.4f} | "
            f"{max_force:.4f} | {oscillation:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 图表",
            "",
            "- `success_rate.png`：三类患者成功率；",
            "- `tracking_force_comparison.png`：跟踪误差和峰值交互力；",
            "- `parameter_stability.png`：导纳参数振荡率。",
            "",
            "## 解释与限制",
            "",
            "SAC/PPO 使用配置指定的训练患者进行短周期可复现实验，再在三类患者上确定性评估。"
            "真实硬件阈值、人体实验和统计显著性检验不由该仿真脚本替代；"
            "部署前必须重新完成硬件验证。",
            "患者主动做功占比是由仿真 `human_power_w` 与辅助能量分项构造的比较代理指标，"
            "不应直接解释为临床生理测量。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_comparison(
    *,
    root: Path,
    config: Phase10Config,
    output_dir: Path,
    methods: tuple[str, ...] | None = None,
    patients: tuple[str, ...] | None = None,
    seeds: tuple[int, ...] | None = None,
    episodes_per_condition: int | None = None,
    training_timesteps: int | None = None,
) -> tuple[list[EpisodeMetrics], list[dict[str, str | int | float]]]:
    """Run the configured comparison matrix and write all Phase 10 artifacts."""

    selected_methods = methods or config.methods
    selected_patients = patients or config.patient_profiles
    selected_seeds = seeds or config.seeds
    episodes = (
        config.episodes_per_condition if episodes_per_condition is None else episodes_per_condition
    )
    timesteps = config.training_timesteps if training_timesteps is None else training_timesteps
    if not selected_methods or not selected_patients or not selected_seeds:
        raise ValueError("methods, patients, and seeds must each contain at least one item")
    if episodes <= 0:
        raise ValueError("episodes_per_condition must be positive")
    if timesteps < 0:
        raise ValueError("training_timesteps must be non-negative")
    if any(method not in VALID_METHODS for method in selected_methods):
        raise ValueError("selected methods contain an unknown method")
    sac_path = config.rl_config if config.rl_config.is_absolute() else root / config.rl_config
    sac_config = load_sac_config(sac_path)
    rows: list[EpisodeMetrics] = []
    model_metadata: dict[str, list[str]] = {"sac": [], "ppo": []}
    for seed in selected_seeds:
        classical_methods = [
            method
            for method in selected_methods
            if method in ("fixed_admittance", "rule_adaptive", "fuzzy_control")
        ]
        for method in classical_methods:
            for patient_index, patient in enumerate(selected_patients):
                for episode in range(episodes):
                    episode_seed = seed + patient_index * 1000 + episode
                    rows.append(
                        evaluate_classical_episode(
                            method=method,
                            task_name=config.task_name,
                            patient_profile=patient,
                            seed=episode_seed,
                            episode=episode,
                            policy_parameters=config.policy_parameters,
                        )
                    )
        for method in ("sac", "ppo"):
            if method not in selected_methods:
                continue
            artifact = train_rl_artifact(
                method=method,
                sac_config=sac_config,
                training_patient=config.training_patient_profile,
                task_name=config.task_name,
                seed=seed,
                timesteps=timesteps,
                device=config.device,
                output_dir=output_dir / "models",
            )
            model_metadata[method].append(str(artifact.model_path.name))
            try:
                for patient_index, patient in enumerate(selected_patients):
                    for episode in range(episodes):
                        episode_seed = seed + 200_000 + patient_index * 1000 + episode
                        rows.append(
                            _evaluate_rl_episode(
                                artifact=artifact,
                                sac_config=sac_config,
                                task_name=config.task_name,
                                patient_profile=patient,
                                seed=episode_seed,
                                episode=episode,
                            )
                        )
            finally:
                artifact.close()
    metadata = {
        "phase": 10,
        "task_name": config.task_name,
        "methods": list(selected_methods),
        "patients": list(selected_patients),
        "seeds": list(selected_seeds),
        "episodes_per_condition": episodes,
        "training_timesteps": timesteps,
        "git_commit": _git_commit(root),
        "config_sha256": _config_digest(root),
        "models": model_metadata,
    }
    summary = write_experiment_outputs(
        rows=rows,
        output_dir=output_dir,
        metadata=metadata,
        plot_dpi=config.plot_dpi,
    )
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return rows, summary
