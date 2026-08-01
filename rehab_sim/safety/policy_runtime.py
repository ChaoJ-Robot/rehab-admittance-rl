"""Model-agnostic deterministic policy runtime behind the safety supervisor."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import ArrayLike

from rehab_sim.safety.supervisor import SafetyDecision, SafetyObservation, SafetySupervisor


class PredictablePolicy(Protocol):
    """Minimal policy contract; no Stable-Baselines3 dependency is required."""

    def predict(self, observation: Any, deterministic: bool = True) -> Any:
        """Return an action or an ``(action, recurrent_state)`` pair."""


PolicyLoader = Callable[[Path], PredictablePolicy]
Clock = Callable[[], float]


class SafePolicyRuntime:
    """Run deterministic policy inference and always pass through supervision."""

    def __init__(
        self,
        supervisor: SafetySupervisor,
        policy: PredictablePolicy | Callable[[Any], Any] | None = None,
        clock: Clock = time.perf_counter,
    ) -> None:
        self.supervisor = supervisor
        self._policy = policy
        self._clock = clock
        self._model_loaded = policy is not None
        self._last_error: str | None = None

    @property
    def model_loaded(self) -> bool:
        """Whether a policy object is currently available for inference."""

        return self._model_loaded

    @property
    def last_error(self) -> str | None:
        """Most recent model-loading or inference error."""

        return self._last_error

    def load_model(self, model_path: str | Path, loader: PolicyLoader) -> bool:
        """Load a model through an injected loader and fail closed on errors."""

        path = Path(model_path)
        self._policy = None
        self._model_loaded = False
        self._last_error = None
        if not path.is_file():
            self._last_error = f"model file does not exist: {path}"
            return False
        try:
            policy = loader(path)
            if policy is None:
                raise ValueError("model loader returned None")
        except Exception as error:  # noqa: BLE001 - deployment boundary must fail closed
            self._last_error = f"model load failed: {error}"
            return False
        self._policy = policy
        self._model_loaded = True
        return True

    def unload_model(self, reason: str = "model_unloaded") -> None:
        """Disable inference; the next action will select fallback parameters."""

        self._policy = None
        self._model_loaded = False
        self._last_error = reason

    @staticmethod
    def _action_from_prediction(prediction: Any) -> np.ndarray:
        if isinstance(prediction, tuple):
            prediction = prediction[0]
        action = np.asarray(prediction, dtype=np.float64)
        if action.shape == (1, 4):
            action = action[0]
        return action

    def _predict(self, observation: Any) -> np.ndarray:
        if self._policy is None:
            raise RuntimeError("policy is not loaded")
        if hasattr(self._policy, "predict"):
            prediction = self._policy.predict(observation, deterministic=True)
        elif callable(self._policy):
            prediction = self._policy(observation)
        else:
            raise TypeError("policy must provide predict() or be callable")
        return self._action_from_prediction(prediction)

    def act(
        self,
        observation: Any,
        current_parameters: ArrayLike,
        safety_observation: SafetyObservation,
    ) -> SafetyDecision:
        """Infer one action, measure latency and return a supervised decision."""

        if not self._model_loaded:
            reasons = (
                ("model_load_failed", "model_not_loaded")
                if self._last_error
                else ("model_not_loaded",)
            )
            return self.supervisor.fallback(
                current_parameters,
                np.zeros(4, dtype=np.float64),
                reasons,
                safety_observation,
            )
        started = self._clock()
        try:
            raw_action = self._predict(observation)
        except Exception as error:  # noqa: BLE001 - inference failure is a safety event
            self._last_error = f"policy inference failed: {error}"
            return self.supervisor.fallback(
                current_parameters,
                np.zeros(4, dtype=np.float64),
                ("policy_inference_error",),
                safety_observation,
            )
        elapsed = self._clock() - started
        return self.supervisor.supervise(
            raw_action,
            current_parameters,
            safety_observation,
            inference_elapsed_s=elapsed,
            model_loaded=True,
        )
