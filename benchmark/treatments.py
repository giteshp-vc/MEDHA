from enum import Enum

from pydantic import ConfigDict, model_validator

from models.execution_policy import ExecutionPolicy, OrchestrationPolicy
from models.schemas import BaseSchema


class BenchmarkArm(str, Enum):
    BASELINE = "BASELINE"
    INDEPENDENT_PARALLEL = "INDEPENDENT_PARALLEL"
    DEPENDENCY_AWARE = "DEPENDENCY_AWARE"
    FULL_MEDHA = "FULL_MEDHA"


class TreatmentPreset(BaseSchema):
    """Validated benchmark-arm settings mapped to generic core policies."""

    model_config = ConfigDict(frozen=True)

    arm: BenchmarkArm
    uses_plan: bool
    specialists_enabled: bool
    policy: OrchestrationPolicy

    @model_validator(mode="after")
    def validate_arm_preset(self):
        expected = _preset_values(self.arm)
        actual = {
            "uses_plan": self.uses_plan,
            "specialists_enabled": self.specialists_enabled,
            "policy": self.policy.model_dump(mode="json"),
        }
        if actual != expected:
            raise ValueError(f"Treatment settings do not match benchmark arm {self.arm}.")
        return self


def get_treatment_preset(arm: BenchmarkArm | str) -> TreatmentPreset:
    """Return the immutable canonical treatment configuration for an arm."""
    selected_arm = BenchmarkArm(arm)
    values = _preset_values(selected_arm)
    return TreatmentPreset(
        arm=selected_arm,
        uses_plan=values["uses_plan"],
        specialists_enabled=values["specialists_enabled"],
        policy=OrchestrationPolicy.model_validate(values["policy"]),
    )


def _preset_values(arm: BenchmarkArm | str):
    arm_value = str(getattr(arm, "value", arm))
    direct_policy = OrchestrationPolicy(
        execution=ExecutionPolicy(
            honor_dependencies=False,
            inject_upstream_results=False,
            route_messages=False,
        ),
        deliberation_enabled=False,
        verification_enabled=False,
        synthesis_enabled=False,
    )
    independent_policy = OrchestrationPolicy(
        execution=ExecutionPolicy(
            honor_dependencies=False,
            inject_upstream_results=False,
            route_messages=False,
        ),
        deliberation_enabled=False,
        verification_enabled=True,
        synthesis_enabled=True,
    )
    dependency_policy = OrchestrationPolicy(
        execution=ExecutionPolicy(),
        deliberation_enabled=False,
        verification_enabled=True,
        synthesis_enabled=True,
    )
    full_policy = OrchestrationPolicy()
    presets = {
        BenchmarkArm.BASELINE.value: {
            "uses_plan": False,
            "specialists_enabled": False,
            "policy": direct_policy.model_dump(mode="json"),
        },
        BenchmarkArm.INDEPENDENT_PARALLEL.value: {
            "uses_plan": True,
            "specialists_enabled": True,
            "policy": independent_policy.model_dump(mode="json"),
        },
        BenchmarkArm.DEPENDENCY_AWARE.value: {
            "uses_plan": True,
            "specialists_enabled": True,
            "policy": dependency_policy.model_dump(mode="json"),
        },
        BenchmarkArm.FULL_MEDHA.value: {
            "uses_plan": True,
            "specialists_enabled": True,
            "policy": full_policy.model_dump(mode="json"),
        },
    }
    try:
        return presets[arm_value]
    except KeyError as error:
        raise ValueError(f"Unsupported benchmark arm: {arm_value}") from error