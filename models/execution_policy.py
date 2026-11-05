from pydantic import ConfigDict, Field, model_validator

from models.schemas import BaseSchema


class ExecutionPolicy(BaseSchema):
    """Generic controls for scheduling and inter-agent input/output routing."""

    model_config = ConfigDict(frozen=True)

    honor_dependencies: bool = True
    inject_upstream_results: bool = True
    route_messages: bool = True

    @model_validator(mode="after")
    def validate_dependency_policy(self):
        if self.inject_upstream_results and not self.honor_dependencies:
            raise ValueError("Upstream results require dependency scheduling.")
        return self


class OrchestrationPolicy(BaseSchema):
    """Treatment-neutral controls for orchestration stages."""

    model_config = ConfigDict(frozen=True)

    execution: ExecutionPolicy = Field(default_factory=ExecutionPolicy)
    deliberation_enabled: bool = True
    verification_enabled: bool = True
    synthesis_enabled: bool = True

    @model_validator(mode="after")
    def validate_stage_policy(self):
        if self.synthesis_enabled and not self.verification_enabled:
            raise ValueError("Synthesis requires verification for multi-agent orchestration.")
        if self.deliberation_enabled and not (
            self.execution.honor_dependencies
            and self.execution.inject_upstream_results
            and self.execution.route_messages
            and self.verification_enabled
            and self.synthesis_enabled
        ):
            raise ValueError(
                "Deliberation requires dependency execution, upstream context, message routing, "
                "verification, and synthesis."
            )
        combination = (
            self.execution.honor_dependencies,
            self.execution.inject_upstream_results,
            self.execution.route_messages,
            self.deliberation_enabled,
            self.verification_enabled,
            self.synthesis_enabled,
        )
        supported_combinations = {
            (False, False, False, False, False, False),
            (False, False, False, False, True, True),
            (True, True, True, False, True, True),
            (True, True, True, True, True, True),
        }
        if combination not in supported_combinations:
            raise ValueError("Orchestration policy does not match a supported treatment combination.")
        return self