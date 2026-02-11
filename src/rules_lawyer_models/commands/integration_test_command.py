from dataclasses import dataclass

from rules_lawyer_models.core import RunContext


@dataclass
class IntegrationTestCommand:
    def execute(self, ctxt: RunContext) -> None: ...
