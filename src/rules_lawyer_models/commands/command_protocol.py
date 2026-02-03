from __future__ import annotations

from typing import Protocol

from rules_lawyer_models.core import RunContext


class CommmandProtocol(Protocol):
    def execute(self, ctxt: RunContext) -> None: ...
