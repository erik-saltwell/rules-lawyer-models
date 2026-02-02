from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

from rich.console import Console
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn
from rich.status import Status
from rich.traceback import Traceback

from rules_lawyer_models.utils import (
    BaseModelName,
    CommonPaths,
    DatasetName,
    LoggingProtocol,
)
from rules_lawyer_models.utils.logging_protocol import ProgressTask, StatusHandle


@dataclass(slots=True)
class RunContext(LoggingProtocol):
    model_name: str
    base_model_name: BaseModelName
    dataset_name: DatasetName
    common_paths: CommonPaths.ModelPaths = field(init=False)
    _console: Console = field(default_factory=Console)

    def __post_init__(self) -> None:
        self.common_paths = CommonPaths.ModelPaths(model_name=self.model_name)
        self.common_paths.ensure_all_dirs_exist()

    def report_message(self, message: str) -> None:
        self._console.print(message)

    def report_warning(self, message: str) -> None:
        self._console.print(f"[yellow]WARNING[/yellow] {message}")

    def report_error(self, message: str) -> None:
        self._console.print(f"[red]ERROR[/red] {message}")

    def report_exception(self, context: str, exc: BaseException) -> None:
        # Report the *passed* exception with full traceback, regardless of whether we're inside an except block.
        self._console.print(f"[red]EXCEPTION[/red] {context}")
        tb = Traceback.from_exception(type(exc), exc, exc.__traceback__)
        self._console.print(tb)
        # Do NOT raise here; let the caller decide (and preserve traceback with `raise` at call site).

    # ----- status/progress -----

    @contextmanager
    def status(self, message: str) -> Iterator[StatusHandle]:
        st = Status(message, console=self._console)
        st.start()

        class _Handle:
            _closed = False

            def update(self, message: str) -> None:
                st.update(message)

            def close(self) -> None:
                if not self._closed:
                    st.stop()
                    self._closed = True

        handle: StatusHandle = _Handle()
        try:
            yield handle
        finally:
            handle.close()

    @contextmanager
    def progress(self, description: str, total: int | None = None) -> Iterator[ProgressTask]:
        prog = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=self._console,
            transient=True,
        )
        prog.start()

        try:
            task_id = prog.add_task(description, total=total)

            class _Task:
                def advance(self, n: int = 1) -> None:
                    prog.advance(task_id, n)

                def set_total(self, total: int | None) -> None:
                    prog.update(task_id, total=total)

                def set_completed(self, completed: int) -> None:
                    prog.update(task_id, completed=completed)

                def set_description(self, description: str) -> None:
                    prog.update(task_id, description=description)

                def close(self) -> None:
                    # Context manager owns lifecycle.
                    pass

            task: ProgressTask = _Task()
            yield task
        finally:
            prog.stop()
