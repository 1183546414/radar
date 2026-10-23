# -*- coding: utf-8 -*-
"""将 stdout/stderr 同时写入运行日志文件（每次执行前清空）。"""
from __future__ import annotations

import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator, TextIO


class _TeeStream:
    """终端与日志文件双写。"""

    def __init__(self, stream: TextIO, log_file: TextIO) -> None:
        self._stream = stream
        self._log_file = log_file

    def write(self, data: str) -> int:
        self._stream.write(data)
        self._log_file.write(data)
        self._log_file.flush()
        return len(data)

    def flush(self) -> None:
        self._stream.flush()
        self._log_file.flush()

    def isatty(self) -> bool:
        return self._stream.isatty()


@contextmanager
def run_with_log(log_path: Path, script_name: str = "run") -> Iterator[Path]:
    """
    每次进入时以写入模式打开日志（清空旧内容），
    将 stdout/stderr 同步写入该文件与终端。
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w", encoding="utf-8") as log_file:
        started = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header = f"===== {script_name} started at {started} =====\n"
        log_file.write(header)
        log_file.flush()

        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = _TeeStream(old_stdout, log_file)
        sys.stderr = _TeeStream(old_stderr, log_file)
        try:
            yield log_path
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            ended = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            footer = f"\n===== {script_name} finished at {ended} =====\n"
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(footer)
