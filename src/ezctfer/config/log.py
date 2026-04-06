"""Console logging helpers with thread labels and prompt redraw support."""

from __future__ import annotations

import builtins
import os
import sys
import threading
from dataclasses import dataclass


class Colors:
    YELLOW = "\033[93m"
    GREEN = "\033[92m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"
    BLUE = "\033[94m"
    BLACK = "\033[30m"
    WHITE_BG = "\033[47m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


THREAD_EMOJIS = {
    1: "🟢",
    2: "🔵",
    3: "🟡",
    4: "🟣",
    5: "🟠",
}


_ERASE_LINE = "\033[2K"
_CURSOR_UP = "\033[1A"
_CARRIAGE_RETURN = "\r"

_console_lock = threading.RLock()
_prompt_condition = threading.Condition(_console_lock)
_vt_enabled = False
_debug_enabled = False


@dataclass
class _PromptState:
    header_lines: list[str]
    prompt: str
    style: str
    reset_style: str
    buffer: str = ""
    rendered_lines: int = 0


_active_prompt: _PromptState | None = None


def _enable_virtual_terminal_if_needed() -> None:
    global _vt_enabled

    if _vt_enabled:
        return

    if os.name != "nt" or not sys.stdout.isatty():
        _vt_enabled = True
        return

    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)
    except Exception:
        pass

    _vt_enabled = True


def _write_raw(text: str) -> None:
    _enable_virtual_terminal_if_needed()
    sys.stdout.write(text)
    sys.stdout.flush()


def _get_thread_prefix(thread_id: int = 0) -> str:
    if thread_id > 0:
        return f"{THREAD_EMOJIS.get(thread_id, str(thread_id))} "
    return ""


def _format_message(message: str, thread_id: int = 0) -> str:
    return f"{_get_thread_prefix(thread_id)}{message}"


def _clear_prompt_locked() -> None:
    global _active_prompt

    if _active_prompt is None or _active_prompt.rendered_lines == 0:
        return

    rendered_lines = _active_prompt.rendered_lines
    for index in range(rendered_lines):
        _write_raw(f"{_CARRIAGE_RETURN}{_ERASE_LINE}")
        if index < rendered_lines - 1:
            _write_raw(_CURSOR_UP)
    _write_raw(_CARRIAGE_RETURN)
    _active_prompt.rendered_lines = 0


def _render_prompt_locked() -> None:
    if _active_prompt is None:
        return

    state = _active_prompt
    _write_raw("\n")
    for line in state.header_lines:
        _write_raw(f"{state.style}{line}{state.reset_style}\n")
    _write_raw(f"{state.style}{state.prompt}{state.buffer}{state.reset_style}")
    state.rendered_lines = len(state.header_lines) + 1


def _read_prompt_char() -> str:
    if os.name == "nt":
        import msvcrt

        while True:
            ch = msvcrt.getwch()
            if ch in ("\x00", "\xe0"):
                msvcrt.getwch()
                continue
            return ch

    import termios
    import tty

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            while True:
                next_ch = sys.stdin.read(1)
                if not next_ch or next_ch.isalpha() or next_ch == "~":
                    break
            return ""
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def console_print(message: str = "", end: str = "\n", flush: bool = True) -> None:
    with _console_lock:
        has_prompt = _active_prompt is not None
        if has_prompt:
            _clear_prompt_locked()
        builtins.print(message, end=end, flush=flush)
        if has_prompt:
            _render_prompt_locked()


def prompt_input(
    header_lines: list[str],
    prompt: str,
    *,
    style: str = "",
    reset_style: str = Colors.RESET,
) -> str:
    global _active_prompt

    if not sys.stdin.isatty() or not sys.stdout.isatty():
        for line in header_lines:
            console_print(f"{style}{line}{reset_style}")
        return builtins.input(f"{style}{prompt}")

    with _prompt_condition:
        while _active_prompt is not None:
            _prompt_condition.wait()
        _active_prompt = _PromptState(
            header_lines=header_lines,
            prompt=prompt,
            style=style,
            reset_style=reset_style,
        )
        _render_prompt_locked()

    try:
        while True:
            ch = _read_prompt_char()
            with _console_lock:
                state = _active_prompt
                if state is None:
                    raise RuntimeError("Prompt state lost")

                if ch in ("\r", "\n"):
                    value = state.buffer
                    _clear_prompt_locked()
                    builtins.print("", flush=True)
                    return value

                if ch == "\x03":
                    _clear_prompt_locked()
                    builtins.print("", flush=True)
                    raise KeyboardInterrupt

                if ch == "\x1a":
                    _clear_prompt_locked()
                    builtins.print("", flush=True)
                    raise EOFError

                if ch in ("\b", "\x7f"):
                    state.buffer = state.buffer[:-1]
                elif ch and ch.isprintable():
                    state.buffer += ch
                else:
                    continue

                _clear_prompt_locked()
                _render_prompt_locked()
    finally:
        _write_raw(Colors.RESET)
        with _prompt_condition:
            _active_prompt = None
            _prompt_condition.notify_all()


def prompt_wait_key(
    header_lines: list[str],
    prompt: str,
    *,
    style: str = "",
    reset_style: str = Colors.RESET,
) -> str:
    global _active_prompt

    if not sys.stdin.isatty() or not sys.stdout.isatty():
        for line in header_lines:
            console_print(f"{style}{line}{reset_style}")
        return builtins.input(f"{style}{prompt}")

    with _prompt_condition:
        while _active_prompt is not None:
            _prompt_condition.wait()
        _active_prompt = _PromptState(
            header_lines=header_lines,
            prompt=prompt,
            style=style,
            reset_style=reset_style,
        )
        _render_prompt_locked()

    try:
        while True:
            ch = _read_prompt_char()
            with _console_lock:
                if _active_prompt is None:
                    raise RuntimeError("Prompt state lost")

                if ch == "":
                    continue
                if ch == "\x03":
                    _clear_prompt_locked()
                    builtins.print("", flush=True)
                    raise KeyboardInterrupt
                if ch == "\x1a":
                    _clear_prompt_locked()
                    builtins.print("", flush=True)
                    raise EOFError

                _clear_prompt_locked()
                builtins.print("", flush=True)
                return ch
    finally:
        _write_raw(Colors.RESET)
        with _prompt_condition:
            _active_prompt = None
            _prompt_condition.notify_all()


def log_info(message: str, thread_id: int = 0) -> None:
    console_print(_format_message(message, thread_id))


def set_debug_enabled(enabled: bool) -> None:
    global _debug_enabled
    _debug_enabled = enabled


def is_debug_enabled() -> bool:
    return _debug_enabled


def log_debug(message: str, thread_id: int = 0) -> None:
    if not _debug_enabled:
        return
    console_print(f"{Colors.DIM}{_format_message(message, thread_id)}{Colors.RESET}")


def log_warning(message: str, thread_id: int = 0) -> None:
    console_print(f"{Colors.YELLOW}⚠️  {_format_message(message, thread_id)}{Colors.RESET}")


def log_error(message: str, thread_id: int = 0) -> None:
    console_print(f"{Colors.RED}❌ {_format_message(message, thread_id)}{Colors.RESET}")


def log_success(message: str, thread_id: int = 0) -> None:
    console_print(f"{Colors.GREEN}✅ {_format_message(message, thread_id)}{Colors.RESET}")


def log_color(message: str, color: str, thread_id: int = 0, bold: bool = False) -> None:
    prefix = _get_thread_prefix(thread_id)
    style = f"{color}{Colors.BOLD}" if bold else color
    console_print(f"{prefix}{style}{message}{Colors.RESET}")


def log_bold(message: str, thread_id: int = 0) -> None:
    console_print(f"{Colors.BOLD}{_format_message(message, thread_id)}{Colors.RESET}")


def log_separator(char: str = "=", length: int = 60, thread_id: int = 0) -> None:
    console_print(_format_message(char * length, thread_id))


def log_empty_line(thread_id: int = 0) -> None:
    console_print()
