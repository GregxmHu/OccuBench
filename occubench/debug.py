"""
Global debug flag for OccuBench.
"""

DEBUG = False
_log_file = None


def set_debug(enabled: bool, log_path: str = None):
    global DEBUG, _log_file
    DEBUG = enabled
    if enabled and log_path:
        _log_file = open(log_path, "w", encoding="utf-8")


def debug_print(msg: str):
    if DEBUG:
        print(msg, flush=True)
        if _log_file:
            _log_file.write(msg + "\n")
            _log_file.flush()


def close_debug():
    global _log_file
    if _log_file:
        _log_file.close()
        _log_file = None
