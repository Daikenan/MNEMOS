"""
统一从项目根目录加载 .env，且 override=True，确保 .env 中的 key 覆盖 shell/IDE 中已设置的值。
所有 API Key 仅由此处与 .env 管理，避免旧 key 残留在环境变量中生效。
"""
from pathlib import Path


def get_project_root() -> Path:
    """从当前文件向上查找包含 pyproject.toml 或 .env 的目录作为项目根。"""
    path = Path(__file__).resolve().parent
    for _ in range(6):
        if (path / "pyproject.toml").exists() or (path / ".env").exists():
            return path
        parent = path.parent
        if parent == path:
            break
        path = parent
    return Path(__file__).resolve().parents[1]


def load_env(*, override: bool = True) -> None:
    """从项目根目录加载 .env，override=True 时 .env 覆盖已有环境变量。"""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    root = get_project_root()
    env_file = root / ".env"
    if env_file.exists():
        load_dotenv(env_file, override=override)


# 供 run_eval 等非 mnemos 子包使用：已知项目根为 Mnemos 时可用
def get_mnemos_project_root() -> Path:
    """从 KnowMeBench/evaluate 或 scripts 等位置推断 Mnemos 项目根。"""
    path = Path(__file__).resolve()
    # mnemos/env_loader.py -> 项目根 = path.parents[1]
    return path.parents[1]
