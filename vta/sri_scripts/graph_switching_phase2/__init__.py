"""
Phase 2: Quick Win Approach for Multi-Model Execution

Shared parameter pool with zero-copy linking for fast model switching.
"""

from .shared_param_manager import SharedParamManager
from .multi_runtime_executor import (
    MultiRuntimeExecutor,
    MultiRuntimeExecutorBuilder,
    CompiledModelInfo
)
from .phase2_builder import Phase2Builder, quick_build

__all__ = [
    'SharedParamManager',
    'MultiRuntimeExecutor',
    'MultiRuntimeExecutorBuilder',
    'CompiledModelInfo',
    'Phase2Builder',
    'quick_build'
]

__version__ = '1.0.0'

