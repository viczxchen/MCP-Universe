from .function_call import FunctionCall
from .basic import BasicAgent
from .workflow import WorkflowAgent
from .react import ReAct
from .harmony_agent import HarmonyReAct
from .reflection import Reflection
from .explore_and_exploit import ExploreAndExploit
from .base import BaseAgent
from .claude_code import ClaudeCodeAgent

# OpenAIAgentSDK (based on openai-agents) is optional and has its own
# dependency constraints. Import it lazily so that environments which
# don't satisfy those constraints can still use the core agents (ReAct, etc.).
try:
    from .openai_agent_sdk import OpenAIAgentSDK  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    OpenAIAgentSDK = None  # type: ignore

__all__ = [
    "FunctionCall",
    "BasicAgent",
    "WorkflowAgent",
    "ReAct",
    "HarmonyReAct",
    "Reflection",
    "BaseAgent",
    "ClaudeCodeAgent",
    # OpenAIAgentSDK is optional; export only if available
]

if OpenAIAgentSDK is not None:  # type: ignore
    __all__.append("OpenAIAgentSDK")
