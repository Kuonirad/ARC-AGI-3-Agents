from typing import Type, cast
from dotenv import load_dotenv

from .agent import Agent, Playback
from .nova_autarky import NovaAutarkyAgent
from .nova_omni import NovaOmniAgent
from .nova_perceptron import NovaPerceptronAgent
from .nova_polymath import NovaPolymathAgent
from .kevin_kull import KevinKullAgent
from .kevin_kull import NovaSingularityAgent as KevinKull
from .recorder import Recorder
from .swarm import Swarm
from .templates.langgraph_functional_agent import LangGraphFunc, LangGraphTextOnly
from .templates.langgraph_random_agent import LangGraphRandom
from .templates.langgraph_thinking import LangGraphThinking
from .templates.llm_agents import LLM, FastLLM, GuidedLLM, ReasoningLLM
from .templates.random_agent import Random
from .templates.reasoning_agent import ReasoningAgent
from .templates.smolagents import SmolCodingAgent, SmolVisionAgent
from .nova_autarky import NovaAutarkyAgent
from .nova_omni import NovaOmniAgent
from .nova_perceptron import NovaPerceptronAgent
from .nova_polymath import NovaPolymathAgent

# NOVA PROTOCOL
from .nova_autarky import NovaAutarkyAgent
from .nova_omni import NovaOmniAgent
from .nova_perceptron import NovaPerceptronAgent
from .nova_polymath import NovaPolymathAgent
from .kevin_kull import KevinKullAgent # <--- v4.0-OMEGA

# NOVA PROTOCOL
from .nova_autarky import NovaAutarkyAgent
from .nova_omni import NovaOmniAgent
from .nova_perceptron import NovaPerceptronAgent
from .nova_polymath import NovaPolymathAgent
from .kevin_kull import KevinKullAgent # <--- v6.0-NEXUS

# NOVA PROTOCOL
from .nova_autarky import NovaAutarkyAgent
from .nova_omni import NovaOmniAgent
from .nova_perceptron import NovaPerceptronAgent
from .nova_polymath import NovaPolymathAgent
from .kevin_kull import KevinKullAgent # <--- v6.0-NEXUS

# NOVA PROTOCOL
from .nova_autarky import NovaAutarkyAgent
from .nova_omni import NovaOmniAgent
from .nova_perceptron import NovaPerceptronAgent
from .nova_polymath import NovaPolymathAgent
from .kevin_kull import KevinKullAgent # <--- v7.0-ZENITH

# NOVA PROTOCOL
from .nova_autarky import NovaAutarkyAgent
from .nova_omni import NovaOmniAgent
from .nova_perceptron import NovaPerceptronAgent
from .nova_polymath import NovaPolymathAgent
# from .nova_singularity import NovaSingularityAgent
from .kevin_kull import KevinKullAgent # <--- v9.1-COSMOS

# NOVA PROTOCOL
from .nova_autarky import NovaAutarkyAgent
from .nova_omni import NovaOmniAgent
from .nova_perceptron import NovaPerceptronAgent
from .nova_polymath import NovaPolymathAgent
# from .nova_singularity import NovaSingularityAgent
from .kevin_kull import KevinKullAgent # <--- v9.1-COSMOS

# NOVA PROTOCOL
from .nova_autarky import NovaAutarkyAgent
from .nova_omni import NovaOmniAgent
from .nova_perceptron import NovaPerceptronAgent
from .nova_polymath import NovaPolymathAgent
# from .nova_singularity import NovaSingularityAgent
from .kevin_kull import KevinKullAgent # <--- v9.1-COSMOS

load_dotenv()

AVAILABLE_AGENTS: dict[str, Type[Agent]] = {
    cls.__name__.lower(): cast(Type[Agent], cls)
    for cls in Agent.__subclasses__()
    if cls.__name__ != "Playback"
}
AVAILABLE_AGENTS["nova_autarky"] = NovaAutarkyAgent
AVAILABLE_AGENTS["nova_omni"] = NovaOmniAgent
AVAILABLE_AGENTS["nova_perceptron"] = NovaPerceptronAgent
# AVAILABLE_AGENTS["nova_singularity"] = NovaSingularityAgent # Deprecated
AVAILABLE_AGENTS["kevin_kull"] = KevinKullAgent
AVAILABLE_AGENTS["nova_polymath"] = NovaPolymathAgent
AVAILABLE_AGENTS["kevin_kull"] = KevinKull

# Explicit Registration
AVAILABLE_AGENTS["nova_autarky"] = NovaAutarkyAgent
AVAILABLE_AGENTS["nova_omni"] = NovaOmniAgent
AVAILABLE_AGENTS["nova_perceptron"] = NovaPerceptronAgent
AVAILABLE_AGENTS["nova_polymath"] = NovaPolymathAgent
AVAILABLE_AGENTS["kevin_kull"] = KevinKullAgent

# Explicit Registration
AVAILABLE_AGENTS["nova_autarky"] = NovaAutarkyAgent
AVAILABLE_AGENTS["nova_omni"] = NovaOmniAgent
AVAILABLE_AGENTS["nova_perceptron"] = NovaPerceptronAgent
AVAILABLE_AGENTS["nova_polymath"] = NovaPolymathAgent
AVAILABLE_AGENTS["kevin_kull"] = KevinKullAgent

# Explicit Registration
AVAILABLE_AGENTS["nova_autarky"] = NovaAutarkyAgent
AVAILABLE_AGENTS["nova_omni"] = NovaOmniAgent
AVAILABLE_AGENTS["nova_perceptron"] = NovaPerceptronAgent
AVAILABLE_AGENTS["nova_polymath"] = NovaPolymathAgent
AVAILABLE_AGENTS["kevin_kull"] = KevinKullAgent

# Explicit Registration
AVAILABLE_AGENTS["nova_autarky"] = NovaAutarkyAgent
AVAILABLE_AGENTS["nova_omni"] = NovaOmniAgent
AVAILABLE_AGENTS["nova_perceptron"] = NovaPerceptronAgent
AVAILABLE_AGENTS["nova_polymath"] = NovaPolymathAgent
AVAILABLE_AGENTS["kevin_kull"] = KevinKullAgent

AVAILABLE_AGENTS["nova_autarky"] = NovaAutarkyAgent
AVAILABLE_AGENTS["nova_omni"] = NovaOmniAgent
AVAILABLE_AGENTS["nova_perceptron"] = NovaPerceptronAgent
AVAILABLE_AGENTS["nova_polymath"] = NovaPolymathAgent
# AVAILABLE_AGENTS["nova_singularity"] = NovaSingularityAgent
AVAILABLE_AGENTS["kevin_kull"] = KevinKullAgent

AVAILABLE_AGENTS["nova_autarky"] = NovaAutarkyAgent
AVAILABLE_AGENTS["nova_omni"] = NovaOmniAgent
AVAILABLE_AGENTS["nova_perceptron"] = NovaPerceptronAgent
AVAILABLE_AGENTS["nova_polymath"] = NovaPolymathAgent
# AVAILABLE_AGENTS["nova_singularity"] = NovaSingularityAgent
AVAILABLE_AGENTS["kevin_kull"] = KevinKullAgent

AVAILABLE_AGENTS["nova_autarky"] = NovaAutarkyAgent
AVAILABLE_AGENTS["nova_omni"] = NovaOmniAgent
AVAILABLE_AGENTS["nova_perceptron"] = NovaPerceptronAgent
AVAILABLE_AGENTS["nova_polymath"] = NovaPolymathAgent
# AVAILABLE_AGENTS["nova_singularity"] = NovaSingularityAgent
AVAILABLE_AGENTS["kevin_kull"] = KevinKullAgent

for rec in Recorder.list():
    AVAILABLE_AGENTS[rec] = Playback

AVAILABLE_AGENTS["reasoningagent"] = ReasoningAgent

__all__ = [
    "Swarm",
    "Random",
    "LangGraphFunc",
    "LangGraphTextOnly",
    "LangGraphThinking",
    "LangGraphRandom",
    "LLM",
    "FastLLM",
    "ReasoningLLM",
    "GuidedLLM",
    "ReasoningAgent",
    "SmolCodingAgent",
    "SmolVisionAgent",
    "Agent",
    "Recorder",
    "Playback",
    "AVAILABLE_AGENTS",
]
