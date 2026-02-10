import json
from Arena import Arena
from Agent import Agent
from DeceptiveSeller import DeceptiveSeller
from LLM import GemmaLLM, LLamaLLM
from openai import OpenAI

LLamaLLM.set_model("llama-3.3-70b-versatile")

arena = Arena.load_session(
    "DealingProblem/Context/Scenario2.json",
).loadAgents(
    Agent.fromJSON(
        path="DealingProblem/Context/Scenario2.json",
        agentType="buyers", 
        name="neutral-concise-buyer",
        isJSON=True,
        client=LLamaLLM
    )
).loadAgents(
    DeceptiveSeller.fromJSON_DeceptiveSeller(
        path="DealingProblem/Context/Scenario3.json",
        agentType="sellers", 
        name="desperate-discursive-seller",
        isJSON=True,
        client=LLamaLLM
    )
).set_fileName(
    "DealingProblem/TEST/TEST.json"
)

arena.negotiate(maxRounds=10)