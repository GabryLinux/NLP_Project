import hashlib
import json
from math import ceil
import re
import time
from typing import OrderedDict

from Actor import Actor
from Agent import Agent
from Critic import Critic
import Formatter
from LLM import GemmaLLM, LLM, LLM_Evaluator
from pathlib import Path

from Utilities import Utilities

# The Arena class represents the environment in which the agents negotiate.
# It keeps track of the history of the negotiation and the agents involved in it.
# It is responsible for running the negotiation session, saving the history of the session, and evaluating the session at the end.
class Arena:
    def __init__(self, agentList: list[Agent], context, sessionName: str, LLMClient : LLM = LLM_Evaluator):
        self.__savePath = sessionName
        self.__agents = agentList
        self.__history = [{"role": "seller", "text": context}]
        self.__LLMClient = LLMClient


    def _nextRound(self):
        for i in range(len(self.__agents)):
            mess = self.__agents[i].respond(self.__history)
            self.__history.append(mess)
            if self.__agents[i].getAgreement():
                return


    def negotiate(self, maxRounds: int = 10):
        for _ in range(maxRounds):
            self._nextRound()
            allAgreed = any(agent.getAgreement() for agent in self.__agents)
            if allAgreed:
                break
        
        
        self.save_history(self.__savePath)
        return


    def set_fileName(self, name: str):
        self.__savePath = name
        return self

    # Load session data from files
    # contextPath is the path to the actor context. It has to be a json file formatted in the correct way.
    @staticmethod
    def load_session(contextPath : str, client : LLM = LLM_Evaluator):
        with open(contextPath, 'r') as context_file:
            actorInits = json.load(context_file)
        
        return Arena([], actorInits['scenario'], contextPath, client)
    
    def _load_savepath(self, path: str):
        agents = []
        file_path = Path(path)

        file_path.parent.mkdir(parents=True, exist_ok=True)
        if not file_path.exists():
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump({}, f)

        with open(file_path, "r+", encoding="utf-8") as f:
            return json.load(f)
        
    def save_history(self, path: str):
        raw = self._load_savepath(path)

        JSON = OrderedDict()
        JSON['scenario'] = raw.get('scenario', self.__history[0]['text'])
        JSON['sessions'] = raw.get('sessions', [])
        agentDescriptions = []

        # Remove rules from agent descriptions before saving, to avoid redundancy
        for agent in self.__agents: 
            agent.getDescription().pop('rules', None)
            agentDescriptions.append(agent.getDescription())  
        
        session = {
                "id" : self._generateHashcode(),
                "agents":  agentDescriptions,
                "history": self.getHistory()[1:] # The first message of the history is the context, we can omit it.
            }

        try:
            eval = self.evaluateHistory(self.getHistory())
        except Exception as e:
            print(f"Error during evaluation: {e}")
            eval = {}
        
        session["evaluation"] = eval

        self._add_and_remove(JSON, session)
        
        with open(Path(self.__savePath), "w", encoding="utf-8") as f:
            json.dump(JSON, f, indent=4, sort_keys=False)
        return
    
    def _add_and_remove(self, JSON: dict, element: dict):
        for s in JSON['sessions']:
            if s['id'] == element['id']:
                print("Session already exists in history file. Overwriting.")
                JSON['sessions'].remove(s)
                break

        JSON['sessions'].append(element)
    

    def _generateHashcode(self) -> int:
        s = "".join(
            [agent.getDescription()['name'] for agent in self.__agents]
        ).join(
            [agent.getDescription()['role'] for agent in self.__agents]
        )
        return int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16) % (2**64)
    
    def getHistory(self):
        return self.__history
    
    def loadAgents(self, agent: Agent):
        self.__agents.append(agent)
        agent.reset()
        return self
    
    def evaluateHistory(self, history):
        with open("DealingProblem/Rules.json", 'r') as f:
            evaluatorDescription = json.load(f)['Evaluator']
        evaluator = Actor(evaluatorDescription, self.__LLMClient)
        evaluationResponse = evaluator.ask(history)
        try:
            evaluationResponse = Utilities.extract_json(evaluationResponse)
        except json.JSONDecodeError:
            print("Failed to parse JSON from evaluator response: " + evaluationResponse)
            evaluationResponse = {"Result": "ERROR", "initial_price": "NaN", "initial_buyer_offer": "NaN", "Error": "JSONDecodeError"}
        analysis = {
                "result": evaluationResponse['Result'],
                "analysis": [
                    agent.analyzeSession(history, additionalInfo=evaluationResponse) for agent in self.__agents
                ],
                "rounds" : ceil(len(history) / 2),
                "final_price": str(float('nan'))
            }
        
        analysis["final_price"] = evaluationResponse.get('final_price', str(float('nan')))
            
        return analysis
    

    @staticmethod
    def null_arena():
        return Arena([], "","nullArena", LLMClient=LLM_Evaluator)
    