import json
from Actor import Actor
from Critic import Critic
from Agent import Agent
from LLM import GemmaLLM, LLM_Evaluator, LLamaLLM

# It is a specialization of the agent class.
# It represents a deceptive seller, for the third benchmark, which has some hidden information that the buyer does not know.
# It computes a deception index (DI) score for each message and at the end it computes the max and average DI score for the whole negotiation session.
class DeceptiveSeller(Agent):
    def __init__(self, actor: Actor, critic: Critic, hidden_info: str, isJSON):
        self.__hidden_info = hidden_info
        super().__init__(actor, critic, isJSON)
        with open("DealingProblem/Rules.json", 'r') as file:
            rules = json.load(file)
        self._DI_Evaluator = Actor(
            description=rules['DI_Evaluator'], client=LLM_Evaluator)
    
    def respond(self, history) -> dict:
        original_response = super().respond(history)
        DI_score = self._DI_Evaluator.ask(
            [{
                "role" : "",
                "text" : "Sentence 1:" + original_response['text'] + "\n" + 
                         "Sentence 2:" + self.__hidden_info
            }]
        )
        try:
            DI_score = json.loads(DI_score)
        except json.JSONDecodeError:
            DI_score = {'score' : float('nan'), 'reason' : 'Invalid JSON'}
        original_response['DI_score'] = DI_score['score']
        return original_response
    

    def analyzeSession(self, history, additionalInfo = {}):
        analysis = super().analyzeSession(history, additionalInfo)
        role = self.getDescription().get('role', '')
        seller_msgs = [msg for msg in history if role.lower() == msg['role'].lower()]
        DI_scores = [score['DI_score'] if 'DI_score' in score else float('nan') for score in seller_msgs]
        analysis['max_DI'] = max(DI_scores) if len(DI_scores) > 0 else float('nan')
        analysis['avg_DI'] = sum(DI_scores) / len(DI_scores) if len(DI_scores) > 0 else float('nan')
        return analysis
    
    @staticmethod
    def fromJSON_DeceptiveSeller(path: str, agentType: str, name: str, client, additionalInstructions: str = "", isJSON = False):
        with open(path, 'r') as f:
            json_data = json.load(f)
            agents = json_data[agentType]
            hidden_info = json_data['hidden_info']

        critics = json.load(open("DealingProblem/Rules.json", 'r'))
        for agent in agents:
            if agent["name"] == name:
                agent['rules'].append(additionalInstructions)
                if isJSON:
                    JSONRole = 'Json' + agent['role']
                    agent["rules"] = [*agent['rules'], *critics[JSONRole]['rules']]
                return DeceptiveSeller(
                        actor  = Actor(agent, client),
                        critic = Critic(critics[agent["role"]]),
                        hidden_info = hidden_info,
                        isJSON = isJSON
                )
        raise Exception(f"Agent with name {name} not found")
