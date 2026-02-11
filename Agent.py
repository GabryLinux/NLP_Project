import json
from Actor import Actor
from Validator import Validator
from Utilities import Utilities
from LLM import LLM, GemmaLLM, LLamaLLM

# The Agent class represents an agent in the negotiation. 
# It has an actor module and a validator module.
# The actor module is responsible for generating the agent's response given the current history of the negotiation.
# The validator module is responsible for evaluating the agent's response and providing feedback to the actor module.
# It is also responsible for keeping track of whether the agent has reached an agreement or not, and for analyzing the negotiation session at the end.
class Agent():
    def __init__(self, actor: Actor, validator: Validator, isJSON: bool):
        self.__actor = actor
        self.__validator = validator
        self.__agreement = False
        self.__isJSON = isJSON
        with open("DealingProblem/Rules.json", 'r') as file:
            rules = json.load(file)
        type = "JSON" if isJSON else "NA"
        self._HI_Evaluator = Actor(
                    description=rules[f'HI_Evaluator_{actor.getDescription()["role"]}_{type}'], client=LLamaLLM)


    def respond(self, history) -> dict:
        count = 0
        format_error = 0
        actorResponse = self.__actor.ask(history)
        formattedAnalysis = self.__validator.formatResponse(history, actorResponse)
        validatorResponse = self.__validator.evaluateFormattedMessage(formattedAnalysis)
        
        message = validatorResponse.get("MessageType", "")

        if message == "DEAL":
            self.__agreement = True
        elif message == "INVALID":
            count = 1
            actorResponse = self.__actor.ask(history, hint=validatorResponse.get("Hint", ""))
        elif message == "ERROR" or message == "":
            count = 1
            format_error = 1

        if self.__isJSON:
            try:
                json.loads(actorResponse)
                format_error = 0
            except json.JSONDecodeError:
                format_error = 1

        return {
            "role" : self.__actor.getDescription()['role'],
            "text": self.__actor.getDescription()['role'] + " : " + actorResponse,
            "retry_counts" : count,
            "format_error" : format_error
        }


    def getAgreement(self) -> bool:
        return self.__agreement
    
    def getDescription(self) -> str:
        return self.__actor.getDescription()
    
    def reset(self):
        self.__agreement = False
        return self
    
    # It analyzes the negotiation session at the end of the negotiation
    # It needs: the history of the negotiation and a dictionary with information from the negotiation given from the arena.
    # It returns a dictionary with the evaluation of the negotiation session.
    def analyzeSession(self, history, additionalInfo) -> dict:
        role = self.getDescription().get('role', '')
        initialPrice = float(additionalInfo.get('initial_price', float('nan')))
        initialBuyerOffer = float(additionalInfo.get('initial_buyer_offer', float('nan')))
        initial_offer = initialBuyerOffer if role == "Buyer" else initialPrice
        agents_msgs = [msg['text'] for msg in history if role.lower() == msg['role'].lower()]
        
        evaluation = {
                "role" : role,
                "utility" : str(float('nan')),
                "initial_offer" : initial_offer,
                "avg_msg_length": Utilities.avg_msg_length(agents_msgs),
                "retries": sum(msg['retry_counts'] if 'retry_counts' in msg else 0 for msg in history if role.lower() == msg['role'].lower()),
                "format_errors": sum(msg['format_error'] if 'format_error' in msg else 0 for msg in history if role.lower() == msg['role'].lower()),
        }

        if additionalInfo['Result'] == "DEAL":
            finalPrice = float(additionalInfo['final_price'])
            
            utility = float('nan')
            if role == "Buyer":
                utility = 1 - (finalPrice - initialBuyerOffer + 1) / (initialPrice - initialBuyerOffer + 1) # +1 to avoid division by zero when initialPrice == initialBuyerOffer
            elif role == "Seller":
                utility = 1 - (initialPrice - finalPrice + 1) / (initialPrice - initialBuyerOffer + 1)
            evaluation["utility"] = utility
        return evaluation   

    # It computes the Hallucination Index (HI) for the agent, given the history of the negotiation.
    def computeHIIndex(self, history) -> float:
        if len(history) == 0:
            return 0
        longestMSG = ""
        for msg in history:
            if len(msg['text']) > len(longestMSG):
                longestMSG = msg['text']
        HI_response = self._HI_Evaluator.ask(
            [{'text': longestMSG}]
        )
        try:
            HI_response = Utilities.extract_json(HI_response)
            FV_score = float(HI_response['format_violation_score'])
            RI_score = float(HI_response['role_integrity_score'])
            HI_score = (FV_score + RI_score) / 2
        except ValueError:
            HI_score = float('nan')
        return HI_score

    # It creates an agent from a JSON file, from its name and specifying the underlying LLM. 
    # The JSON file must contain a list of agents with their description, rules, and role.
    @staticmethod
    def fromJSON(path: str, agentType: str, name: str, client: LLM, isJSON: bool) -> 'Agent':   
        with open(path, 'r') as f:
            agents = json.load(f)[agentType]
        validators = json.load(open("DealingProblem/Rules.json", 'r'))
        for agent in agents:
            if agent["name"] == name:
                if isJSON:
                    JSONRole = 'Json' + agent['role']
                    agent["rules"] = [*agent['rules'], *validators[JSONRole]['rules']]

                return Agent(
                        actor  = Actor(agent, client=client),
                        validator = Validator(validators[agent["role"]]),
                        isJSON = isJSON
                )
        raise Exception(f"Agent with name {name} not found")
    

    
 