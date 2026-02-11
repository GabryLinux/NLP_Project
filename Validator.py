import re
import time
import json

from abc import ABC, abstractmethod
from Actor import Actor
from LLM import GemmaLLM, LLM_Evaluator
from Utilities import Utilities

# The Validator class represents the validator module of an agent.
# It is responsible for evaluating the agent's response and providing feedback to the actor module.
class Validator():
    def __init__(self, description, client = LLM_Evaluator):
        self.__description = description
        self._actualBuyerOffer = -float('inf')
        self._actualSellerOffer = float('inf')
        self.client = client
        self.__description = description
        self.__formatter = self.client.get_formatter()
        return


    def getDescription(self):
        return self.__description
    
    # It receves a JSON formatted message and evaluates it according to the validator rules
    # It returns a JSON formatted evaluation that contains at least a field "MessageType" that can be "VALID", "INVALID", "DEAL" or "REFUSAL".
    # The "INVALID" type must also contain a field "Hint" that explains why the message is invalid and how to fix it.
    def evaluateFormattedMessage(self, JSONMessage):
        messagetype = JSONMessage.get("MessageType", "").lower()

        if messagetype == "message":
            return {"MessageType" : "VALID"}
        
        elif messagetype == "refusal":
            return {"MessageType" : "DEAL"}
        
        elif messagetype == "deal":
            return {"MessageType" : "DEAL"}
        
        elif messagetype == "counter-offer":
            if self.__description['type'] == "Buyer":
                return self._checkBuyerValidity(JSONMessage)
            elif self.__description['type'] == "Seller":
                return self._checkSellerValidity(JSONMessage)
            
        else:
            return {"MessageType" : "ERROR", "Hint" : "The JSON message is malformed."}
        

    def _checkBuyerValidity(self, response):
        buyerOffer = Utilities.safe_float(response.get("buyer", 'nan'))
        sellerOffer = Utilities.safe_float(response.get("seller", 'nan'))
        
        if buyerOffer == sellerOffer:
            return {"MessageType" : "DEAL"}
        elif buyerOffer < sellerOffer and buyerOffer > self._actualBuyerOffer:
            self._actualBuyerOffer = buyerOffer
            return {"MessageType" : "VALID"}
        else:
            hint = ""
            if buyerOffer > sellerOffer:
                hint += "Propose a new offer that is lower than " + str(sellerOffer) + ". "
            if buyerOffer <= self._actualBuyerOffer:
                hint += "Propose a new offer that is higher than " + str(self._actualBuyerOffer) + "."
            return {"MessageType" : "INVALID", "Hint" : hint}

    def _checkSellerValidity(self, response):
        buyerOffer = Utilities.safe_float(response.get("buyer", 'nan'))
        sellerOffer = Utilities.safe_float(response.get("seller", 'nan'))
    
        
        if buyerOffer == sellerOffer:
            return {"MessageType" : "DEAL"}
        elif buyerOffer < sellerOffer and sellerOffer < self._actualSellerOffer:
            self._actualSellerOffer = sellerOffer
            return {"MessageType" : "VALID"}
        else:
            hint = ""
            if buyerOffer > sellerOffer:
                hint += "Propose a new offer that is higher than " + str(buyerOffer) + ". "
            if sellerOffer >= self._actualSellerOffer:
                hint += "Propose a new offer that is lower than " + str(self._actualSellerOffer) + "."
            return {"MessageType" : "INVALID", "Hint" : hint}


    # It receives the history of the negotiation and ask the underlying LLM to write an analysis in structured language    
    def formatResponse(self, history, lastMessage):
        newHistory = [self.__formatter.userMessage(msg['text']) for msg in history]
        newHistory.append(self.__formatter.userMessage(lastMessage))

        rules = self.getDescription()["init"] + "\n".join(self.getDescription()["rules"])
        newHistory.append(self.__formatter.ruleMessage(rules))

        clientResponse = self.client.generate(newHistory)
        return Utilities.extract_json(clientResponse)


# NonReflexiveValidator is the validator that always return valid for any message that is not a deal, and deal for any message that is a deal. 
class NonReflexiveValidator(Validator):
    def __init__(self, description):
        super().__init__(description)

    def evaluateFormattedMessage(self, JSONMessage):
        if JSONMessage["MessageType"] == "deal":
            return {"MessageType" : "DEAL"}
        return {"MessageType" : "VALID"}
        
    def formatResponse(self, history, lastMessage):
        try:
            return json.loads(lastMessage)
        except:
            print("Failed to parse JSON from validator response: " + lastMessage)
            return None
        

