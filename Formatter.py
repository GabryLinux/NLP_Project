from abc import ABC, abstractmethod


# Base class for formatters. 
# A formatter is an entity that, given a message, generate the correct JSON formatted message 
# for an LLM Agent. A message can be written by the user, by the model, or be a rule for the agent to follow.
# There is also a generic method to convert a message to the correct format, given the type of the message.
class Formatter(ABC):
    @abstractmethod
    def userMessage(self, message) -> dict:
        pass

    @abstractmethod
    def modelMessage(self, message) -> dict:
        pass

    @abstractmethod
    def ruleMessage(self, message) -> dict:
        pass

    @abstractmethod
    def messageToPrompt(self, message, type) -> dict:
        pass

# The Gemma formatter, used for the Gemma LLM.
class GemmaFormatter(Formatter):
    def __init__(self):
        pass

    def messageToPrompt(self, message, type) -> dict:
        if type != 'model' and type != 'user':
            raise Exception("Invalid message type for GemmaFormatter. It must be 'model' or 'user'.")
        return {"role": type, "parts": [{"text": message}]}
    
    def ruleMessage(self, message) -> dict:
        return self.messageToPrompt(message, type='user')
    
    def modelMessage(self, message) -> dict:
        return self.messageToPrompt(message, type='model')
    
    def userMessage(self, message) -> dict:
        return self.messageToPrompt(message, type='model')
    

# The LLama formatter, used for the LLama LLM.
class LLamaFormatter(Formatter):
    def __init__(self):
        pass

    def messageToPrompt(self, message, type) -> dict:
        if type != 'assistant' and type != 'user' and type != 'system':
            raise Exception("Invalid message type for LLamaFormatter. It must be 'assistant', 'user', or 'system'.")
        return {"role": type, "content": message}
    
    
    def ruleMessage(self, message) -> dict:
        return self.messageToPrompt(message, type='system')
    
    def modelMessage(self, message) -> dict:
        return self.messageToPrompt(message, type='assistant')
    
    def userMessage(self, message) -> dict:
        return self.messageToPrompt(message, type='user')
    