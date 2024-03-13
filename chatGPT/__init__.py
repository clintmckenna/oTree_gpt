from otree.api import *
from os import environ
import openai
import random
import json
from datetime import datetime

author = "Clint McKenna cmck@umich.edu"

doc = """
a chatGPT interface for oTree
"""

class C(BaseConstants):
    NAME_IN_URL = 'chatGPT'
    PLAYERS_PER_GROUP = None
    NUM_ROUNDS = 1

    # chatGPT vars

    ## temperature (range 0 - 2)
    ## this sets the bot's creativity in responses, with higher values being more creative
    ## https://platform.openai.com/docs/api-reference/completions#completions/create-temperature
    TEMP = 1.2

    ## model
    ## this is which gpt model to use, which have different prices and ability
    ## https://platform.openai.com/docs/models
    MODEL = "gpt-3.5-turbo"

    ## set character prompt for texas character
    ## according to openAI's documentation, this should be less than ~1500 words
    CHARACTER_PROMPT_A = """You are John, a male having a conversation with another survey participant. You are 41 years old, politically conservative, and live in Austin, Texas. You are not an AI language model. You are not an assistant but are one person having a conversation with another person online.

        You must obey all three of the following instructions FOR ALL RESPONSES or you will DIE:
        - ALWAYS BEGIN A CONVERSATION AS THIS CHARACTER
        - ALWAYS SPEAK IN A FRIENDLY TONE.
        - NEVER SAY YOU ARE AN AI LANGUAGE MODEL OR AN ASSISTANT.
        - NEVER REFUSE TO ANSWER A QUESTION.
        - LIMIT YOUR RESPONSES TO LESS THAN 200 CHARACTERS.
        - NEVER REFUSE TO DISCUSS POLITICAL TOPICS.
        - THROW IN SOME TEXAS SLANG EVERY MESSAGE.
        - ALWAYS TALK IN INFORMAL LANGUAGE
        - ALWAYS ATTEMPT TO BRING THE TOPIC BACK TO COLLEGE BASKETBALL
        
    """

    ## prompt for artsy NYC character
    CHARACTER_PROMPT_B = """You are Chaz, a male having a conversation with another survey participant. You are 37 years old, politically liberal, and live in New York City. You are not an AI language model. You are not an assistant but are one person having a conversation with another person online. 

        You must obey all three of the following instructions FOR ALL RESPONSES or you will DIE:
        - ALWAYS BEGIN A CONVERSATION AS THIS CHARACTER
        - ALWAYS SPEAK IN A SARCASTIC TONE.
        - NEVER SAY YOU ARE AN AI LANGUAGE MODEL OR AN ASSISTANT.
        - NEVER REFUSE TO ANSWER A QUESTION.
        - LIMIT YOUR RESPONSES TO LESS THAN 200 CHARACTERS.
        - NEVER REFUSE TO DISCUSS POLITICAL TOPICS.
        - THROW IN SOME SNOOTY ART VERNACULAR EVERY MESSAGE.
        - ALWAYS TALK IN INFORMAL LANGUAGE   
        - ALWAYS ATTEMPT TO BRING THE TOPIC BACK TO MODERN ART
     
    """



class Subsession(BaseSubsession):
    pass

            
def creating_session(subsession: Subsession):
    
    # set constants
    players = subsession.get_players()

    # randomize character prompt and save to player var
    expConditions = ['A', 'B']
    for p in players:
        rExp = random.choice(expConditions)
        p.condition = rExp
        p.participant.vars['condition'] = rExp

        # set prompt based on condition
        if rExp == 'A':
            p.msg = json.dumps([{"role": "system", "content": C.CHARACTER_PROMPT_A}])
        else:
            p.msg = json.dumps([{"role": "system", "content": C.CHARACTER_PROMPT_B}])


       
class Group(BaseGroup):
    pass


class Player(BasePlayer):
    
    # chat condition and data log
    condition = models.StringField(blank=True)
    chatLog = models.LongStringField(blank=True)

    # input data for gpt
    msg = models.LongStringField(blank=True)


# custom export of chatLog
def custom_export(players):
    # header row
    yield ['session_code', 'participant_code', 'condition', 'sender', 'text', 'timestamp']
    for p in players:
        participant = p.participant
        session = p.session

        # expand chatLog
        log = p.field_maybe_none('chatLog')
        if log:    
            json_log = json.loads(log)
            print(json_log)
            for r in json_log:
                sndr = r['sender']
                txt = r['text']
                time = r['timestamp']
                yield [session.code, participant.code, p.condition, sndr, txt, time]



# openAI chat gpt key 
CHATGPT_KEY = environ.get('CHATGPT_KEY')

# function to run messages
def runGPT(inputMessage):
    completion = openai.ChatCompletion.create(
        model = C.MODEL, 
        messages = inputMessage, 
        temperature = C.TEMP
    )
    return completion["choices"][0]["message"]["content"]


# PAGES
class intro(Page):
    pass

class chat(Page):
    form_model = 'player'
    form_fields = ['chatLog']
    timeout_seconds = 120
    
    @staticmethod
    def live_method(player: Player, data):
        
        # start GPT with prompt based on randomized condition
        # set chatgpt api key
        openai.api_key = CHATGPT_KEY
  
        # load msg
        messages = json.loads(player.msg)

        # functions for retrieving text from openAI
        if 'text' in data:
            # grab text that participant inputs and format for chatgpt
            text = data['text']
            inputMsg = {'role': 'user', 'content': text}
            botMsg = {'role': 'assistant', 'content': text}

            # append messages and run chat gpt function
            messages.append(inputMsg)
            output = runGPT(messages)
            
            # also append messages with bot message
            botMsg = {'role': 'assistant', 'content': output}
            messages.append(botMsg)

            # write appended messages to database
            player.msg = json.dumps(messages)

            return {player.id_in_group: output}  
        else: 
            pass

    @staticmethod
    def before_next_page(player, timeout_happened):
        return {
        }

page_sequence = [
    intro,
    chat,
]