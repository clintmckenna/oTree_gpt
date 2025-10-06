from otree.api import *
from os import environ
from openai import AsyncOpenAI
import random
import json
from pydantic import BaseModel 
from datetime import datetime, timezone

doc = """
LLM chat with reactions and structured output
"""

author = 'Clint McKenna clint@calsocial.org'

########################################################
# Constants                                            #
########################################################

class C(BaseConstants):
    NAME_IN_URL = 'chat_complex'
    PLAYERS_PER_GROUP = None
    NUM_ROUNDS = 1

    # emoji reactions used for chat
    ALLOW_REACTIONS = True
    EMOJIS = ['👍', '👎', '❤️',]

    # LLM vars
    ## bot label and temperature

    ### temperature (range 0 - 2)
    ### this sets the bot's creativity in responses, with higher values being more creative and less deterministic
    ### https://platform.openai.com/docs/api-reference/completions#completions/create-temperature
    #### moved this to function input

    ### pariticpant bot info
    BOT_LABEL = 'Bot'
    BOT_TEMP = 1.0
    
    ## openAI key
    OPENAI_KEY = environ.get('OPENAI_KEY')

    ## model
    ## this is which gpt model to use, which have different prices and ability
    ## https://platform.openai.com/docs/models
    ## IMPORTANT: for this app, you must use a model that supports structured output
    MODEL = "gpt-4o-mini"

    ## set system prompt for agents
    ## according to OpenAI's documentation, this should be less than ~1500 words
    SYS_BOT = f"""You are Alex, a human participant taking part in an online discussion. Always limit messages to less than 200 characters and speak in an informal language. 

    Each user input will be a list of json objects containing:
    - their sender identifer, which shows who sent the message (string)
    - message Identifer (string)
    - instructions for responding (string)
    - tone to use (string)
    - text you will be responding to (string)
    - reactions that users have made to different messages (in the 'reactions' field) (string)

    IMPORTANT: This list will be the entire message history between all actors in a conversation. Messages sent by you are labeled in the 'Sender' field as {BOT_LABEL}. Other actors will be labeled differently (e.g., 'P1', 'B1', etc.).
    
    You must actively monitor and acknowledge reactions to messages. The following reactions are possible: {', '.join(EMOJIS)}
    When you see any of these reactions in the json, incorporate them naturally into your responses.
    
    As output, you MUST provide a json object with:
    - 'sender': your assigned sender identifier
    - 'msgId': your assigned message ID
    - 'tone': your assigned tone
    - 'text': your response (limit to 300 characters)
    - 'reactions': your assigned reactions value"""


########################################################
# LLM Setup                                            #
########################################################

# specify json schema for bot messages
class MsgOutputSchema(BaseModel):
    sender: str
    msgId: str
    tone: str
    text: str
    reactions: str

# function to run messages 
async def runGPT(inputMessage, tone):

    # grab bot vars from constants
    botTemp = C.BOT_TEMP
    botLabel = C.BOT_LABEL
    botPrompt = C.SYS_BOT
    
    # assign message id and bot label
    dateNow = str(datetime.now(tz=timezone.utc).timestamp())
    botMsgId = botLabel + '-' + str(dateNow)

    # grab text that participant inputs and format for chatgpt
    reactionsDict = {emoji: 0 for emoji in C.EMOJIS}
    instructions = f"""
        Provide a json object with the following schema (DO NOT CHANGE ASSIGNED VALUES):
            'sender': {botLabel} (string),
            'msgId': {botMsgId} (string), 
            'tone': {tone} (string), 
            'text': Your response to the user's message in a {tone} tone (string), 
            'reactions': {json.dumps(reactionsDict)} (string)
    """

    # overwrite instructions for each dictionary
    for x in inputMessage:
        x['instructions'] = json.dumps(instructions)

    # combine input message with assigned prompt
    inputMsg = [{'role': 'system', 'content': botPrompt}] + inputMessage

    # openai client and response creation
    client = AsyncOpenAI(api_key=C.OPENAI_KEY)
    response = await client.chat.completions.create(
        model=C.MODEL,
        temperature=botTemp,
        messages=inputMsg,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "msg_output_schema",
                "schema": MsgOutputSchema.model_json_schema(),
            }
        }
    )

    # grab text output
    msgOutput = response.choices[0].message.content

    # return the response json
    return msgOutput


########################################################
# Models                                               #
########################################################

# subsession vars
class Subsession(BaseSubsession):
    pass

# creating session functions
def creating_session(subsession: Subsession):
    
    # grab players in session
    players = subsession.get_players()

    # iterate through players
    for p in players:

        # randomize tone for the conversation
        # tones = ['friendly', 'sarcastic', 'UNHINGED']
        tones = ['friendly', ]
        tone = random.choice(tones)
        p.tone = tone

# group vars
class Group(BaseGroup):
    pass    

# player vars
class Player(BasePlayer):

    # tone for the bot
    tone = models.StringField()

    # cache of all messages in conversation
    cachedMessages = models.LongStringField(initial='[]')

########################################################
# Extra models                                         #
########################################################

# message information
class MessageData(ExtraModel):
    # data links
    player = models.Link(Player)

    # msg info
    msgId = models.StringField()
    timestamp = models.StringField()
    sender = models.StringField()
    tone = models.StringField()
    fullText = models.StringField()
    msgText = models.StringField()

# message reaction information
class MsgReactionData(ExtraModel):
    # data links
    player = models.Link(Player)

    # reaction info
    msgId = models.StringField()
    msgReactionId = models.StringField()
    timestamp = models.StringField()
    sender = models.StringField()
    target = models.StringField()
    emoji = models.StringField()
    

########################################################
# Custom export                                        #
########################################################

# custom export of chatLog
def custom_export(players):
    # header row
    yield [
        'sessionId', 
        'subjectId',
        'msgId',
        'timestamp',
        'sender',
        'tone',
        'fullText',
        'msgText',
        'reactionData'
    ]

    # get MessageData model
    mData = MessageData.filter()
    for m in mData:

        # get player info
        player = m.player
        participant = player.participant
        session = player.session

        # full text field
        try:
            fullText = json.loads(m.fullText)['content']
        except:
            fullText = m.fullText

        # get message reaction info as well
        try:
            mReactions = MsgReactionData.filter(player=player, msgId=m.msgId)
            reaction_list = [
                {
                    'sender': r.sender,
                    'msgReactionId': r.msgReactionId,
                    'timestamp': r.timestamp,
                    'target': r.target,
                    'emoji': r.emoji,
                } for r in mReactions
            ]
            # save as a json dictionary to column
            # you will have to unnest it afterwards since I don't think you can have multiple exports
            reacts = json.dumps(reaction_list)
        except:
            reacts = '[]'
    

        # write to csv
        yield [
            session.code,
            participant.code,
            m.msgId,
            m.timestamp,
            m.sender,
            m.tone,
            fullText,
            m.msgText,
            reacts,
        ]


########################################################
# Pages                                                #
########################################################

# chat page 
class chat(Page):
    form_model = 'player'
    timeout_seconds = 300

    # vars that we will pass to chat.html
    @staticmethod
    def js_vars(player):

        # playerId as seen in chat
        currentPlayer = 'P' + str(player.id_in_group)
        
        # send player info and emojis to page
        return dict(
            id_in_group = player.id_in_group,
            playerId = currentPlayer,
            emojis = C.EMOJIS,
            allow_reactions = C.ALLOW_REACTIONS,
        )

    # live method functions
    @staticmethod
    async def live_method(player: Player, data):
        
        # if no new data, just return cached messages
        if not data:
            yield {player.id_in_group: dict(
                messages=json.loads(player.cachedMessages),
                reactions=[]
            )}
        
        # if we have new data, process it and update cache
        messages = json.loads(player.cachedMessages)

        # create current player identifier
        currentPlayer = 'P' + str(player.id_in_group)

        # grab tone from data
        tone = player.tone
        
        # handle different event types
        if 'event' in data:

            # grab event type
            event = data['event']
            
            # handle player input logic
            if event == 'text':
                
                # create message id
                dateNow = str(datetime.now(tz=timezone.utc).timestamp())
                msgId = currentPlayer + '-' + str(dateNow)
                
                # grab text and phase info
                text = data['text']

                # create message content with reactions and save to database
                reactionsDict = {emoji: 0 for emoji in C.EMOJIS}
                content = dict(
                    sender=currentPlayer,
                    msgId=msgId,
                    instructions='',
                    tone=tone,
                    text=text,
                    reactions=json.dumps(reactionsDict),
                )
                
                # create message in LLM format
                msg = {'role': 'user', 'content': json.dumps(content)}

                # save to database
                MessageData.create(
                    player=player,
                    msgId=msgId,
                    timestamp=dateNow,
                    sender='Subject',
                    tone=tone,
                    fullText=json.dumps(msg),
                    msgText=text,
                )

                # add message to list
                messages.append(msg)
                
                # update cache
                player.cachedMessages = json.dumps(messages)
                
                # return output to chat.html
                yield {player.id_in_group: dict(
                    event='text',
                    selfText=text,
                    sender=currentPlayer,
                    msgId=msgId,
                    tone=tone,
                )}

            # handle bot messages
            elif event == 'botMsg':

                # grab constants bot info
                botId = C.BOT_LABEL

                # run llm on input text
                dateNow = str(datetime.now(tz=timezone.utc).timestamp())
                botText = await runGPT(messages, tone)
                
                # grab bot response data
                botContent = json.loads(botText)
                outputText = botContent['text']
                botMsgId = botContent['msgId']
                
                # create bot message
                botMsg = {'role': 'assistant', 'content': botText}
                
                # save to database
                MessageData.create(
                    player=player,
                    sender=botId,
                    msgId=botMsgId,
                    timestamp=dateNow,
                    tone=tone,
                    fullText=json.dumps(botMsg),
                    msgText=outputText,
                )

                # update cache with bot message
                messages.append(botMsg)
                player.cachedMessages = json.dumps(messages)

                # return output to chat.html
                yield {player.id_in_group: dict(
                    event='botText',
                    sender=botId,
                    botMsgId=botMsgId,
                    tone=tone,
                    text=outputText,
                )}


            # handle emoji reaction logic
            elif event == 'reaction':

                # create reaction id
                dateNow = str(datetime.now(tz=timezone.utc).timestamp())
                msgReactionId = currentPlayer + '-' + str(dateNow)
                
                # grab data
                msgId = data['msgId']
                trgt = data['target']
                emoji = data['emoji']

                # check if reaction already exists
                existingReactions = MsgReactionData.filter(
                    player=player,
                    msgId=msgId,
                    sender=currentPlayer,
                    emoji=emoji
                )
                
                # create new reaction in database if not existing
                if not existingReactions:
                    MsgReactionData.create(
                        player=player,
                        sender=currentPlayer,
                        msgId=msgId,
                        msgReactionId=msgReactionId,
                        timestamp=dateNow,
                        target=trgt,
                        emoji=emoji,
                    )

                    # update reaction counts in message cache
                    # this function looks through the database to make sure that players can only react once for each emoji/message
                    for i, msg in enumerate(messages):
                        content = json.loads(msg['content'])
                        if content.get('msgId') == msgId:
                            reactionCounts = {emoji: 0 for emoji in C.EMOJIS}
                            msgReactions = MsgReactionData.filter(player=player, msgId=msgId)
                            countedUsers = {emoji: set() for emoji in C.EMOJIS}
                            for reaction in msgReactions:
                                if reaction.target not in countedUsers[reaction.emoji]:
                                    reactionCounts[reaction.emoji] += 1
                                    countedUsers[reaction.emoji].add(reaction.target)
                            content['reactions'] = json.dumps(reactionCounts)
                            messages[i]['content'] = json.dumps(content)
                            break

                    # update cache
                    player.cachedMessages = json.dumps(messages)

                    # return output to chat.html
                    yield {player.id_in_group: dict(
                        event='msgReaction',
                        playerId=currentPlayer,
                        msgId=msgId,
                        msgReactionId=msgReactionId,
                        target=trgt,
                        emoji=emoji
                    )}

            


# page sequence
page_sequence = [
    chat,
]