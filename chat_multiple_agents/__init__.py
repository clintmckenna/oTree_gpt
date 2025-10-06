from otree.api import *
from os import environ
from openai import AsyncOpenAI
import random
import json
from pydantic import BaseModel 
from datetime import datetime, timezone

doc = """
LLM chat with multiple agents, based on chat_complex
"""

author = 'Clint McKenna clint@calsocial.org'

########################################################
# Constants                                            #
########################################################

class C(BaseConstants):
    NAME_IN_URL = 'chat_multiple_agents'
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

    ### pariticpant bot
    BOT_LABEL1 = 'Participant Bot'
    BOT_TEMP1 = 1
    ### moderator bot
    BOT_LABEL2 = 'Moderator Bot'
    BOT_TEMP2 = 0.5
    BOT_MSG_FREQUENCY = 6
    
    ## openAI key
    OPENAI_KEY = environ.get('OPENAI_KEY')

    ## model
    ## this is which gpt model to use, which have different prices and ability
    ## https://platform.openai.com/docs/models
    ## IMPORTANT: for this app, you must use a model that supports structured output
    MODEL = "gpt-4o-mini"

    ## set system prompt for agents
    ## according to OpenAI's documentation, this should be less than ~1500 words
    ## set system prompt for participant
    SYS_PARTICIPANT = f"""You are Alex, a human PhD student in social psychology who is taking part in an online discussion. Always speak in an informal language. Your job is to debate the user on the merits of cognitive dissonance vs. self-perception theory. You will favor cognitive dissonance over self-perception theory in all cases. In EVERY response except your greeting, be sure to include a citation from the field of social psychology to support your point. As a greeting, just state that you are excited to begin the discussion.
    
    You will also be moderated by another moderator agent. Pay attention to the messages from both your debate partner and the moderator.

    Each user input will be a nested list of json objects containing:
    - their sender identifer, which shows who sent the message
    - instructions for responding
    - tone to use
    - text you will be responding to
    - reactions that users have made to different messages (in the 'reactions' field)

    IMPORTANT: This list will be the entire message history between all actors in a conversation. Messages sent by you are labeled in the 'Sender' field as {BOT_LABEL1}. Other actors will be labeled differently (e.g., 'P1', 'B1', etc.).
    
    IMPORTANT: You must actively monitor and acknowledge reactions to messages. The following reactions are possible: {', '.join(EMOJIS)}
    When you see any of these reactions in the json, incorporate them naturally into your responses.
    
    As output, you MUST provide a json object with:
    - 'sender': your assigned sender identifier
    - 'msgId': your assigned message ID
    - 'tone': your assigned tone
    - 'text': your response (limit to 300 characters)
    - 'reactions': your assigned reactions value"""

    SYS_MODERATOR = f"""You are Moderator bot, a helpful dialogue coach analyzing conversations to improve discussion quality. You will be moderating a PhD-level debate between two participants on the merits of cognitive dissonance vs. self-perception theory. As a greeting, state that you are excited to moderate the discussion. For each intervention:

    1. Analyze the dialogue for:
        - Communication clarity and engagement
        - Quality of reasoning and tone
        - Areas needing deeper exploration

    2. Provide ONE specific, actionable suggestion using phrases like:
        - "Consider exploring..."
        - "It might help to clarify..."
        - "The discussion could benefit from..."
    
    Each user input will be a list of json objects containing:
    - their sender identifer, which shows who sent the message
    - instructions for responding
    - tone to use
    - text you will be responding to
    - reactions that users have made to different messages (in the 'reactions' field)

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
## when triggered, this function will run the system prompt and the user message, which will contain the entire message history, rather than building on dialogue one line at a time

# participant bot llm function
async def runParticipantGPT(inputMessage, tone):

    # grab bot vars from constants
    botTemp = C.BOT_TEMP1
    botLabel = C.BOT_LABEL1
    botPrompt = C.SYS_PARTICIPANT

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


    # create input message with a nested structure
    nestedInput = [{'role': 'user', 'content': json.dumps(inputMessage)}]
    
    # combine input message with assigned prompt
    inputMsg = [{'role': 'system', 'content': botPrompt}] + nestedInput


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


# run moderator llm function
async def runModeratorGPT(inputMessage):

    # grab bot vars from constants
    botTemp = C.BOT_TEMP2
    botLabel = C.BOT_LABEL2
    botPrompt = C.SYS_MODERATOR

    # assign message id and bot label
    dateNow = str(datetime.now(tz=timezone.utc).timestamp())
    botMsgId = botLabel + '-' + str(dateNow)

    # grab text that participant inputs and format for chatgpt
    reactionsDict = {emoji: 0 for emoji in C.EMOJIS}
    instructions = f"""
        Provide a json object with the following schema (DO NOT CHANGE ASSIGNED VALUES):
            'sender': {botLabel} (string),
            'msgId': {botMsgId} (string), 
            'tone': '' (string), 
            'text': Your response to the messages since your last message (string), 
            'reactions': {json.dumps(reactionsDict)} (string)
    """

    # overwrite instructions for each dictionary
    for x in inputMessage:
        x['instructions'] = json.dumps(instructions)


    # create input message with a nested structure
    nestedInput = [{'role': 'user', 'content': json.dumps(inputMessage)}]

    # combine input message with assigned prompt
    inputMsg = [{'role': 'system', 'content': botPrompt}] + nestedInput


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
        tones = ['neutral', ]
        tone = random.choice(tones)
        p.tone = tone

# group vars
class Group(BaseGroup):
    pass    

# player vars
class Player(BasePlayer):

    # tone for the bot
    tone = models.StringField()

    # phase number
    phase = models.IntegerField(initial=0)

    # cache of all messages in conversation
    cachedMessages = models.LongStringField(initial='[]')

    # turn tracking 
    ## message count when participant bot last spoke
    lastParticipantBotMsg = models.IntegerField(initial=0)  
    ## message count when moderator bot last spoke
    lastModeratorBotMsg = models.IntegerField(initial=0)   
    ## message count when user last spoke
    lastUserMsg = models.IntegerField(initial=0)           
    ## total message count
    messageCount = models.IntegerField(initial=0)          

# function to check if it's a bot's turn to speak
def can_bot_speak(player: Player, bot_type: str) -> bool:
    
    # grab message count
    msg_count = player.messageCount
    
    # during initialization (first 2 messages), both bots should speak
    if msg_count < 2:
        return True
        
    # participant bot
    if bot_type == C.BOT_LABEL1:  
        # can speak if user has spoken since its last message
        messages_since_last = msg_count - player.lastParticipantBotMsg
        return player.lastUserMsg > player.lastParticipantBotMsg
    # moderator bot
    else:  
        # can speak if at least 6 messages have occurred since its last message
        messages_since_last = msg_count - player.lastModeratorBotMsg
        # print('=============================================')
        # print(f'Messages since last moderator: {messages_since_last}')
        # print('Current message count:', msg_count)
        # print('Last moderator message:', player.lastModeratorBotMsg)
        # print('=============================================')
        return messages_since_last >= C.BOT_MSG_FREQUENCY

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
            # you will have to expand it afterwards
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
        
        return dict(
            id_in_group=player.id_in_group,
            playerId=currentPlayer,
            allow_reactions=C.ALLOW_REACTIONS,
            emojis=C.EMOJIS,
            bot_label1=C.BOT_LABEL1,
            bot_label2=C.BOT_LABEL2,
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
                
                # get data from request
                text = data.get('text', '')
                currentPlayer = 'P' + str(player.id_in_group)
                messages = json.loads(player.cachedMessages)
                
                # create message id
                dateNow = str(datetime.now(tz=timezone.utc).timestamp())
                msgId = currentPlayer + '-' + dateNow
                
                # create message
                msg = {'role': 'user', 'content': json.dumps({
                    'sender': currentPlayer,
                    'msgId': msgId,
                    'tone': tone,
                    'text': text,
                    'reactions': json.dumps({emoji: 0 for emoji in C.EMOJIS})
                })}
                
                # save to database
                MessageData.create(
                    player=player,
                    sender=currentPlayer,
                    msgId=msgId,
                    timestamp=dateNow,
                    tone=tone,
                    fullText=json.dumps(msg),
                    msgText=text,
                )
                
                # update message tracking
                player.lastUserMsg = player.messageCount
                player.messageCount += 1
                
                # append to messages
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
                    phase=player.phase
                )}

            # handle bot messages
            elif event == 'botMsg':

                # get data from request
                botId = data.get('botId')
                isGreeting = data.get('isGreeting', False)
                
                # Skip if no botId provided
                if not botId:  
                    yield {player.id_in_group: dict()}
                    
                # get messages
                messages = json.loads(player.cachedMessages)
                
                # handle bot greetings (first message)
                if isGreeting:
                    dateNow = str(datetime.now(tz=timezone.utc).timestamp())
                    
                    # run function to generate greetings
                    if botId == C.BOT_LABEL1:
                        botText = await runParticipantGPT(messages, tone)
                    else:
                        botText = await runModeratorGPT(messages)
                        
                    # extract output
                    botContent = json.loads(botText)
                    outputText = botContent['text']
                    botMsgId = botContent['msgId']
                    botMsg = {'role': 'assistant', 'content': botText}
                    
                    MessageData.create(
                        player=player,
                        sender=botId,
                        msgId=botMsgId,
                        timestamp=dateNow,
                        tone=tone,
                        fullText=json.dumps(botMsg),
                        msgText=outputText,
                    )
                    messages.append(botMsg)
                    
                    # update player message count
                    if botId == C.BOT_LABEL1:
                        player.lastParticipantBotMsg = player.messageCount
                    else:
                        player.lastModeratorBotMsg = player.messageCount
                    
                    # increment message count
                    player.messageCount += 1
                    player.cachedMessages = json.dumps(messages)
                    
                    # return data to chat.html
                    yield {player.id_in_group: dict(
                        event='botText',
                        botMsgId=botMsgId,
                        text=outputText,
                        tone=tone,
                        sender=botId,
                        phase=player.phase
                    )}
                
                # regular bot messages - check if bot can speak
                if can_bot_speak(player, botId):
                    
                    # get last message
                    lastMsg = messages[-1]
                    lastMsg = json.loads(lastMsg['content'])
                    lastSender = lastMsg['sender']
                    
                    # if last message was from this bot, wait
                    if lastSender == botId:
                        yield {player.id_in_group: dict()}
                    
                    # if not, generate bot response

                    # run appropriate bot
                    if botId == C.BOT_LABEL1:
                        botText = await runParticipantGPT(messages, tone)
                        player.lastParticipantBotMsg = player.messageCount
                    else:
                        botText = await runModeratorGPT(messages)
                        player.lastModeratorBotMsg = player.messageCount
                    
                    # process bot response
                    botContent = json.loads(botText)
                    outputText = botContent['text']
                    botMsgId = botContent['msgId']
                    botMsg = {'role': 'assistant', 'content': botText}
                    
                    # save to database
                    MessageData.create(
                        player=player,
                        sender=botId,
                        msgId=botMsgId,
                        tone=tone,
                        fullText=json.dumps(botMsg),
                        msgText=outputText,
                    )
                    
                    # update message count and cache
                    player.messageCount += 1
                    messages.append(botMsg)
                    player.cachedMessages = json.dumps(messages)
                    
                    # return data to chat.html
                    yield {player.id_in_group: dict(
                        event='botText',
                        botMsgId=botMsgId,
                        text=outputText,
                        tone=tone,
                        sender=botId,
                        phase=player.phase
                    )}
                else:
                    # if its not bot's turn to speak, pass
                    # print(f'Not {botId} turn to speak yet...')
                    
                    yield {player.id_in_group: dict()}
            
            # handle reaction logic
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

            # handle phase updates
            elif event == 'phase':
                
                # update phase
                player.phase = data.get('phase', 0)
                yield {player.id_in_group: dict(
                    event='phase',
                    phase=player.phase
                )}


# page sequence
page_sequence = [
    chat,
]