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

    # chat history on refresh
    SHOW_HISTORY = True

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

    ## reasoning level for supported models
    ## this can be set to 'none', 'minimal', 'low', 'medium', or 'high'
    REASONING_LVL = 'none'
    
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
    - their sender identifer, which shows who sent the message (string)
    - instructions for responding (string)
    - tone to use (string)
    - the conversation message history so far (string)
    - reactions that users have made to different messages (in the 'reactions' field) (string)

    IMPORTANT: This list will be the entire message history between all actors in a conversation. Messages sent by you are labeled in the 'Sender' field as your assigned label. Other actors will be labeled differently (e.g., 'P1', 'B1', etc.).
    
    Participants may have added reactions to different messages. The following reactions are possible: {', '.join(EMOJIS)}

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
    - their sender identifer, which shows who sent the message (string)
    - instructions for responding (string)
    - tone to use (string)
    - the conversation message history so far (string)
    - reactions that users have made to different messages (in the 'reactions' field) (string)

    As output, you MUST provide a json object with:
    - 'sender': your assigned sender identifier (string)
    - 'msgId': your assigned message ID (string)
    - 'tone': your assigned tone (string)
    - 'text': your response (limit to 300 characters) (string)
    - 'reactions': your assigned reactions value (string)"""


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
async def runParticipantGPT(inputDat):

    # grab bot vars from constants and inputDat
    botTemp = C.BOT_TEMP1
    botPrompt = C.SYS_PARTICIPANT
    botLabel = inputDat['botLabel']
    tone = inputDat['tone']

    # assign message id and bot label
    dateNow = str(datetime.now(tz=timezone.utc).timestamp())
    botMsgId = botLabel + '-' + str(dateNow)

    # grab text that participant inputs and format for chatgpt
    reactionsDict = {emoji: 0 for emoji in C.EMOJIS}
    instructions = f"""
        You are {botLabel}. Provide a json object with the following schema (DO NOT CHANGE ASSIGNED VALUES):
            'sender': {botLabel} (string),
            'msgId': {botMsgId} (string), 
            'tone': {tone} (string), 
            'text': Your response to the user's message in a {tone} tone (string), 
            'reactions': {json.dumps(reactionsDict)} (string)
    """

    # add instructions to inputDat
    inputDat['instructions'] = instructions

    # combine input message with assigned prompt
    inputMsg = [{'role': 'system', 'content': botPrompt}, {'role': 'user', 'content': json.dumps(inputDat)}]

    # openai client and response creation
    client = AsyncOpenAI(api_key=C.OPENAI_KEY)

    # responses api with retries in case of rate limits
    max_retries = 9
    for attempt in range(max_retries):
        try:
            response = await client.responses.parse(
                model=C.MODEL,
                input=inputMsg,
                text_format=MsgOutputSchema,
            )

            # if model supports reasoning, include, otherwise dont
            reasoning = {'effort': C.REASONING_LVL} if 'gpt-5' in C.MODEL else None
            response.reasoning = reasoning

            # if model supports temperature, include, otherwise dont
            temperature = botTemp if 'gpt-4' in C.MODEL else None
            response.temperature = temperature
            return response.output_parsed

        except Exception as e:
            if attempt < max_retries - 1:
                # exponential backoff with larger delays and jitter; honor server hint if present
                base_delay = min(64, 2 ** (attempt + 1))
                hinted = None
                try:
                    m = re.search(r"Please try again in\s+([0-9]+(?:\.[0-9]+)?)s", str(e))
                    if m:
                        hinted = float(m.group(1))
                except Exception:
                    hinted = None
                delay = max(base_delay, hinted or 0) + random.uniform(0, 1.0)
                botLabel = inputDat.get('botLabel', 'UNKNOWN')
                try:
                    print(f"[LLM][retry] runGPT for {botLabel}: {e}. Retrying in {delay:.1f}s (attempt {attempt+1}/{max_retries})")
                except Exception:
                    pass
                await asyncio.sleep(delay)
            else:
                botLabel = inputDat.get('botLabel', 'UNKNOWN')
                try:
                    print(f"[LLM][error] runGPT for {botLabel}: giving up after {max_retries} attempts. Last error: {e}")
                except Exception:
                    pass
                raise


# run moderator llm function
async def runModeratorGPT(inputDat):

    # grab bot vars from constants and inputDat
    botTemp = C.BOT_TEMP2
    botPrompt = C.SYS_MODERATOR
    botLabel = inputDat['botLabel']
    tone = inputDat['tone']

    # assign message id and bot label
    dateNow = str(datetime.now(tz=timezone.utc).timestamp())
    botMsgId = botLabel + '-' + str(dateNow)

    # grab text that participant inputs and format for chatgpt
    reactionsDict = {emoji: 0 for emoji in C.EMOJIS}
    instructions = f"""
        You are {botLabel}, the MODERATOR. You do NOT participate in the debate. Your role is to facilitate and coach the discussion. Provide a json object with the following schema (DO NOT CHANGE ASSIGNED VALUES):
            'sender': {botLabel} (string),
            'msgId': {botMsgId} (string), 
            'tone': moderator (string), 
            'text': Your moderation feedback on the recent exchange (string), 
            'reactions': {json.dumps(reactionsDict)} (string)
    """
    # add instructions to inputDat
    inputDat['instructions'] = instructions

    # combine input message with assigned prompt
    inputMsg = [{'role': 'system', 'content': botPrompt}, {'role': 'user', 'content': json.dumps(inputDat)}]

    # openai client and response creation
    client = AsyncOpenAI(api_key=C.OPENAI_KEY)

    # responses api with retries in case of rate limits
    max_retries = 9
    for attempt in range(max_retries):
        try:
            response = await client.responses.parse(
                model=C.MODEL,
                input=inputMsg,
                text_format=MsgOutputSchema,
            )

            # if model supports reasoning, include, otherwise dont
            reasoning = {'effort': C.REASONING_LVL} if 'gpt-5' in C.MODEL else None
            response.reasoning = reasoning

            # if model supports temperature, include, otherwise dont
            temperature = botTemp if 'gpt-4' in C.MODEL else None
            response.temperature = temperature
            return response.output_parsed

        except Exception as e:
            if attempt < max_retries - 1:
                # exponential backoff with larger delays and jitter; honor server hint if present
                base_delay = min(64, 2 ** (attempt + 1))
                hinted = None
                try:
                    m = re.search(r"Please try again in\s+([0-9]+(?:\.[0-9]+)?)s", str(e))
                    if m:
                        hinted = float(m.group(1))
                except Exception:
                    hinted = None
                delay = max(base_delay, hinted or 0) + random.uniform(0, 1.0)
                botLabel = inputDat.get('botLabel', 'UNKNOWN')
                try:
                    print(f"[LLM][retry] runGPT for {botLabel}: {e}. Retrying in {delay:.1f}s (attempt {attempt+1}/{max_retries})")
                except Exception:
                    pass
                await asyncio.sleep(delay)
            else:
                botLabel = inputDat.get('botLabel', 'UNKNOWN')
                try:
                    print(f"[LLM][error] runGPT for {botLabel}: giving up after {max_retries} attempts. Last error: {e}")
                except Exception:
                    pass
                raise


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

    # cache of all messages in conversation
    cachedMessages = models.LongStringField(initial='[]')

    # turn tracking 
    
    ## phase number
    phase = models.IntegerField(initial=0)
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
    ## check if bot contains MODERATOR_LABEL
    if 'M' not in bot_type:
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

    # vars that we will pass to chat.html (javascript)
    @staticmethod
    def js_vars(player):
        # playerId as seen in chat
        currentPlayer = 'P' + str(player.id_in_group)
        botLabel = 'B' + str(player.id_in_group)
        modLabel = 'M' + str(player.id_in_group)
        
        return dict(
            id_in_group=player.id_in_group,
            playerId=currentPlayer,
            allow_reactions=C.ALLOW_REACTIONS,
            emojis=C.EMOJIS,
            bot_label1=botLabel,
            bot_label2=modLabel,
        )

    # vars that we will pass to chat.html
    @staticmethod
    def vars_for_template(player):
        # playerId as seen in chat
        currentPlayer = 'P' + str(player.id_in_group)
        botLabel = 'B' + str(player.id_in_group)
        modLabel = 'M' + str(player.id_in_group)
        return dict(
            cached_messages = json.loads(player.cachedMessages),
            show_history = C.SHOW_HISTORY,
            currentPlayer = currentPlayer,
            bot_label1 = botLabel,
            bot_label2 = modLabel,
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
        botLabel = 'B' + str(player.id_in_group)
        modLabel = 'M' + str(player.id_in_group)

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
      
                # save to database
                MessageData.create(
                    player=player,
                    sender=currentPlayer,
                    msgId=msgId,
                    timestamp=dateNow,
                    tone=tone,
                    msgText=text,
                )

                
                # add message to list
                messages.append({
                    'sender': 'user',
                    'label': currentPlayer,
                    'msgId': msgId,
                    'text': text,
                    'reactions': json.dumps(reactionsDict),
                })
                
                # update cache
                player.cachedMessages = json.dumps(messages)

                # update message tracking
                player.lastUserMsg = player.messageCount
                player.messageCount += 1
                                
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
                    
                # create inputDat and run api function
                inputDat = dict(
                    botLabel = botId,
                    messages = messages,
                    tone = tone,
                )
                
                # handle bot greetings (first message)
                if isGreeting:
                    dateNow = str(datetime.now(tz=timezone.utc).timestamp())
                    
                    # run function to generate greetings
                    if botId == botLabel:
                        botText = await runParticipantGPT(inputDat)
                    else:
                        botText = await runModeratorGPT(inputDat)
                        
                    # grab bot response data
                    outputText = botText.text
                    botMsgId = botText.msgId
                    botTone = botText.tone
                    botReactions = botText.reactions

                    # save to database
                    MessageData.create(
                        player=player,
                        sender=botId,
                        msgId=botMsgId,
                        timestamp=dateNow,
                        tone=botTone,
                        msgText=outputText,
                    )
                    # update cache with bot message
                    sndr = f'assistant ({botId})' if 'M' not in botId else 'assistant (Moderator)'
                    messages.append({
                        'sender': sndr,
                        'label': botId,
                        'msgId': botMsgId,
                        'text': outputText,
                        'reactions': json.dumps(botReactions),
                    })
                    player.cachedMessages = json.dumps(messages)

                    # update player message count
                    if botId == botLabel:
                        player.lastParticipantBotMsg = player.messageCount
                    else:
                        player.lastModeratorBotMsg = player.messageCount
                    
                    # increment message count
                    player.messageCount += 1
                    
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
                    lastSender = lastMsg['sender']
                    
                    # if last message was from this bot, wait
                    if lastSender == botId:
                        yield {player.id_in_group: dict()}
                    
                    # if not, generate bot response

                    # run appropriate bot
                    dateNow = str(datetime.now(tz=timezone.utc).timestamp())
                    if botId == botLabel:
                        botText = await runParticipantGPT(inputDat)
                        player.lastParticipantBotMsg = player.messageCount
                    else:
                        botText = await runModeratorGPT(inputDat)
                        player.lastModeratorBotMsg = player.messageCount
                    
                    # grab bot response data
                    outputText = botText.text
                    botMsgId = botText.msgId
                    botTone = botText.tone
                    botReactions = botText.reactions

                    # save to database
                    MessageData.create(
                        player=player,
                        sender=botId,
                        msgId=botMsgId,
                        timestamp=dateNow,
                        tone=botTone,
                        msgText=outputText,
                    )
                    
                    # update message count and cache
                    sndr = f'assistant ({botId})' if 'M' not in botId else 'assistant (Moderator)'
                    player.messageCount += 1
                    messages.append({
                        'sender': sndr,
                        'label': botId,
                        'msgId': botMsgId,
                        'text': outputText,
                        'reactions': json.dumps(botReactions),
                    })
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
                        reactions = json.loads(msg['reactions'])
                        if msg.get('msgId') == msgId:
                            reactionCounts = {emoji: 0 for emoji in C.EMOJIS}
                            msgReactions = MsgReactionData.filter(player=player, msgId=msgId)
                            countedUsers = {emoji: set() for emoji in C.EMOJIS}
                            for reaction in msgReactions:
                                if reaction.target not in countedUsers[reaction.emoji]:
                                    reactionCounts[reaction.emoji] += 1
                                    countedUsers[reaction.emoji].add(reaction.target)
                            msg['reactions'] = json.dumps(reactionCounts)
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