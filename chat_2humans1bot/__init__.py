from otree.api import *
from os import environ
from openai import AsyncOpenAI
import random
import re
import asyncio
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
    NAME_IN_URL = 'chat_2humans1bot'
    PLAYERS_PER_GROUP = 2
    NUM_ROUNDS = 1

    # emoji reactions used for chat
    ALLOW_REACTIONS = True
    EMOJIS = ['👍', '👎', '❤️',]

    # chat history on refresh
    SHOW_HISTORY = True

    # LLM vars
    ## moderator bot label and temperature

    ### temperature (range 0 - 2)
    ### this sets the bot's creativity in responses, with higher values being more creative and less deterministic
    ### https://platform.openai.com/docs/api-reference/completions#completions/create-temperature

    MOD_LABEL = 'Moderator'
    MOD_TEMP = 0.5
    MOD_MSG_FREQUENCY = 6
    
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

    ## set system prompt for moderator bot
    ## according to OpenAI's documentation, this should be less than ~1500 words
    SYS_MODERATOR = f"""You are Moderator bot, a helpful dialogue coach analyzing conversations to improve discussion quality. You will be moderating a discussion between two human participants on the merits of cognitive dissonance vs. self-perception theory. As a greeting, state that you are excited to moderate the discussion. For each intervention:

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

# moderator llm function
async def runModeratorGPT(inputDat):

    # grab bot vars from constants and inputDat
    botTemp = C.MOD_TEMP
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

# group vars (shared conversation state)
class Group(BaseGroup):

    # cache of all messages in conversation (shared by both players)
    cachedMessages = models.LongStringField(initial='[]')

    # turn tracking
    ## phase number
    phase = models.IntegerField(initial=0)
    ## message count when moderator bot last spoke
    lastModeratorBotMsg = models.IntegerField(initial=0)
    ## total message count
    messageCount = models.IntegerField(initial=0)

# player vars
class Player(BasePlayer):

    # tone for the conversation
    tone = models.StringField()

# function to check if moderator bot should speak
def can_mod_speak(group: Group) -> bool:
    msg_count = group.messageCount
    # moderator can speak if enough messages have occurred since its last message
    messages_since_last = msg_count - group.lastModeratorBotMsg
    return messages_since_last >= C.MOD_MSG_FREQUENCY

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

# wait page so both players arrive before chat begins
class ChatWaitPage(WaitPage):
    pass

# chat page 
class chat(Page):
    form_model = 'player'
    timeout_seconds = 300

    # vars that we will pass to chat.html (javascript)
    @staticmethod
    def js_vars(player):
        currentPlayer = 'P' + str(player.id_in_group)
        
        return dict(
            id_in_group=player.id_in_group,
            playerId=currentPlayer,
            allow_reactions=C.ALLOW_REACTIONS,
            emojis=C.EMOJIS,
            mod_label=C.MOD_LABEL,
        )

    # vars that we will pass to chat.html
    @staticmethod
    def vars_for_template(player):
        group = player.group
        currentPlayer = 'P' + str(player.id_in_group)
        return dict(
            cached_messages = json.loads(group.cachedMessages),
            show_history = C.SHOW_HISTORY,
            currentPlayer = currentPlayer,
            mod_label = C.MOD_LABEL,
        )

    # live method functions
    @staticmethod
    async def live_method(player: Player, data):
        group = player.group
        
        # if no new data, just return cached messages
        if not data:
            yield {player.id_in_group: dict(
                messages=json.loads(group.cachedMessages),
                reactions=[]
            )}
            return
        
        # if we have new data, process it and update cache
        messages = json.loads(group.cachedMessages)

        # create current player identifier
        currentPlayer = 'P' + str(player.id_in_group)
        modLabel = C.MOD_LABEL

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
                
                # update group cache and message count
                group.cachedMessages = json.dumps(messages)
                group.messageCount += 1
                                
                # broadcast to all players in group
                response = dict(
                    event='text',
                    text=text,
                    sender=currentPlayer,
                    msgId=msgId,
                    tone=tone,
                    phase=group.phase
                )
                yield {p.id_in_group: response for p in group.get_players()}

            # handle moderator bot messages
            elif event == 'botMsg':

                # get data from request
                isGreeting = data.get('isGreeting', False)

                # create inputDat and run api function
                inputDat = dict(
                    botLabel = modLabel,
                    messages = messages,
                    tone = tone,
                )
                
                # handle moderator greeting (first message)
                if isGreeting:
                    dateNow = str(datetime.now(tz=timezone.utc).timestamp())
                    botText = await runModeratorGPT(inputDat)
                        
                    # grab bot response data
                    outputText = botText.text
                    botMsgId = botText.msgId
                    botTone = botText.tone
                    botReactions = botText.reactions

                    # save to database
                    MessageData.create(
                        player=player,
                        sender=modLabel,
                        msgId=botMsgId,
                        timestamp=dateNow,
                        tone=botTone,
                        msgText=outputText,
                    )
                    # update cache with bot message
                    messages.append({
                        'sender': 'assistant (Moderator)',
                        'label': modLabel,
                        'msgId': botMsgId,
                        'text': outputText,
                        'reactions': json.dumps(botReactions),
                    })
                    group.cachedMessages = json.dumps(messages)
                    group.lastModeratorBotMsg = group.messageCount
                    group.messageCount += 1
                    
                    # broadcast to all players in group
                    response = dict(
                        event='botText',
                        botMsgId=botMsgId,
                        text=outputText,
                        tone=tone,
                        sender=modLabel,
                        phase=group.phase
                    )
                    yield {p.id_in_group: response for p in group.get_players()}
                
                # regular moderator messages - check if it should speak
                elif can_mod_speak(group):
                    
                    # if last message was from moderator, skip
                    if messages and messages[-1].get('label') == modLabel:
                        yield {player.id_in_group: dict()}
                        return
                    
                    # generate moderator response
                    dateNow = str(datetime.now(tz=timezone.utc).timestamp())
                    botText = await runModeratorGPT(inputDat)
                    
                    # grab bot response data
                    outputText = botText.text
                    botMsgId = botText.msgId
                    botTone = botText.tone
                    botReactions = botText.reactions

                    # save to database
                    MessageData.create(
                        player=player,
                        sender=modLabel,
                        msgId=botMsgId,
                        timestamp=dateNow,
                        tone=botTone,
                        msgText=outputText,
                    )
                    
                    # update group cache and message count
                    group.lastModeratorBotMsg = group.messageCount
                    group.messageCount += 1
                    messages.append({
                        'sender': 'assistant (Moderator)',
                        'label': modLabel,
                        'msgId': botMsgId,
                        'text': outputText,
                        'reactions': json.dumps(botReactions),
                    })
                    group.cachedMessages = json.dumps(messages)

                    # broadcast to all players in group
                    response = dict(
                        event='botText',
                        botMsgId=botMsgId,
                        text=outputText,
                        tone=tone,
                        sender=modLabel,
                        phase=group.phase
                    )
                    yield {p.id_in_group: response for p in group.get_players()}
                else:
                    # if its not moderator's turn to speak, pass
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

                # check if reaction already exists (search current player's reactions)
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
                    # search across ALL players in group for accurate counts
                    for i, msg in enumerate(messages):
                        if msg.get('msgId') == msgId:
                            reactionCounts = {e: 0 for e in C.EMOJIS}
                            countedUsers = {e: set() for e in C.EMOJIS}
                            for gp in group.get_players():
                                msgReactions = MsgReactionData.filter(player=gp, msgId=msgId)
                                for reaction in msgReactions:
                                    if reaction.sender not in countedUsers[reaction.emoji]:
                                        reactionCounts[reaction.emoji] += 1
                                        countedUsers[reaction.emoji].add(reaction.sender)
                            msg['reactions'] = json.dumps(reactionCounts)
                            break

                    # update group cache
                    group.cachedMessages = json.dumps(messages)

                    # broadcast reaction to all players in group
                    response = dict(
                        event='msgReaction',
                        playerId=currentPlayer,
                        msgId=msgId,
                        msgReactionId=msgReactionId,
                        target=trgt,
                        emoji=emoji
                    )
                    yield {p.id_in_group: response for p in group.get_players()}

            # handle phase updates
            elif event == 'phase':
                
                # update group phase and broadcast to all players
                group.phase = data.get('phase', 0)
                response = dict(
                    event='phase',
                    phase=group.phase
                )
                yield {p.id_in_group: response for p in group.get_players()}


# page sequence
page_sequence = [
    ChatWaitPage,
    chat,
]