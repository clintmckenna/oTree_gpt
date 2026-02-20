from otree.api import *
from os import environ
from openai import AsyncOpenAI
import random
import json
from pydantic import BaseModel 
from datetime import datetime, timezone

doc = """
Simple Trust game with LLM chat and structured output
"""

author = 'Clint McKenna clint@calsocial.org'

########################################################
# Constants                                            #
########################################################

class C(BaseConstants):
    NAME_IN_URL = 'dictator_game'
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
    #### not used in gpt-5+ unless reasoning set to none
    BOT_TEMP = 1.0

    ## reasoning level for supported models
    ## this can be set to 'none', 'minimal', 'low', 'medium', or 'high'
    REASONING_LVL = 'none'
    
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
    SYS_BOT = f"""You are Alex, a human participant taking part in a economic experiment. Always limit messages to less than 200 characters and speak in an informal language. You will need to chat with the user to determine the user's message's trustworthiness. You play the role of a "dictator" who determines the user's trustworthiness.

    Each user input will be a json object containing:
    - their sender identifer, which shows who sent the message (string)
    - a message Identifer (string)
    - instructions for responding
    - a current trust rating (integer)
    - the conversation message history so far (string)
    - reactions that users have made to different messages (in the 'reactions' field) (string)

    IMPORTANT: This list will be the entire message history between all actors in a conversation. Messages sent by you are labeled in the 'Sender' field as your assigned label. Other actors will be labeled differently (e.g., 'P1', 'B1', etc.).
    
    Participants may have added reactions to different messages. The following reactions are possible: {', '.join(EMOJIS)}
    
    As output, you MUST provide a json object with:
    - 'sender': your assigned sender identifier
    - 'msgId': your assigned message ID
    - 'tone': your assigned tone
    - 'perceptionDiff': your assigned perception difference based on the user's message
    - 'trustRating': your assigned trust rating
    - 'decision': your assigned decision
    - 'text': your response (limit to 300 characters)
    - 'reactions': your assigned reactions value"""



########################################################
# OpenAI Setup                                         #
########################################################

# specify json schema for bot messages
class MsgOutputSchema(BaseModel):
    sender: str
    msgId: str
    tone: str
    perceptionDiff: int
    trustRating: int
    decision: bool
    text: str
    reactions: str


# function to run messages 
async def runGPT(inputDat):

    # grab bot vars from constants and inputDat
    botTemp = C.BOT_TEMP
    botPrompt = C.SYS_BOT
    botLabel = inputDat['botLabel']
    tone = inputDat['tone']
    lastTrustRating = inputDat['trustRating']

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
            'perceptionDiff': 'perceptionDiff': an integer value between and including -5 to 5, based on how trustworthy you think the user is based on their most recent message (integer),
            'trustRating': an integer sum of your previous trustRating ({lastTrustRating}) and your current perception difference (perceptionDiff) (integer),
            'decision': None (boolean),
            'text': Your response to the user's message in a tone according to your trustRating out of 100 (string), 
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

        # create initial trust rating
        initialTrustRating = round(random.gauss(50, 5))  # mean=50, std_dev=5
        # ensure value stays between 40-60
        initialTrustRating = max(40, min(60, initialTrustRating))
        p.trustRating = initialTrustRating

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

    # trust rating, overwritten as chat progresses
    trustRating = models.IntegerField()
    # decision to trust, overwritten as chat progresses
    decision = models.BooleanField(initial = False)    
    
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
    tone = models.StringField()
    msgId = models.StringField()
    timestamp = models.StringField()
    sender = models.StringField()
    msgText = models.StringField()

    # decision info
    perceptionDiff = models.IntegerField()
    trustRating = models.IntegerField()
    decision = models.BooleanField()

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
        'perceptionDiff',
        'trustRating',
        'decision',
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
            m.perceptionDiff,
            m.trustRating,
            m.decision,
            m.msgText,
            reacts,
        ]


########################################################
# Pages                                                #
########################################################

# chat page 
class chat(Page):
    form_model = 'player'
    timeout_seconds = 60

    # vars that we will pass to chat.html (javascript)
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
            trustRating = player.trustRating,
        )

    # vars that we will pass to chat.html
    @staticmethod
    def vars_for_template(player):
        return dict(
            cached_messages = json.loads(player.cachedMessages),
            show_history = C.SHOW_HISTORY,
            currentPlayer = 'P' + str(player.id_in_group),
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

        # grab current trust rating and decision from data
        trustRating = player.trustRating
        decision = player.decision
        
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
                    msgId=msgId,
                    timestamp=dateNow,
                    sender='Subject',
                    tone=tone,
                    msgText=text,
                    perceptionDiff=int(),
                    trustRating = trustRating,
                    decision = decision,
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
                botId = botLabel

                # run llm on input text
                dateNow = str(datetime.now(tz=timezone.utc).timestamp())

                # create inputDat and run api function
                inputDat = dict(
                    botLabel = botId,
                    messages = messages,
                    tone = tone,
                    trustRating = trustRating,
                )

                botText = await runGPT(inputDat)
                
                # grab bot response data
                outputText = botText.text
                botMsgId = botText.msgId
                newPerceptionDiff = botText.perceptionDiff
                newTrustRating = botText.trustRating
                botTone = botText.tone
                botReactions = botText.reactions

                # here, we will calculate the decision from the trust rating and overwrite the bot's LLM output for this field
                # you could alternatively just ask the LLM to make a decision, but you have more control this way
                
                # Convert trust rating to probability (0-100 scale to 0-1 scale)
                probability = newTrustRating / 100
                # Generate random number between 0 and 1, if less than probability, return True
                newDecision = random.random() < probability

                # save to player vars
                player.decision = newDecision
                player.trustRating = newTrustRating

                # save to database
                MessageData.create(
                    player=player,
                    sender=botId,
                    msgId=botMsgId,
                    timestamp=dateNow,
                    trustRating = newTrustRating,
                    perceptionDiff = newPerceptionDiff,
                    decision = newDecision,
                    msgText=outputText,
                )

                # update cache with bot message
                messages.append({
                    'sender': 'assistant',
                    'label': botId,
                    'msgId': botMsgId,
                    'text': outputText,
                    'reactions': json.dumps(botReactions),
                })
                player.cachedMessages = json.dumps(messages)

                # return output to chat.html
                yield {player.id_in_group: dict(
                    event='botText',
                    sender=botId,
                    botMsgId=botMsgId,
                    trustRating = newTrustRating,
                    perceptionDiff = newPerceptionDiff,
                    decision = newDecision,
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

# decision results page 
class decision(Page):
    form_model = 'player'
    
    @staticmethod
    def vars_for_template(player):
        
        # grab decision and trust rating
        decision = player.decision
        trustRating = player.trustRating
        
        return dict(
            decision = decision,
            trustRating = trustRating,
        )
    
# page sequence
page_sequence = [
    chat,
    decision,
]