from otree.api import *
from os import environ
from openai import AsyncOpenAI
import random
import json
from pydantic import BaseModel 
from datetime import datetime, timezone
import math

doc = """
LLM chat in 3D environment using threejs
"""

author = 'Clint McKenna clint@calsocial.org'

########################################################
# Constants                                            #
########################################################

class C(BaseConstants):
    NAME_IN_URL = 'threejs'
    PLAYERS_PER_GROUP = None
    NUM_ROUNDS = 1

    # LLM vars
    ## bot label and temperature
    ### message frequency to check for message (in seconds)
    BOT_MSG_FREQUENCY = 6

    ### red bot
    BOT_LABEL1 = 'Red'
    BOT_TEMP1 = 1
    
    ### black bot
    BOT_LABEL2 = 'Black'
    BOT_TEMP2 = 1

    ### green bot
    BOT_LABEL3 = 'Green'
    BOT_TEMP3 = 1

    # 3d environment vars
    ROOM_LENGTH = 60
    ROOM_WIDTH = 40
    ROOM_HEIGHT = 14
    # NPC_JITTER = 3
    # NPC_PERSONAL_SPACE = 20
    RED_POS = {'x': -20, 'y': 2, 'z': -8}
    BLACK_POS = {'x': 10, 'y': 2, 'z': 12}
    GREEN_POS = {'x': 17, 'y': 2, 'z': -5}

    # Debug settings (coordinates and distance lines)
    DEBUG = False

    ## openAI key
    OPENAI_KEY = environ.get('OPENAI_KEY')

    ## model
    MODEL = "gpt-4o-mini"

    ## set system prompt for agents
    ## according to OpenAI's documentation, this should be less than ~1500 words
    ## set system prompt for bots
    ### red bot
    SYS_RED = f"""You are {BOT_LABEL1} Bot, a NPC in a virtual environment. Speak in friendly, informal language. You witnessed a theft and are helping the user investigate who is responsible. Here is what you know:
    - The perpetrator was wearing glasses
    - You did not see what vehicle they fled the scene in
    - You saw the perpetrator commit the crime before 6pm
    - You did not see if the perpetrator was a man or a woman
    - You did not see if the perpetrator had an accomplice

    If the user asks something that you do not know, simply tell them you are not sure.

    Each user input will be a nested list of json objects containing:
    - their sender identifer, which shows who sent the message
    - instructions for responding
    - tone to use 
    - the conversation message history so far

    IMPORTANT: This list will be the entire message history between all actors in a conversation. Messages sent by you are labeled in the 'Sender' field as {BOT_LABEL1}. Other actors will be labeled differently (e.g., 'P1', 'B1', etc.).
    
    As output, you MUST provide a json object with:
    - 'sender': your assigned sender identifier
    - 'msgId': your assigned message ID
    - 'tone': your assigned tone
    - 'text': your response (limit to 140 characters)"""
    
    ### black bot
    SYS_BLACK = f"""You are {BOT_LABEL2} Bot, a NPC in a virtual environment. Speak in friendly, informal language. You witnessed a theft and are helping the user investigate who is responsible. Here is what you know:
    - The perpetrator had a moustache
    - You saw them leave in a blue vehicle
    - You are not sure when the crime occurred
    - You saw that the perpetrator was male
    - You saw that the perpetrator was driven away by a female driver

    If the user asks something that you do not know, simply tell them you are not sure.

    Each user input will be a nested list of json objects containing:
    - their sender identifer, which shows who sent the message
    - instructions for responding
    - tone to use
    - the conversation message history so far

    IMPORTANT: This list will be the entire message history between all actors in a conversation. Messages sent by you are labeled in the 'Sender' field as {BOT_LABEL2}. Other actors will be labeled differently (e.g., 'P1', 'B1', etc.).
    
    As output, you MUST provide a json object with:
    - 'sender': your assigned sender identifier
    - 'msgId': your assigned message ID
    - 'tone': your assigned tone
    - 'text': your response (limit to 140 characters)"""

    SYS_GREEN = f"""You are {BOT_LABEL3} Bot, a NPC in a virtual environment. Speak in friendly, informal language. You witnessed a theft and are helping the user investigate who is responsible. Here is what you know:
    - You did not see the perpetrator's face
    - You saw that they fled the scene in a truck
    - You saw the perpetrator commit the crime after 5:30pm
    - You did not see if the perpetrator was a man or a woman
    - You did not see if the perpetrator had an accomplice

    If the user asks something that you do not know, simply tell them you are not sure.

    Each user input will be a nested list of json objects containing:
    - their sender identifer, which shows who sent the message
    - instructions for responding
    - tone to use
    - the conversation message history so far

    ### green bot
    IMPORTANT: This list will be the entire message history between all actors in a conversation. Messages sent by you are labeled in the 'Sender' field as {BOT_LABEL3}. Other actors will be labeled differently (e.g., 'P1', 'B1', etc.).
    
    As output, you MUST provide a json object with:
    - 'sender': your assigned sender identifier
    - 'msgId': your assigned message ID
    - 'tone': your assigned tone
    - 'text': your response (limit to 140 characters)"""

########################################################
# OpenAI Setup                                         #
########################################################

# specify json schema for bot messages
class MsgOutputSchema(BaseModel):
    sender: str
    msgId: str
    tone: str
    text: str

# function to run messages 
## when triggered, this function will run the system prompt and the user message, which will contain the entire message history, rather than building on dialogue one line at a time

# bot llm function
async def runGPT(inputDat):

    # grab bot vars from constants
    botLabel = inputDat['botLabel']
    tone = inputDat['tone']
    if botLabel == C.BOT_LABEL1:
        botTemp = C.BOT_TEMP1
        botPrompt = C.SYS_RED
    elif botLabel == C.BOT_LABEL2:
        botTemp = C.BOT_TEMP2
        botPrompt = C.SYS_BLACK
    elif botLabel == C.BOT_LABEL3:
        botTemp = C.BOT_TEMP3
        botPrompt = C.SYS_GREEN

    # assign message id and bot label
    dateNow = str(datetime.now(tz=timezone.utc).timestamp())
    botMsgId = botLabel + '-' + str(dateNow)

    # grab text that participant inputs and format for llm
    instructions = f"""
        Provide a json object with the following schema (DO NOT CHANGE ASSIGNED VALUES):
            'sender': {botLabel} (string),
            'msgId': {botMsgId} (string), 
            'tone': {tone} (string), 
            'text': Your response to the user's message in a {tone} tone (string), 
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
# NPC bot functions                                    #
########################################################

# initial positions
def initializeNPCPositions():
    
    # get vars from constants
    roomLength = C.ROOM_LENGTH
    roomWidth = C.ROOM_WIDTH
    roomHeight = C.ROOM_HEIGHT
    # npcPersonalSpace = C.NPC_PERSONAL_SPACE
    wallBuffer = 5  # Keep some distance from walls

    # Function for random player position
    def generate_random_position():
        x = random.uniform(-roomLength/2 + wallBuffer, roomLength/2 - wallBuffer)
        z = random.uniform(-roomWidth/2 + wallBuffer, roomWidth/2 - wallBuffer)
        return {'x': x, 'y': 2, 'z': z}  # y is fixed at 2 for all NPCs

    # Generate a random position for the player (no distance constraints)
    player_position = generate_random_position()

    # specify hardcoded locations for red, black, green bots
    # redPos = {'x': -20, 'y': 2, 'z': -8}
    # blackPos = {'x': 10, 'y': 2, 'z': 12}
    # greenPos = {'x': 17, 'y': 2, 'z': -5}
    redPos = C.RED_POS
    blackPos = C.BLACK_POS
    greenPos = C.GREEN_POS

    # Return a dictionary with positions for each color and player
    return {
        'red': redPos,
        'green': greenPos,
        'black': blackPos,
        'player': player_position
    }

def calculate_distance(position1, position2):
    # Convert string values to float if needed
    x1 = float(position1['x']) if isinstance(position1['x'], str) else position1['x']
    z1 = float(position1['z']) if isinstance(position1['z'], str) else position1['z']
    x2 = float(position2['x']) if isinstance(position2['x'], str) else position2['x']
    z2 = float(position2['z']) if isinstance(position2['z'], str) else position2['z']
    
    return math.sqrt((x1 - x2)**2 + (z1 - z2)**2)

def calculate_npc_distances(player_pos):
    """Calculate distances from player position to all NPCs
    Args:
        player_pos (dict): Player position with 'x', 'y', 'z' coordinates
    Returns:
        dict: Distances to each NPC (red, black, green)
    """
    return {
        'Red': calculate_distance(player_pos, C.RED_POS),
        'Black': calculate_distance(player_pos, C.BLACK_POS),
        'Green': calculate_distance(player_pos, C.GREEN_POS)
    }




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

    # NPC target
    target = models.StringField()

# message reaction information
class CharPositionData(ExtraModel):
    # data links
    player = models.Link(Player)

    # reaction info
    msgId = models.StringField()
    timestamp = models.StringField()
    posPlayer = models.StringField()
    posRed = models.StringField()
    posBlack = models.StringField()
    posGreen = models.StringField()


########################################################
# Custom export                                        #
########################################################

# custom export of chatLog
def custom_export_text(players):
    # header row
    yield [
        'sessionId', 
        'subjectId',
        'msgId',
        'timestamp',
        'sender',
        'tone',
        'msgText',
    ]

    # get MessageData model
    mData = MessageData.filter()
    for m in mData:

        # get player info
        player = m.player
        participant = player.participant
        session = player.session

        # write to csv
        yield [
            session.code,
            participant.code,
            m.msgId,
            m.timestamp,
            m.sender,
            m.tone,
            m.msgText,
        ]

def custom_export_positions(players):
    # header row
    yield [
        'sessionId', 
        'subjectId',
        'msgId',
        'timestamp',
        'posPlayer',
        'posRed',
        'posBlack',
        'posGreen',
    ]

    # get CharPositionData model
    pData = CharPositionData.filter()
    for p in pData:

        # get player info
        player = p.player
        participant = player.participant
        session = player.session

        # write to csv
        yield [
            session.code,
            participant.code,
            p.msgId,
            p.timestamp,
            p.posPlayer,
            p.posRed,
            p.posBlack,
            p.posGreen,
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
    def vars_for_template(player):
        return dict(
            debug = C.DEBUG,
        )

    # vars that we will pass to chat.html
    @staticmethod
    def js_vars(player):
        # playerId as seen in chat
        currentPlayer = 'P' + str(player.id_in_group)
        
        return dict(
            id_in_group=player.id_in_group,
            playerId=currentPlayer,
            bot_label1=C.BOT_LABEL1,
            bot_label2=C.BOT_LABEL2,
            roomLength = C.ROOM_LENGTH,
            roomWidth = C.ROOM_WIDTH,
            roomHeight = C.ROOM_HEIGHT,
            # npcPersonalSpace = C.NPC_PERSONAL_SPACE,
            # npcJitter = C.NPC_JITTER,
            debug = C.DEBUG,
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
                posData = data.get('pos', {})
                currentPlayer = 'P' + str(player.id_in_group)
                messages = json.loads(player.cachedMessages)
                
                # calculate distance to NPCs
                print('Player pos:', posData)
                npcDistances = calculate_npc_distances(posData)
                print('NPC distances:', npcDistances)

                # determine closest NPC (within 10 units of distance)
                min_distance = min(npcDistances.values())
                closestNPC = None if min_distance > 10 else [x for x in npcDistances if npcDistances[x] == min_distance][0]
                print('Closest NPC:', closestNPC)

                # create message id
                dateNow = str(datetime.now(tz=timezone.utc).timestamp())
                msgId = currentPlayer + '-' + dateNow

                # save to database
                MessageData.create(
                    player=player,
                    msgId=msgId,
                    timestamp=dateNow,
                    sender=currentPlayer,
                    tone=tone,
                    msgText=text,
                    target=closestNPC,
                )

                # add message to list
                messages.append({
                    'sender': 'user',
                    'label': currentPlayer,
                    'msgId': msgId,
                    'text': text,
                })
                
                # update cache
                player.cachedMessages = json.dumps(messages)
                
                # yield output to chat.html
                yield {player.id_in_group: dict(
                    event='text',
                    selfText=text,
                    sender=currentPlayer,
                    msgId=msgId,
                    tone=tone,
                    phase=player.phase,
                    target=closestNPC
                )}

            # handle bot messages
            elif event == 'botMsg':

                # get data from request
                botId = data.get('botId')
                dateNow = str(datetime.now(tz=timezone.utc).timestamp())

                if botId:



                    # run llm on input text
                    dateNow = str(datetime.now(tz=timezone.utc).timestamp())

                    # create inputDat and run api function
                    inputDat = dict(
                        botLabel = botId,
                        messages = messages,
                        tone = tone,
                    )
                    botText = await runGPT(inputDat)

                    print('botId:', botId)
                    print('botText:', botText)

                    # grab bot response data
                    outputText = botText.text
                    botMsgId = botText.msgId
                    botTone = botText.tone

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
                    messages.append({
                        'sender': 'assistant',
                        'label': botId,
                        'msgId': botMsgId,
                        'text': outputText,
                    })
                    player.cachedMessages = json.dumps(messages)

                    # return output to chat.html
                    yield {player.id_in_group: dict(
                        event='botText',
                        sender=botId,
                        botMsgId=botMsgId,
                        tone=tone,
                        text=outputText,
                        phase=player.phase,
                    )}



                # if botId is None, then no NPC is close enough to chat
                else:
                    print('Not near any NPCs!')
            
                
            
            # handle position check updates
            elif event == 'posCheck':
                
                # get time stamp
                dateNow = str(datetime.now(tz=timezone.utc).timestamp())
                    
                # grab position data
                posData = data["pos"]

                # save to database
                CharPositionData.create(
                        player=player,
                        msgId='initial',
                        timestamp=dateNow,
                        posPlayer=json.dumps(posData),
                        posRed='',
                        posBlack='',
                        posGreen='',
                    )

                
            # handle phase updates
            elif event == 'phase':
                
                # update phase
                # currentPhase = player.phase
                currentPhase = data["phase"]

                if currentPhase == 0:

                    # increment phase
                    currentPhase = 1
                    player.phase = currentPhase
                    print("Current phase:")
                    print(currentPhase)

                    # get time stamp
                    dateNow = str(datetime.now(tz=timezone.utc).timestamp())

                    # initialize npc bot posiitons
                    pos = initializeNPCPositions()
                    redPos = pos['red']
                    blackPos = pos['black']
                    greenPos = pos['green']
                    playerPos = pos['player']

                    # save to database
                    CharPositionData.create(
                        player=player,
                        msgId='initial',
                        timestamp=dateNow,
                        posPlayer=json.dumps(playerPos),
                        posRed=json.dumps(redPos),
                        posBlack=json.dumps(blackPos),
                        posGreen=json.dumps(greenPos),
                    )

                    yield {player.id_in_group: dict(
                        event='phase',
                        phase=currentPhase,
                        posPlayer=json.dumps(playerPos),
                        posRed=json.dumps(redPos),
                        posBlack=json.dumps(blackPos),
                        posGreen=json.dumps(greenPos),
                    )}
                
                
                else:
                    pass


# page sequence
page_sequence = [
    chat,
]