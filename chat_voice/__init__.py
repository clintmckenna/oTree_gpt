from otree.api import *
from os import environ
from openai import AsyncOpenAI
import random
import json
from pydantic import BaseModel 
from datetime import datetime, timezone
import httpx
import base64
import boto3
import asyncio
import requests as _requests

doc = """
Chat with voice via Whisper API and ElevenLabs. Based on chat_complex.
"""

author = 'Clint McKenna clint@calsocial.org'

########################################################
# Constants                                            #
########################################################

class C(BaseConstants):
    NAME_IN_URL = 'chat_voice'
    PLAYERS_PER_GROUP = None
    NUM_ROUNDS = 1

    # settings for text transcript and voice recording
    ## show text transcript of what is spoken
    SHOW_TEXT_TRANSCRIPT = True

    ## save user audio as webm file
    SAVE_USER_AUDIO = True
    
    ## allow emoji reactions?
    ALLOW_REACTIONS = True
    EMOJIS = ['👍', '👎', '❤️',]

    # chat history on refresh
    SHOW_HISTORY = True

    ## save using s3 vs locally
    ### unless using locally in lab, you should always save to s3 bucket or similar for security
    ### if set to false, audio will be saved locally to static folder
    AMAZON_S3 = False 

    ## Amazon S3 keys
    AMAZON_S3_KEY = environ.get('AMAZON_S3_KEY')
    AMAZON_S3_SECRET = environ.get('AMAZON_S3_SECRET')

    # LLM vars
    ## openAI key
    OPENAI_KEY = environ.get('OPENAI_KEY')

    ### temperature (range 0 - 2)
    ### this sets the bot's creativity in responses, with higher values being more creative and less deterministic
    ### https://platform.openai.com/docs/api-reference/completions#completions/create-temperature
    #### not used in gpt-5+ unless reasoning set to none
    BOT_TEMP = 1.0

    ## reasoning level for supported models
    ## this can be set to 'none', 'minimal', 'low', 'medium', or 'high'
    REASONING_LVL = 'none'

    ## model
    ## this is which gpt model to use, which have different prices and ability
    ## https://platform.openai.com/docs/models
    ## IMPORTANT: you must use a model that supports structured output
    MODEL = "gpt-4o-mini"

    ## set system prompt for agents
    ## according to OpenAI's documentation, this should be less than ~1500 words
    SYS_BOT = f"""You are Alex, a human participant taking part in an online discussion. Always limit messages to less than 200 characters and speak in an informal language. 

        Each user input will be a list of json objects containing:
        - their sender identifer, which shows who sent the message (string)
        - message Identifer (string)
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

    ## ElevenLabs vars
    ELEVENLABS_KEY = environ.get('ELEVENLABS_KEY')

    ## set elevenlabs voice id
    ### this one is Sarah: A young, serious sounding crisp British female. Great for a podcast.
    VOICE_ID = 'rf6Kp06FksMr0VCBn1Pf' 


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
async def runGPT(inputDat):

    # grab bot vars from constants and inputDat
    botTemp = C.BOT_TEMP
    botPrompt = C.SYS_BOT
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


########################################################
# ElevenLabs Setup                                     #
########################################################

# httpx client for whisper transcription
HTTPX_CLIENT = httpx.AsyncClient(timeout=30)

# function to get audio from elevenlabs via REST API
def _call_elevenlabs(inputMessage: str, voice_id: str) -> bytes:
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    key = C.ELEVENLABS_KEY.strip() if C.ELEVENLABS_KEY else ""
    headers = {
        "xi-api-key": key,
        "Content-Type": "application/json",
    }
    payload = {
        "text": inputMessage,
        "model_id": "eleven_multilingual_v2",
    }
    resp = _requests.post(
        url,
        headers=headers,
        json=payload,
        params={"output_format": "mp3_44100_128"},
        timeout=30,
    )
    if resp.status_code != 200:
        print(f"[ElevenLabs] Error {resp.status_code}: {resp.text}")
    resp.raise_for_status()
    return resp.content

async def runVoiceAPI(inputMessage: str, voice_id: str) -> bytes:
    return await asyncio.to_thread(_call_elevenlabs, inputMessage, voice_id)

# for further prompt formatting, check out this page:
# https://elevenlabs.io/docs/best-practices/prompting
# in this app, we adjust the tone of the voice by adding a prefic like this:
# <sarcastic>: Hello, nice to meet you.



########################################################
# Amazon S3 Setup                                      #
########################################################

# load s3 bucket environment
s3_client = boto3.client(
    's3',
    aws_access_key_id=C.AMAZON_S3_KEY,
    aws_secret_access_key=C.AMAZON_S3_SECRET,
    region_name='us-east-2',
)

s3_signer = boto3.client(
    's3',
    aws_access_key_id=C.AMAZON_S3_KEY,
    aws_secret_access_key=C.AMAZON_S3_SECRET,
    region_name='us-east-2',
)

# save audio to s3 function
async def saveToS3(bucket: str, filename: str, audio: bytes) -> bool:
    content_type = 'audio/mpeg' if filename.endswith('.mp3') else 'audio/webm'

    def _put():
        s3_client.put_object(
                Bucket=bucket,
                Key=filename,
                Body=audio,
                ContentType=content_type,
                ContentDisposition='inline',
                CacheControl='no-cache',
            )
        return True

    try:
        return await asyncio.to_thread(_put)
    except Exception as e:
        print(f'Error saving to S3: {e}')
        return False

# grab s3 url function
def get_s3_url(bucket, filename, expiration=3600):
    try:
        content_type = 'audio/mpeg' if filename.endswith('.mp3') else 'audio/webm'
        url = s3_signer.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket, 'Key': filename, 'ResponseContentType': content_type},
            ExpiresIn=expiration,
        )
        print(f"Generated S3 URL for {filename} with content type {content_type}")
        return url
    except Exception as e:
        print(f"Error generating S3 URL: {str(e)}")
        return None


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
        
        # send player info and emojis to page
        return dict(
            id_in_group = player.id_in_group,
            playerId = currentPlayer,
            emojis = C.EMOJIS,
            allow_reactions = C.ALLOW_REACTIONS,
            showTextTranscript = C.SHOW_TEXT_TRANSCRIPT,
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

        # grab tone from data
        tone = player.tone
        
        # handle different event types
        if 'event' in data:


            # grab event type
            event = data['event']
            
            # handle player input logic
            if event == 'text':

                # # grab base64 text and decode
                voiceInput = data['text']
                print("Received base64 text length:", len(voiceInput))
                b64 = base64.b64decode(voiceInput)
                print("Decoded base64 length:", len(b64))

                # get session code
                sessionCode = player.session.code

                # create filename format: (sessioncode)_(player id in group).webm
                filename = str(player.session.code) + '_' + str(player.id_in_group) + '.webm'
                
                # Check if we have the OpenAI key
                if not C.OPENAI_KEY:
                    print("ERROR: OpenAI API key is not set!")
                    yield {player.id_in_group: {'error': 'OpenAI API key is not configured'}}

                try:
                    
                    headers = {
                        'Authorization': f'Bearer {C.OPENAI_KEY}',
                    }
                    data = {
                        'model': 'whisper-1',
                    }
                    files = {
                        'file': (filename, b64, 'audio/webm'),
                    }

                    resp = await HTTPX_CLIENT.post(
                        'https://api.openai.com/v1/audio/transcriptions',
                        headers=headers,
                        data=data,
                        files=files,
                        timeout=60,
                    )
                    resp.raise_for_status()
                    llmText = resp.json()['text']
                    print("LLM Text:", llmText)

                # debug if there was a problem with transcription
                except Exception as e:
                    print("Error during transcription:", str(e))
                    yield {player.id_in_group: {'error': f'Transcription failed: {str(e)}'}}
                    return
                
                # create message id
                dateNow = str(datetime.now(tz=timezone.utc).timestamp())
                msgId = currentPlayer + '-' + str(dateNow)

                # write user audio to file if enabled
                if C.SAVE_USER_AUDIO:
                    ## if amazon s3 setting, save to s3, otherwise save to static folder
                    filename = f'{sessionCode}_{msgId}.webm'
                    if C.AMAZON_S3:
                        # change to whatever you named your S3 bucket
                        await saveToS3('otree-gpt', filename, b64)
                    else:
                        # or save to static folder
                        audioFilePath = f'_static/chat_voice/recordings/{filename}'
                        with open(audioFilePath, 'wb') as f:
                            f.write(b64)
                else:
                    pass
                
                # grab text and phase info
                text = llmText

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
                )

                botText = await runGPT(inputDat)
                
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
                messages.append({
                    'sender': 'assistant',
                    'label': botId,
                    'msgId': botMsgId,
                    'text': outputText,
                    'reactions': json.dumps(botReactions),
                })
                player.cachedMessages = json.dumps(messages)

                # set voice id
                ## this one is Sarah: A young, serious sounding crisp British female. Great for a podcast.
                voiceId = C.VOICE_ID

                # append outputText with tone
                ## format for elevenlabs is <tone>: text
                textForVoice = f"<{tone}>: {outputText}"

                # run elevenlabs on the botText
                audioDat = await runVoiceAPI(textForVoice, voiceId)
                
                # write audio to file and stream from chat.html
                ## if amazon s3 setting, save to s3, otherwise save to static folder
                sessionCode = player.session.code
                filename = f'{sessionCode}_{botMsgId}.mp3'
                if C.AMAZON_S3:
                    if await saveToS3('otree-gpt', filename, audioDat):
                        audioURL = get_s3_url('otree-gpt', filename)
                        if not audioURL:
                            print("Failed to generate S3 URL, falling back to local storage")
                            audioFilePath = f'_static/chat_voice/recordings/{filename}'
                            with open(audioFilePath, 'wb') as f:
                                f.write(audioDat)
                            audioURL = filename
                    else:
                        print("Failed to save to S3!")
                else:
                    audioFilePath = f'_static/chat_voice/recordings/{filename}'
                    with open(audioFilePath, 'wb') as f:
                        f.write(audioDat)
                    audioURL = filename

                # return output to chat.html
                yield {player.id_in_group: dict(
                    event='botText',
                    sender=botId,
                    botMsgId=botMsgId,
                    tone=tone,
                    text=outputText,
                    audioFilePath = audioURL,

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

            
# page sequence
page_sequence = [
    chat,
]