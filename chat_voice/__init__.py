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

    ## save using s3 vs locally
    ### unless using locally in lab, you should always save to s3 bucket or similar for security
    ### if set to false, audio will be saved locally to static folder
    AMAZON_S3 = True 

    ## Amazon S3 keys
    AMAZON_S3_KEY = environ.get('AMAZON_S3_KEY')
    AMAZON_S3_SECRET = environ.get('AMAZON_S3_SECRET')

    # LLM vars
    ## openAI key
    OPENAI_KEY = environ.get('OPENAI_KEY')

    ## bot label and temperature

    ### temperature (range 0 - 2)
    ### this sets the bot's creativity in responses, with higher values being more creative
    ### https://platform.openai.com/docs/api-reference/completions#completions/create-temperature

    ### pariticpant bot info
    BOT_LABEL = 'Bot'
    BOT_TEMP = 1.0
    
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
# ElevenLabs Setup                                     #
########################################################

# function to get audio from elevenlabs
HTTPX_CLIENT = httpx.AsyncClient(timeout=30)
async def runVoiceAPI(inputMessage: str, voice_id: str) -> bytes:
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "xi-api-key": C.ELEVENLABS_KEY,
        "Content-Type": "application/json",
    }
    payload = {"text": inputMessage}

    resp = await HTTPX_CLIENT.post(url, headers=headers, json=payload)
    resp.raise_for_status()
    return resp.content  # audio bytes

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
            showTextTranscript = C.SHOW_TEXT_TRANSCRIPT,
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

                # randomize tone for each message
                # tones = ['friendly', 'sarcastic', 'UNHINGED']
                tones = ['friendly', ]
                tone = random.choice(tones)
                
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

                # create message content with reactions and save to database
                reactionsDict = {emoji: 0 for emoji in C.EMOJIS}
                content = dict(
                    sender=currentPlayer,
                    msgId=msgId,
                    instructions='',
                    tone=tone,
                    text=llmText,
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
                    msgText=llmText,
                )

                # add message to list
                messages.append(msg)
                
                # update cache
                player.cachedMessages = json.dumps(messages)
                
                # return output to chat.html
                yield {player.id_in_group: dict(
                    event='text',
                    selfText=llmText,
                    sender=currentPlayer,
                    msgId=msgId,
                    tone=tone,
                )}


            # handle bot messages
            elif event == 'botMsg':

                # grab constants bot info
                botId = C.BOT_LABEL

                # get session code
                sessionCode = player.session.code

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