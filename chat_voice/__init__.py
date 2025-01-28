from otree.api import *
from os import environ
import litellm
import random
import json
from pydantic import BaseModel 
from datetime import datetime, timezone
import requests
import base64
import boto3

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
    AMAZON_S3 = False 

    ## Amazon S3 keys
    AMAZON_S3_KEY = environ.get('AMAZON_S3_KEY')
    AMAZON_S3_SECRET = environ.get('AMAZON_S3_SECRET')

    # LLM vars
    ## openAI key
    OPENAI_KEY = environ.get('CHATGPT_KEY')

    ## temperature (range 0 - 2)
    ## this sets the bot's creativity in responses, with higher values being more creative
    ## https://platform.openai.com/docs/api-reference/completions#completions/create-temperature
    TEMP = 1.0

    ## model
    ## this is which gpt model to use, which have different prices and ability
    ## https://platform.openai.com/docs/models
    ## IMPORTANT: you must use a model that supports structured output
    MODEL = "gpt-4o-mini"

    ## set system prompt for agents
    ## according to OpenAI's documentation, this should be less than ~1500 words
    ## moved this to subsession creation

    ## ElevenLabs vars
    ELEVENLABS_KEY = environ.get('ELEVENLABS_KEY')

    ## set elevenlabs voice id
    ### this one is Sarah: A young, serious sounding crisp British female. Great for a podcast.
    VOICE_ID = 'rf6Kp06FksMr0VCBn1Pf' 


########################################################
# LiteLLM Setup                                        #
########################################################

# set litellm OpenAI key 
litellm.api_key = C.OPENAI_KEY

# log messages in console
litellm.set_verbose = True

# validate json schema setting
litellm.enable_json_schema_validation = True

# specify json schema for bot messages
class MsgOutputSchema(BaseModel):
    msgId: str
    tone: str
    text: str
    reactions: str

# function to run messages
def runGPT(inputMessage):

    # run LiteLLM completion function
    response = litellm.completion(
        model = C.MODEL,
        messages = inputMessage,
        temperature = C.TEMP,
        response_format = MsgOutputSchema
    )
    # return the response json
    return response.choices[0].message.content


########################################################
# ElevenLabs Setup                                     #
########################################################

# function to get audio from elevenlabs
def runVoiceAPI(inputMessage, voice_id):
    
    # using requests package again
    response = requests.post(
        f'https://api.elevenlabs.io/v1/text-to-speech/{voice_id}', 
        headers = {
            'xi-api-key': C.ELEVENLABS_KEY,
            'Content-Type': 'application/json',
        }, 
        json = {
            'text': inputMessage
        }
    )

    # return audio
    return response.content

# for further prompt formatting, check out this page:
# https://elevenlabs.io/docs/best-practices/prompting
# in this app, we adjust the tone of the voice by adding a prefic like this:
# <sarcastic>: Hello, nice to meet you.



########################################################
# Amazon S3 Setup                                      #
########################################################

# load s3 bucket environment
s3_client = boto3.client('s3',
    aws_access_key_id = C.AMAZON_S3_KEY,
    aws_secret_access_key = C.AMAZON_S3_SECRET,
    region_name = 'us-east-2'  # Match your bucket's region
)

# save audio to s3 function
def saveToS3(bucket, filename, audio):
    """Save file to S3 with appropriate content type"""
    try:
        # Determine content type based on file extension
        content_type = 'audio/mpeg' if filename.endswith('.mp3') else 'audio/webm'
        
        # save to s3 with content type
        s3_client.put_object(
            Bucket=bucket,
            Key=filename,
            Body=audio,
            ContentType=content_type,
            ContentDisposition='inline',
            CacheControl='no-cache'
        )
        return True
    except Exception as e:
        print(f"Error saving to S3: {str(e)}")
        return False

# grab s3 url function
def get_s3_url(bucket, filename, expiration=3600):
    """
    Generate a pre-signed URL for an S3 object
    Args:
        bucket (str): S3 bucket name
        filename (str): Object key/filename in S3
        expiration (int): URL expiration time in seconds (default 1 hour)
    Returns:
        str: Pre-signed URL for the S3 object
    """
    try:
        # Get the content type based on file extension
        content_type = 'audio/mpeg' if filename.endswith('.mp3') else 'audio/webm'
        
        url = s3_client.generate_presigned_url(
            'get_object',
            Params={
                'Bucket': bucket,
                'Key': filename,
                'ResponseContentType': content_type
            },
            ExpiresIn=expiration
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

        ## set system prompt
        SYS = f"""You are Alex, a human participant taking part in an online discussion. Always limit messages to less than 200 words and speak in an informal language. 

        Each user input will be a json object containing:
        - instructions for responding
        - tone to use
        - text you will be responding to
        - reactions that users have made to different messages (in the 'reactions' field)
        
        IMPORTANT: You must actively monitor and acknowledge reactions to messages. The following reactions are possible: {', '.join(C.EMOJIS)}
        When you see any of these reactions in the json, incorporate them naturally into your responses.
        
        As output, you MUST provide a json object with:
        - 'msgId': your assigned message ID
        - 'tone': your assigned tone
        - 'text': your response (incorporating reaction acknowledgments)
        - 'reactions': your assigned reactions value"""

        # system prompt
        sysPrompt = {'role': 'system', 'content': SYS}
            
        # creating message id from current time
        dateNow = str(datetime.now(tz=timezone.utc).timestamp())
        currentPlayer = str(p.id_in_group)
        msgId = currentPlayer + '-' + str(dateNow)

        # create initial message extra model
        MessageData.create(
            player = p,
            msgId = msgId,
            timestamp = dateNow,
            sender = 'System',
            tone = '',
            fullText = json.dumps(sysPrompt),
            msgText = SYS,
            filepath = '',
        )


# group vars
class Group(BaseGroup):
    pass

# player vars
class Player(BasePlayer):
    pass    


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
    filepath = models.StringField()

# message reaction information
class MsgReactionData(ExtraModel):
    # data links
    player = models.Link(Player)
    sender = models.StringField()

    # reaction info
    msgId = models.StringField()
    msgReactionId = models.StringField()
    timestamp = models.StringField()
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
        'filepath',
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
            m.filepath,
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
            allowReactions = C.ALLOW_REACTIONS,
            showTextTranscript = C.SHOW_TEXT_TRANSCRIPT,
        )


    @staticmethod
    def live_method(player: Player, data):
        
        # create empty list for messages and reactions
        messages = []
        reactions = []

        # load existing message data and add to list
        mData = MessageData.filter(player=player)
        for x in mData:
            msg = json.loads(x.fullText)
            content = None
            
            if msg['role'] == 'assistant':
                # For bot messages, parse the content to get the actual message
                content = json.loads(msg['content'])
            elif msg['role'] == 'user':
                # For user messages, the content is already in the right format
                content = json.loads(msg['content'])
            
            # Update reaction counts for both bot and user messages
            if content and 'reactions' in content:
                reactionCounts = {emoji: 0 for emoji in C.EMOJIS}
                msgReactions = MsgReactionData.filter(player=player, msgId=content['msgId'])
                # Count unique reactions per emoji (only counting each user once per emoji)
                counted_users = {emoji: set() for emoji in C.EMOJIS}
                for reaction in msgReactions:
                    # Only count if this user hasn't used this emoji yet
                    if reaction.target not in counted_users[reaction.emoji]:
                        reactionCounts[reaction.emoji] += 1
                        counted_users[reaction.emoji].add(reaction.target)
                content['reactions'] = json.dumps(reactionCounts)
                msg['content'] = json.dumps(content)
            
            messages.append(msg)
        
        # functions for retrieving text from openAI
        if 'text' in data:


            # get session code
            sessionCode = player.session.code


            # grab base64 text and decode
            voiceInput = data['text']
            b64 = base64.b64decode(voiceInput)

            try:

                # create filename format: (sessioncode)_(player id in group).webm
                filename = str(player.session.code) + '_' + str(player.id_in_group) + '.webm'

                # Send the file-like object to Whisper API for transcription using requests
                response = requests.post(
                    url = 'https://api.openai.com/v1/audio/transcriptions',
                    headers = {'Authorization': f'Bearer {C.OPENAI_KEY}'},
                    files={"file": (filename, b64, "audio/webm")}, # base64 decoded data sent directly
                    data={'model': 'whisper-1'}
                )

                # extract text from response
                text = dict(response.json())
                llmText = text['text']
                
                # randomize tone for each message
                # tones = ['friendly', 'sarcastic', 'UNHINGED']
                tones = ['friendly', ]
                tone = random.choice(tones)

                # assign message id for user and bot
                dateNow = str(datetime.now(tz=timezone.utc).timestamp())
                currentPlayer = 'P' + str(player.id_in_group)
                msgId = currentPlayer + '-' + str(dateNow)
                BotMsgId = 'B' + '-' + str(dateNow)

                # write user audio to file if enabled
                if C.SAVE_USER_AUDIO:
                    ## if amazon s3 setting, save to s3, otherwise save to static folder
                    filename = f'{sessionCode}_{msgId}.webm'
                    if C.AMAZON_S3:
                        saveToS3('otree-gpt', filename, b64)
                    else:
                        audioFilePath = f'_static/global/chat_voice/recordings/{filename}'
                        with open(audioFilePath, 'wb') as f:
                            f.write(b64)
                else:
                    pass

                # grab text that participant inputs and format for chatgpt
                reactionsDict = {emoji: 0 for emoji in C.EMOJIS}
                instructions = f"""
                    Provide a json object with the following schema (DO NOT CHANGE ASSIGNED VALUES):
                        'msgId': {BotMsgId} (string), 
                        'tone': {tone} (string), 
                        'text': Your response to the user's message in a {tone} tone (string), 
                        'reactions': {json.dumps(reactionsDict)} (string)
                """

                content = dict(
                    msgId = msgId,
                    instructions = instructions,
                    tone = tone,
                    text = text,
                    reactions = json.dumps(reactionsDict),
                    currentPlayer = currentPlayer,
                )
                inputMsg = {'role': 'user', 'content': json.dumps(content)}
                # inputMsg = {'role': 'user', 'content': f"Your tone is {tone}. Provide a json object with a 'tone' and 'text' field, which contains your assigned tone and your text response to this user message: {text} (Message Id: {msgId})"}

                # create message log
                MessageData.create(
                    player = player,
                    msgId = msgId,
                    timestamp = dateNow,
                    sender = 'Subject',
                    tone = tone,
                    fullText = json.dumps(inputMsg),
                    msgText = llmText,
                    filepath = filename,
                )

                # append messages and run llm request
                messages.append(inputMsg)
                botText = runGPT(messages)
                
                # extract fields from json
                outputText = json.loads(botText)['text']
                
                # also append messages with bot message
                botMsg = {'role': 'assistant', 'content': botText}
                messages.append(botMsg)

                # set voice id
                ## this one is Sarah: A young, serious sounding crisp British female. Great for a podcast.
                voiceId = C.VOICE_ID

                # append outputText with tone
                ## format for elevenlabs is <tone>: text
                textForVoice = f"<{tone}>: {outputText}"

                # run elevenlabs on the botText
                audioDat = runVoiceAPI(textForVoice, voiceId)
                
                # write audio to file and stream from chat.html
                ## if amazon s3 setting, save to s3, otherwise save to static folder
                filename = f'{sessionCode}_{BotMsgId}.mp3'
                if C.AMAZON_S3:
                    if saveToS3('otree-gpt', filename, audioDat):
                        audioURL = get_s3_url('otree-gpt', filename)
                        if not audioURL:
                            print("Failed to generate S3 URL, falling back to local storage")
                            audioFilePath = f'_static/global/chat_voice/recordings/{filename}'
                            with open(audioFilePath, 'wb') as f:
                                f.write(audioDat)
                            audioURL = filename
                    else:
                        print("Failed to save to S3, falling back to local storage")
                        audioFilePath = f'_static/global/chat_voice/recordings/{filename}'
                        with open(audioFilePath, 'wb') as f:
                            f.write(audioDat)
                        audioURL = filename
                else:
                    audioFilePath = f'_static/global/chat_voice/recordings/{filename}'
                    with open(audioFilePath, 'wb') as f:
                        f.write(audioDat)
                    audioURL = filename

                # log message data for bot   
                MessageData.create(
                    player = player,
                    msgId = BotMsgId,
                    timestamp = dateNow,
                    sender = 'Bot',
                    tone = tone,
                    fullText = json.dumps(botMsg),
                    msgText = outputText,
                    filepath = filename,
                )


                # dictionary for html page
                output = dict(
                    event = 'text',
                    selfText = llmText,
                    botText = outputText,
                    msgId = msgId,
                    botMsgId = BotMsgId,
                    tone = tone,
                    currentPlayer = currentPlayer,
                    audioFilePath = audioURL
                )

                # return output to chat.html
                return {player.id_in_group: output}  
            

            except Exception as e:
                print("Error processing audio:", str(e))

                
        
        # function for handling reactions
        elif 'msgId' in data:

            # get current player and time to create unique msg id
            currentPlayer = 'P' + str(player.id_in_group)
            dateNow = str(datetime.now(tz=timezone.utc).timestamp() * 1000)
            msgReactionId = currentPlayer + '-' + str(dateNow)

            # grab msgId and emoji
            msgId = data['msgId']
            trgt = data['target']
            emoji = data['emoji']  

            # add reaction to list for player
            reactions.append({currentPlayer: emoji})

            # grab message from identifier and add reaction data
            for m in messages:
                # skip system messages
                if m['role'] == 'system':
                    pass
                else:
                    # load message
                    msg = json.loads(m['content'])
                    
                    # if the current message is the one being reacted to
                    if msg['msgId'] == msgId:
                        reacts = json.loads(msg['reactions'])
                        
                        # if the current emoji has already been made by the sender, do nothing
                        if emoji in [r.emoji for r in MsgReactionData.filter(player=player, msgId=msgId)]:
                            pass
                        # if the current emoji has not been made by the sender, add it to the information in the message
                        else:
                            # add +1 to reaction count
                            reacts[emoji] += 1

                            # update the reactions in the message
                            msg['reactions'] = json.dumps(reacts)
                            
                            # Find the index of the current message in the messages list
                            msg_index = messages.index(m)
                            # Update the message at that index
                            messages[msg_index] = {'role': m['role'], 'content': json.dumps(msg)}

            
            # write values to MsgReactionData
            MsgReactionData.create(
                player = player,
                sender = currentPlayer,
                msgId = msgId,
                msgReactionId = msgReactionId,
                timestamp = dateNow,
                target = trgt,
                emoji = emoji,
            )


            # create output json
            output = dict(
                event = 'msgReaction',
                playerId = currentPlayer,
                msgId = msgId,
                msgReactionId = msgReactionId,
                target = trgt,
                emoji = emoji,  
            )
                     
            # return output to chat.html
            return {player.id_in_group: output}  
        
        else: 
            pass

    
# page sequence
page_sequence = [
    chat,
]