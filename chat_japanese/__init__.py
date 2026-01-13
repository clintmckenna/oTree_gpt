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
    NAME_IN_URL = 'chat_japanese'
    PLAYERS_PER_GROUP = None
    NUM_ROUNDS = 1

    # emoji reactions used for chat
    ALLOW_REACTIONS = True
    EMOJIS = ['👍', '👎', '❤️',]

    # 論点のリスト
    TOPICS = [
        {
            'id': 'nuclear_power',
            'title': '日本は原子力発電を継続すべきである',
            'description': '東日本大震災による福島第一原子力発電所事故以降、日本のエネルギー政策は大きな転換点を迎えています。原子力発電の継続については、安全性、エネルギー安全保障、経済性、環境への影響など、様々な観点から議論されています。',
            'bot_context': '今回の討論のテーマは「日本は原子力発電を継続すべきである」です。このテーマについて、参加者と建設的な対話を行ってください。'
        },
        {
            'id': 'okinawa_base',
            'title': '沖縄の米軍基地を縮小すべきである',
            'description': '沖縄県には在日米軍施設の約７割が集中しており、基地による騒音、事故、環境問題が深刻です。一方で、東アジアの安全保障や日米同盟の観点からは、米軍基地の必要性も主張されています。',
            'bot_context': '今回の討論のテーマは「沖縄の米軍基地を縮小すべきである」です。このテーマについて、参加者と建設的な対話を行ってください。'
        },
        {
            'id': 'separate_surnames',
            'title': '選択的夫婦別姓制度を導入すべきである',
            'description': '選択的夫婦別姓制度とは、夫婦が望む場合には、結婚後も夫婦がそれぞれ結婚前の姓を名乗ることを認める制度です。伝統的な家族観や子どもの姓の問題、一方で個人のアイデンティティや男女平等の観点から議論されています。',
            'bot_context': '今回の討論のテーマは「選択的夫婦別姓制度を導入すべきである」です。このテーマについて、参加者と建設的な対話を行ってください。'
        },
        {
            'id': 'death_penalty',
            'title': '日本は死刑制度を廃止すべきである',
            'description': '日本は先進国の中で数少ない死刑制度を維持している国の一つです。国際的には廃止の潮流がある一方で、日本国内では凶悪犯罪への対応や犯罪抑止効果、遺族の感情などの観点から、死刑存置を支持する声も大きいです。',
            'bot_context': '今回の討論のテーマは「日本は死刑制度を廃止すべきである」です。このテーマについて、参加者と建設的な対話を行ってください。'
        },
    ]

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
    SYS_BOT = f"""あなたはアレックスという名前で、オンライン討論に参加している一般市民です。常にメッセージは200文字以内に制限し、カジュアルな日本語で話してください。

    今回の討論のテーマは「日本は原子力発電を継続すべきである」です。このテーマについて、参加者と建設的な対話を行ってください。

    各ユーザー入力は、以下を含むjsonオブジェクトのリストです：
    - 送信者識別子（メッセージを送信した人を示す文字列）
    - メッセージ識別子（文字列）
    - 応答の指示（文字列）
    - 使用するトーン（文字列）
    - 応答するテキスト（文字列）
    - ユーザーが異なるメッセージに対して行ったリアクション（'reactions'フィールド内の文字列）

    重要：このリストは、会話内のすべての参加者間のメッセージ履歴全体です。あなたが送信したメッセージは、'Sender'フィールドに{BOT_LABEL}とラベル付けされています。他の参加者は異なるラベルが付けられます（例：'P1'、'B1'など）。
    
    メッセージに対するリアクションを積極的に監視し、認識する必要があります。可能なリアクションは次のとおりです：{', '.join(EMOJIS)}
    jsonでこれらのリアクションを見た場合は、自然に応答に組み込んでください。
    
    出力として、以下のjsonオブジェクトを提供する必要があります：
    - 'sender': 割り当てられた送信者識別子
    - 'msgId': 割り当てられたメッセージID
    - 'tone': 割り当てられたトーン
    - 'text': あなたの応答（300文字まで）
    - 'reactions': 割り当てられたリアクション値"""


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
async def runGPT(player, inputMessage, tone):

    # grab bot vars from constants
    botTemp = C.BOT_TEMP
    botLabel = C.BOT_LABEL
    
    # AIの立場に基づいた説得指示を追加
    if player.ai_stance == 'support':
        persuasion_instruction = f"""
        あなたの目標は、参加者を「{player.topic_title}」という立場に賛成するように説得することです。
        賛成する理由やメリットを強調し、反対意見に対する反論を提供してください。
        ただし、強引になりすぎず、建設的な対話を心がけてください。
        """
    else:  # oppose
        persuasion_instruction = f"""
        あなたの目標は、参加者を「{player.topic_title}」という立場に反対するように説得することです。
        反対する理由やリスク、問題点を強調し、賛成意見に対する反論を提供してください。
        ただし、強引になりすぎず、建設的な対話を心がけてください。
        """
    
    # プレイヤーの論点情報と説得指示を使用してシステムプロンプトを構築
    botPrompt = f"""あなたはアレックスという名前で、オンライン討論に参加している一般市民です。常にメッセージは200文字以内に制限し、カジュアルな日本語で話してください。

    {player.topic_bot_context}
    
    {persuasion_instruction}

    各ユーザー入力は、以下を含むjsonオブジェクトのリストです：
    - 送信者識別子（メッセージを送信した人を示す文字列）
    - メッセージ識別子（文字列）
    - 応答の指示（文字列）
    - 使用するトーン（文字列）
    - 応答するテキスト（文字列）
    - ユーザーが異なるメッセージに対して行ったリアクション（'reactions'フィールド内の文字列）

    重要：このリストは、会話内のすべての参加者間のメッセージ履歴全体です。あなたが送信したメッセージは、'Sender'フィールドに{botLabel}とラベル付けされています。他の参加者は異なるラベルが付けられます（例：'P1'、'B1'など）。
    
    メッセージに対するリアクションを積極的に監視し、認識する必要があります。可能なリアクションは次のとおりです：{', '.join(C.EMOJIS)}
    jsonでこれらのリアクションを見た場合は、自然に応答に組み込んでください。
    
    出力として、以下のjsonオブジェクトを提供する必要があります：
    - 'sender': 割り当てられた送信者識別子
    - 'msgId': 割り当てられたメッセージID
    - 'tone': 割り当てられたトーン
    - 'text': あなたの応答（300文字まで）
    - 'reactions': 割り当てられたリアクション値"""
    
    # assign message id and bot label
    dateNow = str(datetime.now(tz=timezone.utc).timestamp())
    botMsgId = botLabel + '-' + str(dateNow)

    # grab text that participant inputs and format for chatgpt
    reactionsDict = {emoji: 0 for emoji in C.EMOJIS}
    instructions = f"""
        以下のスキーマでjsonオブジェクトを提供してください（割り当てられた値は変更しないでください）：
            'sender': {botLabel} (文字列),
            'msgId': {botMsgId} (文字列), 
            'tone': {tone} (文字列), 
            'text': {tone}のトーンでユーザーのメッセージに対するあなたの応答（文字列）, 
            'reactions': {json.dumps(reactionsDict)} (文字列)
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

        # 論点をランダムに選択
        topic = random.choice(C.TOPICS)
        p.topic_id = topic['id']
        p.topic_title = topic['title']
        p.topic_description = topic['description']
        p.topic_bot_context = topic['bot_context']

# group vars
class Group(BaseGroup):
    pass    

# player vars
class Player(BasePlayer):

    # 論点情報
    topic_id = models.StringField()
    topic_title = models.LongStringField()
    topic_description = models.LongStringField()
    topic_bot_context = models.LongStringField()

    # AIの立場（'support' = 賛成、'oppose' = 反対）
    ai_stance = models.StringField()

    # tone for the bot
    tone = models.StringField()

    # cache of all messages in conversation
    cachedMessages = models.LongStringField(initial='[]')

    # 争点に関する質問の回答（5段階評価）
    pre_chat_opinion = models.IntegerField(
        label="この争点についてどう思いますか？",
        choices=[
            [1, "強く反対"],
            [2, "やや反対"],
            [3, "どちらでもない"],
            [4, "やや賛成"],
            [5, "強く賛成"]
        ],
        widget=widgets.RadioSelect
    )
    
    post_chat_opinion = models.IntegerField(
        label="この争点についてどう思いますか？",
        choices=[
            [1, "強く反対"],
            [2, "やや反対"],
            [3, "どちらでもない"],
            [4, "やや賛成"],
            [5, "強く賛成"]
        ],
        widget=widgets.RadioSelect
    )

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
                botText = await runGPT(player, messages, tone)
                
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

            


# イントロページ
class Introduction(Page):
    form_model = 'player'

# プレチャット質問ページ
class PreChatQuestion(Page):
    form_model = 'player'
    form_fields = ['pre_chat_opinion']

    @staticmethod
    def before_next_page(player: Player, timeout_happened):
        # 参加者の回答に基づいてAIの立場を決定
        opinion = player.pre_chat_opinion
        
        if opinion <= 2:  # 反対（強く反対またはやや反対）
            # 参加者が反対なので、AIは賛成の立場を取る
            player.ai_stance = 'support'
        elif opinion >= 4:  # 賛成（やや賛成または強く賛成）
            # 参加者が賛成なので、AIは反対の立場を取る
            player.ai_stance = 'oppose'
        else:  # 中立（opinion == 3）
            # 中立の場合はランダムにAIの立場を決定
            player.ai_stance = random.choice(['support', 'oppose'])

# ポストチャット質問ページ
class PostChatQuestion(Page):
    form_model = 'player'
    form_fields = ['post_chat_opinion']

# 終わりのページ
class EndPage(Page):
    form_model = 'player'

# page sequence
page_sequence = [
    Introduction,
    PreChatQuestion,
    chat,
    PostChatQuestion,
    EndPage,
]