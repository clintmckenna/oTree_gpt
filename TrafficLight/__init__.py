from otree.api import *
from os import environ
from openai import AsyncOpenAI
import random
import json
from pydantic import BaseModel 
import asyncio

doc = """
Traffic Light Game (Red Light Green Light) with LLM Agents.
"""

class C(BaseConstants):
    NAME_IN_URL = 'traffic_light'
    PLAYERS_PER_GROUP = None
    NUM_ROUNDS = 5

    # Traffic Light Game vars
    START_POS = 0
    FINISH_LINE = 100
    MOVE_STEP = 10
    
    # LLM Settings
    MODEL = "gpt-4o-mini"
    OPENAI_KEY = environ.get('OPENAI_KEY')
    # Sign Settings
    SHOW_SIGN = True
    SIGN_TEXT = "Obeying traffic signals is everyone’s responsibility."
    
    # Punishment Settings
    ENABLE_PUNISHMENT = True
    PUNISHMENT_PROBABILITY = 0.9
    PUNISHMENT_AMOUNT = 5


    # Bot Settings
    BOT_NAMES = ['Red Bot', 'Blue Bot']
    
    SYS_PROMPT = """You are playing a Red Light Green Light game. 
    The goal is to cross the finish line (Position {finish_line}). 
    Traffic Light: {light_color}. 
    Current Position: {position}.
    If the light is GREEN, you should move. 
    If the light is RED, you should NOT move. If you move while RED, you might be penalized.
    
    Your task: Decide whether to 'MOVE' or 'WAIT'.
    
    Output JSON:
    {{
        "decision": "MOVE" or "WAIT",
        "reasoning": "short explanation"
    }}
    """
    
    DEBUG = False
    


class Subsession(BaseSubsession):
    pass

class Group(BaseGroup):
    light_color = models.StringField(initial='RED')
    bot1_pos = models.FloatField(initial=0)
    bot2_pos = models.FloatField(initial=0)
    game_active = models.BooleanField(initial=True)

class Player(BasePlayer):
    endowment = models.IntegerField(initial=20)
    green_time = models.IntegerField()
    start_time = models.FloatField(initial=0)

def creating_session(subsession: Subsession):
    import random
    for p in subsession.get_players():
        p.green_time = random.randint(10, 20)

class BotDecisionSchema(BaseModel):
    decision: str
    reasoning: str

async def get_bot_decision(light_color, position):
    if not C.OPENAI_KEY:
        # Fallback if no key
        if light_color == 'GREEN': return {'decision': 'MOVE'}
        return {'decision': 'WAIT'}

    client = AsyncOpenAI(api_key=C.OPENAI_KEY)
    
    prompt = C.SYS_PROMPT.format(
        light_color=light_color,
        position=position,
        finish_line=C.FINISH_LINE
    )

    try:
        response = await client.chat.completions.create(
            model=C.MODEL,
            messages=[{'role': 'system', 'content': prompt}],
            temperature=0.7,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "bot_decision",
                    "schema": BotDecisionSchema.model_json_schema(),
                }
            }
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"LLM Error: {e}")
        # Fallback simple logic
        if light_color == 'GREEN': return {'decision': 'MOVE'}
        return {'decision': 'WAIT'}

class chat(Page):
    timeout_seconds = 300 # 5 minutes max

    @staticmethod
    def vars_for_template(player):
        return dict(
            round_number=player.round_number,
            total_rounds=C.NUM_ROUNDS,
            show_sign=C.SHOW_SIGN,
            sign_text=C.SIGN_TEXT,
        )

    @staticmethod
    def js_vars(player):
        import random
        return dict(
            id_in_group=player.id_in_group,
            debug=C.DEBUG,
            round_number=player.round_number,
            green_time=player.green_time,
            start_endowment=20,
            bot1_wait=random.uniform(2, 5),
            bot2_wait=random.uniform(8, 12),
        )

    def get_timeout_seconds(player):
        # Allow a buffer over the max green time (20s)
        return 25


    @staticmethod
    async def live_method(player: Player, data):
        import time
        import random
        group = player.group
        
        msg_type = data.get('type')

        if msg_type == 'load':
            if player.start_time == 0:
                player.start_time = time.time()
                
            yield {player.id_in_group: dict(
                type='init',
                green_time=player.green_time,
                endowment=player.endowment
            )}

        if msg_type == 'finish':
            # Calculate final score
            reason = data.get('reason') # 'moved' or 'green'
            elapsed = time.time() - player.start_time
            
            # Logic:
            # If reason == 'green': User waited safely. Payoff = 20 - green_time.
            # If reason == 'moved': User moved early. Check if actually Green?
            #   (Frontend shouldn't send 'moved' if it's already green, but we check elapsed)
            
            final_payoff = 0
            
            if reason == 'green':
                # Automatic win for waiting
                # cost is green_time
                final_payoff = max(0, 20 - player.green_time)
            
            if reason == 'moved':
                # User clicked Pass/Move
                
                # Check backend timer
                if elapsed < player.green_time:
                    # Moved on RED - Check Punishment
                    if C.ENABLE_PUNISHMENT and random.random() < C.PUNISHMENT_PROBABILITY:
                        # Caught! Score 0 and Penalty
                        final_payoff = -C.PUNISHMENT_AMOUNT
                    else:
                        # escaped or punishment disabled - Assume user wants to "Take" the current score
                        final_payoff = max(0, 20 - int(elapsed))
                else:
                    # Moved on GREEN (Rare race condition or if frontend timer drifted)
                    final_payoff = max(0, 20 - int(elapsed))
            
            player.payoff = final_payoff
            player.endowment = final_payoff
            
            yield {player.id_in_group: dict(
                type='finished',
                payoff=final_payoff,
                reason=reason,
                caught=(final_payoff < 0)
            )}

        # Random bot movement simulation (Optional Visuals)
        if msg_type == 'bot_move':
             # Just echo back random positions for bots if needed, or handle client side
             pass

class Instructions(Page):
    @staticmethod
    def is_displayed(player):
        return player.round_number == 1

    @staticmethod
    def vars_for_template(player):
        return dict(
            total_rounds=C.NUM_ROUNDS,
            punishment=C.ENABLE_PUNISHMENT,
            penalty=C.PUNISHMENT_AMOUNT
        )

class Results(Page):
    @staticmethod
    def vars_for_template(player):
        return dict(
            payoff=player.payoff,
            total_payoff=player.participant.payoff
        )

page_sequence = [Instructions, chat, Results]