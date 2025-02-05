from os import environ


SESSION_CONFIGS = [
     dict(
         name='chat_simple',
         app_sequence=['chat_simple',],
         num_demo_participants=1,
     ),
     dict(
         name='chat_complex',
         app_sequence=['chat_complex',],
         num_demo_participants=1,
     ),
     dict(
         name='chat_voice',
         app_sequence=['chat_voice',],
         num_demo_participants=1,
     ),
     dict(
         name='dictator_game',
         app_sequence=['dictator_game',],
         num_demo_participants=1,
     ),
     dict(
         name='chat_multiple_agents',
         app_sequence=['chat_multiple_agents',],
         num_demo_participants=1,
     ),
     dict(
         name='threejs',
         app_sequence=['threejs',],
         num_demo_participants=1,
     ),
]
# if you set a property in SESSION_CONFIG_DEFAULTS, it will be inherited by all configs
# in SESSION_CONFIGS, except those that explicitly override it.
# the session config can be accessed from methods in your apps as self.session.config,
# e.g. self.session.config['participation_fee']

SESSION_CONFIG_DEFAULTS = dict(
    real_world_currency_per_point=1.00, participation_fee=0.00, doc=""
)

PARTICIPANT_FIELDS = []
SESSION_FIELDS = []


# rooms
ROOMS = [
    dict(
        name='studyRoom1',
        display_name='Study Room 1',
    ),
]


# ISO-639 code
# for example: de, fr, ja, ko, zh-hans
LANGUAGE_CODE = 'en'

# e.g. EUR, GBP, CNY, JPY
REAL_WORLD_CURRENCY_CODE = 'USD'
USE_POINTS = True

ADMIN_USERNAME = 'admin'
# for security, best to set admin password in an environment variable
ADMIN_PASSWORD = environ.get('OTREE_ADMIN_PASSWORD')

DEMO_PAGE_INTRO_HTML = """ """

SECRET_KEY = '6929828123368'
