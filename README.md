# oTree GPT 

This is a simple chat app for [oTree](https://www.otree.org/) that allows participants to chat with a predefined ChatGPT personality using OpenAI's API. Please feel free to leave any feedback or open an issue if you spot a problem. A short blogpost on using this app can be found [here](https://clintmckenna.com/blog/2023-03-29/)!

## System Prompt 
You can use the system prompt to adjust the tendencies of the personality you want participants to speak to. Here is an example system prompt:

![embedded](https://clintmckenna.com/img/2023/2023-03-29/prompt_texas.png)

And what this looks like when chatting:

![embedded](https://clintmckenna.com/img/2023/2023-03-29/texas.png)

The app is set to randomize participants to view one of two personalities. You can adjust these or provide more/less as desired.

## API key
To use this, you will need to acquire a key from [OpenAI's API](https://openai.com/product). Add this as an environment variable to your local environment. If using on Heroku, you can use this command to add it to your application:

---
> <i>heroku config:add CHATGPT_KEY=sk-.....</i>
---

## OpenAI package requirements
When using locally, you will also need to install openai's Python package. Be sure to add this to your requrements.txt file before using online.

---
> <i>openai==0.27.0</i>
---

## Model Parameters
Currently, I have this set up to use gpt-3.5-turbo. You can adjust this model and the temperature in the \__init__.py file:

![embedded](https://clintmckenna.com/img/2023/2023-03-29/constants.png)

## Data Output
The text logs are saved in participant fields, but I also made a simple custom export function. This can be accessed in the "data" tab in oTree and will show the chat logs as a long-form csv.


## Citation
As part of oTree's [installation agreement](https://otree.readthedocs.io/en/master/install.html), be sure to cite their paper: 

- Chen, D.L., Schonger, M., Wickens, C., 2016. oTree - An open-source platform for laboratory, online and field experiments. Journal of Behavioral and Experimental Finance, vol 9: 88-97.

If this app was helpful, you may consider citing this github repository as well.

- McKenna, C., (2023). oTree GPT. https://github.com/clintmckenna/oTree_gpt

