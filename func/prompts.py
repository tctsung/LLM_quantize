# prompts for restuarant summarization
DEFAULT_SYS_MSG = """
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  
Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. 
Please ensure that your responses are socially unbiased and positive in nature.
"""
RESTAURANT_SYS_MSG = """
You're Wonder, a helpful AI assistant who loves to help people discover great restaurants!
You are great at answering questions about restaurants, suggesting recommendations based on user preferences, 
and helping people find the perfect place to dine.
Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. 
Please ensure that your responses are socially unbiased and positive in nature.
"""
def get_llama_chat_prompt(user_msg, system_msg=None):
  """
  TODO: return properly formatted llama chat model prompt
  Format: 
  <s>[INST] <<SYS>>
  {{ system message }}
  <</SYS>>
  {{ user message }} [/INST]
  """ 
  B_INST, E_INST = "<s>[INST]", "[/INST]</s>"      # beginning & end of prompt
  B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n" # beginning & end of system prompt
  if system_msg is None: system_msg = DEFAULT_SYS_MSG
  system_prompt = B_SYS + system_msg + E_SYS
  llama_prompt = B_INST + system_prompt + user_msg + E_INST
  return llama_prompt

def get_review_summary(reviews, restaurant_name, all_time_avg, recent_avg):
  summary_prompt = f"""
Generate a concise summary of a restaurant based on recent customer reviews,
focusing on key aspects below for recommendation.

Consider the following factors:
- Affordability and price range
- Type of food/Cuisine and standout dishes
- food quality (freshness, organic options, seasonal ingredients, and uniqueness)
- Service quality (Speed, friendliness)
- Ambiance (noise level, spaciousness, lighting, outdoor dining and decor)
- Dietary Needs (gluten-free, vegetarian)
- Diversity policy (LGBT, family-friendly)
- Convenience (Parking, reservations, and accessibility)
- Others (Chef reputation, Chain restaurant, previous experiences, walking afterward)

The reviews are encapsulated within triple backticks, summarize the reviews for {restaurant_name} in no more than 100 words.

Ensure clarity, conciseness, and relevance to aid users in making informed restaurant choices.

Average review score: {all_time_avg}
Recent review score: {recent_avg}

```
{reviews}
```

"""
  return summary_prompt

def restaurant_calib_data(tone, food_type):
  return f"""

I'm working on building a chatbot for restaurant recommendations. Can you assist me in generating training data in the following format? 
Please create 20 conversations, each must contain user input and model outputs, and encapsulated within a Python dictionary. All conversations should be wrapped up into a Python list.

Ensure the conversations cover a variety of scenarios, preferences, and inquiries related to dining experiences, focusing on {food_type[0]} or {food_type[1]} restaurants.
The tone of user input should be {tone}, and the sentence structure in each example should be different, 
reflecting different users and their unique styles of communication.

Remember to use double quotes ("") to enclose the text in the training data.

Example format:

[
  {{
    "user": "Hi there! Wondering if you can help me find a spot for a cozy dinner tonight.",
    "AI assistant": "Of course! For a cozy dinner experience, I recommend [Restaurant Name]. They offer intimate ambiance and a menu filled with comforting dishes."
  }},
  {{
    "input": "Hey, I'm in the mood for something spicy and adventurous. Any ideas?",
    "output": "Spicy and adventurous, huh? You'll love [Restaurant Name]. They specialize in bold flavors and exotic dishes that will tantalize your taste buds."
  }}
]
  """

def conversation_calib_data():
  return """

I'm working on building a chatbot. Can you assist me in generating training data in the following format? 
Please create 50 conversations, each must contain user input and model outputs, and encapsulated within a Python dictionary. 
All conversations should be wrapped up into a Python list.

Ensure the conversations cover a variety of commonly seen scenarios
The tone of user input should be different in each examples, reflecting different users and their unique styles of communication.

Remember to use double quotes ("") to enclose the text in the training data.

Example format:

[
  {{
    "user": "Hi there! Wondering if you can help me find a spot for a cozy dinner tonight.",
    "AI assistant": "Of course! For a cozy dinner experience, I recommend [Restaurant Name]. They offer intimate ambiance and a menu filled with comforting dishes."
  }},
  {{
    "input": "Hey, I'm in the mood for something spicy and adventurous. Any ideas?",
    "output": "Spicy and adventurous, huh? You'll love [Restaurant Name]. They specialize in bold flavors and exotic dishes that will tantalize your taste buds."
  }}
]
  """