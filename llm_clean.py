import openai
import os
import requests
import json
from prompt_meta import demo_prompt, system_prompt, new_task_prompt, interact_prompt
import numpy as np
from mw import make
import time
import re

model = ['gpt-4', 'gpt-4-32k', 'gpt-35-turbo']
model_choice = "gpt-4-32k"
API_KEY = os.environ.get("OPENAI_API_KEY")
API_ENDPOINT = f"https://gcrgpt4aoai5c.openai.azure.com/openai/deployments/{model_choice}/chat/completions?api-version=2023-03-15-preview"
headers = {'Content-Type': 'application/json', 'api-key': API_KEY}
max_wait_gpt4_time = 40

def reset(self):
    pass


def get_input(obs, action):
    input_data = {
        "messages": [  
        {"role": "system", "content": system_prompt},  
        {"role": "assistant", "content": demo_prompt},
        {"role": "user", "content": new_task_prompt},
        # {"role": "system", "content": "1"},  
        # {"role": "assistant", "content": "1"},
        # {"role": "user", "content": "1"},
    ],  
    "max_tokens": 500,
    "temperature": 0.7,
    }

    obs = np.array(obs)
    action = np.array(action)
    interact_prompt_in = interact_prompt.format(current_observation = obs, previous_action = action)
    add_line = {"role": "user", "content": interact_prompt_in}
    input_data["messages"].append(add_line)
    return input_data
    # obs = np.array(obs)
    # action = np.array(action)
    # interact_prompt_in = interact_prompt.format(current_observation = obs, previous_action = action)
    # add_line = {"role": "user", "content": interact_prompt_in}
    # input_data["messages"].append(add_line)
    # print(input_data)
    # return input_data

def get_first_input(obs):
    obs = np.array(obs)
    new_task_prompt_new = new_task_prompt.format(observation = obs)
    input_data = {
        "messages": [  
        {"role": "system", "content": system_prompt},  
        {"role": "assistant", "content": demo_prompt},
        {"role": "user", "content": new_task_prompt_new},
    ],  
    "max_tokens": 500,
    "temperature": 0.7,
}
    return input_data
    
# def llm_init():
#     model = ['gpt-4', 'gpt-4-32k', 'gpt-35-turbo']
#     model_choice = "gpt-4"
#     API_KEY = os.environ.get("OPENAI_API_KEY")
#     API_ENDPOINT = f"https://gcrgpt4aoai5c.openai.azure.com/openai/deployments/{model_choice}/chat/completions?api-version=2023-03-15-preview"
#     headers = {'Content-Type': 'application/json', 'api-key': API_KEY}


# def make_metaworld():
#     train_env = make(name='door-close', frame_stack=3, action_repeat=2, seed=1,
#                 train=True, device_id=-1)
#     time_step = train_env.reset()

def interact(llm_output):
    action_list = []
    for i in action_str.split('Output: [')[1].split('],')[0].split(','):
        i = i.replace(']', '').strip()
        if i.isspace()==True :
            i = i.strip()
            action_list.append(float(i))
        else:
            action_list.append(float(i))
    action = np.array(action_list)
    return action

    
if __name__ == "__main__":
    train_env = make(name='door-close', frame_stack=3, action_repeat=2, seed=1,
                train=True, device_id=-1)
    time_step = train_env.reset()
 
# TODO 这里交互逻辑有点怪，应该是先拿第一个obs，然后进入llm拿action，action输入envs再拿下一个obs，然后这个obs和action一起进入llm得到下一个obs
# TODO 也许也可以再封装一下
    count = 0
    while not time_step.last() or time_step['success'] == 1:
        if  count == 0:
            observation = time_step.observation
            obs = np.array(observation)
            response = requests.post(API_ENDPOINT, json=get_first_input(obs=obs), headers=headers)
            llm_output = response.json()
            print(llm_output)
            if 'error' in llm_output:
                message = llm_output['error']['message']
                # print(message)
                sleep_time = int(re.findall(r'Please retry after (\w+) second', message)[0])
                sleep_time = min(sleep_time, max_wait_gpt4_time)
                time.sleep(sleep_time + 1.0)
            else:
                action_str = llm_output['choices'][0]['message']['content']
                action_list = []
                for i in action_str.split('Output: [')[1].split('],')[0].split(','):
                    i = i.replace(']', '').strip()
                    if i.isspace()==True :
                        i = i.strip()
                        action_list.append(float(i))
                    else:
                        action_list.append(float(i))
                action = np.array(action_list)
                time_step = train_env.step(action)
                observation = np.array(time_step.observation)
                count += 1
    
        else:
            observation = time_step.observation
            response = requests.post(API_ENDPOINT, json=get_input(obs=observation,action=action), headers=headers)
            llm_output = response.json()
            print(llm_output)
            if 'error' in llm_output:
                message = llm_output['error']['message']
                # print(message)
                sleep_time = int(re.findall(r'Please retry after (\w+) second', message)[0])
                sleep_time = min(sleep_time, max_wait_gpt4_time)
                time.sleep(sleep_time + 1.0)
            else:
                action_str = llm_output['choices'][0]['message']['content']
                action_list = []
                for i in action_str.split('Output: [')[1].split('],')[0].split(','):
                    i = i.replace(']', '').strip()
                    if i.isspace()==True :
                        i = i.strip()
                        action_list.append(float(i))
                    else:
                        action_list.append(float(i))
                action = np.array(action_list)
                time_step = train_env.step(action)
                observation = time_step.observation
                observation = np.array(observation)
                count += 1

        



# 下面都是test的部分
    # # metaworld_env.get_input_data()
    # # metaworld_env.get_input()
    # # metaworld_env.get_output()
    # print(llm_output)
    # # I want to save the response.json() to a json file
    # # please help me to do this
    # save_path = "/home/ziyu/code/side_codes/Dynamic_model/Prompt2meta/try.json"
    # with open(save_path, "w") as file:
    #     json.dump(response.json(), file)
    
    # print(response.json())
    # response = requests.post(API_ENDPOINT, json=get_input(obs,action), headers=headers)
    # print(response.json())
    # obs = [-0.4954247, 0.60209962, 0.19214157, 1.0, -0.20142042, 0.43168769, 0.15003595, 0.39201098, -0.5883548, 0.39171508, 0.58883387, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.4978892, 0.60481796, 0.19447884, 1.0, -0.21001911, 0.43393005, 0.15003595, 0.38529471, -0.5927748, 0.38499338, 0.59325047, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.27786269, 0.7108399, 0.15000001]
    # action = [0.0, 0.0, 0.0, 0.0]
    # response = requests.post(API_ENDPOINT, json=get_input(obs,action), headers=headers)
    # print(response.json())
    #llm_input = llm.get_input_data()
    # llm_input = get_input(obs, action)
    # print(llm_input)

        