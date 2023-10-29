import openai
import textwrap
import os
import requests
from prompt_meta import control_prompt
from mw import make
import numpy as np

API_KEY = os.environ.get("OPENAI_API_KEY")

def get_new_prompt(obs, action):
    prompt_history = get_input_data(obs, action)
    return control_prompt.format(prompt_history=prompt_history)


def get_input_data(obs, action):
    obs_text = "observation: " + str(obs) + "\n"
    action_text = "action: " + str(action) + "\n"

    input_data = input_data['messages'].append([{'role': 'user', 'content': obs_text}, {'role': 'user', 'content': action_text}])
    return input_data


def llm_init():
    openai.api_key = os.environ.get("OPENAI_API_KEY")

    global prompt_system, prompt_history, query_count, control_prompt

    prompt_system = '''
    You are the controller of a sawyer robot with 100 Hz. You can control the robot to do some tasks in MetaWorld environment from deepmind.
    Please inference the output.

    The observation space is 39-dimensional. 
    The first 18 dimensions 
    3D Position of Gripper (3 values): (x, y, z) coordinates of the gripper.
    Normalized Gripper State (1 value): A single normalized measure of the gripper state.
    3D Position of First Object (3 values): (x, y, z) coordinates of the first object.
    4D Quaternion of First Object (4 values): Orientation of the first object represented as a quaternion (a mathematical way to describe orientations in 3D space).
    3D Position of Second Object (3 values): (x, y, z) coordinates of the second object.
    4D Quaternion of Second Object (4 values): Orientation of the second object represented as a quaternion.
    The first and second object will be illustrated according to the task, and will be notified in the description of the task.
    Previous Time Step Gripper State (Next 18 values):
    Similar to the first 18 values, representing the same information but at the previous time step. This information is likely crucial for understanding the dynamics of the system.
    Goal Position (Last 3 values): (x, y, z) coordinates of the goal position.
   
    
    After we have the observation, we will infer about the action, and then output the action.

    The action space is 4-dimensional, with all actions in range [−1, 1] with every number in action divide to ten decimal places. 
    The first three action dimensions correspond to the change in position of the gripper,
    The final value represents the normalized force applied by the gripper.

    We will firstly provide some descriptions of other tasks and success trajectories from these tasks as demonstration.
    Then we will give the observation and description of current task for you to inference the action.
    The following are success trajectories and descriptions of other tasks.
    
    Now you will need to use the dynmaics you learned from previous task to solve a new task in this metaworld environment. 
    {new task description}
    Please try to control the robot. You will receive a reward and current observation after each step. And you will need to use the observation to decide the next action.
    The action is a 4-dim vector, and each dimension is a float number between -1 and 1. You answer should be a list of 4 float numbers.

    The output format is as follows:
    The previous action is [], The current action is [],
    Output: [0.0, 0.0, 0.0, 0.0], output the action here, numbers should be divided to 3 decimal places.
    {new task description}
    '''
    prompt_system = textwrap.dedent(prompt_system)
    prompt_system = prompt_system.split('\n', 1)[1]

    prompt_history = ''
    # prompt_history = control_prompt.format(prompt_history=prompt_history)

    query_count = 0


def llm_query(msg, call_api=True):
    global prompt_system, prompt_history, query_count, control_prompt
    # prompt_history = prompt_history + msg + '\n'

    prompt_history = prompt_history + msg + '\n'
    model = ['gpt-4', 'gpt-4-32k', 'gpt-35-turbo']
    model_choice = "gpt-4-32k"
    API_ENDPOINT = f"https://gcrgpt4aoai5c.openai.azure.com/openai/deployments/{model_choice}/chat/completions?api-version=2023-03-15-preview"

    # if call_api:
    #     completion = openai.ChatCompletion.create(
    #         model="gpt-4-0613", # "gpt-3.5-turbo-16k"
    #         messages=[
    #             {"role": "system", "content": prompt_system},
    #             {"role": "user", "content": prompt_history}
    #         ],
    #         temperature=0.0
    #     )

    # if call_api:
    #     input_data = {  
    #     "messages": [  
    #         {"role": "system", "content": "prompt_system"},  
    #         {"role": "user", "content": get_new_prompt()},
    #     ],  
    #     "max_tokens": 500,
    #     "temperature": 0.7,
    # }

    # Test for the API
    # TODO 把prompt改成交互式封装的prompt，需要加一个新的计数参数
    if call_api:
        input_data = {  
        "messages": [  
            {"role": "system", "content": "You are the controller of a sawyer robot with 100 Hz. You will need to control the robot to complete a task in MetaWorld, which is a environment in RL. At the beginning, you will receive an explaintion of observation and action of the environment. Then you will get some successful trajectories of other task. After, you will recive the desciption of new task. You should learn about the successful task, and then infer about the new task follow the output format. The observation space is 39-dimensional. Description of the observation space: \ The first 18 dimensions, which are the same as the Sawyer robot, are: 3D Position of Gripper (3 values): (x, y, z) coordinates of the gripper. Normalized Gripper State (1 value): A single normalized measure of the gripper state. 3D Position of First Object (3 values): (x, y, z) coordinates of the first object. 4D Quaternion of First Object (4 values): Orientation of the first object represented as a quaternion. 3D Position of Second Object (3 values): (x, y, z) coordinates of the second object. 4D Quaternion of Second Object (4 values): Orientation of the second object represented as a quaternion. The first and second object will be illustrated according to the task, and will be notified in the description of the task. Previous Time Step Gripper State (Next 18 values): Similar to the first 18 values, representing the same information but at the previous time step. This information is likely crucial for understanding the dynamics of the system. Goal Position (Last 3 values): (x, y, z) coordinates of the goal position. Then is the decsription of the action. The action space is 4-dimensional, with all actions in range [−1, 1] with every number in action divide to ten decimal places. The first three action dimensions correspond to the change in position of the gripper, The final value represents the normalized force applied by the gripper. The following are success trajectories and descriptions of other tasks.\nTask1: Door-open.Description: In this task,Open a door with a revolving joint. In this task, you will need to get closer to the door and use force to open it. The first object in obervation will be the door. There are no second object in this task.\nTrajectories: observation: [0.00591, 0.3997, 0.19498, 1.0, -0.02386, 0.67426, 0.114, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00591, 0.3997, 0.19498, 1.0, -0.03096, 0.65573, 0.114, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.13044, 0.81615, 0.125] action: [-1.0, 1.0, 1.0, 1.0] observation: [0.0037, 0.40408, 0.19734, 0.98286, -0.05956, 0.64115, 0.114, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00547, 0.40072, 0.1954, 0.99868, -0.05956, 0.64115, 0.114, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.13044, 0.81615, 0.125] action: [-1.0, 1.0, 1.0, 1.0] observation: [-0.00517, 0.41651, 0.20577, 0.92307, -0.05956, 0.64115, 0.114, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00019, 0.40956, 0.20087, 0.95727, -0.05956, 0.64115, 0.114, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.13044, 0.81615, 0.125] action: [-1.0, 1.0, 0.9556999802589417, 1.0] observation: [-0.02073, 0.43241, 0.21858, 0.83279, -0.05956, 0.64115, 0.114, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.01223, 0.42428, 0.21176, 0.88128, -0.05956, 0.64115, 0.114, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.13044, 0.81615, 0.125] action: [-1.0, 1.0, 0.6353899836540222, 1.0]observation: [-0.04141, 0.44879, 0.23321, 0.71888, -0.05956, 0.64115, 0.114, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0304, 0.44059, 0.22584, 0.77842, -0.05956, 0.64115, 0.114, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.13044, 0.81615, 0.125] action: [-1.0, 1.0, 0.2697499990463257, 1.0]observation: [-0.06527, 0.46457, 0.24703, 0.58676, -0.05956, 0.64115, 0.114, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.05335, 0.45683, 0.24042, 0.6548, -0.05956, 0.64115, 0.114, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.13044, 0.81615, 0.125] action: [-0.8572499752044678, 1.0, -0.07575999945402145, 1.0]observation: [-0.08756, 0.47975, 0.25712, 0.4407, -0.05956, 0.64115, 0.114, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.07674, 0.47214, 0.25268, 0.51524, -0.05956, 0.64115, 0.114, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.13044, 0.81615, 0.125] action: [-0.29993000626564026, 1.0, -0.3278999924659729, 1.0]observation: [-0.1056, 0.49546, 0.26203, 0.32173, -0.05956, 0.64115, 0.114, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.09734, 0.4875, 0.26025, 0.36889, -0.05956, 0.64115, 0.114, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.13044, 0.81615, 0.125] action: [0.15102000534534454, 1.0, -0.45087000727653503, 1.0]observation: [-0.11619, 0.51232, 0.26198, 0.26412, -0.05956, 0.64115, 0.114, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.11197, 0.50371, 0.26258, 0.28365, -0.05956, 0.64115, 0.114, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.13044, 0.81615, 0.125] action: [0.41585999727249146, 1.0, -0.44958001375198364, 1.0]observation: [-0.11825, 0.53074, 0.25819, 0.27495, -0.05956, 0.64115, 0.114, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.11825, 0.52132, 0.26045, 0.26712, -0.05956, 0.64115, 0.114, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.13044, 0.81615, 0.125] action: [0.46720999479293823, 1.0, -0.35464999079704285, 1.0]observation: [-0.11343, 0.55074, 0.25237, 0.28746, -0.05956, 0.64115, 0.114, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.11651, 0.54056, 0.25542, 0.28213, -0.05956, 0.64115, 0.114, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.13044, 0.81615, 0.125] action: [0.3467499911785126, 1.0, -0.20916999876499176, 1.0]observation: [-0.10514, 0.57196, 0.24627, 0.29337, -0.05956, 0.64115, 0.114, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.10948, 0.56123, 0.24925, 0.29107, -0.05956, 0.64115, 0.114, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.13044, 0.81615, 0.125]action: [0.13954000174999237, 1.0, -0.05663999915122986, 1.0]observation: [-0.097, 0.59379, 0.24127, 0.29556, -0.05956, 0.64115, 0.114, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.10085, 0.58284, 0.24356, 0.29475, -0.05956, 0.64115, 0.114, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.13044, 0.81615, 0.125]action: [-0.0640999972820282, 1.0, 0.06814999878406525, 1.0]\nNow you have all the previous information you have! And you will receive the description of the new task."},  
            {"role": "user", "content": "Here is the discription of the new task\nNew task: Door-close\nDescription: Close a door with a revolving joint. In this task, you will need to get closer to the door and use force to open it. The first object in obervation will be the door. There are no second object in this task.\ Please try to control the robot. You will receive current observation after each step. And you will need to use the observation to decide the next action. The action is a 4-dim vector, and each dimension is a float number between -1 and 1. The following is the current observation:[-0.49889606,  0.60583909,  0.19519936,  1.        , -0.21919795,0.43268491,  0.15003595,  0.38193382, -0.59494583,  0.38162979,0.59541979,  0.        ,  0.        ,  0.        ,  0.        ,0.        ,  0.        ,  0.        , -0.49889606,  0.60583909,0.19519936,  1.        , -0.25043907,  0.44631121,  0.15003595,0.38193382, -0.59494583,  0.38162979,  0.59541979,  0.        ,0.        ,  0.        ,  0.        ,  0.        ,  0.        ,0.        ,  0.27294333,  0.70840615,  0.15000001] The output format is as follows: The previous action is [], The current action is [], Output: [0.0, 0.0, 0.0, 0.0], numbers should be divided to 3 decimal places."},
        ],  
        "max_tokens": 500,
        "temperature": 0.7,
    }
       

    headers = {'Content-Type': 'application/json', 'api-key': API_KEY}  
    response = requests.post(API_ENDPOINT, json=input_data, headers=headers)
    res = response.json()
    print(res)



        # res = completion.choices[0].message.content
        # res = res.split('\n', 1)[0]

    # query_count += 1

    # if query_count > 50:
    #     prompt_history = prompt_history.split('\n', 1)[1]
    #     prompt_history = prompt_history.split('\n', 1)[1]

    if call_api:
        return res
    else:
        return None

if __name__ == '__main__':
    llm_init()
train_env = make(name='door-close', frame_stack=3, action_repeat=2, seed=1,
                 train=True, device_id=-1)
time_step = train_env.reset()
while not time_step.last() or time_step['success'] == 1:
    # todo: 每次拿到新obs之后，把obs和action加到prompt里面
    msg = prompt_history 
    # llm_output = llm_query(msg, call_api=True)
    # action_str = llm_output['choices'][0]['message']['content']
    # action_list = []
    # for i in action_str.split('Output: [')[1].split('],')[0].split(','):
    #     i = i.replace(']', '').strip()
    #     if i.isspace()==True :
    #         i = i.strip()
    #         action_list.append(float(i))
    #     else:
    #         action_list.append(float(i))
    action_list = [0.9, -1.0, -0.9, -1.0]
    #action_list = [float(i) for i in action_str.split('Output: [')[1].split('],')[0].split(',')]
    print(action_list)
    action = np.array(action_list)
    #time_step = train_env.step(action)
    #obs = time_step.observation
    #print(type(obs))
    obs = [-0.4954247, 0.60209962, 0.19214157, 1.0, -0.20142042, 0.43168769, 0.15003595, 0.39201098, -0.5883548, 0.39171508, 0.58883387, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.4978892, 0.60481796, 0.19447884, 1.0, -0.21001911, 0.43393005, 0.15003595, 0.38529471, -0.5927748, 0.38499338, 0.59325047, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.27786269, 0.7108399, 0.15000001]
    action_text = "action: " + str(action_list) + "\n"
    obs_text = "observation: " + str(obs) + "\n"
    msg = msg + obs_text + action_text
    input_data = input_data['messages'].append([{'role': 'user', 'content': obs_text}, {'role': 'user', 'content': action_text}])