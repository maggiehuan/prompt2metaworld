from .mw import make

train_env = make(name='door-open', frame_stack=3, action_repeat=2, seed=1,
                 train=True, device_id=-1)
time_step = train_env.reset()
while not time_step.last() or time_step['success'] == 1:
    action = '' # LLM output
    time_step = train_env.step(action)