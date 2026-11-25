import os, sys, time, json
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
dev = sys.argv[1]; N = int(sys.argv[2])
if dev=='cpu': os.environ['CUDA_VISIBLE_DEVICES']='-1'
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import numpy as np, tensorflow as tf
for g in tf.config.list_physical_devices('GPU'): tf.config.experimental.set_memory_growth(g, True)
from knight_env import KnightTourEnv
import dqn_agent  
from tensorflow.keras.optimizers import Adam as _A
dqn_agent.Adam = lambda **k: _A(jit_compile=False, **k)
from dqn_agent import DQNAgent
res={'device':dev,'board':N,'gpus':len(tf.config.list_physical_devices('GPU'))}
env=KnightTourEnv(board_size=N); np.random.seed(0)
# env throughput (random valid moves)
t=time.perf_counter(); steps=0
for _ in range(500):
    env.reset(); done=False
    while not done:
        m=env._get_valid_moves_mask(); env.get_action_onward_counts()
        if not m.any(): break
        _,_,done,_=env.step(int(np.random.choice(np.where(m)[0]))); steps+=1
res['env_steps_per_s']=round(steps/(time.perf_counter()-t))
res['random_avg_len']=round(steps/500,1)
ag=DQNAgent(env.observation_space.shape,8,epsilon=0.0)
res['params']=ag.model.count_params()
s=env.reset(); m=env._get_valid_moves_mask()
for _ in range(20): ag.act(s,m)
t=time.perf_counter()
for _ in range(300): ag.act(s,m)
res['act_ms']=round((time.perf_counter()-t)/300*1000,3)
for _ in range(2000): ag.remember(s,0,0.1,s,False)
for _ in range(5): ag.replay(64)
t=time.perf_counter()
for _ in range(50): ag.replay(64)
res['replay64_ms']=round((time.perf_counter()-t)/50*1000,2)
# warnsdorff baseline success rate (deterministic tie-break order)
wins=0
for _ in range(200):
    env.reset(); done=False
    while not done:
        c=env.get_action_onward_counts()
        if (c<0).all(): break
        c=np.where(c<0,99,c); a=int(np.random.choice(np.where(c==c.min())[0]))
        _,_,done,info=env.step(a)
    wins+= env.visited_count==env.total_squares
res['warnsdorff_random_tiebreak_winrate']=wins/200
print(json.dumps(res))
