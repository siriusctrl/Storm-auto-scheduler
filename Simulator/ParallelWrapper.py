from multiprocessing import Pool, Queue, Process
import os
import time
import gym

from WordCounting import WordCountingEnv

queue = Queue()

class ParalllelWrapper():
    
    def __init__(self, env, n_env=4) -> None:
        if type(env) is str:
            self.envs = [gym.make(env) for _ in range(n_env)]
            self.n_env = 4
        elif type(env) is list:
            self.envs = env
            self.n_env = len(env)
        else:
            raise ValueError(f'Unknown env type {type(env)}')
    
    def step_multiple(self, assignments):
        assert(type(assignments) is list)
        assert(len(assignments) == self.n_env)

        p_list = []
        for i in range(len(self.envs)):
            p = Process(target=self.call_step, args=(self.envs[i], assignments[i],))
            p.start()
            p_list.append(p)
        
        for p in p_list:
            p.join()

        res = []
        for i in range(len(self.envs)):
            res.append(queue.get())
        
        return res
            
    @staticmethod
    def call_step(env, a):
        queue.put(env.step(a))

    def once_multiple(self,):
        return self.pool.map(self.call_once, self.envs)
    
    def call_once(self, env):
        return env.once()



if __name__ == '__main__':
    """
    Practice Sample
    """
    # q_in = Queue()
    # q_out = Queue()
    # # p = Pool(4, f, (q_in, q_out,))
    # p = Pool(4)
    # total = 10

    # # time.sleep(1)

    # # print(new_data(q_in, q_out, total))
    # # print(new_data(q_in, q_out, total+10))
    # a = [test(), test(), test(), test()]
    # print(p.map(call_test, a))
    # print(a)

    """
    Parallel ENV test
    """
    n_envs = 6
    envs = [WordCountingEnv() for _ in range(n_envs)]
    p = ParalllelWrapper(envs, n_envs)
    actions = []

    for i in range(n_envs):
        actions.append(envs[0].action_space.sample())

    print(p.step_multiple(actions))

    # pack = list(zip(envs, actions))

    # p_list = []
    # for p in pack:
    #     proc = Process(target=call_step, args=p)
    #     p_list.append(proc)
    #     proc.start()
    
    # for proc in p_list:
    #     proc.join()