from multiprocessing import Pool, Queue, Process
import gym

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
            p = Process(target=self.call_step, args=(self.envs[i], assignments[i], i, ))
            p.start()
            p_list.append(p)
        
        for p in p_list:
            p.join()

        res = []
        for i in range(len(self.envs)):
            res.append(queue.get())

        res.sort(key=lambda x:x[-1])
        res = [i[0] for i in res]
        return res
    
    def reset(self):
        p_list = []

        for i in range(len(self.envs)):
            p = Process(target=self.call_reset, args=(self.envs[i], i, ))
            p.start()
            p_list.append(p)
        
        for p in p_list:
            p.join()
        
        res = []
        for i in range(len(self.envs)):
            res.append(queue.get())
        
        res.sort(key=lambda x:x[-1])
        res = [i[0] for i in res]

        return res
            
    @staticmethod
    def call_step(env, a, index):
        queue.put((env.step(a), index))
    
    @staticmethod
    def call_reset(env, index):
        queue.put((env.reset(), index))



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
    # from WordCounting import WordCountingEnv
    from WordCounting_Wolpertinger import WordCountingEnv
    n_envs = 6
    envs = [WordCountingEnv() for _ in range(n_envs)]
    p = ParalllelWrapper(envs, n_envs)
    actions = []

    for i in range(n_envs):
        actions.append(envs[0].random_action())
    print(actions[1])
    print(p.step_multiple(actions)[1])
    # print(p.reset()[0])