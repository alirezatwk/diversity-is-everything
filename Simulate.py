import numpy as np
import pickle
import os
from tqdm import tqdm

from agents import *
from results import *
from visualizers import *
from utility_functions import UtilityFunctionBase
from agents import AgentBase
from environments import EnvironmentBase
from rewards import RewardBase
from typing import List, Tuple

from GetParams import get_args

class Simulate():
    def __init__(self, 
                 AGENTS_SOC_CLASS_NAME: list[str],
                 AGENTS_IND_CLASS_NAME: list[str],
                 AGENTS_ID: list[str],
                 REWARD_VALUES: list[float], 
                 REWARD_PROB_ARMS: list[float], 
                 REWARD_CLASS_NAME: str,
                 ENVIRONMENT_CLASS_NAME: str,
                 AgentCreator_CLASS_NAME: str, 
                 EXPERIMENT_NAME: str,
                 EPISODE_MAX_LENGTH: int,
                 REPETITION: int, 
                 PLOT_MAX_LENGTH: int, 
                 SEED: int, 
                 SAVE_FE: str,
                 SAVE_PREFERENCE: str,
                 SAVE_PATH: str,
                 AGENT_EPSILON: float, 
                 C_FE: float,
                 N0_FE: int,
                 LAMBDA_FE: float,
                 CONJUGATE_PRIOR: str,
                 C_UCB: float,
                 B1_OUCB: float,
                 B2_OUCB: float,
                 EPSILON: float,
                 LR: float,
                 DIVERSITY: bool, 
                 ALPHA_MEAN: float,
                 ALPHA_STD: float,
                 BETA_MEAN: float,
                 BETA_STD: float,
                 GAMMA_MEAN: float,
                 GAMMA_STD: float,
                 N_SAMPLES_EXP_U: int, 
                 ) -> None:

        self.AGENTS_SOC_CLASS_NAME = AGENTS_SOC_CLASS_NAME
        self.AGENTS_IND_CLASS_NAME = AGENTS_IND_CLASS_NAME
        self.AGENTS_ID = AGENTS_ID
        self.AGENTS_COUNT = len(AGENTS_SOC_CLASS_NAME)
        self.REWARD_VALUES = REWARD_VALUES
        self.REWARD_PROB_ARMS = [] 
        for prob in REWARD_PROB_ARMS:
            self.REWARD_PROB_ARMS.append([prob, 1 - prob])
        self.REWARD_CLASS_NAME = REWARD_CLASS_NAME
        self.ENVIRONMENT_CLASS_NAME = ENVIRONMENT_CLASS_NAME
        self.AgentCreator_CLASS_NAME = AgentCreator_CLASS_NAME
        self.EXPERIMENT_NAME = EXPERIMENT_NAME 
        self.EPISODE_MAX_LENGTH = EPISODE_MAX_LENGTH
        self.PLOT_MAX_LENGTH = PLOT_MAX_LENGTH
        self.REPETITION = REPETITION
        self.SEED = SEED
        
        self.AGENT_EPSILON = AGENT_EPSILON
        self.C_FE = C_FE
        self.N0_FE = N0_FE
        self.LAMBDA_FE = LAMBDA_FE
        self.EPSILON = EPSILON
        self.LR = LR
        self.CONJUGATE_PRIOR = CONJUGATE_PRIOR
        self.C_UCB = C_UCB
        self.B1_OUCB = B1_OUCB
        self.B2_OUCB = B2_OUCB
        self.DIVERSITY = DIVERSITY

        self.ALPHA_MEAN = ALPHA_MEAN
        self.ALPHA_STD = ALPHA_STD
        self.BETA_MEAN = BETA_MEAN
        self.BETA_STD = BETA_STD
        self.GAMMA_MEAN = GAMMA_MEAN
        self.GAMMA_STD = GAMMA_STD
        self.N_SAMPLES_EXP_U = N_SAMPLES_EXP_U

        self.SAVE_FE = SAVE_FE
        self.SAVE_PREFERENCE = SAVE_PREFERENCE
        self.SAVE_PATH = SAVE_PATH
    
        self.ENVIRONMENTS, self.REWARDS, self.REWARDS_NAME = self.set_environments()
        self.POP_AGENTS = self.set_pop_agents()
        self.best_arms()
        print(self.EUS, self.EXP_U_BESTS, self.BEST_ACTIONS)

    def _create_ind_agent(self, agent_class_name: str, agent_id: str, uf: UtilityFunctionBase, part_of_agent: bool) -> AgentBase:
        Ind_Models = ['ThompsonSamplingAgent', 'EpsilonGreedyAgent' , 'UCBAgent'] + ["AlwaysRandomAgent", "AlwaysBestAgent", "AlwaysWorstAgent", "AlwaysSecondBestAgent","PercentBestAgent","-"] 
        assert agent_class_name in Ind_Models

        if agent_class_name == 'ThompsonSamplingAgent':
            agent = ThompsonSamplingAgent(id=agent_id, utility_function= uf, part_of_agent= part_of_agent, conjugate_prior= self.CONJUGATE_PRIOR)

        elif agent_class_name == 'EpsilonGreedyAgent':
            agent = EpsilonGreedyAgent(id=agent_id, utility_function= uf, part_of_agent= part_of_agent, epsilon= self.AGENT_EPSILON)

        elif agent_class_name == "UCBAgent":
            agent = UCBAgent(id= agent_id, c_ucb = self.C_UCB, utility_function= uf)

        elif agent_class_name == "AlwaysBestAgent":
            agent = AlwaysBestAgent(id=agent_id, utility_function= uf, part_of_agent= part_of_agent)

        elif agent_class_name == "AlwaysRandomAgent":
            agent = AlwaysRandomAgent(id=agent_id, utility_function= uf, part_of_agent= part_of_agent)

        elif agent_class_name == "AlwaysWorstAgent":
            agent = AlwaysWorstAgent(id=agent_id, utility_function= uf, part_of_agent= part_of_agent) 

        elif agent_class_name == "AlwaysSecondBestAgent":     
            agent = AlwaysSecondBestAgent(id=agent_id, utility_function= uf, part_of_agent= part_of_agent)

        return agent 
    
    def _create_agent(self, agent_soc_class_name:str, agent_ind_class_name:str, agent_id:str, uf: UtilityFunctionBase) -> AgentBase:
        Ind_Models = ['ThompsonSamplingAgent', 'EpsilonGreedyAgent', 'UCBAgent'] + ["AlwaysRandomAgent", "AlwaysBestAgent", "AlwaysWorstAgent", "AlwaysSecondBestAgent","PercentBestAgent","-"] 
        FE_Models = ["FreeEnergySocialAgent_M1_1_1","FreeEnergySocialAgent_M1_1_2",
                     "FreeEnergySocialAgent_M2_1_1", "FreeEnergySocialAgent_M2_1_2",
                     "FreeEnergySocialAgent_M3_1_1", "FreeEnergySocialAgent_M3_1_2"]
        Other_Social_Models = ["PreferenceBasedSocialAgent", "TUCBAgent", "OUCBAgent","-"]
        Soc_Models = FE_Models + Other_Social_Models

        assert agent_soc_class_name in Soc_Models
        assert agent_ind_class_name in Ind_Models
        
        if agent_soc_class_name in FE_Models:
            if agent_soc_class_name == "FreeEnergySocialAgent_M1_1_1":
                ind_agent = self._create_ind_agent(agent_ind_class_name, agent_id, uf, part_of_agent= True)
                agent = FreeEnergySocialAgent_M1_1_1(id= agent_id, individual_agent= ind_agent, c= self.C_FE, n0= self.N0_FE, 
                                                 lamda= self.LAMBDA_FE, conjugate_prior= self.CONJUGATE_PRIOR, epsilon= self.EPSILON)
            elif agent_soc_class_name == "FreeEnergySocialAgent_M1_1_2":
                ind_agent = self._create_ind_agent(agent_ind_class_name, agent_id, uf, part_of_agent= True)
                agent = FreeEnergySocialAgent_M1_1_2(id= agent_id, individual_agent= ind_agent, c= self.C_FE, n0= self.N0_FE, 
                                                 lamda= self.LAMBDA_FE, conjugate_prior= self.CONJUGATE_PRIOR, epsilon= self.EPSILON)
            elif agent_soc_class_name == "FreeEnergySocialAgent_M2_1_1":
                ind_agent = self._create_ind_agent(agent_ind_class_name, agent_id, uf, part_of_agent= True)
                agent = FreeEnergySocialAgent_M2_1_1(id= agent_id, individual_agent= ind_agent, c= self.C_FE, n0= self.N0_FE, 
                                                 lamda= self.LAMBDA_FE, conjugate_prior= self.CONJUGATE_PRIOR, epsilon= self.EPSILON)
            elif agent_soc_class_name == "FreeEnergySocialAgent_M2_1_2":
                ind_agent = self._create_ind_agent(agent_ind_class_name, agent_id, uf, part_of_agent= True)
                agent = FreeEnergySocialAgent_M2_1_2(id= agent_id, individual_agent= ind_agent, c= self.C_FE, n0= self.N0_FE, 
                                                 lamda= self.LAMBDA_FE, conjugate_prior= self.CONJUGATE_PRIOR, epsilon= self.EPSILON)
            elif agent_soc_class_name == "FreeEnergySocialAgent_M3_1_1":
                ind_agent = self._create_ind_agent(agent_ind_class_name, agent_id, uf, part_of_agent= True)
                agent = FreeEnergySocialAgent_M3_1_1(id= agent_id, individual_agent= ind_agent, c= self.C_FE, n0= self.N0_FE, 
                                                 lamda= self.LAMBDA_FE, conjugate_prior= self.CONJUGATE_PRIOR, epsilon= self.EPSILON)
            elif agent_soc_class_name == "FreeEnergySocialAgent_M3_1_2":
                ind_agent = self._create_ind_agent(agent_ind_class_name, agent_id, uf, part_of_agent= True)
                agent = FreeEnergySocialAgent_M3_1_2(id= agent_id, individual_agent= ind_agent, c= self.C_FE, n0= self.N0_FE, 
                                                 lamda= self.LAMBDA_FE, conjugate_prior= self.CONJUGATE_PRIOR, epsilon= self.EPSILON)
                        
        elif agent_soc_class_name == "PreferenceBasedSocialAgent":
            ind_agent = self._create_ind_agent(agent_ind_class_name, agent_id, uf, part_of_agent= True)
            agent = PreferenceBasedSocialAgent(id= agent_id, individual_agent= ind_agent, epsilon= self.EPSILON, lr= self.LR)
        
        elif agent_soc_class_name in "TUCBAgent":
            agent = TUCBAgent(id= agent_id, c_ucb = self.C_UCB, epsilon = self.EPSILON, utility_function= uf)
        
        elif agent_soc_class_name in "OUCBAgent":
            agent = OUCBAgent(id= agent_id, c_ucb = self.C_UCB, b1=self.B1_OUCB , b2= self.B2_OUCB,utility_function= uf)

        else:
            agent = self._create_ind_agent(agent_ind_class_name, agent_id, uf, part_of_agent= False)
        
        return agent 
       
    def set_pop_agents(self):# -> Tuple(List[AgentBase], List[str]):
        pop_agents = []

        AGENT_CREATOR = self.AgentCreator_CLASS_NAME(seed= self.SEED,
                                                     alpha_mean = self.ALPHA_MEAN,
                                                     alpha_std = self.ALPHA_STD,
                                                     beta_mean = self.BETA_MEAN,
                                                     beta_std = self.BETA_STD,
                                                     gamma_mean = self.GAMMA_MEAN,
                                                     gamma_std = self.GAMMA_STD)
    
        self.MAIN_AGENT = AGENT_CREATOR.create_agent()
        print(f'alpha={self.MAIN_AGENT.alpha}, beta={self.MAIN_AGENT.beta}, gamma={self.MAIN_AGENT.gamma}')   
        agent = self._create_agent(agent_soc_class_name= self.AGENTS_SOC_CLASS_NAME[0], agent_ind_class_name= self.AGENTS_IND_CLASS_NAME[0], agent_id= self.AGENTS_ID[0], uf= self.MAIN_AGENT)
        pop_agents.append(agent)

        uf = self.MAIN_AGENT
        for i in range(1, self.AGENTS_COUNT):   
            if self.DIVERSITY:
                uf = AGENT_CREATOR.create_agent()
            agent =  self._create_agent(agent_soc_class_name= self.AGENTS_SOC_CLASS_NAME[i], agent_ind_class_name= self.AGENTS_IND_CLASS_NAME[i], agent_id= self.AGENTS_ID[i], uf= uf)
            pop_agents.append(agent)

        return pop_agents
    
    def best_arms(self):
        self.BEST_ACTIONS = {}
        self.EXP_U_BESTS = {}
        self.EUS = {}
        for REWS, REW_NAME in zip(self.REWARDS,self.REWARDS_NAME):
            EUs = []    

            for reward in REWS:
                EUs.append(reward.get_expected_utility(self.MAIN_AGENT, self.N_SAMPLES_EXP_U))

            EUs = np.array(EUs)
            self.EUS[REW_NAME] = EUs
            BEST_ACTION = np.random.choice(np.flatnonzero(EUs == EUs.max()))
            EXP_U_BEST = EUs.max()
            self.BEST_ACTIONS[REW_NAME] = BEST_ACTION
            self.EXP_U_BESTS[REW_NAME] = EXP_U_BEST

    def get_rewards(self, reward_class, values, arms_probabilities):
        rewards = []
        for arm_probabilities in arms_probabilities:
            reward = reward_class(values=values, probabilities=arm_probabilities)
            rewards.append(reward)
        return rewards
    
    def set_environments(self):# -> Tuple(List[EnvironmentBase], List[List[RewardBase]], List[str]):
        #TODO: write more general
        environments = []
        rewards = []
        rewards_name = []
        for REWARD_VALUES in self.REWARD_VALUES:
            REWS = self.get_rewards(self.REWARD_CLASS_NAME, [REWARD_VALUES, -0.5 * REWARD_VALUES], self.REWARD_PROB_ARMS)
            rewards.append(REWS)
            rewards_name.append(f"Bernoulli_{REWARD_VALUES}")
            environment = self.ENVIRONMENT_CLASS_NAME(rewards=REWS)
            environments.append(environment)

        return environments, rewards, rewards_name
    
    def _simulate(self, environment, exp_us):
        taken_actions = np.zeros((self.REPETITION, self.EPISODE_MAX_LENGTH))
        given_rewards = np.zeros((self.REPETITION, self.EPISODE_MAX_LENGTH))
        after_rewards = np.zeros((self.REPETITION, self.EPISODE_MAX_LENGTH))
        expected_utils = np.zeros((self.REPETITION, self.EPISODE_MAX_LENGTH))
        selected_agents = np.zeros((self.REPETITION, self.EPISODE_MAX_LENGTH))
        free_energies = np.zeros((self.AGENTS_COUNT, self.REPETITION, self.EPISODE_MAX_LENGTH))
        history = []
        
        for r in tqdm(range(self.REPETITION)):
            for agent in self.POP_AGENTS:
                agent.set_environment(environment, exp_us)
            
            environment.submit()

            for trial in range(self.EPISODE_MAX_LENGTH):
                for agent_idx in range(self.AGENTS_COUNT):
                    _, reward, _, info, action = self.POP_AGENTS[agent_idx].take_action()
                    if agent_idx == 0:
                        taken_actions[r][trial] = action
                        given_rewards[r][trial] = reward
                        after_rewards[r][trial] = self.MAIN_AGENT.apply(reward)
                        expected_utils[r][trial] = exp_us[action]
                        selected_agents[r][trial] = info['selected_agent']
                        free_energies[:, r, trial] = info['FE']

            if self.SAVE_PREFERENCE: history.append(self.POP_AGENTS[0].history)

            environment.reset()
        return taken_actions, given_rewards, after_rewards, expected_utils, selected_agents, free_energies, history

    def simulate(self):
        TAKEN_ACTIONS = {}
        AFTER_REWARDS = {}
        EXP_US = {}
        SELECTED_AGENTS = {}
        FREE_ENERGIES = {}
        HISTORY = {}
        for ENV, REW_NAME in zip(self.ENVIRONMENTS, self.REWARDS_NAME) :
            taken_actions, given_rewards, after_rewards, expected_utils, selected_agents, free_energies, history = self._simulate(environment=ENV, exp_us=self.EUS[REW_NAME])
            TAKEN_ACTIONS[REW_NAME] = taken_actions
            AFTER_REWARDS[REW_NAME] = after_rewards
            EXP_US[REW_NAME] = expected_utils
            SELECTED_AGENTS[REW_NAME] = selected_agents
            FREE_ENERGIES[REW_NAME] = free_energies
            HISTORY[REW_NAME] = history 

        # # Writing Results
        # ## Actions
        for REW_NAME in self.REWARDS_NAME:
            main_actions = ActionsResult(repetition = self.REPETITION, trials=self.EPISODE_MAX_LENGTH)
            main_actions.set_actions(TAKEN_ACTIONS[REW_NAME])

            os.makedirs(self.SAVE_PATH + f"/actions/{self.SEED}_{REW_NAME}/", exist_ok = True)
            with open(self.SAVE_PATH + f'/actions/{self.SEED}_{REW_NAME}/{self.AGENTS_ID[0]}.pkl', 'wb') as f:
                pickle.dump(main_actions, f)

        # ## Rewards
        for REW_NAME in self.REWARDS_NAME:
            main_rewards = RewardsResult(repetition = self.REPETITION, trials=self.EPISODE_MAX_LENGTH)
            main_rewards.set_rewards(EXP_US[REW_NAME])

            os.makedirs(self.SAVE_PATH + f"/rewards/{self.SEED}_{REW_NAME}", exist_ok = True)
            with open(self.SAVE_PATH + f'/rewards/{self.SEED}_{REW_NAME}/{self.AGENTS_ID[0]}.pkl', 'wb') as f:
                pickle.dump(main_rewards, f)
        
        # ## Free Energy
        if self.SAVE_FE:
            for REW_NAME in self.REWARDS_NAME:
                main_fes = FreeEnergyResult(repetition = self.REPETITION, trials=self.EPISODE_MAX_LENGTH, n_agents= self.AGENTS_COUNT, agents_id= self.AGENTS_ID)
                main_fes.set_fe(FREE_ENERGIES[REW_NAME])

                os.makedirs(self.SAVE_PATH + f"/free_energy/{self.SEED}_{REW_NAME}/", exist_ok = True)
                with open(self.SAVE_PATH + f'/free_energy/{self.SEED}_{REW_NAME}/{self.AGENTS_ID[0]}.pkl', 'wb') as f:
                    pickle.dump(main_fes, f)

            for REW_NAME in self.REWARDS_NAME:
                main_agents = ActionsResult(repetition = self.REPETITION, trials=self.EPISODE_MAX_LENGTH)
                main_agents.set_actions(SELECTED_AGENTS[REW_NAME])

                os.makedirs(self.SAVE_PATH + f"/selected_agents/{self.SEED}_{REW_NAME}/", exist_ok = True)
                with open(self.SAVE_PATH + f'/selected_agents/{self.SEED}_{REW_NAME}/{self.AGENTS_ID[0]}.pkl', 'wb') as f:
                    pickle.dump(main_agents, f)
        
        # ## Preferences
        if self.SAVE_PREFERENCE:
            for REW_NAME in self.REWARDS_NAME:
                for other_agent in self.POP_AGENTS :
                    other_agent_id = other_agent.id
                    main_history = ActionsResult(repetition = self.REPETITION, trials=self.EPISODE_MAX_LENGTH)
                    h = HISTORY[REW_NAME]
                    h = np.array(history)
                    h = np.where(h == other_agent_id, self.BEST_ACTIONS[REW_NAME], 1)
                    main_history.set_actions(h)
                    os.makedirs(self.SAVE_PATH + f'/selecting_probability/{self.SEED}_{REW_NAME}/', exist_ok = True)
                    with open(self.SAVE_PATH + f'/selecting_probability/{self.SEED}_{REW_NAME}/{self.AGENTS_ID[0]}_{other_agent_id}.pkl', 'wb') as f:
                        pickle.dump(main_history, f)

    def visualize(self):
        # # Visualization   
        for REW_NAME in self.REWARDS_NAME: 

            folder_path = self.SAVE_PATH + f"/actions/{self.SEED}_{REW_NAME}/"

            visualizer = ProbabilityOfChoosingBestActionVisualizer(
                best_action= self.BEST_ACTIONS[REW_NAME],
                max_trial=self.PLOT_MAX_LENGTH,
                experiment_name =  f"Probability of Selection Optimal Action[{self.EXPERIMENT_NAME}_{self.SEED}_{REW_NAME}]",
                data_path= folder_path,
                write_path= folder_path+'/visualization.html',
            )
            visualizer.visualize()


        for REW_NAME in self.REWARDS_NAME: 

            folder_path = self.SAVE_PATH + f"/rewards/{self.SEED}_{REW_NAME}/"

            visualizer2 = RegretVisualizer(
                exp_u_best= self.EXP_U_BESTS[REW_NAME],
                max_trial=self.PLOT_MAX_LENGTH,
                experiment_name =  f"Cumulative Regret[{self.EXPERIMENT_NAME}_{self.SEED}_{REW_NAME}]",
                data_path=folder_path,
                write_path=folder_path+'/visualization2.html',
            )
            visualizer2.visualize()

        if self.SAVE_FE:
            for REW_NAME in self.REWARDS_NAME:

                folder_path = self.SAVE_PATH + f"/free_energy/{self.SEED}_{REW_NAME}/" 
                visualizer3 = FreeEnergyVisualizer(
                                                    max_trial = self.PLOT_MAX_LENGTH,
                                                    experiment_name =  f"Free Energy of Agents[{self.EXPERIMENT_NAME}_{self.SEED}_{REW_NAME}]",
                                                    data_path = folder_path,
                                                    write_path = folder_path+'/visualization3.html',
                                                )
                visualizer3.visualize()

            for REW_NAME in self.REWARDS_NAME:
                folder_path = self.SAVE_PATH + f"/selected_agents/{self.SEED}_{REW_NAME}/"
                
                visualizer5 = ProbabilityOfSelectingAgentVisualizer(
                    n_agents= self.AGENTS_COUNT,
                    agents_id= self.AGENTS_ID,
                    max_trial=self.PLOT_MAX_LENGTH,
                    experiment_name = f"Agent Selection Probablity[{self.EXPERIMENT_NAME}_{self.SEED}_{REW_NAME}]",
                    data_path= folder_path,
                    write_path= folder_path+'/visualization5.html',
                )
                visualizer5.visualize()     

        if self.SAVE_PREFERENCE:
            for REW_NAME in self.REWARDS_NAME:  

                folder_path = self.SAVE_PATH + f"/selecting_probability/{self.SEED}_{REW_NAME}/"
                
                visualizer4 = ProbabilityOfChoosingBestActionVisualizer(
                    best_action = self.BEST_ACTIONS[REW_NAME],
                    max_trial = self.PLOT_MAX_LENGTH,
                    experiment_name =  f"Agent Selection Probablity[{self.EXPERIMENT_NAME}_{self.SEED}_{REW_NAME}]",
                    data_path = folder_path,
                    write_path = folder_path+'/visualization4.html',
                )
                visualizer4.visualize()

if __name__ == '__main__':
    args = get_args()
    simulate = Simulate(
                 AGENTS_SOC_CLASS_NAME= args.AGENTS_SOC_CLASS_NAME,
                 AGENTS_IND_CLASS_NAME= args.AGENTS_IND_CLASS_NAME,
                 AGENTS_ID= args.AGENTS_ID,
                 REWARD_VALUES= args.REWARD_VALUES, 
                 REWARD_PROB_ARMS= args.REWARD_PROB_ARMS, 
                 REWARD_CLASS_NAME= args.REWARD_CLASS_NAME,
                 ENVIRONMENT_CLASS_NAME= args.ENVIRONMENT_CLASS_NAME,
                 AgentCreator_CLASS_NAME= args.AgentCreator_CLASS_NAME, 
                 EXPERIMENT_NAME= args.EXPERIMENT_NAME,
                 EPISODE_MAX_LENGTH= args.EPISODE_MAX_LENGTH,
                 REPETITION= args.REPETITION, 
                 PLOT_MAX_LENGTH= args.PLOT_MAX_LENGTH, 
                 SEED= args.SEED, 
                 SAVE_FE= args.SAVE_FE,
                 SAVE_PREFERENCE= args.SAVE_PREFERENCE,
                 SAVE_PATH= args.SAVE_PATH,
                 AGENT_EPSILON= args.AGENT_EPSILON, 
                 C_FE= args.C_FE,
                 N0_FE= args.N0_FE,
                 LAMBDA_FE= args.LAMBDA_FE,
                 CONJUGATE_PRIOR= args.CONJUGATE_PRIOR,
                 C_UCB= args.C_UCB,
                 B1_OUCB= args.B1_OUCB, 
                 B2_OUCB= args.B2_OUCB,
                 DIVERSITY= args.DIVERSITY, 
                 EPSILON= args.EPSILON,
                 LR= args.LR,
                 ALPHA_MEAN = args.ALPHA_MEAN,
                 ALPHA_STD = args.ALPHA_STD,
                 BETA_MEAN = args.BETA_MEAN,
                 BETA_STD = args.BETA_STD,
                 GAMMA_MEAN = args.GAMMA_MEAN, 
                 GAMMA_STD = args.GAMMA_STD,
                 N_SAMPLES_EXP_U = args.N_SAMPLES_EXP_U
                )
    
    simulate.simulate()
    simulate.visualize()