import argparse
import configargparse
from utils.agent_creator import AgentCreator
from environments.multi_armed_bandit import MultiArmedBanditEnvironment
from rewards.multinomial import MultinomialReward

def get_args():
    """
    For example:
    """
    # TODO: write help of each argument
    # parser = argparse.ArgumentParser()
    parser = configargparse.ArgumentParser()

    parser.add_argument('-c', '--my-config', type = str, is_config_file=True, help='config file path') 

    parser.add_argument('--AGENTS_SOC_CLASS_NAME', action= 'append', required=True)
    parser.add_argument('--AGENTS_IND_CLASS_NAME', action= 'append', required=True)    
    parser.add_argument('--REWARD_VALUES', action= 'append',  type = float, required=True) 
    parser.add_argument('--REWARD_PROB_ARMS', action= 'append', type = float, required=True)
    parser.add_argument('--AGENTS_ID', action= 'append', required=True)
    parser.add_argument('--EXPERIMENT_NAME', type= str, required=True)

    
    parser.add_argument('--SAVE_FE', action= "store_true")
    parser.add_argument('--SAVE_PREFERENCE', action= "store_true")
    parser.add_argument('--SAVE_PATH', type= str, required=True)

    parser.add_argument('--SEED', default = 2048, type= int)
    parser.add_argument('--EPISODE_MAX_LENGTH', default = 200, type= int)
    parser.add_argument('--REPETITION',  default = 50, type= int)
    parser.add_argument('--PLOT_MAX_LENGTH', default = 200, type= int) 

    parser.add_argument('--C_FE', default = 0.5, type= float)
    parser.add_argument('--N0_FE', default = 100, type= int)
    parser.add_argument('--LAMBDA_FE', default = 0.99, type= float)
    parser.add_argument('--AGENT_EPSILON', default = 0.1, type= float)
    parser.add_argument('--C_UCB', default = 2, type= float)
    parser.add_argument('--B1_OUCB', default= 0.5, type= float)
    parser.add_argument('--B2_OUCB', default= 0.5, type= float)
    parser.add_argument('--LR', default =  0.1, type= float)
    parser.add_argument('--DIVERSITY', action= 'store_false') 

    parser.add_argument('--ENVIRONMENT_CLASS_NAME', default= MultiArmedBanditEnvironment, type= str) 
    parser.add_argument('--REWARD_CLASS_NAME', default= MultinomialReward, type= str)
    parser.add_argument('--AgentCreator_CLASS_NAME',  default = AgentCreator, type= str)
    parser.add_argument('--CONJUGATE_PRIOR', default= "Bernoulli", type= str)
    parser.add_argument('--EPSILON', default= 0.00001, type= float)
    parser.add_argument('--ALPHA_MEAN', default= 0.6, type= float)
    parser.add_argument('--ALPHA_STD', default= 0.1, type= float)
    parser.add_argument('--BETA_MEAN', default= 0.6, type= float)
    parser.add_argument('--BETA_STD', default= 0.1, type= float)
    parser.add_argument('--GAMMA_MEAN', default= 1.75, type= float)
    parser.add_argument('--GAMMA_STD', default= 0.25, type= float)
    parser.add_argument('--N_SAMPLES_EXP_U', default= 1000, type= int)

    arguments = parser.parse_args()
    print("-"*10)
    print(parser.format_values())
    return arguments