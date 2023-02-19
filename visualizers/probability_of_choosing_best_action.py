import os
import pickle
from typing import List

import pandas as pd
import plotly.express as px


class ProbabilityOfChoosingBestActionVisualizer:
    def __init__(self, best_action: int, max_trial: int, data_path: str, write_path: str):
        self.best_action = best_action
        self.max_trial = max_trial
        self.data_path = data_path
        self.write_path = write_path

    def _read_results(self) -> dict:  # TODO: Use the correct type hint.
        agents_probabilities = {}
        for file_name in sorted(os.listdir(self.data_path)):
            with open(os.path.join(self.data_path, file_name), 'rb') as pickle_file:
                agent_actions = pickle.load(pickle_file)
                name = ''.join(file_name.split('.')[:-1])
                agents_probabilities[name] = agent_actions.probability_of_choosing_action(self.best_action)
        return agents_probabilities

    def _get_pandas(self, agents_probabilities: dict) -> List[pd.DataFrame]:
        agents_probabilities_pdf = []
        for agent in agents_probabilities:
            agent_max_trial = min(self.max_trial, len(agents_probabilities[agent]))
            probabilities_pdf = pd.DataFrame(
                agents_probabilities[agent][:agent_max_trial],
                columns=['probability of choosing the best action'],
            )
            probabilities_pdf['trial'] = list(range(agent_max_trial))
            probabilities_pdf['agent'] = agent
            agents_probabilities_pdf.append(probabilities_pdf)
        return agents_probabilities_pdf

    def visualize(self):
        agents_probabilities = self._read_results()
        agents_probabilities_pdf = self._get_pandas(agents_probabilities=agents_probabilities)
        agents_pdf = pd.concat(agents_probabilities_pdf)
        fig = px.line(agents_pdf, x='trial', y='probability of choosing the best action', color='agent')
        fig.write_html(self.write_path)


if __name__ == '__main__':
    visualizer = ProbabilityOfChoosingBestActionVisualizer(
        best_action=0,
        max_trial=100,
        data_path='/home/Alireza/results/actions/1',
        write_path='/home/Alireza/results/actions/1/visualization.html',
    )
    visualizer.visualize()
