import argparse
import os
import pickle
from typing import List, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objs as go


class ProbabilityOfSelectingAgentVisualizer:
    def __init__(self, n_agents: int, agents_id: list, max_trial: int, experiment_name:str, data_path: str, write_path: str):
        self.n_agents = n_agents
        self.agents_id = agents_id
        self.max_trial = max_trial
        self.experiment_name = experiment_name
        self.data_path = data_path
        self.write_path = write_path

    def _read_results(self) -> dict:  # TODO: Use the correct type hint.
        agents_probabilities = {}
        agents_errors = {}
        for file_name in sorted(os.listdir(self.data_path)):
            if 'pkl' not in file_name:
                continue
            with open(os.path.join(self.data_path, file_name), 'rb') as pickle_file:
                agent_agents = pickle.load(pickle_file)
                name = ''.join(file_name.split('.')[:-1])
                agents_probabilities[name] = {}
                agents_errors[name] = {}
                for i in range(self.n_agents):
                    agents_probabilities[name][i], agents_errors[name][i] = agent_agents.probability_of_choosing_action(i)
        return agents_probabilities, agents_errors

    def _get_pandas(self, agents_probabilities: dict, agents_errors: dict) -> List[pd.DataFrame]:
        agents_probabilities_pdf = []
        for agent in agents_probabilities:
            for i in range(self.n_agents):
                agent_max_trial = min(self.max_trial, len(agents_probabilities[agent][i]))
                probabilities_pdf = pd.DataFrame(
                    agents_probabilities[agent][i][:agent_max_trial],
                    columns=['probability of selecting agent'],
                )
                probabilities_pdf['trial'] = list(range(agent_max_trial))
                probabilities_pdf['agent'] = agent + f'_{self.agents_id[i]}'
                probabilities_pdf['error'] = agents_errors[agent][i][:agent_max_trial]
                agents_probabilities_pdf.append(probabilities_pdf)
        return agents_probabilities_pdf

    def _get_figure(self, agents_pdf: pd.DataFrame): # TODO: use type hint
        figure_with_error_bars = px.line(agents_pdf, x='trial', y='probability of selecting agent', error_y='error',
                                         color='agent', title= self.experiment_name)
        fig = px.line(agents_pdf, x='trial', y='probability of selecting agent', color='agent', title= self.experiment_name)
        for data in figure_with_error_bars.data:
            x = list(data['x'])
            y_upper = list(data['y'] + data['error_y']['array'])
            y_lower = list(
                data['y'] - data['error_y']['array'] if data['error_y']['arrayminus'] is None else data['y'] -
                                                                                                   data['error_y'][
                                                                                                       'arrayminus'])
            color = f"rgba({tuple(int(data['line']['color'].lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))},.3)".replace(
                '((', '(').replace('),', ',').replace(' ', '')
            fig.add_trace(
                go.Scatter(
                    x=x + x[::-1],
                    y=y_upper + y_lower[::-1],
                    fill='toself',
                    fillcolor=color,
                    line=dict(
                        color='rgba(255,255,255,0)'
                    ),
                    hoverinfo="skip",
                    showlegend=False,
                    legendgroup=data['legendgroup'],
                    xaxis=data['xaxis'],
                    yaxis=data['yaxis'],
                )
            )
        reordered_data = []
        for i in range(int(len(fig.data) / 2)):
            reordered_data.append(fig.data[i + int(len(fig.data) / 2)])
            reordered_data.append(fig.data[i])
        fig.data = tuple(reordered_data)
        return fig
    
    def visualize(self):
        agents_probabilities, agents_errors = self._read_results()
        agents_probabilities_pdf = self._get_pandas(agents_probabilities=agents_probabilities,
                                                    agents_errors=agents_errors)
        agents_pdf = pd.concat(agents_probabilities_pdf)
        fig = self._get_figure(agents_pdf=agents_pdf)
        fig.write_html(self.write_path)


if __name__ == '__main__':
    visualizer = ProbabilityOfSelectingAgentVisualizer(
        n_agents=0,
        max_trial=100,
        data_path='/home/Alireza/results/agents/1',
        write_path='/home/Alireza/results/agents/1/visualization.html',
    )
    visualizer.visualize()
