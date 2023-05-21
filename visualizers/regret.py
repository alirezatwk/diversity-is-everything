import argparse
import os
import pickle
from typing import List, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objs as go

class RegretVisualizer:
    def __init__(self, exp_u_best: float, max_trial: int, experiment_name:str, data_path: str, write_path: str):
        self.exp_u_best = exp_u_best
        self.max_trial = max_trial
        self.experiment_name = experiment_name
        self.data_path = data_path
        self.write_path = write_path

    def _read_results(self) -> dict:  # TODO: Use the correct type hint.
        agents_regret = {}
        agents_regret_eom = {}
        for file_name in sorted(os.listdir(self.data_path)):
            if 'pkl' not in file_name:
                continue
            with open(os.path.join(self.data_path, file_name), 'rb') as pickle_file:
                agent_rewards = pickle.load(pickle_file)
                name = ''.join(file_name.split('.')[:-1])
                agents_regret[name], agents_regret_eom[name]= agent_rewards.exp_pseudo_regret(self.exp_u_best)
        return agents_regret, agents_regret_eom

    def _get_pandas(self, agents_regret: dict, agents_regret_eom: dict) -> List[pd.DataFrame]:
        agents_regret_pdf = []
        for agent in agents_regret:
            agent_max_trial = min(self.max_trial, len(agents_regret[agent]))
            regrets_pdf = pd.DataFrame(
                agents_regret[agent][:agent_max_trial],
                columns=['Regret'],
            )
            regrets_pdf['error'] = agents_regret_eom[agent][:agent_max_trial]
            regrets_pdf['trial'] = list(range(agent_max_trial))
            regrets_pdf['agent'] = agent
            agents_regret_pdf.append(regrets_pdf)
        return agents_regret_pdf

    def _get_figure(self, agents_pdf: pd.DataFrame): # TODO: use type hint
        figure_with_error_bars = px.line(agents_pdf, x='trial', y='Regret', error_y='error',
                                         color='agent', title= self.experiment_name)
        fig = px.line(agents_pdf, x='trial', y='Regret', color='agent', title= self.experiment_name)
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
        agents_regret, agents_regret_eom = self._read_results()
        agents_regret_pdf = self._get_pandas(agents_regret=agents_regret, agents_regret_eom=agents_regret_eom)
        agents_pdf = pd.concat(agents_regret_pdf)
        fig = self._get_figure(agents_pdf=agents_pdf)
        fig.write_html(self.write_path)


if __name__ == '__main__':
    visualizer = RegretVisualizer(
        exp_u_best=0,
        max_trial=100,
        data_path='/home/Alireza/results/rewards/1',
        write_path='/home/Alireza/results/rewards/1/visualization.html',
    )
    visualizer.visualize()
