import argparse
import os
import pickle
from typing import List, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objs as go

class FreeEnergyVisualizer:
    def __init__(self, max_trial: int, experiment_name:str, data_path: str, write_path: str):
        self.max_trial = max_trial
        self.experiment_name = experiment_name
        self.data_path = data_path
        self.write_path = write_path

    def _read_results(self) -> dict:  # TODO: Use the correct type hint.
        agents_fe = {}
        agents_errors = {}
        for file_name in sorted(os.listdir(self.data_path)):
            if 'pkl' not in file_name:
                continue
            with open(os.path.join(self.data_path, file_name), 'rb') as pickle_file:
                agent_fe = pickle.load(pickle_file)
                AGENTS_FE, AGENTS_FE_ERROR = agent_fe.average_free_energy()
                name = ''.join(file_name.split('.')[:-1]) 
                agents_fe[name] = {} 
                agents_errors[name] = {}
                for agent_id in agent_fe.agents_id: 
                    agents_fe[name][agent_id] = AGENTS_FE[agent_fe.agents_id.index(agent_id)]
                    agents_errors[name][agent_id] = AGENTS_FE_ERROR[agent_fe.agents_id.index(agent_id)]
        return agents_fe, agents_errors

    def _get_pandas(self, agents_fe: dict, agents_fe_errors: dict) -> List[pd.DataFrame]:
        agents_fe_pdf = []
        for main_agent_id in agents_fe:
            for agent_id in agents_fe[main_agent_id]:
                agent_max_trial = min(self.max_trial, len(agents_fe[main_agent_id][agent_id]))
                fes_pdf = pd.DataFrame(
                    agents_fe[main_agent_id][agent_id][:agent_max_trial],
                    columns=['Free Energy'],
                )
                fes_pdf['trial'] = list(range(agent_max_trial))
                fes_pdf['name'] = main_agent_id + "_" + agent_id
                fes_pdf['error'] =  agents_fe_errors[main_agent_id][agent_id][:agent_max_trial]
                agents_fe_pdf.append(fes_pdf)
        return agents_fe_pdf
   
    def _get_figure(self, agents_pdf: pd.DataFrame): # TODO: use type hint
        figure_with_error_bars = px.line(agents_pdf, x='trial', y='Free Energy', error_y='error',
                                         color='name', title= self.experiment_name)
        fig = px.line(agents_pdf, x='trial', y='Free Energy', color='name', title= self.experiment_name)
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
        agents_fe, agents_errors = self._read_results()
        agents_fe_pdf = self._get_pandas(agents_fe=agents_fe, agents_fe_errors= agents_errors)
        agents_pdf = pd.concat(agents_fe_pdf)
        fig = self._get_figure(agents_pdf=agents_pdf)
        fig.write_html(self.write_path)


if __name__ == '__main__':
    visualizer = FreeEnergyVisualizer(
        exp_u_best=0,
        max_trial=100,
        data_path='/home/Alireza/results/rewards/1',
        write_path='/home/Alireza/results/rewards/1/visualization.html',
    )
    visualizer.visualize()
