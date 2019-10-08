import sys
from os.path import dirname
from os.path import join, abspath

import numpy as np
import pandas as pd
from bokeh.core.properties import value
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, CustomJS
from bokeh.models.widgets import Slider, Button, DataTable, TableColumn, \
    NumberFormatter, Div
from bokeh.plotting import figure
from bokeh.transform import dodge

from ADvantage.scripts.knapsack import Knapsack

# Constants -------------------------------------------------------------------
cols = ['Keyword', 'daily_impressions_average', 'daily_clicks_average',
        'ad_position_average', 'cpc_average', 'daily_cost_average', 'source']
metric = cols[1:-1]

# Path to files
path = abspath(join(dirname(__file__), 'tmp'))
# read output of scraping and stats
full = pd.read_csv(join(path, sys.argv[1]), usecols=cols)
df = full.dropna()
nan_idx = df.index
nan_df = full[~full.index.isin(nan_idx)]
minimum = min(df.daily_cost_average[df.daily_cost_average > 0])
first_budget = max(minimum, 0.1)
maximum = min(400, max(df.daily_cost_average))
choice = ['GKP', 'Optimized']


# Functions ###################################################################
def optimize_values(data_frame, capacity):
    cost = data_frame.daily_cost_average.copy(deep=True)
    cost[cost == 0] = minimum / 10
    values = (data_frame.daily_impressions_average +
              data_frame.daily_clicks_average) * (1 / cost)
    opt = Knapsack(items_names=data_frame.Keyword.to_list(),
                       values=values.to_list(), capacity=capacity,
                       weights=data_frame.daily_cost_average.tolist(),
                       solve_type=5, name='Branch_n_bound')
    opt.get_results(print_it=True)
    return data_frame[data_frame.Keyword.isin(opt.packed_items)]


def set_table_source(dataframe):
    data = {'Keyword': dataframe.Keyword,
            'ad_position_average': dataframe.ad_position_average,
            'cpc_average': dataframe.cpc_average,
            'daily_clicks_average': dataframe.daily_clicks_average,
            'daily_cost_average': dataframe.daily_cost_average,
            'daily_impressions_average': dataframe.daily_impressions_average,
            'source': dataframe.source}
    source = ColumnDataSource(data=data)
    return data, source


# Body of app #################################################################
data, source = set_table_source(df)

data_missing, source_missing = set_table_source(nan_df)
current = optimize_values(df, first_budget)
gkp = optimize_values(df[df.source == 'GKP'], first_budget)
random = df.sample(current.shape[0])
relabel = ['Daily impressions', 'Daily Clicks', 'Ad Pos', 'CPC', 'Daily Cost']
bar_data = {'metric': relabel,
            choice[0]: np.log([gkp[x].sum() for x in metric]),
            choice[1]: np.log([current[x].sum() for x in metric])}

bar_source = ColumnDataSource(data=bar_data)

p = figure(x_range=relabel, y_range=(-5, 15), plot_height=325,
           toolbar_location=None, tools="",
           title='Relative Value change for baskets of words')
p.vbar(x=dodge('metric', -0.25, range=p.x_range), top=choice[0], width=0.2,
       source=bar_source, color="#c9d9d3", legend=value(choice[0]))
p.vbar(x=dodge('metric', 0.25, range=p.x_range), top=choice[1], width=0.2,
       source=bar_source, color="#e84d60", legend=value(choice[1]))
p.x_range.range_padding = 0.1
p.xgrid.grid_line_color = None
p.legend.location = "top_left"
p.legend.orientation = "horizontal"
p.yaxis.axis_label_text_font_size = '18pt'
p.xaxis.axis_label_text_font_size = '18pt'


def update():
    print('Slider Value', slider.value)
    current = optimize_values(df, slider.value)
    gkp = optimize_values(df[df.source == 'GKP'], slider.value)
    impressions = current.daily_impressions_average
    source.data = {'Keyword': current.Keyword,
                   'ad_position_average': current.ad_position_average,
                   'cpc_average': current.cpc_average,
                   'daily_clicks_average': current.daily_clicks_average,
                   'daily_cost_average': current.daily_cost_average,
                   'daily_impressions_average': impressions,
                   'source': current.source
                   }
    bar_data[choice[0]] = np.log([gkp[x].sum() for x in metric])
    bar_data[choice[1]] = np.log([current[x].sum() for x in metric])
    bar_source.data = bar_data


slider = Slider(title="Daily budget", start=minimum, end=maximum,
                value=first_budget, step=0.1, format="0,0")
slider.on_change('value', lambda attr, old, new: update())

button = Button(label="Download", button_type="success", width=400)
button.callback = CustomJS(args=dict(source=source),
                           code=open(join(dirname(__file__), "download.js")
                                     ).read())

columns = [
    TableColumn(field="Keyword", title="Keyword"),
    TableColumn(field="daily_cost_average", title="Cost",
                formatter=NumberFormatter(format="$0,0.00")),
    TableColumn(field="ad_position_average", title="Position"),
    TableColumn(field="daily_clicks_average", title="Clicks"),
    TableColumn(field="cpc_average", title="CPC"),
    TableColumn(field="daily_impressions_average", title='Impressions'),
    TableColumn(field="source", title='Source')
]

div = Div(text="""<b>Optimized keywords</b>""", width=400, height=20)
div_missing = Div(text="""<b>Keywords without data</b>""", width=400, height=20
                  )
data_table = DataTable(source=source, columns=columns, width=450, height=200)

data_table_nans = DataTable(source=source_missing, columns=columns,
                            width=450, height=200)

layout = row(column(div, data_table, button, sizing_mode="scale_width"),
             column(p, sizing_mode="scale_width"))
curdoc().add_root(slider)
curdoc().add_root(layout)
curdoc().add_root(row(column(div_missing, data_table_nans,
                             sizing_mode="scale_width"),
                      sizing_mode="scale_width"))
curdoc().title = "Export CSV"

update()
