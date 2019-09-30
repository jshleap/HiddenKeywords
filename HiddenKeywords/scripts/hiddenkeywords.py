"""
**hiddenkeywords.py**
**Copyright** 2019  Jose Sergio Hleap
Given a landing page, and the Google Keyword planner for your campaign,
create a dashboard that allows you to explore basket of words based on daily
budget
"""
from HiddenKeywords.HiddenKeywords.scripts.gimmewords import *
from HiddenKeywords.HiddenKeywords.scripts.knapsack import *
import os
from bokeh.io import curdoc
from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource, CustomJS
from bokeh.models.widgets import Slider, Button, DataTable, TableColumn, NumberFormatter
import sys
landing_page, stats, budget = sys.argv[1:4]
num, stop = 100, 10
# def main(landing_page, stats, budget, max_results=100, stop=10):
if isfile('pages.dmp'):
    with open('pages.dmp') as p, open('landing.dmp') as l:
        text = [line for line in p]
        land = ' '.join([line for line in l])
else:
    now = time.time()
    pages = GetPages(landing_page, num, stop)
    elapsed = (time.time() - now) / 60
    print("Crawling done in", elapsed, 'minutes')
    to_str = lambda x: x if isinstance(x, str) else '\n'.join(x)
    text = pages.text
    land = to_str(pages.landing_page)
    with open('pages.dmp', 'w') as p, open('landing.dmp', 'w') as l:
        p.write(to_str(text))
        l.write(land)
iw = IdentifyWords(text, stats, land)
s = set([x[0] for y in iw.pre_keywords if y for x in y for y in
         iw.pre_keywords if y] + iw.landing_kw)
if os.path.isfile('df_checkpoint.csv'):
    df = pd.read_csv('df_checkpoint.csv', index_col=0)
else:
    stats = get_stats(list(s), 'jshleap', 'xtmH9EEGFRr5bqgJ')
    df = pd.DataFrame(stats[0].values(), index=stats[0].keys())
values = (df.daily_impressions_average + df.daily_clicks_average) * \
         (1/df.daily_cost_average)
ks = Knapsack(items_names=df.index.to_list(), values=values.to_list(),
              weights=df.daily_cost_average.tolist(), capacity=budget,
              solve_type=5, name='Branch_n_bound')
ks.get_results()
selected = df[df.index.isin(ks.packed_items)]




#path = join(abspath(join(dirname(__file__), pardir)), 'resources')
df_opt = dict(skiprows=[0, 1], encoding=detect_encoding(mock_GKP_result), sep='\t')
gkp = pd.read_csv(stats, **df_opt)
source = ColumnDataSource(data=dict())
slider = Slider(title="offset", value=5.0, start=0.0, end=100.0, step=1)


def update():
    ks.capacity = slider.value
    ks.get_results()
    current = df[df.index.isin(ks.packed_items)]
    source.data = {
        'ad_position_average': current.ad_position_average,
        'cpc_average': current.cpc_average,
        'daily_clicks_average': current.daily_clicks_average,
        'daily_cost_average': current.daily_cost_average,
        'daily_impressions_average': current.daily_impressions_average,
    }


slider.on_change('value', lambda attr, old, new: update())

button = Button(label="Download", button_type="success")
button.callback = CustomJS(args=dict(source=source),
                           code=open(join(dirname(__file__),
                                          "download.js")).read())

columns = [
    TableColumn(field="ad_position_average", title="Ad Position"),
    TableColumn(field="cpc_average", title="Cost per Click (CPC)",
                formatter=NumberFormatter(format="$0,0.00")),
    TableColumn(field="daily_clicks_average", title="Daily Clicks"),
    TableColumn(field="daily_cost_average", title="Daily Cost",
                formatter=NumberFormatter(format="$0,0.00")),
    TableColumn(field="daily_impressions_average", title="Daily Impression"
                                                         "s")
]

data_table = DataTable(source=source, columns=columns, width=800)
controls = column(slider, button)
curdoc().add_root(row(controls, data_table))
curdoc().title = "Export CSV"
update()

#if __name__ == '__main__':

# main(sys.argv[1], sys.argv[2], sys.argv[3])

