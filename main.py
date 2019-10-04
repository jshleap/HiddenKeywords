from os.path import dirname, join

import pandas as pd

from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource, CustomJS
from bokeh.models.widgets import RangeSlider, Button, DataTable, TableColumn, NumberFormatter
from bokeh.io import curdoc

df_opt = dict(skiprows=[0, 1], encoding='utf-16', sep='\t')
stats = join(
    dirname(__file__), 'data', 'Paperspace_KW_Stats_2019-09-23_at_14_37_49.csv'
)
gkp = pd.read_csv(stats, **df_opt)
df = pd.read_csv(join(dirname(__file__), 'df_checkpoint.csv'), index_col=0)

source = ColumnDataSource(data=dict())

def update():
    current = df[(df['salary'] >= slider.value[0]) & (df['salary'] <= slider.value[1])].dropna()
    source.data = {
        'name'             : current.name,
        'salary'           : current.salary,
        'years_experience' : current.years_experience,
    }

slider = RangeSlider(title="Max Salary", start=10000, end=110000, value=(10000, 50000), step=1000, format="0,0")
slider.on_change('value', lambda attr, old, new: update())

button = Button(label="Download", button_type="success")
button.callback = CustomJS(args=dict(source=source),
                           code=open(join(dirname(__file__), "download.js")).read())

columns = [
    TableColumn(field="name", title="Employee Name"),
    TableColumn(field="salary", title="Income", formatter=NumberFormatter(format="$0,0.00")),
    TableColumn(field="years_experience", title="Experience (years)")
]

data_table = DataTable(source=source, columns=columns, width=800)

controls = column(slider, button)

curdoc().add_root(row(controls, data_table))
curdoc().title = "Export CSV"

update()