# drift_report.py
import pandas as pd
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metrics import DataDriftTable, TextOverviewTable
import sys

baseline, live, out_html = sys.argv[1], sys.argv[2], sys.argv[3]

ref = pd.read_csv(baseline)     # columns: text,label
cur = pd.read_csv(live)

report = Report(metrics=[TextOverviewTable(), DataDriftTable()])
report.run(reference_data=ref, current_data=cur,
           column_mapping=ColumnMapping(target="label", text_features=["text"]))
report.save_html(out_html)

