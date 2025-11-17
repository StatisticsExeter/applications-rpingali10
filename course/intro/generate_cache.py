# course/intro/generate_cache.py

import os
from course.intro.pipeline_functions import calculate_correlation, fit_regression, tyler_viglen, plot_scatter

# Make sure cache folder exists
CACHE_DIR = 'course/intro/cache'
os.makedirs(CACHE_DIR, exist_ok=True)

# 1. Load data (Tyler Vigen example)
df = tyler_viglen()

# 2. Compute correlation
corr, pval = calculate_correlation(df, 'Kerosene', 'DivorceRate')
with open(os.path.join(CACHE_DIR, 'correlation.txt'), 'w') as f:
    f.write(f"Correlation coefficient: {corr:.3f}\n")
    f.write(f"P-value: {pval:.5f}\n")
print("correlation.txt generated.")

# 3. Fit regression model
model = fit_regression(df, 'Kerosene', 'DivorceRate')
with open(os.path.join(CACHE_DIR, 'regression_summary.txt'), 'w') as f:
    f.write(model.summary().as_text())
print("regression_summary.txt generated.")

# 4. Generate scatterplot
fig = plot_scatter(df, 'Kerosene', 'DivorceRate')
fig.write_html(os.path.join(CACHE_DIR, 'scatterplot.html'))
print("scatterplot.html generated.")
