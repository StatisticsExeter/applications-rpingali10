from course.intro.pipeline_functions import (
    tyler_viglen,
    calculate_correlation,
    fit_regression,
    plot_scatter
)
import os

# Make sure cache folder exists
cache_dir = 'course/intro/cache'
os.makedirs(cache_dir, exist_ok=True)

# Load Tyler Vigen data
df = tyler_viglen()

# 1️⃣ Calculate correlation and save
corr, p_value = calculate_correlation(df, 'Kerosene', 'DivorceRate')
with open(os.path.join(cache_dir, 'correlation.txt'), 'w') as f:
    f.write(f"Pearson correlation: {corr:.3f}\n")
    f.write(f"p-value: {p_value:.3e}\n")

# 2️⃣ Fit regression and save summary
model = fit_regression(df, 'Kerosene', 'DivorceRate')
with open(os.path.join(cache_dir, 'regression_summary.txt'), 'w') as f:
    f.write(model.summary().as_text())

# 3️⃣ Generate scatterplot and save HTML
fig = plot_scatter(df, 'Kerosene', 'DivorceRate')
fig.write_html(os.path.join(cache_dir, 'scatterplot.html'))

print("Cache files generated successfully!")
