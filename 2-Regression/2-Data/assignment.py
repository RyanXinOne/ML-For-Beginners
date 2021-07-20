'''There are several different libraries that are available for data visualization. Create some visualizations using the Pumpkin data in this lesson with matplotlib and seaborn in a sample notebook. Which libraries are easier to work with?'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

new_pumpkins = pd.read_csv("new_pumpkins.csv")
print(new_pumpkins.head())

# plot line chart using matplotlib
plt.plot(sorted(new_pumpkins.Month.unique()), new_pumpkins.groupby(['Month'])['Low Price'].mean(), color='blue', label="Low Price")
plt.plot(sorted(new_pumpkins.Month.unique()), new_pumpkins.groupby(['Month'])['High Price'].mean(), color='orange', label="High Price")
plt.xticks(sorted(new_pumpkins.Month.unique()))
plt.xlabel("Month")
plt.ylabel("Pumpkin Price")
plt.legend()
plt.show()

# plot line chart using seaborn

# sns.lineplot(x='Month', y='Low Price', data=new_pumpkins.groupby(['Month']).mean())
# sns.lineplot(x='Month', y='High Price', data=new_pumpkins.groupby(['Month']).mean())
# plt.legend(labels=['Low Price', 'High Price'])
sns.lineplot(data=new_pumpkins.groupby(['Month'])[['Low Price', 'High Price']].mean())

plt.xticks(sorted(new_pumpkins.Month.unique()))
plt.ylabel("Pumpkin Price")
plt.show()
