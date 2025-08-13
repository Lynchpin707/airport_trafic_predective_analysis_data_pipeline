import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def clean_data(path='./data/trafic_ma_long.csv'):
    #load the raw data from the respective file
    df = pd.read_csv(path)

    #Transforming the tables to compare Jan-Jun traffic to that of the entire year (cumulative)
    df_pivot = df.pivot_table(index=["Airport", "Year"], 
                            columns="Month", 
                            values="Traffic").reset_index()
    #Renaming columns to respect our naming conventions
    df_pivot = df_pivot.rename(columns={
        "Dec": "cumulative_annual",
        "Jun": "s1",
        "Airport":"airport",
        "Year":"year"
    })
    df_pivot['airport']=df_pivot['airport'].str.lower()

    #Create a new column s2 that only contains the traffic generated in Jun-Dec
    df_pivot["s2"] = df_pivot["cumulative_annual"] - df_pivot["s1"]

    #Transform our table for visualisation
    df_long = pd.melt(df_pivot, 
                    id_vars=["airport", "year"], 
                    value_vars=["s1", "s2"], 
                    var_name="semester", 
                    value_name="traffic")
    df_long["period"] = df_long["year"].astype(str) + "_" + df_long["semester"]

    df_long = df_long.sort_values(by=["airport", "year", "semester"])
    df_long = df_long.reset_index(drop=True)

    df_long.to_csv('./cleaned_data/cleaned_trafic_ma_long.csv')

    #Visualise the traffic trends per airport per year
    sns.set_style("darkgrid")
    ax = sns.lineplot(data=df_long,x="period",y="traffic",hue="airport")
    ax.tick_params(axis='x', rotation=90)

    #save the figure
    plt.tight_layout()  # avoid cutting off labels
    plt.savefig("./docs/outputs/traffic_by_airport.png", dpi=300, bbox_inches="tight")
    plt.close()