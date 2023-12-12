""" Assignment 2: Statistics and trends. """
import pandas as pd
import numpy as np
import requests

from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
# Set the style for seaborn plots
sns.set(style="whitegrid")

base_url = 'http://api.worldbank.org/v2/'

# mapping of feature codes to more meaningful names
indicators_map = {
    "SP.POP.TOTL": "Total Population",
    "SP.POP.TOTL.FE.IN": "Female Population",
    "SP.POP.TOTL.MA.IN": "Male Population",
    "SP.DYN.CBRT.IN": "Birth Rate",
    "SP.DYN.CDRT.IN": "Death Rate",
    "EG.USE.ELEC.KH.PC": "Electric Power Consumption(kWH per capita)",
    "EG.FEC.RNEW.ZS": "Renewable Energy Consumption (%)",
    "EG.USE.COMM.FO.ZS": "Fossil Fuel Consumption (%)"
}
indicators_code = list(indicators_map.keys())

# Mapping of country codes to their actual names
countries_map = {
    "CN": "China",
    "GB": "Great Britain",
    "IN": "India",
    "JP": "Japan",
    "ZA": "South Africa",
    "US": "USA",
}
country_list = list(countries_map.values())
country_codes = list(countries_map.keys())

params = dict()
params['format'] = 'json'
params['per_page'] = '100'
params['date'] = '1970:2020'

# Function to get JSON data from the endpoint


def load_JSON_data(country_code):
    dataList = []

    for indicator in indicators_code:

        # form the URL in the desired format
        url = base_url + 'countries/' + country_code.lower() + '/indicators/' + indicator
        # print(url)
        # send the request using the resquests module
        response = requests.get(url, params=params)
        # fetch result
        if response.status_code == 200 and (
                "message" not in response.json()[0].keys()):
            # list of values for one feature
            indicatorVals = []
            # check if the length of the response is >1
            if len(response.json()) > 1:
                # each object gives one single value for each year
                for obj in response.json()[1]:
                    # handle empty values
                    if obj['value'] == "" or obj['value'] is None:
                        indicatorVals.append(None)
                    else:
                        # add value it to the list of indicator values
                        indicatorVals.append(float(obj['value']))
                dataList.append(indicatorVals)
        else:
            print("Error in Loading the data. Status Code: " +
                  str(response.status_code))

    # This API returns the indicator values from the most recent year, years
    # in reverse order
    dataList.append([year for year in range(2020, 1969, -1)])
    # return the list of lists of feature values
    return dataList

# DataFrame for each country


def getCountrywiseDF(country_code):
    # create a list of column names from the indicators_map
    col_list = list(indicators_map.values())
    # append the year column name
    col_list.append('Year')
    # from the API
    dataList = load_JSON_data(country_code)

    # transform the list of lists of features into a DataFrame=
    df = pd.DataFrame(np.column_stack(dataList), columns=col_list)

    # add the country column by from the map using the country code
    df['Country'] = countries_map[country_code]

    # print(df.head())

    # return the formed dataframe for the given country
    return df


def fetch_data(country_codes):
    # Call the getCountrywiseDF function to fetch data
    # list to hold all countries dataframe
    df_list = list()
    print('Fetching data...')
    for country_code in country_codes:
        df_country = getCountrywiseDF(country_code)
        df_list.append(df_country)
    print("Done Fetching Data.")
    return df_list

# identify missing features and remove features that aren't useful


def remove_missing_features(df, remove_threshold_percent=70):
    # validation for dataframe
    if df is None:
        print("No DataFrame received!")
        return
    # create a copy of the dataframe to avoid changing the original
    df_cp = df.copy()

    print("Removing missing features for: " + df_cp.iloc[0]['Country'])

    # find features with non-zero missing values
    n_missing_vals = df.isnull().sum()

    # get the index list of the features with non-zero missing values
    n_missing_index_list = list(n_missing_vals.index)

    # calculate the percentage of missing values
    missing_percentage = n_missing_vals[n_missing_vals !=
                                        0] / df.shape[0] * 100
    # list to maintain the columns to drop
    cols_to_trim = []
    # iterate over each key value pair
    for i, val in enumerate(missing_percentage):
        # if percentage value is > missing_percentage
        if val > remove_threshold_percent:
            # add the corresponding column to the list of cols_to_trim
            cols_to_trim.append(n_missing_index_list[i])

    if len(cols_to_trim) > 0:
        df_cp = df_cp.drop(columns=cols_to_trim)
        print("Dropped Columns:" + str(cols_to_trim))
    else:
        print("No columns dropped")

    return df_cp


# fill missing values with mean values for columns
def fill_missing_values(df):
    # validation for dataframes
    if df is None:
        print("No DataFrame received")
        return
    # create a copy
    df_cp = df.copy()

    print("Filling missing features for: " + df_cp.iloc[0]['Country'])

    # get the list of columns in the dataframe
    cols_list = list(df_cp.columns)
    # ignore country
    cols_list.pop()

    # replace all None values with NaN, fillna only works on nans
    # df_cp.fillna(value=np.nan, inplace=True)
    # replace all NaN values with the mean of the column values
    for col in cols_list:
        df_cp[col].fillna((df_cp[col].mean()), inplace=True)

    print("Filling missing values completed.")
    return df_cp

# change Year and COuntry to object and other values to float


def change_data_types(df):
    cols = df.columns
    for col in cols:
        if col in ['Year', 'Country']:
            df[col] = df[col].astype(str)
        else:
            df[col] = df[col].astype(float)
    return df


df_list = fetch_data(country_codes)
df_list = list(map(remove_missing_features, df_list))
df_list = list(map(fill_missing_values, df_list))
df_list = list(map(change_data_types, df_list))

# create a combined dataframe
full_df = pd.concat(df_list, ignore_index=True)
# show different statistics
print(full_df.head())
print(full_df.info())
print(full_df.describe())
print(full_df.isnull().sum())

# save the full dataframe to a csv file
# full_df.to_csv("full_df.csv")

# correlation matrix of all countries


def corr_all(df):
    # Exclude the categorical features from the matrix
    df.drop(['Year', 'Country'], inplace=True, axis='columns')
    # plot a correlation matrix
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(df.corr(), cmap='RdBu', center=0, ax=ax)
    plt.title("Correaltion of features for all countries")
    plt.show()


def compare_population(df):
    list_countries = country_list
    df1 = pd.DataFrame()
    df2 = pd.DataFrame()

    for i, df in enumerate(df):
        # pick the value of Total Population for year 2000 and 2018
        df1[list_countries[i]] = df[df['Year'] == '1970']["Total Population"]
        df2[list_countries[i]] = df[df['Year'] == '2020']["Total Population"]
    df1 = df1.T
    df2 = df2.T

    # set other global format
    pd.options.display.float_format = '{:,.1f}'.format

    # rename the columns to the year
    df1 = df1.rename(columns={50: 1970})
    df2 = df2.rename(columns={0: 2020})

    # join the dataframes for both the years
    df_comp = df1.join(df2)
    # the index is the Country name, hence we add it as a column into the data
    # frame.
    df_comp['Countries'] = df_comp.index

    # drop the original index
    df_comp.reset_index(drop=True)
    return df_comp


def population_bar_plot(df):
    # population in 1970  vs in 2020
    # plt.figure(figsize=(10, 6))
    # plot the chart using matplotlib.pyplot library
    df.plot(kind='bar', x='Countries', y=[1970, 2020])

    plt.title("Countries Population in 1970 vs 2020(in 50 Years).")
    plt.xlabel("Countries")
    plt.ylabel("Population")
    plt.show()


def group_population_bar_df(df, feature):
    df_grouped = pd.DataFrame()
    # find average for each country
    df_grouped['Avg. ' + feature] = df.groupby('Country')[feature].mean()
    # set the index as a column - countries
    df_grouped['Country'] = df_grouped.index
    # drop the index
    df_grouped.reset_index(drop=True, inplace=True)
    # sort the rows based of Avg Birth rate
    df_grouped.sort_values('Avg. ' + feature, inplace=True, ascending=False)

    return df_grouped


def plot_bar(df, x_feature, y_feature):
    # bar plot
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=df,
        x=x_feature,
        y="Avg. " + y_feature)
    plt.title("Average " + y_feature + " for Countries")
    plt.xlabel(x_feature)
    plt.ylabel("Avg. " + y_feature)


def group_population_line_df(df, feature):
    # create a new dataframe
    df_grouped = pd.DataFrame()

    # find average for each country
    df_grouped['Avg. ' + feature] = df.groupby('Year')[feature].mean()

    # set the index as a column - countries
    df_grouped['Year'] = df_grouped.index

    # drop the index
    df_grouped.reset_index(drop=True, inplace=True)

    # sort the rows based of Avg Birth rate
    df_grouped.sort_values('Avg. ' + feature, inplace=True, ascending=False)

    print("Avg. " + feature)
    # display(df_grouped)

    return df_grouped


def plot_line(df, x_feature, y_feature):
    # bar plot
    plt.figure(figsize=(10, 6))
    ax = sns.lineplot(
        data=df,
        x=x_feature,
        y="Avg. " + y_feature)

    plt.title("Average " + y_feature + " Over Time")
    plt.xlabel(x_feature)
    plt.ylabel("Avg. " + y_feature)


def group_population_line_df2(df, feature):
    # create a new dataframe
    df_grouped = pd.DataFrame()
    # find average for each country
    df_grouped['Avg. ' +
               feature] = df.groupby(['Year', 'Country'])[feature].mean()
    # reset index to have 'Year' and 'Country' as columns
    df_grouped.reset_index(inplace=True)
    # sort the rows based on Avg feature
    df_grouped.sort_values(['Avg. ' + feature, 'Year'],
                           inplace=True, ascending=[False, True])

    return df_grouped


def plot_line2(df, x_feature, y_feature, hue_feature=None):
    # line plot with multiple lines for each country
    plt.figure(figsize=(10, 6))
    # Calculate the tick interval dynamically
    unique_years = df['Year'].unique()
    total_years = len(unique_years)
    max_years = 12
    tick_interval = max(1, total_years // max_years)
    ax = sns.lineplot(
        data=df,
        x=x_feature,
        y="Avg. " + y_feature,
        hue=hue_feature,
        marker="o",
        markersize=5
    )
    # set the x-axis ticks and labels based on the chosen interval
    plt.xticks(range(0, total_years, tick_interval),
               unique_years[::tick_interval])
    plt.title("Average " + y_feature + " Over Time")
    plt.xlabel(x_feature)
    plt.ylabel("Avg. " + y_feature)
    if hue_feature is not None:
        plt.legend(
            title=hue_feature,
            loc='upper left',
            bbox_to_anchor=(
                0.80,
                1))


def plot_population_energy_line(df):
    # Line plot for Population and Total Electric Power Consumption
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df.groupby('Year').sum().reset_index(),
        x='Total Population',
        y='Electric Power Consumption(kWH per capita)',
        label='Population vs Electric Power Consumption')

    plt.title('Population vs Electric Power Consumption 1970-2020 Years')
    plt.xlabel('Population')
    plt.ylabel('Electric Power Consumption (kWH per capita)')
    plt.show()


def plot_power_consumption_bar(df):
    # Group data by country and calculate the total electric power consumption
    # for each country
    df_grouped = df.groupby('Country')[
        'Electric Power Consumption(kWH per capita)'].sum().reset_index()

    # Bar plot for Total Electric Power Consumption
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x='Country',
        y='Electric Power Consumption(kWH per capita)',
        data=df_grouped)
    plt.title('Total Electric Power Consumption for Each Country (Over 50 Years)')
    plt.xlabel('Country')
    plt.ylabel('Total Electric Power Consumption (kWH per capita)')
    plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
    plt.show()


def plot_power_consumption_bar2(df):
    # Group data by country and calculate the total percentages for renewable
    # and fossil fuel consumption for each country
    df_grouped = df.groupby('Country')[
        ['Renewable Energy Consumption (%)', 'Fossil Fuel Consumption (%)']].sum().reset_index()

    # Melt the DataFrame to have 'Energy Type' as a separate column
    df_melted = pd.melt(
        df_grouped,
        id_vars=['Country'],
        value_vars=[
            'Renewable Energy Consumption (%)',
            'Fossil Fuel Consumption (%)'],
        var_name='Energy Type',
        value_name='Percentage')

    # Bar plot for Renewable and Fossil Fuel Consumption Percentages
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Country', y='Percentage', hue='Energy Type', data=df_melted)
    plt.title('Total Renewable and Fossil Fuel Consumption for Each Country')
    plt.xlabel('Country')
    plt.ylabel('Energy Consumption (%)')
    plt.legend(title='Energy Type')
    plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
    plt.show()


def plot_power_consumption_line(df):
    # Group data by year and calculate the total electric power consumption
    # for each year
    df_grouped = df.groupby(
        'Year')['Electric Power Consumption(kWH per capita)'].sum().reset_index()

    # Line plot for Total Electric Power Consumption
    plt.figure(figsize=(10, 6))
    # Calculate the tick interval dynamically
    unique_years = df['Year'].unique()
    total_years = len(unique_years)
    max_years = 12
    tick_interval = max(1, total_years // max_years)
    sns.lineplot(
        x='Year',
        y='Electric Power Consumption(kWH per capita)',
        data=df_grouped)
    # set the x-axis ticks and labels based on the chosen interval
    plt.xticks(range(0, total_years, tick_interval),
               unique_years[::tick_interval])
    plt.title('Total Electric Power Consumption Over 50 Years')
    plt.xlabel('Year')
    plt.ylabel('Total Electric Power Consumption (kWH per capita)')
    plt.show()


def plot_power_consumption_line2(df):
    # Group data by year and calculate the total percentages for renewable and
    # fossil fuel consumption for each year
    df_grouped = df.groupby('Year')[
        ['Renewable Energy Consumption (%)', 'Fossil Fuel Consumption (%)']].sum().reset_index()

    # Line plot for Renewable and Fossil Fuel Consumption Percentages
    plt.figure(figsize=(10, 6))
    # Calculate the tick interval dynamically
    unique_years = df['Year'].unique()
    total_years = len(unique_years)
    max_years = 12
    tick_interval = max(1, total_years // max_years)
    sns.lineplot(
        x='Year',
        y='Renewable Energy Consumption (%)',
        data=df_grouped,
        label='Renewable Energy')
    # set the x-axis ticks and labels based on the chosen interval
    plt.xticks(range(0, total_years, tick_interval),
               unique_years[::tick_interval])
    sns.lineplot(
        x='Year',
        y='Fossil Fuel Consumption (%)',
        data=df_grouped,
        label='Fossil Fuel')
    plt.title('Total Renewable and Fossil Fuel Consumption Over 50 Years')
    plt.xlabel('Year')
    plt.ylabel('Energy Consumption (%)')
    plt.legend(title='Energy Type')
    plt.show()


df = full_df.copy()
corr_all(df)

df_comp = compare_population(df_list)
population_bar_plot(df_comp)

df_birth = group_population_bar_df(full_df.copy(), 'Birth Rate')
plot_bar(df_birth, 'Country', 'Birth Rate')

df_death = group_population_bar_df(full_df.copy(), 'Death Rate')
plot_bar(df_death, 'Country', 'Death Rate')

df_birth2 = group_population_line_df(full_df.copy(), 'Birth Rate')
plot_line2(df_birth2, 'Year', 'Birth Rate')

df_death2 = group_population_line_df(full_df.copy(), 'Death Rate')
plot_line2(df_death2, 'Year', 'Death Rate')

df_birth3 = group_population_line_df2(full_df.copy(), 'Birth Rate')
plot_line2(df_birth3, 'Year', 'Birth Rate', hue_feature='Country')

df_death3 = group_population_line_df2(full_df.copy(), 'Death Rate')
plot_line2(df_death3, 'Year', 'Death Rate', hue_feature='Country')

plot_population_energy_line(full_df.copy())

plot_power_consumption_bar(full_df.copy())

plot_power_consumption_bar2(full_df.copy())

plot_power_consumption_line(full_df.copy())

plot_power_consumption_line2(full_df.copy())
