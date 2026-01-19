import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title ='DC Bike Rental',layout= 'wide')

#Loading and preprocessing the data
@st.cache_data
def load_data():
    data = pd.read_csv("train.csv")

    # Convert datetime
    data['datetime'] = pd.to_datetime(data['datetime'])
    data['year'] = data['datetime'].dt.year
    data['month'] = data['datetime'].dt.month
    data['day_of_the_week'] = data['datetime'].dt.day_name()
    data['hour'] = data['datetime'].dt.hour

    # Season mapping
    season_map = {1: 'spring', 2: 'summer', 3: 'fall', 4: 'winter'}
    data['season'] = data['season'].map(season_map)

    # Month mapping
    month_map = {
        1:'jan', 2:'feb', 3:'mar', 4:'apr', 5:'may', 6:'jun',
        7:'jul', 8:'aug', 9:'sep', 10:'oct', 11:'nov', 12:'dec'
    }
    data['month'] = data['month'].map(month_map)

    # Day period
    bins = [0,6,12,18,24]
    labels = ['night','morning','afternoon','evening']
    data['day_period'] = pd.cut(data['hour'], bins=bins, labels=labels, right=False)

    return data

data = load_data()

# Sidebar filters

st.sidebar.header("Filters")

years = st.sidebar.multiselect(
    "Select Year(s)",
    options=sorted(data["year"].unique()),
    default=sorted(data["year"].unique())
)

seasons = st.sidebar.multiselect(
    "Select Season(s)",
    options=data["season"].unique(),
    default=list(data["season"].unique())
)

workingday_filter = st.sidebar.selectbox(
    "Working Day Filter",
    ["Both", "Working Day Only", "Non-working Day Only"]
)

# Apply filters
df = data[data["year"].isin(years) & data["season"].isin(seasons)]

if workingday_filter == "Working Day Only":
    df = df[df["workingday"] == 1]
elif workingday_filter == "Non-working Day Only":
    df = df[df["workingday"] == 0]


#  Title

st.title("🚴 Washington D.C. Bike Rentals Dashboard")
st.markdown("Interactive dashboard summarizing bike rental patterns (2011–2012).")


# KPIs

col1, col2, col3 = st.columns(3)
col1.metric("Total Rentals", int(df["count"].sum()))
col2.metric("Average Hourly Rentals", round(df["count"].mean(), 2))
col3.metric("Max Hourly Rentals", int(df["count"].max()))

st.markdown("---")


# Plot 1: Mean rentals by season
st.subheader("Mean Hourly Rentals by Season")
season_group = df.groupby("season")["count"].mean().reset_index()

fig1, ax1 = plt.subplots()
sns.barplot(data=season_group, x="season", y="count", ax=ax1)
st.pyplot(fig1)

# Plot 2: Mean rentals by hour
st.subheader("Mean Hourly Rentals by Hour of Day")
hour_group = df.groupby("hour")["count"].mean().reset_index()

fig2, ax2 = plt.subplots()
sns.lineplot(data=hour_group, x="hour", y="count", marker="o", ax=ax2)
st.pyplot(fig2)

# Plot 3: Working vs Non-working day
st.subheader("Working vs Non-working Day Rentals")
wd_group = df.groupby("workingday")["count"].mean().reset_index()
wd_group["day_type"] = wd_group["workingday"].map({0:"Non-working", 1:"Working"})

fig3, ax3 = plt.subplots()
sns.barplot(data=wd_group, x="day_type", y="count", ax=ax3)
st.pyplot(fig3)


# Plot 4: Monthly rentals
st.subheader("Mean Hourly Rentals by Month")
month_group = df.groupby("month")["count"].mean().reset_index()

fig4, ax4 = plt.subplots()
sns.lineplot(data=month_group, x="month", y="count", marker="o", ax=ax4)
st.pyplot(fig4)


# Plot 5: Weather category rentals
st.subheader("Mean Rentals by Weather Category")
weather_group = df.groupby("weather")["count"].mean().reset_index()

fig5, ax5 = plt.subplots()
sns.barplot(data=weather_group, x="weather", y="count", ax=ax5)
st.pyplot(fig5)


# Plot 6: Correlation heatmap
st.subheader("Correlation Heatmap")
numeric_cols = df.select_dtypes(include=['int64','float64'])
corr = numeric_cols.corr()

fig6, ax6 = plt.subplots(figsize=(10,6))
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax6)
st.pyplot(fig6)


