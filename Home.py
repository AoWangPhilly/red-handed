import streamlit as st

with st.container() as intro:
    st.title("Red-Handed :oncoming_police_car: :cop:")
    st.write("By: Ao Wang")

    st.subheader("NYC Crime Data Analysis")
    st.write(
        "For my final project in DSCI 521 - Data Analysis and Visualization, I wanted to analyze the NYC crime data "
        "and create a web app to visualize the results. Previously, I've done a similar project with Philadelphia crime data, "
        "but I never got around to finish it. So I decided to continue with the NYC Complaint Data, which can be found "
        "[here](https://data.cityofnewyork.us/Public-Safety/NYPD-Complaint-Data-Historic/qgea-i56i), "
        "providing the public safety data from 2006 to 2021 and a very thorough data dictionary. "
    )
    st.image(
        "https://media0.giphy.com/media/3oFyD4xKncK6ptR7qg/giphy.gif?cid=6c09b9525cjjeh2e5ud7e6034s0s59h21e1y9zzdxxvu0v78&rid=giphy.gif&ct=g",
        use_column_width=True,
    )

with st.container() as inquiries:
    st.subheader("Inquiries")
    st.write(
        "A lot of the questions I had in mind came to me thinking how safe it would be to live in NYC. Thinking of moving there in a few years, I'm wondering how bad it would be compared to Philadelphia. Perhaps that'll be another project in the future. :wink: "
    )
    st.write(
        "1. How has NYC crime rates been over the years? Has it been increasing, decreasing, or stagnant? Are we able to predict crime rates? Perhaps linear regression or SARIMA model? \n"
        "2. Is there any external factors on crime rate? After analyzing and plotting the data, there seems to be some seasonality. Perhaps there's a correlation between temperature/weather and crime rates. There's less crime during the winter months and more crime during the summer months. Perhaps precipitation and snow fall would also have some effect.\n"
        "3. Given an area, such as a Borough or Police Precinct, what's the most common types of crime? And what're the frequencies for crimes to occur (monthly, daily, weekly, and hourly). Additionally, how fast are crimes being reported and how long do crimes take to occur?\n"
        "\n_All of these questions will be answered on the sidebar tabs on the left. Enjoy! :wave:_"
    )
