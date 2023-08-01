import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap
from mplsoccer import Pitch, VerticalPitch
from helpers import *


def get_match_label(match_id):
    global team_matches
    match = team_matches[team_matches['match_id'] == match_id]
    home_team = match['home_team'].iloc[0]
    home_score = match['home_score'].iloc[0]
    away_team = match['away_team'].iloc[0]
    away_score = match['away_score'].iloc[0]
    stage = match['competition_stage'].iloc[0]
    return f"{home_team} {home_score} - {away_score} {away_team} ({stage})"


# Start web page
st.title('Visualizaciones Futbol')

with st.sidebar:
    competitions = get_free_competitions()
    competition_regions = filter_competition_regions(competitions)
    intl_index = np.where(competition_regions.values == 'International')[0].item()
    selected_competition_region = st.selectbox('Selecciona una región para ver sus torneos', competition_regions,
                                               index=intl_index)

    country_competitions = filter_country_competitions(competitions, selected_competition_region)
    unique_competitions = country_competitions['competition_name'].drop_duplicates().sort_values()
    world_cup_index = np.where(unique_competitions.values == 'FIFA World Cup')[0].item()
    selected_competition_name = st.selectbox('Selecciona un torneo', unique_competitions, index=world_cup_index)

    selected_competitions = filter_competitions_by_name(country_competitions, selected_competition_name)
    selected_season_name = st.selectbox('Selecciona un año o temporada', selected_competitions['season_name'])
    selected_season = selected_competitions[selected_competitions['season_name'] == selected_season_name]

    matches = get_matches(selected_competitions['competition_id'].iloc[0], selected_season['season_id'].iloc[0])
    teams = pd.concat([matches['home_team'], matches['away_team']]).drop_duplicates().sort_values()
    selected_team = st.selectbox('Selecciona un equipo', teams)

    home_team_matches = matches[matches['home_team'] == selected_team]
    away_team_matches = matches[matches['away_team'] == selected_team]
    team_matches = pd.concat([home_team_matches, away_team_matches]).sort_values(by='match_date')
    selected_match_id = st.selectbox('Selecciona un equipo', team_matches, format_func=get_match_label,
                                     index=len(team_matches) - 1)

match_events = get_match_events(selected_match_id)
selected_match = team_matches[team_matches['match_id'] == selected_match_id]
home_team = selected_match['home_team'].iloc[0]
away_team = selected_match['away_team'].iloc[0]

st.subheader(f'Partido {get_match_label(selected_match_id)} de {selected_competition_name} en {selected_season_name}')
st.markdown('Selecciona el partido y torneo que deseas ver en el menú lateral. Si el menú está oculto presiona la \
flecha (>) de arriba a la izquierda')

st.subheader('Disparos a gol')
st.text('Los más oscuros representan goles')
st.pyplot(get_shots_fig(match_events, home_team, away_team))

passes = match_events.loc[(match_events['type'] == 'Pass') & (match_events['pass_type'] != 'Throw-in')]

st.subheader('Pases')
st.pyplot(get_passes_fig(passes, home_team, away_team))

st.subheader('Redes de pases')
st.pyplot(get_passes_network_fig(match_events, passes, home_team))
st.text(f'Centralization index: {get_centralization_index(match_events, passes, home_team)}')
st.pyplot(get_passes_network_fig(match_events, passes, away_team))
st.text(f'Centralization index: {get_centralization_index(match_events, passes, away_team)}')

st.subheader('Pases de peligro')
home_team_danger_passes = get_danger_passes_df(match_events, home_team)
st.pyplot(get_danger_passes_heatmap(home_team_danger_passes, home_team))
st.bar_chart(home_team_danger_passes.groupby(["player"]).x.count())

away_team_danger_passes = get_danger_passes_df(match_events, away_team)
st.pyplot(get_danger_passes_heatmap(away_team_danger_passes, away_team))
st.bar_chart(away_team_danger_passes.groupby(["player"]).x.count())
