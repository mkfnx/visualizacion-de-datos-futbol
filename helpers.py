import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from statsbombpy import sb
from mplsoccer import Pitch


@st.cache_data
def get_free_competitions():
    return sb.competitions()


@st.cache_data
def get_matches(competition_id, season_id):
    return sb.matches(competition_id, season_id)


@st.cache_data
def get_match_events(match_id):
    return sb.events(match_id)


@st.cache_data
def get_lineups(match_id):
    return sb.lineups(match_id)


def filter_competition_regions(competitions):
    return competitions['country_name'].drop_duplicates().sort_values()


def filter_country_competitions(competitions, competition_region):
    return competitions[competitions['country_name'] == competition_region]


def filter_competitions_by_name(country_competitions, competition_name):
    return country_competitions[country_competitions['competition_name'] == competition_name]


def get_shots_fig(match_events, home_team, away_team):
    shots = match_events.loc[match_events['type'] == 'Shot']

    pitch = Pitch(line_color='black')
    fig, ax = pitch.grid(grid_height=0.9, title_height=0.06, axis=False,
                         endnote_height=0.04, title_space=0, endnote_space=0)
    for i, shot in shots.iterrows():
        x = shot['location'][0]
        y = shot['location'][1]

        if shot["shot_outcome"] == 'Goal':
            if shot['team'] == home_team:
                pitch.scatter(x, y, alpha=1, s=500, color="red", ax=ax['pitch'])
                pitch.annotate(shot["player"], (x + 1, y - 2), ax=ax['pitch'], fontsize=12)
            else:
                pitch.scatter(120 - x, 80 - y, alpha=1, s=500, color="blue", ax=ax['pitch'])
                pitch.annotate(shot["player"], (120 - x + 1, 80 - y - 2), ax=ax['pitch'], fontsize=12)
        else:
            if shot['team'] == home_team:
                pitch.scatter(x, y, alpha=0.2, s=500, color="red", ax=ax['pitch'])
            else:
                pitch.scatter(120 - x, 80 - y, alpha=0.2, s=500, color="blue", ax=ax['pitch'])

    fig.suptitle(f"Disparos de {home_team} (rojo) y {away_team} (azul)", fontsize=26)
    return fig


def get_passes_fig(passes, home_team, away_team):
    home_team_players = passes[passes['team'] == home_team]['player'].sort_values().unique()
    away_team_players = passes[passes['team'] == away_team]['player'].sort_values().unique()
    pass_players = np.concatenate([home_team_players, away_team_players])
    selected_player = st.selectbox('Selecciona un jugador', pass_players)
    player_passes = passes[passes['player'] == selected_player]

    pitch = Pitch(line_color='black')
    fig, ax = pitch.grid(grid_height=0.9, title_height=0.06, axis=False,
                         endnote_height=0.04, title_space=0, endnote_space=0)

    for i, player_pass in player_passes.iterrows():
        player_x = player_pass['location'][0]
        player_y = player_pass['location'][1]
        pass_end_x = player_pass['pass_end_location'][0]
        pass_end_y = player_pass['pass_end_location'][1]

        pitch.arrows(player_x, player_y,
                     pass_end_x, pass_end_y, color="blue", ax=ax['pitch'])
        pitch.scatter(player_x, player_y, alpha=0.2, s=500, color="blue", ax=ax['pitch'])
        fig.suptitle(f"Pases de {selected_player}", fontsize=26)

    return fig


def get_first_substitution_index(match_events, team):
    substitutions = match_events.loc[match_events["type"] == "Substitution"]
    return substitutions.loc[match_events["team"] == team].iloc[0]["index"]


def get_passes_network_fig(match_events, passes, team):
    first_sub_index = get_first_substitution_index(match_events, team)
    passes_for_network = passes[
        (passes['team'] == team) & (passes.index < first_sub_index) & (passes['pass_outcome'].isnull())]
    passes_for_network = passes_for_network[['location', 'pass_end_location', "player", "pass_recipient"]]
    passes_for_network['x'] = passes_for_network['location'].apply(lambda x: x[0])
    passes_for_network['y'] = passes_for_network['location'].apply(lambda x: x[1])
    passes_for_network['end_x'] = passes_for_network['pass_end_location'].apply(lambda x: x[0])
    passes_for_network['end_y'] = passes_for_network['pass_end_location'].apply(lambda x: x[1])

    scatter_df = pd.DataFrame()
    for i, name in enumerate(passes_for_network["player"].unique()):
        pass_x = passes_for_network.loc[passes_for_network["player"] == name]["x"].to_numpy()
        rec_x = passes_for_network.loc[passes_for_network["pass_recipient"] == name]["end_x"].to_numpy()
        pass_y = passes_for_network.loc[passes_for_network["player"] == name]["y"].to_numpy()
        rec_y = passes_for_network.loc[passes_for_network["pass_recipient"] == name]["end_y"].to_numpy()
        scatter_df.at[i, "player"] = name
        # make sure that x and y location for each circle representing the player is the average of passes and receptions
        scatter_df.at[i, "x"] = np.mean(np.concatenate([pass_x, rec_x]))
        scatter_df.at[i, "y"] = np.mean(np.concatenate([pass_y, rec_y]))
        # calculate number of passes
        scatter_df.at[i, "no"] = passes_for_network.loc[passes_for_network["player"] == name].count().iloc[0]

    # adjust the size of a circle so that the player who made more passes
    scatter_df['marker_size'] = (scatter_df['no'] / scatter_df['no'].max() * 1500)

    # counting passes between players
    passes_for_network["pair_key"] = passes_for_network.apply(
        lambda x: "_".join(sorted([x["player"], x["pass_recipient"]])), axis=1)
    lines_df = passes_for_network.groupby(["pair_key"]).x.count().reset_index()
    lines_df.rename({'x': 'pass_count'}, axis='columns', inplace=True)
    # setting a threshold. You can try to investigate how it changes when you change it.
    lines_df = lines_df[lines_df['pass_count'] > 2]

    pitch = Pitch(line_color='grey')
    fig, ax = pitch.grid(grid_height=0.9, title_height=0.06, axis=False,
                         endnote_height=0.04, title_space=0, endnote_space=0)
    pitch.scatter(scatter_df.x, scatter_df.y, s=scatter_df.marker_size, color='red', edgecolors='grey', linewidth=1,
                  alpha=1, ax=ax["pitch"], zorder=3)
    for i, row in scatter_df.iterrows():
        pitch.annotate(row.player, xy=(row.x, row.y), c='black', va='center', ha='center', weight="bold", size=16,
                       ax=ax["pitch"], zorder=4)

    for i, row in lines_df.iterrows():
        player1 = row["pair_key"].split("_")[0]
        player2 = row['pair_key'].split("_")[1]
        # take the average location of players to plot a line between them
        try:
            player1_x = scatter_df.loc[scatter_df["player"] == player1]['x'].iloc[0]
            player1_y = scatter_df.loc[scatter_df["player"] == player1]['y'].iloc[0]
            player2_x = scatter_df.loc[scatter_df["player"] == player2]['x'].iloc[0]
            player2_y = scatter_df.loc[scatter_df["player"] == player2]['y'].iloc[0]
        except:
            continue
        num_passes = row["pass_count"]
        # adjust the line width so that the more passes, the wider the line
        line_width = (num_passes / lines_df['pass_count'].max() * 10)
        # plot lines on the pitch
        pitch.lines(player1_x, player1_y, player2_x, player2_y,
                    alpha=1, lw=line_width, zorder=2, color="red", ax=ax["pitch"])

    fig.suptitle('Red de pases de ' + team, fontsize=26)
    return fig


def get_centralization_index(match_events, passes, team):
    first_sub_index = get_first_substitution_index(match_events, team)
    team_passes = passes[
        (passes['team'] == team) & (passes.index < first_sub_index) & (passes['pass_outcome'].isnull())]
    # calculate number of successful passes by player
    no_passes = team_passes.groupby(['player']).location.count().reset_index()
    no_passes.rename({'location': 'pass_count'}, axis='columns', inplace=True)
    max_no = no_passes["pass_count"].max()
    # calculate the denominator - 10*the total sum of passes
    denominator = 10 * no_passes["pass_count"].sum()
    numerator = (max_no - no_passes["pass_count"]).sum()
    centralisation_index = numerator / denominator
    return centralisation_index


def get_danger_passes_df(match_events, team):
    danger_passes = pd.DataFrame()
    for period in [1, 2]:
        mask_pass = ((match_events.team == team) & (match_events.type == "Pass")
                     & (match_events.pass_outcome.isnull()) & (match_events.period == period)
                     & (match_events.pass_type.isnull()))
        passes = match_events.loc[mask_pass, ["location", "pass_end_location", "minute", "second", "player"]]
        mask_shot = (match_events.team == team) & (match_events.type == "Shot") & (
                match_events.period == period)
        shots = match_events.loc[mask_shot, ["minute", "second"]]
        shot_times = shots['minute'] * 60 + shots['second']
        shot_window = 15
        # find starts of the window
        shot_start = shot_times - shot_window
        # condition to avoid negative shot starts
        shot_start = shot_start.apply(lambda i: i if i > 0 else (period - 1) * 45)
        pass_times = passes['minute'] * 60 + passes['second']
        # check if pass is in any of the windows for this half
        pass_to_shot = pass_times.apply(lambda x: True in ((shot_start < x) & (x < shot_times)).unique())

        # keep only danger passes
        danger_passes_period = passes.loc[pass_to_shot]
        # concatenate dataframe with a previous one to keep danger passes from the whole tournament
        danger_passes = pd.concat([danger_passes, danger_passes_period], ignore_index=True)
    return danger_passes


def get_danger_passes_location_fig(danger_passes, team_name):
    danger_passes['x'] = danger_passes['location'].apply(lambda x: x[0])
    danger_passes['y'] = danger_passes['location'].apply(lambda x: x[1])
    danger_passes['end_x'] = danger_passes['pass_end_location'].apply(lambda x: x[0])
    danger_passes['end_y'] = danger_passes['pass_end_location'].apply(lambda x: x[1])

    pitch = Pitch(line_color='black')
    fig, ax = pitch.grid(grid_height=0.9, title_height=0.06, axis=False,
                         endnote_height=0.04, title_space=0, endnote_space=0)
    pitch.scatter(danger_passes.x, danger_passes.y, s=100, color='blue', edgecolors='grey', linewidth=1, alpha=0.2,
                  ax=ax["pitch"])
    # pitch.arrows(danger_passes.x, danger_passes.y, danger_passes.end_x, danger_passes.end_y, color="blue",
    #              ax=ax['pitch'])
    fig.suptitle('Origen de pases de peligro de ' + team_name, fontsize=26)
    return fig


def get_danger_passes_heatmap(danger_passes, team_name, no_games=1):
    danger_passes['x'] = danger_passes['location'].apply(lambda x: x[0])
    danger_passes['y'] = danger_passes['location'].apply(lambda x: x[1])
    danger_passes['end_x'] = danger_passes['pass_end_location'].apply(lambda x: x[0])
    danger_passes['end_y'] = danger_passes['pass_end_location'].apply(lambda x: x[1])

    pitch = Pitch(line_zorder=2, line_color='black')
    fig, ax = pitch.grid(grid_height=0.9, title_height=0.06, axis=False,
                         endnote_height=0.04, title_space=0, endnote_space=0)
    # get the 2D histogram
    bin_statistic = pitch.bin_statistic(danger_passes.x, danger_passes.y, statistic='count', bins=(6, 5),
                                        normalize=False)
    # normalize by number of games
    bin_statistic["statistic"] = bin_statistic["statistic"] / no_games
    pcm = pitch.heatmap(bin_statistic, cmap='Reds', edgecolor='grey', ax=ax['pitch'])
    # legend to our plot
    ax_cbar = fig.add_axes((1, 0.093, 0.03, 0.786))
    cbar = plt.colorbar(pcm, cax=ax_cbar)
    fig.suptitle('Mapa de calor de pases de peligro de ' + team_name, fontsize=26)
    return fig


def get_danger_passes_players_hist(danger_passes, no_games=1):
    # count passes by player and normalize them
    pass_count = danger_passes.groupby(["player"]).x.count() / no_games
    # make a histogram
    ax = pass_count.plot.bar(pass_count)
    # make legend
    ax.set_xlabel("")
    ax.set_ylabel("Pases de peligro por jugador")
    st.bar_chart()
