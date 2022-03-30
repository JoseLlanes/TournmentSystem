#!/usr/bin/env python
# coding: utf-8

# ## Criterios de clasificación:
# 
# Al terminar cada ronda de esta fase, se aplicarán los siguientes criterios para obtener la clasificación del torneo:
# 
# 1) Puntuación: 
#     La suma de la puntuación viene dada por:
#     ○     G-E-P: Partidos ganados, empatados y perdidos. Cada partido ganado sumará +3pts, cada partido empatado +1pto y cada partido perdido +0pts.
# 
# 2) TB: 
#     Número de partidos ganados contra los oponentes empatados.
# 
# 3) Buchholz: 
#     
#     Sistema de desempate Buchholz Medio. En un sistema suizo los participantes no juegan contra los mismos oponentes, por lo que con este sistema se consigue tener en cuenta que unos habrán jugado contra mejores adversarios (que han finalizado con más puntos).
#     Para aplicar este sistema se suman los puntos obtenidos por los equipos contra los que se ha jugado y se hace la media con los partidos disputados por ese equipo (así se elimina la posibilidad de que un equipo juegue menos partidos, y por eso, tengo un peor criterio que los demás, ej: cuando alguien gane porque su oponente no esta). Se priorizará al equipo que tenga un valor más alto en este campo.
# 
# 4) Menor número de faltas graves.
# 
# 5) Azar.
# 

# In[1]:


import json
import time
import os
import random
import pandas as pd
import numpy as np
import itertools

import warnings
warnings.filterwarnings('ignore')

# ##################
# ### PARAMETERS ###
# ##################

max_foults_num_th = 6
rdm_th = 0

# #################
# ### FUNCTIONS ###
# #################

def get_dicts_by_line(line):
    dict_help1 = {
        "played_teams":[],
        "history":[],
        "vict":0,
        "draw":0,
        "lose":0,
        "FP":0,
        "AP":0,
        "Foults":0,
        "rank0":0
    }
    
    dict_help2 = dict_help1.copy()
    
    both_teams, both_points = line.split(" ; ")[0], line.split(" ; ")[1]
    both_foults = line.split(" ; ")[2]
    
    team1, team2 = both_teams.split(" - ")[0], both_teams.split(" - ")[1]

    points1, points2 = int(both_points.split(" - ")[0]), int(both_points.split(" - ")[1])

    foults1, foults2 = int(both_foults.split(" - ")[0]), int(both_foults.split(" - ")[1])
    
    if points1 > points2:
        team1_result, team2_result = "V", "L"
    elif points1 < points2:
        team1_result, team2_result = "L", "V"
    else:
        team1_result, team2_result = "D", "D"
    
    dict_help1["played_teams"] = team2
    dict_help1["history"] = team1_result
    dict_help1["vict"] += 1 if team1_result == "V" else 0
    dict_help1["draw"] += 1 if team1_result == "D" else 0
    dict_help1["lose"] += 1 if team1_result == "L" else 0
    dict_help1["FP"] += points1
    dict_help1["AP"] += points2
    dict_help1["Foults"] += foults1
    
    dict_help2["played_teams"] = team1
    dict_help2["history"] = team2_result
    dict_help2["vict"] += 1 if team2_result == "V" else 0
    dict_help2["draw"] += 1 if team2_result == "D" else 0
    dict_help2["lose"] += 1 if team2_result == "L" else 0
    dict_help2["FP"] += points2
    dict_help2["AP"] += points1
    dict_help2["Foults"] += foults2
    
    return team1, dict_help1, team2, dict_help2

# ########################
# ### To get TB metric ###
# ########################
def get_TB_metric(df):
    tb_metric_list = []
    for i in range(df.shape[0]):

        team = df.loc[i, "team"]
        points = df.loc[i, "points"]
        played_teams_list = df.loc[i, "played_teams"]
        history_list = df.loc[i, "history"]

        df_rest = df[(df["points"] == points) & (df["team"] != team)]
        if df_rest.shape[0] == 0:
            tb_metric_list.append(0)
        else:
            teams_point_draw = df_rest["team"].to_list()
            counter = 0
            for played_team_i, played_team in enumerate(played_teams_list):
                if played_team in teams_point_draw and history_list[played_team_i] == "V":
                    counter += 1

            tb_metric_list.append(counter)
            
    return tb_metric_list

# ##############################
# ### To get Buchholz metric ###
# ##############################
def get_Buchholz_metric(df):
    buchholz_list = []
    for i in range(df.shape[0]):

        team = df.loc[i, "team"]
        points = df.loc[i, "points"]
        played_teams_list = df.loc[i, "played_teams"]

        counter = 0
        for played_team in played_teams_list:
            total_points = df[df["team"] == played_team]["points"].values[0]
            played_teams2 = df[df["team"] == played_team]["played_teams"].values[0]
            amount_matches = len(played_teams2)

            mean_buch = total_points/amount_matches
            counter += mean_buch

        buchholz_list.append(counter)
    
    return buchholz_list

# #####################
# ### To get Random ###
# #####################
random.seed(rdm_th)
def get_random_metric(df):
    df_list = []
    for k, df_sub in df.groupby(["points", "Buchholz", "Foults", "TB"]):
        # display(df_sub)
        if df_sub.shape[0] > 1:
            N = df_sub.shape[0]
            randomlist = random.sample(range(0, N), N)
            # print(randomlist)
            df_sub_rdm = df_sub.iloc[np.array(randomlist)]
            df_list.append(df_sub_rdm)
        else:
            df_list.append(df_sub)
    return pd.concat(df_list).reset_index(drop=True)

# #####################
# ### To get Foults ###
# #####################
def check_foults(df, max_foults_num=max_foults_num_th, verbose=True):
    df_foult = df[df["Foults"] >= max_foults_num]
    if df_foult.shape[0] > 0:
        if verbose:
            for i in range(df_foult.shape[0]):
                print("Equipo ", df_foult["team"].iloc[i], 
                      "descalificado. El equipo tiene " + str(max_foults_num) + " o más faltas" )
        
        df_no_foult = df[df["Foults"] < max_foults_num]
        df = pd.concat([df_no_foult, df_foult]).reset_index(drop=True)
        
    else:
        if verbose:
            print("Ningún equipo con " + str(max_foults_num) + " o más faltas")
        
    return df

# #####################################
# ### Generate all possible matches ###
# #####################################

# Combinatory algorithm.
def generate_groups(lst, n, conditions, comb_th = 100 * 10 **3):
    if not lst:
        yield []
    else:
        ct_comb = 0
        for group in (((lst[0],) + xs) for xs in itertools.combinations(lst[1:], n-1)):
            if group not in conditions:
                for groups in generate_groups([x for x in lst if x not in group], n, conditions):
                    
                    if ct_comb >= comb_th: 
                        break
                    
                    ct_comb += 1
                    
                    yield [group] + groups
             
            if ct_comb >= comb_th: 
                break
                    
# Find possible match algorithm.
def find_possible_matches(df):
    
    n_of_teams = df.shape[0]
    list_of_tuples = []
    for i in range(n_of_teams):
        played_teams_list = df.loc[i, "played_teams"]
        for p_team in played_teams_list:
            idx_p_team = df[df["team"] == p_team].index.values[0]
            p_tuple = (i, idx_p_team) if idx_p_team > i else (idx_p_team, i)
            if p_tuple not in list_of_tuples:
                list_of_tuples.append(p_tuple)
                
    t0 = time.time()
    conditions = list_of_tuples
    result1 = list(generate_groups(np.arange(0, n_of_teams).tolist(), 2, conditions))
    print("Number of possible combinations = ", len(result1))
    print("Time spend ", np.round((time.time()-t0)/60, 5), "min")
    
    punt_list = []
    for res in result1:
        list_diff = [np.abs(r[1]-r[0]) for r in res]
        punt_list.append(np.sum(list_diff))
    best_combination = result1[np.argmin(punt_list)]
    
    pair_teams_list = []
    taken_teams_list = []
    for b_iter in best_combination:
    
        team_x = df.loc[b_iter[0], "team"]
        team_y = df.loc[b_iter[1], "team"]

        pair_teams_list.append(team_x + " vs " + team_y)
        taken_teams_list += [team_x, team_y]
        
    return taken_teams_list, pair_teams_list

def do_team_pairments(
    path_to_data="partidos/", 
    path_to_teams_data="equipos.txt",
    possible_files=["partidos1.txt", "partidos2.txt", "partidos3.txt", "partidos4.txt"]
    ):

    t0 = time.time()

    file1 = open(path_to_data + path_to_teams_data,"r+") 
    all_lines = file1.readlines()

    all_teams = [line.replace("\n", "")for line in all_lines]
    dict_result = {}
    for team_line in all_teams:
        
        team_name, rank_pos = team_line.split(" ; ")
        
        dict_result[team_name] = {
            "played_teams":[],
            "history":[],
            "vict":0,
            "draw":0,
            "lose":0,
            "FP":0,
            "AP":0,
            "Foults":0,
            "rank0":int(rank_pos)
        }

    for file_i, file in enumerate(possible_files):
        print("Reading... ", file)
        
        file1 = open(path_to_data + file,"r+") 
        all_lines = file1.readlines()
        
        if len(all_lines) == 0 or "?" in all_lines[0] or len(all_lines[0]) < 5:
            print("Stopped in match", file)
            break
        else:
            all_lines_pro = [line.replace("\n", " ")for line in all_lines]
            for line_i, line in enumerate(all_lines_pro):
                
                team1, dict_help1, team2, dict_help2 = get_dicts_by_line(line)
                
                dict_result[team1]["played_teams"].append(dict_help1["played_teams"])
                dict_result[team1]["history"].append(dict_help1["history"])
                dict_result[team1]["vict"] += dict_help1["vict"]
                dict_result[team1]["draw"] += dict_help1["draw"]
                dict_result[team1]["lose"] += dict_help1["lose"]
                dict_result[team1]["FP"] += dict_help1["FP"]
                dict_result[team1]["AP"] += dict_help1["AP"]
                dict_result[team1]["Foults"] += dict_help1["Foults"]
                
                dict_result[team2]["played_teams"].append(dict_help2["played_teams"])
                dict_result[team2]["history"].append(dict_help2["history"])
                dict_result[team2]["vict"] += dict_help2["vict"]
                dict_result[team2]["draw"] += dict_help2["draw"]
                dict_result[team2]["lose"] += dict_help2["lose"]
                dict_result[team2]["FP"] += dict_help2["FP"]
                dict_result[team2]["AP"] += dict_help2["AP"]
                dict_result[team2]["Foults"] += dict_help2["Foults"]
                
        print(file, "readed")
        print()

    df = pd.DataFrame(dict_result).T
    df.insert(0, "team", df.index.values)
    df = df.reset_index(drop=True)

    df["points"] = 3 * df["vict"] + 1 * df["draw"]
    # df["FAP"] = df["FP"] - df["AP"]

    if file != "partidos1.txt":
        
        df["last_result"] = [df["history"].iloc[i][-1] for i in range(df.shape[0])]
        df["last_team"] = [df["played_teams"].iloc[i][-1] for i in range(df.shape[0])]

        df["Buchholz"] = get_Buchholz_metric(df)
        
        df["TB"] = get_TB_metric(df)
        df = get_random_metric(df)
        
        df = df.sort_values( 
            ["points", "Buchholz", "Foults", "TB"], 
            ascending=(False, False, True, False) 
        ).reset_index(drop=True)
        
        check_foults(df)

        df.to_excel("partidos/Table_round_" + str(file_i) + ".xlsx", index=False)
        print("Time of the program = ", np.round((time.time()-t0)/60, 5), "min" )

    t0 = time.time()

    failed_q = 0
    taken_teams_list, pair_teams_list = [], []
    for i in range(df.shape[0]): 
        
        try:
        
            team_x = df["team"].iloc[i]

            if team_x not in taken_teams_list:

                taken_teams_list.append(team_x)

                # 1) Filter
                # result_last_match = df["history"].iloc[i][-1]
                # df1 = df[df["last_result"] == result_last_match]
                df1 = df.copy()

                # 2) Not same team than ALL last matches and not the self team.
                played_teams_list = df1["played_teams"].iloc[i]
                idx_to_remove_list = [df1[df1["team"] == team].index.values[0] for team in played_teams_list]
                df2 = df1.drop(idx_to_remove_list)
                df2 = df2[df2["team"] != df["team"].iloc[i]]

                # 3) Get the team
                i_iloc = 0
                while True:
                    team_y = df2["team"].iloc[i_iloc]
                    if team_y not in taken_teams_list and team_x != team_y:
                        break
                    i_iloc += 1

                pair_teams_list.append(team_x + " vs " + team_y)
                taken_teams_list.append(team_y)
            
        except:
            print("The program will pair teams computing certain possible combinations.")
            taken_teams_list, pair_teams_list = find_possible_matches(df)
            failed_q = 1

    print("Time of the program = ", np.round((time.time()-t0)/60, 5), "min" )
    print()
    print("***** List of matches: *****")
    print(pair_teams_list)

    file1 = open(path_to_data + file, "w+")
    str_save_list = []
    for pair_teams_i, pair_teams in enumerate(pair_teams_list):
        team1, team2 = pair_teams.split(" vs ")
        str_save = team1 + " - " + team2 + " ; ? - ? ; ? - ?"
        str_save_list.append(str_save + " \n")
    file1.writelines(str_save_list)
    file1.close()

    print("Pairments saved")

    return failed_q


if __name__ == "__main__":
    _ = do_team_pairments()