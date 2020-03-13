#!/bin/python3

from __future__ import print_function
import numpy as np
import pandas as pd
from ortools.sat.python import cp_model
import itertools
import multiprocessing
from google.protobuf import text_format

verbose       = False  # set to True to print extensive output info
verbose_optim = True   # set to True to print running optimization
verbose_plot  = True   # display similarity matrix
save_outputs  = True   # save symposia schedule and (if plotted) similatity matix plot

inputs = pd.read_excel('/network/lustre/iss01/home/daniel.margulies/code/ohbm_optim/input.xlsx',sheet_name=["symposia","sessions","rooms","symposia_constraints","symposia_similarity"])

panel_titles  = list(inputs['symposia'].Name)
timeslotsList = list(inputs['sessions'].Name)
symp_per_slot = list(inputs['sessions'].Capacity)
room_names    = list(inputs['rooms'].Name)
rooms         = list(inputs['rooms'].Capacity)
set_priority   = inputs['symposia'].Set_Priority.replace(to_replace="",value=np.nan)
set_session    = inputs['symposia'].Set_Session.replace(to_replace="",value=np.nan)
set_room       = inputs['symposia'].Set_Room.replace(to_replace="",value=np.nan)
avoid_overlap  = inputs['symposia_constraints'].Avoid_overlap.replace(to_replace="",value=np.nan)
ensure_overlap = inputs['symposia_constraints'].Ensure_overlap.replace(to_replace="",value=np.nan)
symposia1      = inputs['symposia_constraints'].Symposia_1.replace(to_replace="",value=np.nan)
symposia2      = inputs['symposia_constraints'].Symposia_2.replace(to_replace="",value=np.nan)
sim_mat_o = 1. - np.asarray(inputs['symposia_similarity'].set_index('Unnamed: 0')) # dissimilarity matrix
sim_mat = np.zeros((len(sim_mat_o),len(sim_mat_o)),dtype=np.int)
for n,i in enumerate(sim_mat_o * 10000. ):
    for m,j in enumerate(i):
        sim_mat[n,m] = int(j)
        
ns            = len(panel_titles)             # number of symposia
nt            = len(timeslotsList)            # number of time slots
nr            = len(room_names)              # number of rooms

##################################################################################################
# Constraint functions:
##################################################################################################
# add constraints for specific symposia that are already scheduled, eg:
def scheduleSymposia(a,b):
    for j in np.setdiff1d(range(nt),b):
        for r in range(nr):
            m.Add(x[(a,j,r)] == 0) # lock symposium a in list to time slot b
    print('Lock: \"' + panel_titles[a] + '\" to ' + timeslotsList[b])

# prioritize symposia earlier in conference
def prioritizeScheduling(a):
    for c in range(int(nt/2)+1,nt): # put symposium in first half of conference 
        for r in range(nr):
            m.Add(x[(a,c,r)] == 0)
    print('Prioritize: \"' + panel_titles[a] + '\"')
    
# # constrain specific symposia to not be in same time slot, eg:
def avoidSymposiaOverlap(a,b):
    a,b = np.sort([a,b])
    for t in range(nt): 
        m.Add(y[(a,b,t)] == 0)
        m.Add(sum([x[(a,t)],x[(b,t)]]) <= 1)
    print('Don\'t overlap: \"' + panel_titles[a] + '\" with \"' + panel_titles[b] + '\"')
    
# schedule specific symposia together, eg:
def symposiaOverlap(a,b):
    a,b = np.sort([a,b])
    m.Add(sum(y[(a,b,t)] for t in range(nt)) == 1)
    for t1,t2 in list(itertools.combinations(range(nt),2)):
        m.Add(sum([sum(x[(a,t1,r)] for r in range(nr)),sum(x[(b,t2,r)] for r in range(nr))]) <= 1)
        m.Add(sum([sum(x[(a,t2,r)] for r in range(nr)),sum(x[(b,t1,r)] for r in range(nr))]) <= 1)
    print('Schedule together: \"' + panel_titles[a] + '\" and \"' + panel_titles[b] + '\"')

# assign rooms
def assignRooms(a,size):
    # options for size: 'large', 'medium', 'small'
    if size == 'large':
        room_id = np.argmax(rooms)
    if size == 'medium':
        room_id = np.argsort(rooms)[int(len(rooms)/2)]
    if size == 'small':
        room_id = np.argmin(rooms)
    for j in range(nt):
        for i in np.setdiff1d(range(nr),room_id):
            m.Add(x[(a,j,i)] == 0) # assign to largest rooms
    print('Room assignment: \"' + panel_titles[a] + '\" in \"' + room_names[room_id] + '\"')
    
##################################################################################################
# Run optimizer:
##################################################################################################
m = cp_model.CpModel()
x = {}
for s in range(ns):
    for t in range(nt):
        x[(s,t)] = m.NewBoolVar('schedule_s%it%i' % (s,t))
for s in range(ns):
    m.Add(sum(x[(s,t)] for t in range(nt)) == 1)
for t in range(nt):
    m.Add(sum(x[(s,t)] for s in range(ns)) == symp_per_slot[t])

print('===== constraints: =====')
#for i in set_room.dropna("").index.values:
 #   assignRooms(i,set_room.dropna("")[i])
#for i in set_session.dropna("").index.values:
#    scheduleSymposia(i,inputs['sessions'][inputs['sessions'].Name.str.match(set_session.dropna("")[i])].index.values[0])
#for i in set_priority.dropna("").index.values:
#    prioritizeScheduling(i)

i_pairs_list = list(itertools.combinations(range(ns),2))
y = {}
for i,ip in i_pairs_list:
    for t in range(nt):
        y[(i,ip,t)] = m.NewBoolVar('overlap_i%iip%it%i' % (i,ip,t))

if len(np.where(set_room.dropna("") == 'large')[0]) > 1:
    room_large_conflicts = list(itertools.combinations(set_room.dropna("").index.values,2))
    for s1,s2 in room_large_conflicts:
        avoidSymposiaOverlap(s1,s2)

#for i in avoid_overlap.dropna("").index.values:
#    avoidSymposiaOverlap(inputs['symposia'][names.str.match(symposia1[i].replace('(','\(').replace(')','\)'))].index.values[0], 
#                         inputs['symposia'][names.str.match(symposia2[i].replace('(','\(').replace(')','\)'))].index.values[0])
#for i in ensure_overlap.dropna("").index.values:
#    symposiaOverlap(inputs['symposia'][names.str.match(symposia1[i].replace('(','\(').replace(')','\)'))].index.values[0], 
#                    inputs['symposia'][names.str.match(symposia2[i].replace('(','\(').replace(')','\)'))].index.values[0])

for i,ip in i_pairs_list:
    for t in range(nt):
        m.Add(x[i,t] + x[ip,t] - y[i,ip,t] <= 1)
        m.Add(2*y[i,ip,t] - x[i,t] - x[ip,t] <= 0)

m.Maximize(sum(sim_mat[i][ip] * y[(i,ip,t)] for i,ip in i_pairs_list for t in range(nt)))

new_model = cp_model.CpModel()
text_format.Parse(str(m), new_model.Proto())

solver = cp_model.CpSolver()
solver.parameters.num_search_workers  = multiprocessing.cpu_count()
# solver.parameters.max_time_in_seconds = 100.0

solution_printer = cp_model.ObjectiveSolutionPrinter()
status = solver.SolveWithSolutionCallback(new_model, solution_printer)
# status = solver.Solve(new_model)
print(solver.StatusName(status))
print(solver.ObjectiveValue())

# ##################################################################################################
# # Results:
# ##################################################################################################

if status == cp_model.FEASIBLE:

    if status == cp_model.OPTIMAL:
        print("optimal solution!")
    else:
        print("feasible solution")

    # Statistics
    print()
    print('===== statistics: =====')
    print('  - Stats           : %i' % solver.ObjectiveValue())
    print('  - wall time       : %f s' % solver.WallTime())
    print(m.ModelStats())
    print()

    # Schedule
    newOrder = []
    print()
    print("===== Schedule: =====")
    file1 = open("/network/lustre/iss01/home/daniel.margulies/code/ohbm_optim/schedule_2d_rooms.txt","w")
    for t in range(nt):
        print('%s' % timeslotsList[t])
        file1.write('%s\n' % timeslotsList[t])
        for s in range(ns):        
            if solver.Value(x[(s,t)]) == 1:
                print('%s' % panel_titles[s])
                newOrder.append(s)
                file1.write('%s\n' % (panel_titles[s].encode('utf8').decode("utf-8")))               
        print()
        file1.write('\n')
    file1.close()

if verbose_plot:
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    panel_titles_reordered = [panel_titles[i] for i in newOrder]

    fig = plt.figure(figsize=(12,12))
    plt.imshow(sim_mat[:,newOrder][newOrder].squeeze(), interpolation='nearest', cmap=cm.jet_r, vmin=.95*10000.)
    ax = plt.gca()
    ax.set_yticks(np.arange(0, len(panel_titles_reordered), 1))
    ax.set_xticks(np.arange(0, len(panel_titles_reordered), 1))
    ax.set_yticklabels(panel_titles_reordered)
    ax.set_xticklabels(panel_titles_reordered)
    plt.xticks(rotation=90)

    for i in np.cumsum(np.asarray(symp_per_slot))-0.5:
        plt.axvline(x=i, color='white', linewidth=6)
        plt.axhline(y=i, color='white', linewidth=6)
    
    if save_outputs:
        plt.savefig('/network/lustre/iss01/home/daniel.margulies/code/ohbm_optim/symposia_similarity_2d_rooms.png')
    plt.close()

