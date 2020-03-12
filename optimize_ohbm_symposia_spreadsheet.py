#!/bin/python3

import numpy as np
import pandas as pd
import gurobipy as gb
import itertools

verbose       = False  # set to True to print extensive output info
verbose_optim = True  # set to True to print running optimization
verbose_plot  = True   # display similarity matrix
save_outputs  = True   # save symposia schedule and (if plotted) similatity matix plot

inputs = pd.read_excel('input.xlsx',sheet_name=["symposia","sessions","rooms","symposia_constraints","symposia_similarity"])

panel_titles  = list(inputs['symposia'].Name)
timeslotsList = list(inputs['sessions'].Name)
symp_per_slot = list(inputs['sessions'].Capacity)
room_names    = list(inputs['rooms'].Name)
rooms         = list(inputs['rooms'].Capacity)
sim_mat       = 1. - np.asarray(inputs['symposia_similarity'].set_index('Unnamed: 0')) # dissimilarity matrix
                      
ns            = len(panel_titles)             # number of symposia
nt            = len(timeslotsList)            # number of time slots
nr            = len(room_names)              # number of rooms

# print IDs for symposia titles and time slots
if verbose:     
    print('Rooms:')
    for i,n in enumerate(room_names):
        print('%s capacity: %i' % (n,rooms[i]))
    print('')    
    print('Time slots:')
    for i,n in enumerate(timeslotsList):
        print('%s' % n)
    print('')    
    print('Symposia titles:')
    for i,n in enumerate(panel_titles):
        print('%s' % n)
    print('')
        
# add constraints for specific symposia that are already scheduled, eg:
def scheduleSymposia(a,b):
    for j in np.setdiff1d(range(nt),b):
        for r in range(nr):
            m.addConstr(x[a,j,r] == 0) # lock symposium a in list to time slot b
    print('Lock: \"' + panel_titles[a] + '\" to ' + timeslotsList[b])

# prioritize symposia earlier in conference
def prioritizeScheduling(a):
    for c in range(int(nt/2),nt): # put symposium in first half of conference 
        for r in range(nr):
            m.addConstr(x[a,c,r] == 0)
    print('Prioritize: \"' + panel_titles[a] + '\"')
    
# constrain specific symposia to not be in same time slot, eg:
def avoidSymposiaOverlap(a,b):
    for j in range(nt):
        m.addConstr(y[a,b,j] == 0)
        m.addConstr(y[b,a,j] == 0)
        sim_mat[a,b]=0.
        sim_mat[b,a]=0.
    print('Don\'t overlap: \"' + panel_titles[a] + '\" with \"' + panel_titles[b] + '\"')
    
# schedule specific symposia together, eg:
def symposiaOverlap(a,b):
    m.addConstr(gb.quicksum(y[a,b,j] for j in range(nt)) == 1)
    m.addConstr(gb.quicksum(y[b,a,j] for j in range(nt)) == 1)
    sim_mat[a,b]=1.
    sim_mat[b,a]=1.
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
            m.addConstr(x[a,j,i] == 0) # assign to largest rooms
    print('Room assignment: \"' + panel_titles[a] + '\" in \"' + room_names[room_id] + '\"')
    
##################################################################################################
# Run optimizer:
##################################################################################################

m = gb.Model("Symposia") # initiate model
m.setParam('OutputFlag', False)
if verbose_optim==True:
    m.setParam('OutputFlag', True)

# add variables
x = m.addVars(range(ns),range(nt),range(nr),vtype=gb.GRB.BINARY,name='x') # symposia x time slots 
m.addConstrs(x.sum(s,'*','*') == 1 for s in range(ns))                    # only hold symposium once
m.addConstrs(x.sum('*',t,'*') == symp_per_slot[t] for t in range(nt))     # constrain number of symposia per time slot
for j in range(nt):                                                       # no overlapping rooms per session
    for r in range(nr):
        m.addConstr(gb.quicksum(x[s,j,r] for s in range(ns)) <= 1)

print('')
print('===== constraints: =====')
for i in inputs['symposia'].Set_Room.dropna("").index.values:
    assignRooms(i,inputs['symposia'].Set_Room.dropna("")[i])
for i in inputs['symposia'].Set_Session.dropna("").index.values:
    scheduleSymposia(i,inputs['sessions'][inputs['sessions'].Name.str.match(inputs['symposia'].Set_Session.dropna("")[i])].index.values[0])
for i in inputs['symposia'].Set_Priority.dropna("").index.values:
    prioritizeScheduling(i)
    
y = m.addVars(range(ns),range(ns),range(nt),vtype=gb.GRB.BINARY,name='y') # symposia x symposia x time slots
i_pairs_list = list(itertools.combinations(range(ns),2))
# add variables for weighting parallel symposia by similarity matrix
for i,ip in i_pairs_list: # loop through i and ip, where i != ip
    for j in range(nt):
        m.addConstr(gb.quicksum(x[i,j,r] for r in range(nr)) + gb.quicksum(x[ip,j,r] for r in range(nr)) - y[i,ip,j] <= 1)
        m.addConstr(2*y[i,ip,j] - gb.quicksum(x[i,j,r] for r in range(nr)) - gb.quicksum(x[ip,j,r] for r in range(nr)) <= 0)

# more constaints:
for i in inputs['symposia_constraints'].Avoid_overlap.dropna("").index.values:
    avoidSymposiaOverlap(inputs['symposia'][inputs['symposia'].Name.str.match(inputs['symposia_constraints'].Symposia_1[i].replace('(','\(').replace(')','\)'))].index.values[0], 
                         inputs['symposia'][inputs['symposia'].Name.str.match(inputs['symposia_constraints'].Symposia_2[i].replace('(','\(').replace(')','\)'))].index.values[0])
for i in inputs['symposia_constraints'].Ensure_overlap.dropna("").index.values:
    symposiaOverlap(inputs['symposia'][inputs['symposia'].Name.str.match(inputs['symposia_constraints'].Symposia_1[i].replace('(','\(').replace(')','\)'))].index.values[0], 
                    inputs['symposia'][inputs['symposia'].Name.str.match(inputs['symposia_constraints'].Symposia_2[i].replace('(','\(').replace(')','\)'))].index.values[0])
print('')

cost = gb.quicksum(gb.quicksum(sim_mat[i,ip]*y[i,ip,j] for i,ip in i_pairs_list) 
                   for j in range(nt)) # cost minimizes sum(similarity matrix weights)

m.setObjective(cost, gb.GRB.MAXIMIZE)
m.update()
print('')
if verbose:
    print('===== running optimizer: =====')
    print('')
m.optimize()

##################################################################################################
# Results:
##################################################################################################
res_all = [np.int(r) for r in m.X[:ns*nt*nr]]
res = np.reshape(res_all,(ns,nt,nr))
sub,tm,rom = np.where(res)
print('')
print('===== Schedule: ======')
newOrder = []
for j in range(nt):
    print(timeslotsList[j])
    for i in np.where(tm == j)[0]:
        print(room_names[rom[i]]  + ': ' + panel_titles[i])
        newOrder.append(i)
    print('')
print('======================')
print('')

if save_outputs:
    file1 = open("schedule.txt","w")
    file1.write('Schedule:\n\n') 
    for j in range(nt):
        file1.write('%s\n' % timeslotsList[j]) 
        for i in np.where(tm == j)[0]:
            try:
                file1.write('%s: %s\n' % (room_names[rom[i]],panel_titles[i].encode('utf8').decode("utf-8"))) 
            except:
                file1.write('%s: %s\n' % (room_names[rom[i]],panel_titles[i].encode('utf8')))                
        file1.write('\n')
    file1.close() 

%matplotlib inline
if verbose_plot:    
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    panel_titles_reordered = [panel_titles[i] for i in newOrder]

    fig = plt.figure(figsize=(12,12))
    plt.imshow(sim_mat[:,newOrder][newOrder].squeeze(), interpolation='nearest', cmap=cm.jet_r, vmin=0.95)
    ax = plt.gca()
    ax.set_yticks(np.arange(0, len(panel_titles_reordered), 1))
    ax.set_xticks(np.arange(0, len(panel_titles_reordered), 1))
    ax.set_yticklabels(panel_titles_reordered)
    ax.set_xticklabels(panel_titles_reordered)
    plt.xticks(rotation=90)

    for i in np.cumsum(list(np.asarray(symp_per_slot)))-0.5:
        plt.axvline(x=i, color='white', linewidth=6)
        plt.axhline(y=i, color='white', linewidth=6)

    if save_outputs:
        plt.savefig('symposia_similarity.png')
        
    plt.show()

    # squares along diagonal reflext similarity of symposia sessions: