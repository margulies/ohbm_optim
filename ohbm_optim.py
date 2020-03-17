"""
Conference schedule optimization  
"""

# Authors: Daniel Margulies
# License: BSD 3 clause
from __future__ import print_function
import itertools
from oauth2client.service_account import ServiceAccountCredentials

import numpy as np

import gspread
import pandas as pd

from ortools.sat.python import cp_model


def _load_sheets(fname, client_id):
    """ Load spreadsheet data using Google API
    """
    # Open credentials to access google spreadsheet
    scope = ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name(https://mattermost.brainhack.org/brainhack/channels/ohbm_optim, scope)
    gs = gspread.authorize(creds)

    # Download the spreadsheet content
    sheets = gs.open_by_url(fname)
    return sheets


def _parse_sheets(sheets, sheet_names):
    """ Parse and sanitize the spreadsheet
    """
    inputs = {}
    for n,i in enumerate(sheet_names):
        inputs[i]  = pd.DataFrame(sheets.get_worksheet(n).get_all_values()[1:], 
                                 columns=sheets.get_worksheet(n).get_all_values().pop(0))
    data = dict(names=inputs['symposia'].Name,
                panel_titles=list(names),
                timeslotsList=list(inputs['sessions'].Name),
                symp_per_slot=[int(i) for i in list(inputs['sessions'].Capacity)],
                room_names=list(inputs['rooms'].Name),
                rooms=[int(i) for i in list(inputs['rooms'].Capacity)],
                set_priority=inputs['symposia'].Set_Priority.replace(to_replace="",value=np.nan),
                set_session=inputs['symposia'].Set_Session.replace(to_replace="",value=np.nan),
                set_room=inputs['symposia'].Set_Room.replace(to_replace="",value=np.nan),
                avoid_overlap=inputs['symposia_constraints'].Avoid_overlap.replace(to_replace="",value=np.nan),
                ensure_overlap=inputs['symposia_constraints'].Ensure_overlap.replace(to_replace="",value=np.nan),
                symposia1=inputs['symposia_constraints'].Symposia_1.replace(to_replace="",value=np.nan),
                symposia2=inputs['symposia_constraints'].Symposia_2.replace(to_replace="",value=np.nan)
                symposia_similarity=np.asarray(inputs['symposia_similarity'].set_index(''))
               )
    return data


def _sim_data(data):
    sim_mat_o = np.zeros(data["symposia_similarity"].shape)
    for n, i in enumerate(data["symposia_similarity"]):
        for m, j in enumerate(i):
            sim_mat_o[n, m] = 1. - float(j)
            
    sim_mat = np.zeros((len(sim_mat_o),len(sim_mat_o)),dtype=np.int)
    for n,i in enumerate(sim_mat_o * 10000. ):
        for m,j in enumerate(i):
            sim_mat[n,m] = int(j)
    return sim_mat

##################################################################################################
# Constraint functions:
##################################################################################################

def scheduleSymposia(a,b):
    """ Add constraints for specific symposia that are already scheduled
    """
    for j in np.setdiff1d(range(nt),b):
        for r in range(nr):
            m.Add(x[(a,j,r)] == 0) # lock symposium a in list to time slot b
    print('Lock: \"' + panel_titles[a] + '\" to ' + timeslotsList[b])


def prioritizeScheduling(a):
    """ Prioritize symposia earlier in conference
    """
    for c in range(int(nt/2)+1,nt): # put symposium in first half of conference 
        for r in range(nr):
            m.Add(x[(a,c,r)] == 0)
    print('Prioritize: \"' + panel_titles[a] + '\"')
    

def avoidSymposiaOverlap(a,b):
    """ Constrain specific symposia to not be in same time slot
    """
    a,b = np.sort([a,b])
    for t in range(nt): 
        m.Add(y[(a,b,t)] == 0)
        m.Add(sum([sum(x[(a,t,r)] for r in range(nr)),sum(x[(b,t,r)] for r in range(nr))]) <= 1)
    print('Don\'t overlap: \"' + panel_titles[a] + '\" with \"' + panel_titles[b] + '\"')
    

def symposiaOverlap(a,b):
    """ Schedule specific symposia together
    """
    a,b = np.sort([a,b])
    m.Add(sum(y[(a,b,t)] for t in range(nt)) == 1)
    for t1,t2 in list(itertools.combinations(range(nt),2)):
        m.Add(sum([sum(x[(a,t1,r)] for r in range(nr)),sum(x[(b,t2,r)] for r in range(nr))]) <= 1)
        m.Add(sum([sum(x[(a,t2,r)] for r in range(nr)),sum(x[(b,t1,r)] for r in range(nr))]) <= 1)
    print('Schedule together: \"' + panel_titles[a] + '\" and \"' + panel_titles[b] + '\"')


def assignRooms(a,size):
    """ Assign rooms
        Options for size: 'large', 'medium', 'small'
    """
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
