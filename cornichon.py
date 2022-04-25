# cornichon.py of EPQ Project
# Copyright James Robinson 2021-2022
# All rights reserved
#
# James Robinson (8116) - The Burgate School and Sixth Form (58815)

import pickle

def load(path=""):
    pf = open(path, 'rb')
    model = pickle.load(pf)
    pf.close()

    return model

def save(file, path="NO_NAME.pickle"):
    pf = open(path, 'wb')
    pickle.dump(file, pf)
    pf.close()