from __future__ import print_function
from gurobipy import *
import numpy as np
import scipy.spatial


def gb_bond(gb_min, j, jj):
    return j == gb_min and jj == gb_min + 1

def build_model(n):

    gb_min = n // 2 - 1
    bonds = [(1, 0), (0, 1), (1, 1)]

    sites = [(i, j) for j in range(n) for i in range(n)]
    bonds = [(i, j, (i + dx) % n, (j + dy) % n) for j in range(n) for i in range(n) for dx, dy in bonds]

    m = Model("mip1")
    x = {(i,j): m.addVar(vtype=GRB.BINARY) for (i,j) in sites}
    y = {(i,j,ii,jj): m.addVar(vtype=GRB.BINARY) for (i,j,ii,jj) in bonds}
    m.update()
    m.setObjective(0, GRB.MINIMIZE)

    for (i,j,ii,jj) in bonds:
        m.addConstr( y[(i,j,ii,jj)] >= x[(i,j)] + x[(ii, jj)] - 1 )
        m.addConstr( y[(i,j,ii,jj)] <= x[(i,j)] )
        m.addConstr( y[(i,j,ii,jj)] <= x[(ii,jj)] )

    con_ngb = m.addConstr( sum([x[(i,j)] for i,j in sites if j in [gb_min, gb_min + 1] ]) == 0)
    con_kgb = m.addConstr( sum([y[(i,j,ii,jj)] for i,j,ii,jj in bonds if gb_bond(gb_min, j, jj)] ) == 0 )
    con_kc  = m.addConstr( sum([y[(i,j,ii,jj)] for i,j,ii,jj in bonds if not gb_bond(gb_min, j, jj)] ) == 0 )
    con_numsolute = m.addConstr( sum(x.values()) == 0 )

    #symmetry-breaking constraint
    m.addConstr( sum([x[(i,j)] for i,j in sites if j <= gb_min]) >= sum([x[(i,j)] for i,j in sites if j > gb_min]) )
    m.update()
    return m, con_ngb, con_kgb, con_kc, con_numsolute, gb_min, x

def go():

    #choose lattice parameters (size and number of solute atoms)
    n = 12
    num_solute = 28

    #build the MIP model
    num_sites = n**2
    num_bonds = 3 * num_sites
    num_gb = 2 * n
    num_c = num_bonds - num_gb

    model, con_ngb, con_kgb, con_kc, con_numsolute, gb_min, x = build_model(n)
    model.params.OutputFlag = 0
    model.params.Presolve = 2
    model.params.MIPFocus = 1
    model.params.Threads = 8
    con_numsolute.RHS = num_solute

    #find the maximum number of crystalline bonds
    kc_max = -1
    for kc in range(3 * num_solute + 1):
        con_ngb.RHS = 0
        con_kgb.RHS = 0
        con_kc.RHS = kc

        model.optimize()
        status = model.getAttr("Status")
        assert(status in [GRB.OPTIMAL, GRB.INFEASIBLE])
        feasible = status == GRB.OPTIMAL
        if not feasible:
            kc_max = kc - 1
            break
    print("max. kc:", kc_max)
    assert(kc_max != -1)

    #build a list of potentially constructible configurations
    configs = []
    for ngb in range(2 * n + 1):
        kgb_max = [ngb - 1, ngb][ngb == 2 * n]
        for kgb in range(kgb_max + 1):
            for kc in range(3 * num_solute - kgb + 1):
                if kc <= kc_max:
                    configs += [(ngb, kgb, kc)]
    configs = np.array(configs)

    #test configurations for feasibility/constructibility
    fs = []
    xs = dict()
    for i, (ngb, kgb, kc) in enumerate(configs):
        con_ngb.RHS = ngb
        con_kgb.RHS = kgb
        con_kc.RHS = kc
        x[(0, gb_min)].lb = int(ngb >= 1)
        x[(0, gb_min+1)].lb = int(ngb >= 2 and kgb >= 1)

        model.optimize()
        status = model.getAttr("Status")
        assert(status in [GRB.OPTIMAL, GRB.INFEASIBLE])
        feasible = status == GRB.OPTIMAL
        status_string = ['infeasible', '  feasible'][feasible]
        print('config: %s / %s ngb=%s kgb=%s kc=%s status: %s' % (str(i).rjust(5), str(len(configs)).rjust(5), str(ngb).rjust(3), str(kgb).rjust(3), str(kc).rjust(3), status_string))
        fs += [feasible]

        #save feasible configurations
        if feasible:
            c = np.zeros((n,n)).astype(np.int)
            for (i, j), v in x.items():
                c[j,i] = int(round(v.x))
            xs[(ngb, kgb, kc)] = c

    #find extreme configurations (vertices of convex hull)
    print("num. feasible:", sum(fs))
    indices = np.where(fs)[0]
    feasible_configs = configs[indices]
    extreme = np.unique(scipy.spatial.ConvexHull(feasible_configs).simplices)
    print(extreme)
    print(feasible_configs[extreme])

    #print extreme configurations
    for ngb, kgb, kc in feasible_configs[extreme]:
        print(ngb, kgb, kc)
        print(xs[(ngb, kgb, kc)])
        print()
go()
