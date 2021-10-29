from sys import platform
import numpy as np
np.set_printoptions(precision=4,suppress=True)
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from copy import deepcopy

keys = ['iters','BFR','t_base', 'acc_base', 't_icp', 'acc_icp', 'GC', 'prosac', 'conf', 'take', 'coherence']

d = { k:i for i,k in enumerate(keys)}
flds  = [['t_base', 'acc_base'], ['t_icp', 'acc_icp']]

def parse_summary(filename):
    try:
        with open(filename, 'r') as fid:
            text = fid.read().splitlines()
    except:
        return np.empty([1,len(d.keys())])

    alg_name = 'dummy'

    data_list = []
    for line in text:    
        if line.startswith('==>'):
            cur_row = np.zeros(len(keys))
            L = line.replace('.txt','')
            a = L.split()
            b = a[1].split('_')
            cur_row[d['GC']] = (b[1] == 'GC')
            if (b[1] == 'GC'):
                alg_name = 'GC'
                offs = 2
            else:
                alg_name = 'RANSAC'
                offs=0            
            cur_row[d['iters']] = float(b[2+offs].replace('K','000').replace('M','000000'))
            if 'BFR' in b:
                cur_row[d['BFR']] = float(b[b.index('BFR')+1])
            else:
                cur_row[d['BFR']] = -1
            if 'conf' in b:
                cur_row[d['conf']] = float(b[b.index('conf')+1])
            else:
                cur_row[d['conf']] = 0.999 # default
            if 'prosac' in b:
                cur_row[d['prosac']] = 1
            else:
                cur_row[d['prosac']] = 0
            if 'coherence' in b:
                cur_row[d['coherence']] = float(b[b.index('coherence')+1])
            else:
                cur_row[d['coherence']] = 0
            if 'take' in b:
                cur_row[d['take']] = float(b[b.index('take')+1])
            else:
                cur_row[d['take']] = 1

        if line.startswith(alg_name + "+ICP"): 
            a = line.split(',')            
            cur_row[d['t_icp']] = float(a[-1].split()[-1])
            cur_row[d['acc_icp']] = float(a[0].split()[-1][:-1])            
            data_list.append(cur_row)

        elif line.startswith(alg_name):
            a = line.split(',')
            cur_row[d['t_base']] = float(a[-2].split()[-1])
            cur_row[d['acc_base']] = float(a[0].split()[-1][:-1])        

    data = np.vstack(data_list)
    return data

def A_to_B():
    ref_names = ['DGR', 'PointDSC', 'TEASER++', 'MFR+RANSAC', 'DFR+RANSAC']
    ref_data = np.array([[0,0, 44.95, 0.418, 48.07, 0.462],
        [0,0,63.97,0.234,66.78,0.293],
        [0,0,59.88,0.146,71.99,0.213],
        [0,0,66.01,0.137,74.54,0.197],
        [0,0,64.16,0.126,73.88,0.188]])
    ref_data = ref_data[:,[0,1,3,2,5,4]]

    generic('A_to_B', ref_data, ref_names)

def prep_data(name):
    data = parse_summary(f'logs/summary_{name}.txt')
    if name == 'B_to_B':
        more_data = parse_summary('logs/oct28.txt')
        data = np.vstack([data,more_data])
    data, hulls = process_variance(data)
    ord = np.argsort(data[:,d['iters']],axis=0,kind='stable')
    data = data[ord,:]
    ord = np.argsort(data[:,d['conf']],axis=0,kind='stable')
    data = data[ord,:]
    return data, hulls

def draw_references(ref_data, ref_names):
    symbols = ['v','s','p','P','*']
    for j in range(2):
        ax = plt.subplot(1,2,j+1)
        plt.title(flds[j])
        plt.xlabel('sec')        
        for ref_i in range(len(ref_names)):
            if ref_names[ref_i] == 'DGR':
                continue
            plt.plot(ref_data[ref_i][d[flds[j][0]]], ref_data[ref_i][d[flds[j][1]]], symbols[ref_i], label=ref_names[ref_i])
        a = ax.axis()
        teaser_t = ref_data[ref_names.index('TEASER++'),d[flds[j][0]]]
        dsc_t = ref_data[ref_names.index('PointDSC'),d[flds[j][0]]]
        m = min(teaser_t, dsc_t)
        ax.plot([m,m], a[2:],'k--')

def B_to_B():
    ref_names = ['DGR', 'PointDSC', 'TEASER++', 'MFR+RANSAC', 'DFR+RANSAC']
    ref_data = np.array([
        [0,0, 57.91,0.453,61.81,0.494],
        [0,0,80.56,0.236,82.48,0.290],
        [0,0,77.43,0.331,86.88,0.378],
        [0,0,83.37,0.078,88.31,0.133],
        [0,0,82.14,0.109,88.70,0.165]])
    ref_data = ref_data[:,[0,1,3,2,5,4]]

    name = 'B_to_B'
    data, hulls = prep_data(name)

    colors = [['darkviolet', 'turquoise', 'lightsteelblue', 'teal', 'blue'], ['red','lightcoral', 'tomato', 'firebrick', 'maroon']]

    plt.figure()
    draw_all_hulls(hulls)
    draw_line(data, colors[0][0], 'BFR', -1, 'GC', 0, 'iters', 10**6, 'prosac', 0, 'conf', 0.999, label_fields=['BFR','GC','iters'])    
    draw_line(data, colors[0][3], 'BFR', 3, 'GC', 0, 'iters', 1500000, 'prosac', 0, 'conf', 0.999, label_fields=['BFR','GC', 'iters'])

    draw_line(data, colors[1][1], 'BFR', 3, 'GC', 1, 'iters', 10**6, 'prosac', 0, 'conf', 0.999, label_fields=['BFR','GC', 'iters'])
    draw_references(ref_data, ref_names)

    draw_line(data, 'darkorange', 'BFR', -1, 'GC', 1, 'iters', 10**6, 'conf', 0.9995, 'prosac', 0, label_fields=['BFR','GC','iters','conf'])
    draw_line(data, 'black', 'BFR', -1, 'GC', 1, 'iters', 10**6, 'prosac', 1, label_fields=['BFR','GC','iters','prosac'])
    draw_line(data, 'deeppink', 'BFR', -1, 'GC', 1, 'iters', 10**6, 'prosac', 0, 'coherence', 0.975, label_fields=['BFR','GC','iters','coherence'])
    draw_line(data, 'yellow', 'BFR', 3, 'GC', 1, 'iters', 10**6, 'prosac', 0, 'coherence', 0.975, label_fields=['BFR','GC','iters','coherence'])

    for i in range(2):
        ax = plt.subplot(1,2,i+1)
        ax.legend()
        plt.axis([0,None,None,None])
   
    plt.suptitle(name)

def get_subset(data, *args , label_fields=None):
    is_cur = np.ones(data.shape[0], dtype=bool)
    lbl = ''
    for i in range(0,len(args),2):
        prop = args[i]
        val = float(args[i+1])
        is_cur = is_cur & (data[:,d[prop]]==val)
        if (label_fields is None) or (prop in  label_fields):
            if prop=='iters':
                if val % 10**6 == 0:
                    lbl += f"{int(val / 10**6)}M "
                elif val % 10**3 == 0:
                    lbl += f"{int(val / 10**3)}K "
                else:
                    lbl += f"{prop}={val} "    
            elif prop=='BFR':
                lbl += (f"BFR({val}) " if val>0 else "MFR ")
            elif prop == 'GC':
                lbl += ("GC " if val else "RANSAC ")
            else:
                lbl += f"{prop}={val} "

    return is_cur, lbl

def process_variance(cur_data):
    major_keys = list(set(keys) - set(['t_base', 'acc_base', 't_icp', 'acc_icp', 'take']))

    mean_data_list = []    
    hulls = []
    remainder = cur_data
    
    is_multi_take = remainder[:,d['take']]>1    
    while is_multi_take.sum() > 0:

        i = np.where(is_multi_take)[0][0]

        args = []
        for k in major_keys:
            args.append(k)
            args.append(remainder[i,d[k]])

        mask, _ = get_subset(remainder, *args)

        cur_d = remainder[mask,:]
        cur_mean = np.mean(cur_d,axis=0,keepdims=True)
        for k in major_keys:
            cur_mean[0,d[k]] = remainder[i,d[k]]
        mean_data_list.append(cur_mean)
    
        cur_hull = []
        for j in range(2):
            points = np.vstack([cur_d[:,d[flds[j][0]]], cur_d[:,d[flds[j][1]]]]).T
            h = ConvexHull(points)
            cur_hull.append([deepcopy(points), deepcopy(h)])
        cur_hull.append(deepcopy(args))
        hulls.append(cur_hull)

        remainder = remainder[~mask,:]
        is_multi_take = remainder[:,d['take']]>1

    mean_data_list.append(remainder)
    final_data = np.vstack(mean_data_list)
    return final_data, hulls

def draw_all_hulls(hulls, color='k'):

    for hull in hulls:
        for j in range(2):
            ax = plt.subplot(1,2,j+1)
            points, h = hull[j]
            for simplex in h.simplices:
                ax.plot(points[simplex, 0], points[simplex, 1], '-', c=color)
            ax.scatter(points[:,0],points[:,1],color=color,marker='*')

def draw_line(data, color, *args , label_fields=None, print_data=False):
    if 'coherence' not in args:
        args += ('coherence', 0)

    is_cur, lbl = get_subset(data, *args , label_fields=label_fields)

    cur_data = data[is_cur,:]
    if print_data:
        print(keys)
        print(cur_data)
    for j in range(2):
        ax = plt.subplot(1,2,j+1)
        ax.plot(cur_data[:,d[flds[j][0]]],
                cur_data[:,d[flds[j][1]]], 
                'o-', 
                c=color,
                label=lbl
        )

def generic(name, ref_data, ref_names):    

    data = parse_summary(f'logs/summary_{name}.txt')
    if name == 'B_to_B':
        more_data = parse_summary('logs/oct28.txt')
        data = np.vstack([data,more_data])
    data, hulls = process_variance(data)
    ord = np.argsort(data[:,d['iters']],axis=0,kind='stable')
    data = data[ord,:]
    ord = np.argsort(data[:,d['conf']],axis=0,kind='stable')
    data = data[ord,:]

    colors = [['darkviolet', 'turquoise', 'lightsteelblue', 'teal', 'blue'], ['red','lightcoral', 'tomato', 'firebrick', 'maroon']]
    symbols = ['v','s','p','P','*']
    plt.figure()
    draw_all_hulls(hulls)
    BFR_vals = np.unique(data[:,d['BFR']])
    for GC in [0,1]:
        for i, BFR in enumerate(BFR_vals):    
            draw_line(data, colors[GC][i], 'BFR', BFR, 'GC', GC, 'prosac', 0, 'conf', 0.999, label_fields=['BFR','GC']) # AD TODO: add 'coherence', 0
    for j in range(2):
        ax = plt.subplot(1,2,j+1)
        plt.title(flds[j])
        plt.xlabel('sec')        
        for ref_i in range(len(ref_names)):
            if ref_names[ref_i] == 'DGR':
                continue
            plt.plot(ref_data[ref_i][d[flds[j][0]]], ref_data[ref_i][d[flds[j][1]]], symbols[ref_i], label=ref_names[ref_i])


    ax = plt.subplot(1,2,1)
    a = ax.axis()
    teaser_t = ref_data[ref_names.index('TEASER++'),d['t_base']]
    dsc_t = ref_data[ref_names.index('PointDSC'),d['t_base']]
    m = min(teaser_t, dsc_t)
    print(m)
    ax.plot([m,m], a[2:],'k--')

    ax = plt.subplot(1,2,2)
    a = ax.axis()
    teaser_t = ref_data[ref_names.index('TEASER++'),d['t_icp']]
    dsc_t = ref_data[ref_names.index('PointDSC'),d['t_icp']]
    m = min(teaser_t, dsc_t)
    ax.plot([m,m], a[2:],'k--')

    if name == 'B_to_B':
        draw_line(data, 'deeppink', 'BFR', 3, 'GC', 1,  'iters', 10**6, 'prosac', 0, label_fields=['BFR','GC','iters'])
        draw_line(data, 'greenyellow', 'BFR', -1, 'GC', 1, 'conf', 0.99, 'prosac', 0, label_fields=['BFR','GC','conf'])
        draw_line(data, 'darkorange', 'BFR', -1, 'GC', 1, 'iters', 10**6, 'prosac', 0, label_fields=['BFR','GC','iters'])
        draw_line(data, 'black', 'BFR', -1, 'GC', 1, 'iters', 10**6, 'prosac', 1, label_fields=['BFR','GC','iters','prosac'])


    for i in range(2):
        ax = plt.subplot(1,2,i+1)
        ax.legend()
        plt.axis([0,None,None,None])
   
    plt.suptitle(name)


        
#A_to_B()            
B_to_B()
plt.show()





