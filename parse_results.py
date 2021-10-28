from sys import platform
import numpy as np
import matplotlib.pyplot as plt

def parse_summary(d, filename, is_GC):
    try:
        with open(filename, 'r') as fid:
            text = fid.read().splitlines()
    except:
        return np.empty([1,len(d.keys())])

    if is_GC:
        alg_name = 'GC'
        offs = 2
    else:
        alg_name = 'RANSAC'
        offs=0

    data_list = []
    for line in text:    
        if line.startswith('==>'):
            cur_row = np.zeros(len(d.keys()))
            L = line.replace('.txt','')
            a = L.split()
            b = a[1].split('_')
            cur_row[d['iters']] = float(b[2+offs].replace('K','000').replace('M','000000'))
            if b[-2] == 'BFR':
                cur_row[d['BFR']] = float(b[-1])
            else:
                cur_row[d['BFR']] = -1

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
    data[:,d['GC']] = is_GC
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


def B_to_B():
    ref_names = ['DGR', 'PointDSC', 'TEASER++', 'MFR+RANSAC', 'DFR+RANSAC']
    ref_data = np.array([
        [0,0, 57.91,0.453,61.81,0.494],
        [0,0,80.56,0.236,82.48,0.290],
        [0,0,77.43,0.331,86.88,0.378],
        [0,0,83.37,0.078,88.31,0.133],
        [0,0,82.14,0.109,88.70,0.165]])
    ref_data = ref_data[:,[0,1,3,2,5,4]]

    generic('B_to_B', ref_data, ref_names)


def generic(name, ref_data, ref_names):
    d = { k:i for i,k in enumerate(['iters','BFR','t_base', 'acc_base', 't_icp', 'acc_icp', 'GC'])}

    data1 = parse_summary(d, f'logs/res_GC_{name}_BFR.txt', is_GC=True)
    data2 = parse_summary(d, f'logs/res_GC_{name}_MFR.txt', is_GC=True)
    data3 = parse_summary(d, f'logs/res_RANSAC_{name}_BFR.txt', is_GC=False)
    data4 = parse_summary(d, f'logs/res_RANSAC_{name}_MFR.txt', is_GC=False)

    data = np.vstack([data1,data2, data3, data4])

    ord = np.argsort(data[:,d['iters']])
    data = data[ord,:]
    print(data)

    colors = [['darkviolet', 'turquoise', 'lightsteelblue', 'teal', 'blue'], ['red','lightcoral', 'tomato', 'firebrick', 'maroon']]
    symbols = ['v','s','p','P','*']
    plt.figure()
    ax = plt.gca()
    flds  = [['t_base', 'acc_base'], ['t_icp', 'acc_icp']]
    BFR_vals = np.unique(data[:,d['BFR']])
    for j in range(2):
        ax = plt.subplot(1,2,j+1)
        for GC in [0,1]:
            for i, BFR in enumerate(BFR_vals):
                is_cur = (data[:,d['BFR']] == BFR) & (data[:,d['GC']] == GC)
                cur_data = data[is_cur,:]    
                lbl = (f"BFR({BFR})" if BFR>0 else "MFR") + ("+GC" if GC else "+RANSAC")
                ax.plot(cur_data[:,d[flds[j][0]]],
                        cur_data[:,d[flds[j][1]]], 
                        'o-', 
                        c=colors[GC][i],
                        label=lbl)
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

    for i in range(2):
        ax = plt.subplot(1,2,i+1)
        ax.legend()
        plt.axis([0,None,None,None])
    
    plt.suptitle(name)

        
A_to_B()            
B_to_B()
plt.show()
