from sys import platform
import numpy as np
np.set_printoptions(precision=4,suppress=True, linewidth=100)
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import qhull, ConvexHull

from copy import deepcopy

keys = ['iters','BFR','t_base', 'acc_base', 't_icp', 'acc_icp', 'GC', 'prosac', 'conf', 'take', 'coherence', 'distratio', 'edgelen', 'AtoB', 'LO']

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
                cur_row[d['prosac']] = float(b[b.index('prosac')+1])
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
            if 'distratio' in b:
                cur_row[d['distratio']] = float(b[b.index('distratio')+1])
            else:
                cur_row[d['distratio']] = 0
            if 'edgelen' in b:
                cur_row[d['edgelen']] = float(b[b.index('edgelen')+1])
            else:
                cur_row[d['edgelen']] = 0                
            if 'LO' in b:
                cur_row[d['LO']] = float(b[b.index('LO')+1])
            else:
                cur_row[d['LO']] = 1                                
            if 'A_to_B' in line:
                cur_row[d['AtoB']] = 1
            elif 'B_to_B' in line:
                cur_row[d['AtoB']] = 0
            else:
                cur_row[d['AtoB']] = np.nan



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

def A_to_S(show_DGR=False):
    ref_names = ['DGR', 'PointDSC', 'TEASER++', 'MFR+RANSAC', 'DFR+RANSAC']
    empty = np.zeros([1,6])
    ref_data = np.vstack([np.zeros([1,6]),
    np.array([0,0, 76.70, 0.367, 79.01, 0.493]),
    np.array([0,0,73.65,0.176,86.57,0.263]),
    np.zeros([1,6]),
    np.zeros([1,6])])
    ref_data = ref_data[:,[0,1,3,2,5,4]]
    name = 'A_to_S'

    data, hulls = prep_data(name)

    plt.figure()    
    
    draw_line(data, 'red', 'BFR', 2, 'GC', 1, 'iters', 50000, 'conf', 0.999, 'prosac', 1, 'distratio', 1, 'edgelen', 1, label_fields=['BFR','GC','iters','conf','prosac','edgelen'])        
    draw_line(data, 'blue', 'BFR', -1, 'GC', 1, 'iters', 10**6, 'conf', 0.9995, 'coherence', 0, 'prosac', 1, 'distratio', 1, 'edgelen', 1, label_fields=['BFR','GC','iters','conf','prosac','edgelen'])    

    draw_references(ref_data, ref_names, show_DGR=show_DGR)    

    for i in range(2):
        ax = plt.subplot(1,2,i+1)
        ax.legend()    

    plt.suptitle(name)

def A_to_B(show_DGR=False):
    ref_names = ['DGR', 'PointDSC', 'TEASER++', 'MFR+RANSAC', 'DFR+RANSAC']
    ref_data = np.array([[0,0, 44.95, 0.418, 48.07, 0.462],
        [0,0,63.97,0.234,66.78,0.293],
        [0,0,59.88,0.146,71.99,0.213],
        [0,0,66.01,0.137,74.54,0.197],
        [0,0,64.16,0.126,73.88,0.188]])
    ref_data = ref_data[:,[0,1,3,2,5,4]]
    name = 'A_to_B'

    data, hulls = prep_data(name)

    colors = [['red', 'turquoise', 'lightsteelblue', 'teal', 'blue'], ['red','red', 'tomato', 'firebrick', 'maroon']]
    plt.figure()        
    draw_line(data, 'red', 'BFR', 2, 'GC', 1, 'iters', 50000, 'conf', 0.999, 'prosac', 1, 'distratio', 1, 'edgelen', 1, label_fields=['BFR','GC','iters','conf','prosac','edgelen'])            
    draw_line(data, 'blue', 'BFR', -1, 'GC', 1, 'iters', 10**6, 'conf', 0.9995, 'coherence', 0, 'prosac', 1, 'distratio', 1, 'edgelen', 1, label_fields=['BFR','GC','iters','conf','prosac','edgelen'])    
    draw_line(data, 'pink', 'BFR', 2, 'GC', 1, 'iters', 50000, 'conf', 0.999, 'prosac', 1, 'distratio', 1, 'edgelen', 1, 'coherence', 0.975, label_fields=['BFR','GC','iters','conf','prosac','edgelen'])            
    draw_line(data, 'cyan', 'BFR', -1, 'GC', 1, 'iters', 10**6, 'conf', 0.9995, 'coherence', 0.975, 'prosac', 1, 'distratio', 1, 'edgelen', 1, label_fields=['BFR','GC','iters','conf','prosac','edgelen'])    


    draw_references(ref_data, ref_names, show_DGR=show_DGR)

    for i in range(2):
        ax = plt.subplot(1,2,i+1)
        ax.legend()    

    plt.suptitle(name)


def prep_data(names):
    if not 'list' in str(type(names)):
        names = [names]
    data_list = []
    for name in names:
        cur_data = parse_summary(f'logs/summary_{name}.txt')
        data_list.append(cur_data)
    data = np.vstack(data_list)

    data, hulls = process_variance(data)
    
    ord = np.argsort(data[:,d['iters']],axis=0,kind='stable')
    data = data[ord,:]
    ord = np.argsort(data[:,d['conf']],axis=0,kind='stable')
    data = data[ord,:]
    return data, hulls

def draw_references(ref_data, ref_names, plot_separator=True, suffix='', show_DGR=False, set_axes=False):
    if show_DGR:
        ignore_list = ['DFR+RANSAC', 'MFR+RANSAC']
    else:
        ignore_list = ['DGR', 'DFR+RANSAC', 'MFR+RANSAC']
    symbols = ['v','s','p','P','*']
    colors = ['black','green','orange','','']
    for j in range(2):
        ax = plt.subplot(1,2,j+1)
        if j==0:
            plt.title('base')
        elif j==1:
            plt.title('with ICP')
        plt.xlabel('sec')        
        for ref_i in range(len(ref_names)):
            if ref_names[ref_i] in ignore_list:
                continue
            plt.plot(ref_data[ref_i][d[flds[j][0]]], ref_data[ref_i][d[flds[j][1]]], symbols[ref_i], c=colors[ref_i], label=ref_names[ref_i]+' '+suffix)

    if set_axes:
        for i in range(2):
            ax = plt.subplot(1,2,i+1)
            plt.axis([0,None,None,None])
            ax.legend()
    
    for j in range(2):
        if plot_separator:
            ax = plt.subplot(1,2,j+1)
            a = ax.axis()
            teaser_t = ref_data[ref_names.index('TEASER++'),d[flds[j][0]]]
            dsc_t = ref_data[ref_names.index('PointDSC'),d[flds[j][0]]]
            m = min(teaser_t, dsc_t)
            teaser_acc = ref_data[ref_names.index('TEASER++'),d[flds[j][1]]]
            dsc_acc = ref_data[ref_names.index('PointDSC'),d[flds[j][1]]]
            M = max(teaser_acc, dsc_acc)
            ax.plot([m,m], a[2:],'--',c='gray')
            ax.plot(list(a[:2]), [M,M],'--',c='gray')

def B_to_B_variants():
    name = 'B_to_B'
    data, hulls = prep_data(name)

    plt.figure()
   
    draw_line(data, 'blue', 'BFR', -1, 'GC', 1, 'iters', 10**6, 'conf', 0.9995, 'prosac', 1, 'edgelen', 1, label_fields=['BFR', 'prosac', 'edgelen'], hulls=hulls, marker='o', label='+prosac +edge-len')    
    draw_line(data, 'darkorange', 'BFR', -1, 'GC', 1, 'iters', 10**6, 'conf', 0.9995, 'prosac', 1, 'edgelen', 0, label_fields=['BFR', 'prosac', 'edgelen'], hulls=hulls, marker='D', label='+prosac')    
    draw_line(data, 'black', 'BFR', -1, 'GC', 1, 'iters', 10**6, 'conf', 0.9995, 'prosac', 0, 'edgelen', 1, label_fields=['BFR', 'prosac', 'edgelen'], hulls=hulls, marker='d', label='+edge-len')    
    draw_line(data, 'green', 'BFR', -1, 'GC', 1, 'iters', 10**6, 'conf', 0.9995, 'prosac', 0, 'edgelen', 0, label_fields=['BFR', 'prosac', 'edgelen'], hulls=hulls, marker='P', label='local-optimization only')    
    draw_line(data, 'yellow', 'BFR', -1, 'GC', 1, 'iters', 10**6, 'conf', 0.9995, 'prosac', 0, 'edgelen', 1, 'LO', 0, label_fields=['BFR', 'prosac', 'edgelen'], hulls=hulls, marker='P', label='-LO +edge-len')    

    #draw_line(data, 'purple', 'BFR', -1, 'GC', 0, 'iters', 10**6, 'distratio', 0, label_fields=['BFR', 'GC'], hulls=hulls, marker='s')    

    for i in range(2):
        ax = plt.subplot(1,2,i+1)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True)
        #ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), fancybox=True, shadow=True)        
        plt.xlabel("Time(s)")
        if i == 0:
            plt.ylabel("Recall(%)")

    #plt.gcf().subplots_adjust(bottom=0.1, top=0.8)
    plt.gcf().subplots_adjust(bottom=0.3)
def B_to_B(show_DGR=False):
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

    plt.figure()    
    draw_line(data, 'black', 'BFR', 3, 'GC', 1, 'iters', 10**6, 'prosac', 1, 'conf', 0.999, 'distratio', 1, 'edgelen', 1, label_fields=['BFR','GC', 'iters', 'prosac', 'edgelen', 'conf'])          
    draw_line(data, 'red', 'BFR', 2, 'GC', 1, 'iters', 50000, 'prosac', 1, 'conf', 0.999, 'distratio', 1, 'edgelen', 1, label_fields=['BFR','GC', 'iters', 'prosac', 'edgelen', 'conf'])              

    draw_line(data, 'blue', 'BFR', -1, 'GC', 1, 'iters', 10**6, 'conf', 0.9995, 'prosac', 1, 'distratio', 1, 'edgelen', 1, label_fields=['BFR','GC','iters','conf', 'prosac', 'edgelen'])    

    draw_references(ref_data, ref_names, show_DGR=show_DGR)        

    for i in range(2):
        ax = plt.subplot(1,2,i+1)
        ax.legend()    
   
    plt.suptitle(name)

def get_subset(data, *args , label_fields=None, short_label=False):
    is_cur = np.ones(data.shape[0], dtype=bool)
    lbl = ''
    for i in range(0,len(args),2):
        prop = args[i]
        val = float(args[i+1])
        is_cur = is_cur & (data[:,d[prop]]==val)
        if short_label:
            if prop=='BFR':
                lbl += (f"BF " if val>0 else "MF ")
            elif prop == 'GC':
                lbl += ("GC " if val else "open3D ")
            if prop=='AtoB':
                lbl += ("cross " if val else "same ")
        else:
            if (label_fields is None) or (prop in label_fields):
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
                    lbl += ("GC " if val else "open3D ")
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
    
        if cur_d.shape[0]>2:
            cur_hull = []
            for j in range(2):
                points = np.vstack([cur_d[:,d[flds[j][0]]], cur_d[:,d[flds[j][1]]]]).T
                try:
                    h = ConvexHull(points)
                except qhull.QhullError as E:
                    if "input is less than 2-dimensional since all points have the same" in str(E):
                        points += 0.001*np.random.rand(*points.shape)
                        h = ConvexHull(points)
                    else:
                        raise E
                cur_hull.append([deepcopy(points), deepcopy(h)])
            cur_hull.append(deepcopy(args))
            hulls.append(cur_hull)

        remainder = remainder[~mask,:]
        is_multi_take = remainder[:,d['take']]>1

    mean_data_list.append(remainder)
    final_data = np.vstack(mean_data_list)
    return final_data, hulls

def draw_all_hulls(hulls, color='k'):
    if hulls is None:
        return

    for hull in hulls:
        draw_hull(hull, 'k')

def get_hull(hulls, args):
    for j in range(len(hulls)):
        is_same = True
        for i in range(0,len(args),2):
            fld = args[i]
            ind = hulls[j][2].index(fld)
            is_same &= (hulls[j][2][ind+1]==args[i+1])
        if is_same:
            return hulls[j]
    return None

def draw_hull(hull, color, marker):
    if hull is None:
        return
    for j in range(2):            
        ax = plt.subplot(1,2,j+1)
        points, h = hull[j]
        for simplex in h.simplices:
            ax.plot(points[simplex, 0], points[simplex, 1], '--', c=color)
        ax.scatter(points[:,0],points[:,1],color=color,marker=marker,facecolors='none')

def draw_line(data, color, *args , label_fields=None, print_data=False, marker=None, short_label=False, hulls=None, label=None):
    if marker is None:
        marker = 'o'
    if 'coherence' not in args:
        args += ('coherence', 0)
    if 'distratio' not in args:
        args += ('distratio', 1)
    if 'edgelen' not in args:
        args += ('edgelen', 0)
    if 'LO' not in args:
        args += ('LO', 1)        

    is_cur, lbl = get_subset(data, *args , label_fields=label_fields, short_label=short_label)
    if hulls is not None:
        hull = get_hull(hulls, args)
        draw_hull(hull, color, marker)

    if label is not None:
        lbl = label

    cur_data = data[is_cur,:]
    if print_data:
        print(keys)
        print(cur_data)
    for j in range(2):
        ax = plt.subplot(1,2,j+1)
        xs = cur_data[:,d[flds[j][0]]]
        ys = cur_data[:,d[flds[j][1]]]
        if len(xs)==1:
            line = ''
        else:
            line = '-'
        ax.plot(xs,ys,
                marker + line, 
                c=color,
                label=lbl
        )

def compare_all():
    pass            

    ref_names = ['DGR', 'PointDSC', 'TEASER++', 'MFR+RANSAC', 'DFR+RANSAC']
    ref_data = np.array([[0,0, 44.95, 0.418, 48.07, 0.462],
        [0,0,63.97,0.234,66.78,0.293],
        [0,0,59.88,0.146,71.99,0.213],
        [0,0,66.01,0.137,74.54,0.197],
        [0,0,64.16,0.126,73.88,0.188]])
    ref_data = ref_data[:,[0,1,3,2,5,4]]

    ref_data_B_to_B = np.array([
        [0,0, 57.91,0.453,61.81,0.494],
        [0,0,80.56,0.236,82.48,0.290],
        [0,0,77.43,0.331,86.88,0.378],
        [0,0,83.37,0.078,88.31,0.133],
        [0,0,82.14,0.109,88.70,0.165]])
    ref_data_B_to_B = ref_data_B_to_B[:,[0,1,3,2,5,4]]

    data, hulls = prep_data(['A_to_B', 'B_to_B'])


    plt.figure()    

    #draw_line(data, 'teal', 'AtoB', 1, 'BFR', 3, 'GC', 0, 'iters', 500000, 'prosac', 0, 'conf', 0.999, label_fields=['BFR','GC','iters'], marker='P', short_label=True)    
    draw_line(data, 'red', 'AtoB', 1, 'BFR', 2, 'GC', 1, 'iters', 50000, 'conf', 0.999, 'prosac', 1, 'distratio', 1, 'edgelen', 1, label_fields=['BFR','GC','iters','conf','prosac','edgelen'], marker='P', short_label=True)        
    draw_line(data, 'blue', 'AtoB', 1, 'BFR', -1, 'GC', 1, 'iters', 1000000, 'conf', 0.9995, 'coherence', 0, 'prosac', 1, 'distratio', 1, 'edgelen', 1, label_fields=['BFR','GC','iters','conf','prosac','distratio','edgelen'], marker='P', short_label=True)
    #draw_line(data, 'purple', 'AtoB', 1, 'BFR', -1, 'GC', 0, 'iters', 10**6, 'prosac', 0, 'conf', 0.999, 'coherence', 0, label_fields=['BFR','GC'], marker='P', short_label=True)

    draw_line(data, 'red', 'AtoB', 0, 'BFR', 2, 'GC', 1, 'iters', 50000, 'conf', 0.999, 'prosac', 1, 'distratio', 1, 'edgelen', 1, label_fields=['BFR','GC','iters','conf','prosac','edgelen'], short_label=True)        
    draw_line(data, 'blue', 'AtoB', 0, 'BFR', -1, 'GC', 1, 'iters', 1000000, 'conf', 0.9995, 'coherence', 0, 'prosac', 1, 'distratio', 1, 'edgelen', 1, label_fields=['BFR','GC','iters','conf','prosac','distratio','edgelen'], short_label=True)

    # draw_line(data, 'purple', 'AtoB', 0, 'BFR', -1, 'GC', 0, 'iters', 10**6, 'prosac', 0, 'conf', 0.999, label_fields=['BFR','GC','iters'], short_label=True)        
    # draw_line(data, 'teal', 'AtoB', 0, 'BFR', 3, 'GC', 0, 'iters', 1500000, 'prosac', 0, 'conf', 0.999, 'distratio', 1, label_fields=['BFR','GC', 'iters'], short_label=True)
    # draw_line(data, 'red', 'AtoB', 0, 'BFR', 3, 'GC', 1, 'iters', 10**6, 'prosac', 1, 'conf', 0.999, 'distratio', 1, 'edgelen', 1, label_fields=['BFR','GC', 'iters', 'prosac', 'edgelen'], short_label=True)    
    # draw_line(data, 'blue', 'AtoB', 0, 'BFR', -1, 'GC', 1, 'iters', 10**6, 'conf', 0.9995, 'prosac', 1, 'distratio', 1, 'edgelen', 1, label_fields=['BFR','GC','iters','conf', 'prosac', 'edgelen'], short_label=True)    

    draw_references(ref_data, ref_names, plot_separator=False, suffix='cross', set_axes=False, show_DGR=True)
    draw_references(ref_data_B_to_B, ref_names, plot_separator=False, suffix='same', set_axes=False, show_DGR=True)

    for i in range(2):        
        ax = plt.subplot(1,2,i+1)
        a = list(ax.axis())
        if i==0:
            b = a
        elif i==1:
            b = [min(a[0],b[0]), max(a[1],b[1]), min(a[2],b[2]), max(a[3],b[3])]
    ax.legend()
    plt.axis(b)
    plt.subplot(1,2,1)
    plt.axis(b)    

def parse_tables():
    latex_table = [
r"\multirow{3}{*}{A} & A & \textbf{99.00} & \uu{96.78} & 94.02 & 73.10 & 0.15 \\\cline{2-7}",
r"& B & \uu{88.98} & \textbf{92.75} & 88.53 & 85.67 & 0.08 \\\cline{2-7}",
r"& S & \textbf{96.70} & \uu{95.72} & 93.54 & 81.45 & 0.11  \\\Xhline{2\arrayrulewidth}",
r"\multirow{3}{*}{B} & A & \uu{73.88} & \textbf{74.54} & 66.40 & 72.11 & 0.08  \\\cline{2-7}",
r"& B & \textbf{88.70} & \uu{88.31} &  82.37 & 86.88 &  0.12 \\\cline{2-7}",
r"& S & \textbf{82.75} & \uu{81.29} & 75.39 & 79.63 &  0.11  \\\Xhline{2\arrayrulewidth}",
r"\multirow{3}{*}{S} & A & 86.19 & \textbf{87.19} & 79.01 & \uu{86.69} &  0.08  \\\cline{2-7}",
r"& B & 87.42 & \textbf{89.97} & 82.02 & \uu{89.16} &  0.08 \\\cline{2-7}",
r"& S & \textbf{93.98} & \uu{93.75} & 90.59 & 93.29 &  0.12 \\\hline"
]

    def extract_data(text):
        data = []
        for line in text:
            if line.startswith("==>"):
                for tgt in ['A','B','S']:
                    for src in ['A','B','S']:
                        if f"_{src}_to_{tgt}_" in line:
                            cur_src = src
                            cur_tgt = tgt                                            
            
            if "filtered pairs" in line:
                a = line.split('(')
                b = a[1].split()
                inlier_ratio = float(b[0])
            
            if line.startswith("GC+ICP"): 
                a = line.split(',')            
                acc_icp = float(a[0].split()[-1][:-1])            
                data.append([cur_src, cur_tgt, acc_icp, inlier_ratio])

        res = np.vstack(data)
        return res

    with open('logs/GPF_full_table.txt','r') as fid:
        GPF_text = fid.read().splitlines()
    GPF_data = extract_data(GPF_text)

    with open('logs/MFR_full_table.txt','r') as fid:
        MFR_text = fid.read().splitlines()        
    MFR_data = extract_data(MFR_text)

    new_latex_table = []
    latex_row_ind = 0
    for tgt in ['A','B','S']:
        for src in ['A','B','S']:
            print(f'latex_row_ind={latex_row_ind}')
            latex_row = latex_table[latex_row_ind]
            a = latex_row.split('&')
            cur_row_GPF = (GPF_data[:,0] == src) & (GPF_data[:,1] == tgt)
            assert cur_row_GPF.sum() == 1
            GPF_acc = float(GPF_data[cur_row_GPF,2])
            inlier_ratio = float(GPF_data[cur_row_GPF,3])
            cur_row_MFR = (MFR_data[:,0] == src) & (MFR_data[:,1] == tgt)
            assert cur_row_MFR.sum() == 1
            MFR_acc = float(MFR_data[cur_row_MFR,2])
            inlier_ratio2 = float(MFR_data[cur_row_MFR,3]) # AD DEL
            assert (float(inlier_ratio2)==float(inlier_ratio)) # AD DEL
            a[2] = f' {GPF_acc:.2f} '
            a[3] = f' {MFR_acc:.2f} '
            b = a[-1].split()
            b[0] = f' {inlier_ratio:.2f} '
            a[-1] = ''.join(b)         
            if tgt == 'A': # manually insert some new TEASER results (with GPF)
                if src == 'A':
                    a[-2] = ' 96.65 '
                elif src  == 'B':
                    a[-2] = ' 92.62 '
                elif src == 'S':
                    a[-2] = ' 95.16 '

            new_latex_table.append('&'.join(a))
            latex_row_ind += 1
        
    for line in new_latex_table:
        print(line)

def parse_running_times():
    
    def extract_data(data_name):        
        if data_name == "TEASER":
            alg_name = "TEASER"
        else:
            alg_name = "GC"

        with open(f'logs/{data_name}_full_table.txt','r') as fid:
            text = fid.read().splitlines()
        
        data = []
        t_base = None
        for line in text:
            if line.startswith("==>"):
                for tgt in ['A','B','S']:
                    for src in ['A','B','S']:
                        if (f"_{src}_to_{tgt}_" in line) or (f"_{src}_TO_{tgt}_" in line):
                            cur_src = src
                            cur_tgt = tgt             
                         
                        elif (f"train_{src}" in line) and (f"test_{tgt}" in line):
                            cur_src = src
                            cur_tgt = tgt              
            
            if line.startswith(alg_name + "+ICP"): 
                a = line.split(',')            
                t_icp = float(a[-1].split()[-1])        
                if alg_name=="TEASER":
                    t_icp *= 0.75   
                data.append([cur_src, cur_tgt, t_base, t_icp])

            elif line.startswith(alg_name):
                a = line.split(',')
                t_base = float(a[-2].split()[-1])
                if alg_name=="TEASER":
                    t_base *= 0.75               

        res = np.vstack(data)
        return res

    D = {}
    data_names = ['GPF','MFR','TEASER']
    for data_name in data_names:
        D[data_name] = extract_data(data_name)

    base_list = []
    icp_list = []
    for tgt in ['A','B','S']:
        for src in ['A','B','S']:            
            cur_base = []
            cur_icp = []
            for data_name in data_names:
                is_cur_row = (D[data_name][:,0] == src) & (D[data_name][:,1] == tgt)
                assert is_cur_row.sum() == 1
                cur_base.append(D[data_name][is_cur_row,2])
                cur_icp.append(D[data_name][is_cur_row,3])
            base_list.append(cur_base)
            icp_list.append(cur_icp)

    print("tgt | src ||        base        ||        icp         |")
    print("    |     || GPF | MFR | TEASER || GPF | MFR | TEASER |")
    print("------------------------------------------------------|")

    i = 0
    for tgt in ['A','B','S']:
        for src in ['A','B','S']:            
            print(f" {tgt}  | {src}  | {float(base_list[i][0]):.2f} | {float(base_list[i][1]):.2f} | {float(base_list[i][2]):.2f}  || {float(icp_list[i][0]):.2f} | {float(icp_list[i][1]):.2f} | {float(icp_list[i][2]):.2f}")
            i += 1

def bar_kitti_10m():    
    matplotlib.rcParams.update({'font.size': 12})
    names = np.array(['DGR', 'PointDSC', 'HRegNet', 'D3Feat', 'PREDATOR'])
    recalls = np.array([96.9, 98.2, 99.7, 99.8, 99.8])
    nums = np.array([555-17,555-10, 555-2, 555-1, 555-1])
    ord = np.argsort(nums)    
    nums = nums[ord]
    recalls = recalls[ord]
    names = names[ord]
    fig, ax = plt.subplots()
    colors = ['teal','green','blue','darkgoldenrod','m']
    bar_plot = plt.bar(np.arange(len(nums)), recalls, tick_label=names,color=colors)
    #ax = plt.axis()
    #plt.plot(ax[:2], [100]*2, 'k--')
    #plt.xticks(np.arange(len(nums)), names)
    plt.ylabel('Recall (%)')

    def autolabel(rects):
        for idx,rect in enumerate(bar_plot):
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 0.87*height,
                    f"{recalls[idx]}%\n({nums[idx]})",
                    ha='center', va='bottom', rotation=0, c='w')
        
    autolabel(bar_plot)
    plt.axis([None,None,None,100])

#bar_kitti_10m()
#parse_running_times()
#parse_tables()
#B_to_B_variants()
# A_to_S()
# A_to_B()            
# B_to_B()
# A_to_B(show_DGR=True)            
# B_to_B(show_DGR=True)
compare_all()
plt.show()





