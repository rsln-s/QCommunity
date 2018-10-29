import numpy as np
import matplotlib.pyplot as plt
import re
import pandas


def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color, linewidth=4)
    plt.setp(bp['whiskers'], color=color, linewidth=4)
    plt.setp(bp['caps'], color=color, linewidth=4)
    plt.setp(bp['medians'], color=color, linewidth=4)


def two_figures():
    """ Returns two sepetate box plots"""
    # results from gurobi global, version 6.5.0
    gurobi_global = {
        'sociopatterns-infectious': 0.4633343671,
        'moreno_oz_oz': 0.3385620174,
        'contact': 0.1049169921,
        'brunson_revolution_revolution': 0.4310351647,
        'arenas-jazz': 0.3206093636,
        'arenas-meta': 0.31373627450980396
    }
    dwavefile = 'expr_dwave16.txt'
    gurobifile = 'expr_gurobi16.txt'
    qaoafile = 'ibm.csv'
    graphInfoFile = 'graph_info.txt'
    graphInfo = {}
    myfile = open(graphInfoFile, 'r')
    for line in myfile:
        graph, n, m = line.split()
        graphInfo[graph] = {'nodes': int(n), 'edges': int(m)}
    myfile.close()
    dwave_data = {}
    gurobi_data = {}
    qaoa_data = {}
    for i in graphInfo:
        if not i.endswith('email'):
            dwave_data[i] = {'mod': [], 'iter': []}
            gurobi_data[i] = {'mod': [], 'iter': []}
            qaoa_data[i] = {'mod': [], 'iter': []}
    myfile = open(dwavefile, 'r')
    for line in myfile:
        line_split = line.split()
        dwave_data[line_split[0]]['mod'].append(float(line_split[-2]))
        dwave_data[line_split[0]]['iter'].append(int(line_split[-1]))
    myfile.close()
    myfile = open(gurobifile, 'r')
    for line in myfile:
        line_split = line.split()
        gurobi_data[line_split[0]]['mod'].append(float(line_split[-2]))
        gurobi_data[line_split[0]]['iter'].append(int(line_split[-1]))
    myfile.close()
    myfile = open(qaoafile, 'r')
    for line in myfile:
        ls = line.split('_seed_')
        graphname = ls[0].split('.')[-1]
        ls2 = ls[1].split(',')
        mod = 0.25 * float(ls2[-3]) / graphInfo[graphname]['edges']
        it = int(ls2[-2])
        qaoa_data[graphname]['mod'].append(mod)
        qaoa_data[graphname]['iter'].append(it)
    myfile.close()

    # modularity data in list of lists
    dwave_box_mod = [dwave_data[g]['mod'] for g in dwave_data]
    gurobi_box_mod = [gurobi_data[g]['mod'] for g in dwave_data]
    qaoa_box_mod = [qaoa_data[g]['mod'] for g in dwave_data]
    ticks = [re.split('-|_', i)[-1] for i in dwave_data.keys()]
    global_data = [gurobi_global[i] for i in dwave_data]
    myticks = list(xrange(0, len(ticks) * 2, 2))

    # horizontal lines
    for i, val in enumerate(global_data[:-1]):
        plt.hlines(
            y=val,
            xmin=myticks[i] - 0.5,
            xmax=myticks[i] + 0.8,
            linewidth=4,
            color='black')
    last_val = gurobi_global[dwave_data.keys()[-1]]
    m = len(global_data)
    hline = plt.hlines(
        y=last_val,
        xmin=myticks[m - 1] - 0.5,
        xmax=myticks[m - 1] + 0.8,
        linewidth=4,
        color='black')

    # box plots
    bp_gurobi = plt.boxplot(
        gurobi_box_mod,
        positions=np.array(xrange(len(dwave_box_mod))) * 2.0 - 0.2,
        sym='+',
        widths=0.2)
    bp_qaoa = plt.boxplot(
        qaoa_box_mod,
        positions=np.array(xrange(len(dwave_box_mod))) * 2.0 + 0.2,
        sym='+',
        widths=0.2)
    bp_dwave = plt.boxplot(
        dwave_box_mod,
        positions=np.array(xrange(len(dwave_box_mod))) * 2.0 + 0.6,
        sym='+',
        widths=0.2)

    #dwave_color = '#2C7BB6'
    #gurobi_color = '#D7191C'
    #qaoa_color = '#31a354'
    dwave_color = 'aquamarine'
    gurobi_color = 'blue'
    qaoa_color = 'orange'
    set_box_color(bp_gurobi,
                  gurobi_color)  # colors are from http://colorbrewer2.org/
    set_box_color(bp_qaoa, qaoa_color)
    set_box_color(bp_dwave, dwave_color)

    # draw temporary red and blue lines and use them to create a legend
    plt.plot([], c='black', linewidth=4, label='Global Solver')
    plt.plot([], c=gurobi_color, linewidth=4, label='Gurobi')
    plt.plot([], c=qaoa_color, linewidth=4, label='IBM: QAOA')
    plt.plot([], c=dwave_color, linewidth=4, label='D-Wave')
    plt.legend()

    # axes
    plt.xticks(xrange(0, len(ticks) * 2, 2), ticks)
    plt.xlim(-2, len(ticks) * 2)
    plt.ylim(0, 0.5)
    plt.xticks(fontsize=0, rotation=20)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    #plt.xlabel('Graph',fontsize=22)
    #plt.ylabel('Modularity', fontsize=22)
    #plt.savefig('modularity_box.png')
    plt.show()

    # repeat process for number of iterations
    plt.clf()
    dwave_box_it = [dwave_data[g]['iter'] for g in dwave_data]
    gurobi_box_it = [gurobi_data[g]['iter'] for g in dwave_data]
    qaoa_box_it = [qaoa_data[g]['iter'] for g in dwave_data]
    #ticks = [re.split('-|_',i)[-1] for i in dwave_data.keys()]
    bp_gurobi = plt.boxplot(
        gurobi_box_it,
        positions=np.array(xrange(len(dwave_box_mod))) * 2.0 - 0.2,
        sym='+',
        widths=0.2)
    bp_qaoa = plt.boxplot(
        qaoa_box_it,
        positions=np.array(xrange(len(dwave_box_mod))) * 2.0 + 0.2,
        sym='+',
        widths=0.2)
    bp_dwave = plt.boxplot(
        dwave_box_it,
        positions=np.array(xrange(len(dwave_box_mod))) * 2.0 + 0.6,
        sym='+',
        widths=0.2)
    set_box_color(bp_gurobi, gurobi_color)
    set_box_color(bp_qaoa, qaoa_color)
    set_box_color(bp_dwave, dwave_color)

    # draw temporary red and blue lines and use them to create a legend
    plt.plot([], c=gurobi_color, linewidth=4, label='Gurobi')
    plt.plot([], c=qaoa_color, linewidth=4, label='IBM: QAOA')
    plt.plot([], c=dwave_color, linewidth=4, label='D-Wave')
    #plt.legend()
    plt.xticks(xrange(0, len(ticks) * 2, 2), ticks)
    plt.xlim(-2, len(ticks) * 2)
    plt.ylim(5, 40)
    plt.xticks(fontsize=18, rotation=20)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    #plt.xlabel('Graph',fontsize=22)
    #plt.ylabel('Solver Calls', fontsize=22)
    #plt.savefig('solver_box.png')
    plt.show()


def one_figure():
    #dwavefile = 'exp_dwave.txt'
    # results from gurobi global, version 6.5.0
    gurobi_global = {
        'sociopatterns-infectious': 0.4633343671,
        'moreno_oz_oz': 0.3385620174,
        'contact': 0.1049169921,
        'brunson_revolution_revolution': 0.4310351647,
        'arenas-jazz': 0.3206093636,
        'arenas-meta': 0.31373627450980396
    }
    dwavefile = 'expr_dwave16.txt'
    gurobifile = 'expr_gurobi16.txt'
    qaoafile = 'ibm.csv'
    graphInfoFile = 'graph_info.txt'
    graphInfo = {}
    myfile = open(graphInfoFile, 'r')
    for line in myfile:
        graph, n, m = line.split()
        graphInfo[graph] = {'nodes': int(n), 'edges': int(m)}
    myfile.close()
    dwave_data = {}
    gurobi_data = {}
    qaoa_data = {}
    for i in graphInfo:
        if not i.endswith('email'):
            dwave_data[i] = {'mod': [], 'iter': []}
            gurobi_data[i] = {'mod': [], 'iter': []}
            qaoa_data[i] = {'mod': [], 'iter': []}
    myfile = open(dwavefile, 'r')
    for line in myfile:
        line_split = line.split()
        dwave_data[line_split[0]]['mod'].append(float(line_split[-2]))
        dwave_data[line_split[0]]['iter'].append(int(line_split[-1]))
    myfile.close()
    myfile = open(gurobifile, 'r')
    for line in myfile:
        line_split = line.split()
        gurobi_data[line_split[0]]['mod'].append(float(line_split[-2]))
        gurobi_data[line_split[0]]['iter'].append(int(line_split[-1]))
    myfile.close()
    myfile = open(qaoafile, 'r')
    for line in myfile:
        ls = line.split('_seed_')
        graphname = ls[0].split('.')[-1]
        ls2 = ls[1].split(',')
        mod = 0.25 * float(ls2[-3]) / graphInfo[graphname]['edges']
        it = int(ls2[-2])
        qaoa_data[graphname]['mod'].append(mod)
        qaoa_data[graphname]['iter'].append(it)
    myfile.close()

    # modularity data in list of lists
    dwave_box_mod = [dwave_data[g]['mod'] for g in dwave_data]
    gurobi_box_mod = [gurobi_data[g]['mod'] for g in dwave_data]
    qaoa_box_mod = [qaoa_data[g]['mod'] for g in dwave_data]
    ticks = [re.split('-|_', i)[-1] for i in dwave_data.keys()]
    global_data = [gurobi_global[i] for i in dwave_data]

    myticks = list(xrange(0, len(ticks) * 2, 2))
    fig = plt.figure()
    ax1 = fig.add_axes([0.1, 0.5, 0.9, 0.4], ylim=(0.04, 0.5))
    ax2 = fig.add_axes([0.1, 0.06, 0.9, 0.4], ylim=(5, 35))
    # horizontal lines
    for i, val in enumerate(global_data[:-1]):
        ax1.hlines(
            y=val,
            xmin=myticks[i] - 0.5,
            xmax=myticks[i] + 0.8,
            linewidth=4,
            color='black')
    last_val = gurobi_global[dwave_data.keys()[-1]]
    m = len(global_data)

    hline = ax1.hlines(
        y=last_val,
        xmin=myticks[m - 1] - 0.5,
        xmax=myticks[m - 1] + 0.8,
        linewidth=4,
        color='black')

    # box plots

    bp_gurobi = ax1.boxplot(
        gurobi_box_mod,
        positions=np.array(xrange(len(dwave_box_mod))) * 2.0 - 0.2,
        sym='+',
        widths=0.2)
    bp_qaoa = ax1.boxplot(
        qaoa_box_mod,
        positions=np.array(xrange(len(dwave_box_mod))) * 2.0 + 0.2,
        sym='+',
        widths=0.2)
    bp_dwave = ax1.boxplot(
        dwave_box_mod,
        positions=np.array(xrange(len(dwave_box_mod))) * 2.0 + 0.6,
        sym='+',
        widths=0.2)

    #dwave_color = '#2C7BB6'
    #gurobi_color = '#D7191C'
    #qaoa_color = '#31a354'
    dwave_color = 'aquamarine'
    gurobi_color = 'blue'
    qaoa_color = 'orange'
    set_box_color(bp_gurobi,
                  gurobi_color)  # colors are from http://colorbrewer2.org/
    set_box_color(bp_qaoa, qaoa_color)
    set_box_color(bp_dwave, dwave_color)

    # draw temporary red and blue lines and use them to create a legend
    ax1.plot([], c='black', linewidth=4, label='Global Solver')
    ax1.plot([], c=gurobi_color, linewidth=4, label='Gurobi')
    ax1.plot([], c=qaoa_color, linewidth=4, label='IBM: QAOA')
    ax1.plot([], c=dwave_color, linewidth=4, label='D-Wave')
    ax1.legend()

    # axes
    #ax1.xticks(xrange(0, len(ticks) * 2, 2), ticks)
    #ax1.xlim(-2, len(ticks)*2)
    #ax1.ylim(0, 0.5)
    #ax1.xticks(fontsize=0,  rotation=20)
    #plt.yticks(fontsize=18)
    #ax1.tight_layout()
    #plt.xlabel('Graph',fontsize=22)
    #plt.ylabel('Modularity', fontsize=22)
    #plt.savefig('modularity_box.png')
    #plt.show()

    # repeat process for number of iterations
    #plt.clf()
    dwave_box_it = [dwave_data[g]['iter'] for g in dwave_data]
    gurobi_box_it = [gurobi_data[g]['iter'] for g in dwave_data]
    qaoa_box_it = [qaoa_data[g]['iter'] for g in dwave_data]
    #ticks = [re.split('-|_',i)[-1] for i in dwave_data.keys()]
    bp_gurobi = ax2.boxplot(
        gurobi_box_it,
        positions=np.array(xrange(len(dwave_box_mod))) * 2.0 - 0.2,
        sym='+',
        widths=0.2)
    bp_qaoa = ax2.boxplot(
        qaoa_box_it,
        positions=np.array(xrange(len(dwave_box_mod))) * 2.0 + 0.2,
        sym='+',
        widths=0.2)
    bp_dwave = ax2.boxplot(
        dwave_box_it,
        positions=np.array(xrange(len(dwave_box_mod))) * 2.0 + 0.6,
        sym='+',
        widths=0.2)
    set_box_color(bp_gurobi, gurobi_color)
    set_box_color(bp_qaoa, qaoa_color)
    set_box_color(bp_dwave, dwave_color)

    # draw temporary red and blue lines and use them to create a legend
    ax2.plot([], c=gurobi_color, linewidth=4, label='Gurobi')
    ax2.plot([], c=qaoa_color, linewidth=4, label='IBM: QAOA')
    ax2.plot([], c=dwave_color, linewidth=4, label='D-Wave')
    #plt.legend()
    plt.xticks(xrange(0, len(ticks) * 2, 2), ticks)
    #plt.xlim(-1, len(ticks)*2)
    #ax2.ylim(5, 40)
    plt.xticks(fontsize=18, rotation=5)
    #plt.yticks(fontsize=18)
    #plt.tight_layout()
    #plt.xlabel('Network Name',fontsize=22)
    #plt.ylabel('#Solver', fontsize=18)
    #plt.savefig('solver_box.png')
    ax1.set_xticklabels(())
    ax1.set_xlim(-1, len(ticks) * 2)
    ax2.set_xlim(-1, len(ticks) * 2)
    ax2.set_ylim(5, 40)
    #ax1.set_ylabel('Modularity',fontsize=16)
    #ax2.set_ylabel('Number Solver Calls', fontsize=16)
    #ax2.set_xticks(xrange(0, len(ticks) * 2, 2), ticks)
    ax1.tick_params(labelsize=18)
    ax2.tick_params(labelsize=18)
    plt.show()


if __name__ == '__main__':
    #two_figures()
    one_figure()
