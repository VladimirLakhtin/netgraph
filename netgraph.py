import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
import io
from PIL import Image

def find_events(df1):

    """
    Finds start and finish events for each works  

    Arguments:
    df1, pandas DataFrame - table with data

    Returns:
    start - dictionary where keys are works and values are start events
    finish - dictionary where keys are works and values are finish events
    duration - dictionary where keys are works and values are duration of these works
    """

    start = {}
    finish = {}
    duration = {}
    parents_finish = 0
    num_list = [i for i in range(2, 100)] # generate numerical sequence to 

    df = df1.copy()

    #iterate by works 
    for ind, row in df.iterrows():
        duration[row[0]] = row[1]
        parents = row[2].split(', ')

        # process works without parents
        if row[2] == '-' or pd.isna(row[0]):
            start[row[0]] = 1
            continue
        
        parents_start = []
        last_fictional_event = 0

        # process fictional works: iterate by work's parents
        for i, parent in enumerate(parents):      

            #check the same parents of the parents of the work 
            if start[parent] in parents_start:
                
                # if fictional work alredy exists create a new one
                if parent + "'" in start:
                    fictional_work = parent + "''"
 
                else:
                    # if fictional event alredy exists create a new one
                    if last_fictional_event != 0:
                        last_fictional_event += 0.1
                        finish[parent] = np.round(last_fictional_event, 1)
                    else:
                        finish[parent] = start[parent] + 0.1

                    fictional_work = parent + "'"

                # define start, finish events and duration of fictional work
                start[fictional_work] = finish[parent]
                parents[i] = fictional_work
                duration[fictional_work] = 0
                last_fictional_event = finish[parent]
            
            parents_start.append(start[parent])

        for i in parents:
            
            # define a main finish for parents of this work if one of them already has it
            if i in finish:
                parents_finish = finish[i]

            # check that the parents of this work don't have a finish event yet
            elif parents_finish != 0:
                finish[i] = parents_finish
            
            # if parents has not finish event get new event from our list 
            else:
                finish[i] = num_list.pop(0)
                parents_finish = finish[i]

        parents_finish = 0 

        # start event of if work is finish event of one of parents
        start[row[0]] = finish[row[2].split(', ')[0]]

    main_finish = num_list.pop(0) # find last event
    parents_start = []

    # define a finish event for works which don't have it
    for ind, row in df.iterrows():

        if row[0] not in finish:

            # procces fictional works for last works
            if start[row[0]] not in parents_start: 
                parents_start.append(start[row[0]])
                finish[row[0]] = main_finish
                continue
            
            else:
                finish[row[0]] = start[row[0]]  + 0.1
                fictional_work = row[0] + "'"
                start[fictional_work] = finish[row[0]]
                finish[fictional_work] = main_finish
                duration[fictional_work] = 0

    start['finish'] = main_finish

    return start, finish, duration

def family_relationship(start, finish):

    """
    Finds a relationship of each event with it parent event and child event.
    Finds incoming works and outgoing works for each event.

    Arguments:
    start, dict - dictionary where keys are works and values are start events
    finish, dict - dictionary where keys are works and values are finish events

    Returns:
    papa4son - dictionary where keys are event and values are child event
    son4papa - dictionary where keys are event and values are parent event
    exits - dictionary where keys are events and values are outgoings works
    inputs - dictionary where keys are events and values are incoming works
    """

    papa4son, son4papa = {}, {}
    exits, inputs = {}, {}

    # find childs for events
    for i in start:

        if i != 'finish':

            if start[i] not in papa4son:
                papa4son[start[i]] = [finish[i]]
                exits[start[i]] = [i]

            else:
                papa4son[start[i]].append(finish[i])
                exits[start[i]].append(i)

    
    papa4son = {i: sorted(papa4son[i]) for i in papa4son}
            
    # find parents for events
    for i in finish:
        if finish[i] not in son4papa:
            son4papa[finish[i]] = [start[i]] 
            inputs[finish[i]] = [i]
        else:
            son4papa[finish[i]].append(start[i]) 
            inputs[finish[i]].append(i)

    son4papa = {i: sorted(son4papa[i]) for i in son4papa}
    
    return papa4son, son4papa, exits, inputs

def define_pos(start, papa4son, son4papa, len_factor=1, family_factor=1):
    
    """
    Find positions for events on 2D-graph.

    Arguments:
    start, dict - dictionary where keys are works and values are start events
    papa4son, dict - dictionary where keys are event and values are child event
    son4papa, dict - dictionary where keys are event and values are parent event
    len_factor, int - the degree of influence of the factor of the distance of events from each other horizontally on their vertical position
    family_factor, int - the degree of influence of the factor of the number of parents of the event on its vertical position

    Returns:
    pos, dict - dictionary where keys are events and values are tuples with vertical and horizontal positions
    """

    # setting positions with zeros
    pos_x = {i: 0 for i in son4papa}
    pos_y = {i: 0 for i in son4papa}

    # setting position for first event 
    pos_x[1], pos_y[1] = 1, 0

    # sort events
    events = sorted(set([start[i] for i in start]))

    # define horizontal positions 
    for event in events:

        if event == np.max(events):
            continue  

        # event's horizontal positions is a last horizontal position of it's parents plus 1
        for son in papa4son[event]:
            pos_x[son] = pos_x[event] + 1

    # define vertical positions 
    # each parent of the event contributes to the determination of its vertical position depending on 3 factors: 
    # the distance by x, the number of parents of the event and the number of brothers of the event
    for event in events:

        if event == np.max(events):
            continue            
        
        # one child parent wants to bring the position of the event closer to his own 
        if len(papa4son[event]) == 1:
            son = papa4son[event][0]
            pos_y[son] += (pos_y[event] - pos_y[son]) * (1 / len(son4papa[son]))

        # one child parent wants to arrange his children around him
        else: 
            num_sons = len(papa4son[event])
            val = (num_sons / 2) - 0.5

            # define vertical position for each child
            for son in papa4son[event]:
                
                pos_y[son] += ((pos_y[event] + val) - pos_y[son]) * (pos_x[son] - pos_x[event])**len_factor * (1 / len(son4papa[son]))**family_factor
                val -= 1

    pos = {i: [pos_x[i], pos_y[i]] for i in pos_x}
    return pos

def get_critical_path(start, finish, duration, papa4son, son4papa, exits, inputs):
    
    """
    Find critical path in a four - sector way (https://www.pmexamsmartnotes.com/how-to-calculate-critical-path-float-and-early-and-late-starts-and-finishes/3/).

    Arguments:
    start, dict - dictionary where keys are works and values are start events
    finish, dict - dictionary where keys are works and values are finish events
    duration, dict - dictionary where keys are works and values are duration of these works
    papa4son, dict - dictionary where keys are event and values are child event
    son4papa, dict - dictionary where keys are event and values are parent event
    exits - dictionary where keys are events and values are outgoings works
    inputs - dictionary where keys are events and values are incoming works

    Returns:
    critical_edgelist - list with start and finish events of critical works
    critical_nodelist - list with numbers of critical events
    """
    
    Te, Tl, R = {}, {}, {}
    critical_nodelist = [1]
    critical_edgelist = []
    Te[1] = 0

    # calculate the earlier start
    for event in sorted(son4papa):
        Te[event] = np.max([duration[inp] + Te[start[inp]] for inp in inputs[event]])
    
    Tl[event] = Te[event]
    R[event] = 0

    # calculate the latest start and diff
    for event in sorted(papa4son, reverse=True):
        Tl[event] = np.min([Tl[finish[inp]] - duration[inp] for inp in exits[event]])
        R[event] = Tl[event] - Te[event]
    
    # find critical path
    while event != max(R):
        for son in papa4son[event]:
            if R[son] == 0:
                critical_nodelist.append(son)
                critical_edgelist.append((event, son))
                event = son
                break
        
    return critical_edgelist, critical_nodelist

def show_netgraph(start, finish, pos, duration, critical_edgelist, critical_nodelist):

    """
    Show the netgraph. Calculate duration of critical way.

    Arguments:
    start, dict - dictionary where keys are works and values are start events
    finish, dict - dictionary where keys are works and values are finish events
    pos, dict - dictionary where keys are events and values are tuples with vertical and horizontal positions
    duration, dict - dictionary where keys are works and values are duration of these works
    critical_edgelist - list with start and finish events of critical works
    critical_nodelist - list with numbers of critical events

    Returns:
    fig, matplotlib figure - final netgraph
    """
    
    plt.rcParams["figure.figsize"] = (50,20) # set graph size
    edgelist, fictional_edgelist, fictional_critical_edgelist = [], [], []
    nodes_list = sorted(set([start[i] for i in start]))
    scope_x = nodes_list[-1] - nodes_list[0]

    # fill edgelist
    for i in start:

        if i != 'finish':
            edge = (start[i], finish[i])

            if i[-1] == "'":
                fictional_edgelist.append(edge) # detect fictional edge

                if edge in critical_edgelist:
                    fictional_critical_edgelist.append(edge) # detect fictional critical edge
                    critical_edgelist.remove(edge) # remove fictional critical edge from all critical edge    

            else:
                edgelist.append(edge)

    G = nx.DiGraph(edgelist)
            
    options = {"edgecolors": "tab:gray", "node_size": 40000 / scope_x}
    nx.draw_networkx_nodes(G, pos, nodelist=nodes_list, node_color="black", **options) # show all nodes
    nx.draw_networkx_nodes(G, pos, nodelist=critical_nodelist, node_color="darkorchid", **options)# show critical nodes

    nx.draw_networkx_edges(G, pos, edgelist=edgelist, width=7, arrowstyle='-|>', arrowsize=100, edge_color="black") # show all edges
    nx.draw_networkx_edges(G, pos, edgelist=fictional_edgelist, width=7, arrowstyle='-|>', arrowsize=100, edge_color="black", style="dashed") # show fictional edges
    nx.draw_networkx_edges(G, pos, edgelist=critical_edgelist, width=10, arrowstyle='-|>', arrowsize=100, edge_color="darkorchid", alpha=1) # show all critical edges
    nx.draw_networkx_edges(G, pos, edgelist=fictional_critical_edgelist, width=7, arrowstyle='-|>', arrowsize=100, edge_color="darkorchid", style="dashed")# show fictional critical edges

    node_labels = {i: str(i) for i in nodes_list}
    edge_labels = {(start[i], finish[i]): i + ' ' + str(duration[i]) for i in start if i != 'finish'}
    nx.draw_networkx_labels(G, pos, node_labels, font_size=30, font_color="whitesmoke", font_weight=700) # named events
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=22, font_color="black", alpha=1, font_weight=700) # named works

    len_critical_way = np.sum([float(re.findall(r'[0-9]+.[0-9]+|[0-9]', edge_labels[i])[0]) for i in critical_edgelist]) # calculate duration of critical way
    day_name = 'дней' if str(int(len_critical_way))[-1] in ['5','6','7','8','9','0'] or str(int(len_critical_way))[-2:] in ['11','12','13','14'] else ('день' if str(int(len_critical_way))[-1] == '1' else 'дня')

    # set margins for the axes so that nodes aren't clipped
    ax = plt.gca()
    ax.margins(0.1)
    plt.axis("off")
    plt.title(f'Длительность критического пути: {len_critical_way} {day_name}', fontsize=30)
    fig = plt.gcf()
    plt.show()

    return fig

def fig2img(fig):

    """
    Convert a Matplotlib figure to a PIL Image and return it
    """

    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)

    return img

def main(df, len_factor=1, family_factor=1):

    """
    Build netgraph of works from data.

    Arguments:
    df, pandas DataFrame - table with name, duration and parents of works  
    len_factor, int - the degree of influence of the factor of the distance of events from each other horizontally on their vertical position
    family_factor, int - the degree of influence of the factor of the number of parents of the event on its vertical position

    Return:
    fig, PIL Image - netgraph
    """

    df.columns = range(len(df.columns))
    start, finish, duration = find_events(df)
    papa4son, son4papa, exits, inputs = family_relationship(start, finish)
    pos = define_pos(start, papa4son, son4papa, len_factor, family_factor)
    critical_edgelist, critical_nodelist = get_critical_path(start, finish, duration, papa4son, son4papa, exits, inputs)
    fig = show_netgraph(start, finish, pos, duration, critical_edgelist, critical_nodelist)
    image = fig2img(fig)
    return image

