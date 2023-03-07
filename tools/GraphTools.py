import networkx as nx


def get_dag_by_predit_prob(g, score):
    loops = 0
    while not nx.is_directed_acyclic_graph(g):
        cycle = nx.find_cycle(g)
        cycle_prob = [score[c[0], c[1]] for c in cycle]
        cycle = list(zip(cycle, cycle_prob))
        cycle.sort(key=lambda a: a[1])
        if len(cycle) > 2:
            cycle = cycle[:1]
        elif abs(cycle_prob[0] - cycle_prob[1]) < 0.01:
            cycle = cycle[:1]
        for c in cycle:
            g.remove_edge(*c[0])
        loops += 1
    print('Removing loops ', loops, 'times')
    return g
