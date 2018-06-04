"""
    This is for extracting all non-overlapping closed polygons in our annotations.
    Note that every extracted polygon never contains other polygons.
"""


import numpy as np
import os
import os.path as osp
import copy

SOURCE_DIR = './annotations'
EPSILON = 1e-4


def DFS(start_v, current_v, graph, visited, stack, num_steps, isback, cycle_list):
    '''
        finding all cycles in the graph which start at start_v
    '''
    visited[current_v] = True
    stack.append(current_v)
    next_v = _next_adj_v(graph, current_v=current_v, next_v=0)
    num_steps += 1
    # print('current v is {}, next v is {}, steps{}'.format(current_v, next_v, num_steps))
    # print(visited)
    # import IPython; IPython.embed()
    while True:
        if next_v != -1:
            if visited[next_v] is True and next_v == start_v and num_steps == 2:
                # ignore cycle with only 2 vertices
                next_v = _next_adj_v(graph, current_v=current_v, next_v=next_v)
                continue
            elif visited[next_v] is True and next_v == start_v and num_steps > 2:
                # find a cycle containing at least 3 vertices
                cycle = _get_cycle(stack)
                # print('find cycle {}'.format(cycle))
                cycle_list.append(cycle)
                next_v = _next_adj_v(graph, current_v=current_v, next_v=next_v)
                continue
            elif visited[next_v] is False:
                isback, num_steps = DFS(start_v, current_v=next_v, graph=graph, visited=visited, stack=stack,
                                        num_steps=num_steps,
                                        isback=isback,
                                        cycle_list=cycle_list)

            if isback:  # the last search finds no way to go from next_v
                num_steps -= 1
                temp_v = next_v
                # must by-pass all previously visited v
                next_v = _next_adj_v(graph, current_v=current_v, next_v=next_v)
                visited[temp_v] = False
                stack.pop()
                isback = False
                # print(visited)
                # print(stack)
                continue

            next_v = _next_adj_v(graph, current_v=current_v, next_v=next_v)
            # print('4. next v is {}, steps {}'.format(next_v, num_steps))

        else:  # next_v is -1, no more path at current_v, need to backtrack
            isback = True
            return isback, num_steps


def get_graph_cycles(graph):
    # print('the graph is {}'.format(graph))
    converted_graph, v_mapping = convert_graph(graph)
    # print('the converted graph is {}'.format(converted_graph))
    # print('v mapping is {}'.format(v_mapping))

    num_v = len(graph.keys())
    cycles = list()

    for start_v in converted_graph.keys():
        # print('processing loops starting at {}'.format(start_v))
        visited = [False] * (num_v + 1)
        stack = list()
        DFS(start_v, current_v=start_v, graph=converted_graph, visited=visited, stack=stack, num_steps=0, isback=False,
            cycle_list=cycles)

    cycles = convert_cycles(cycles, v_mapping)

    # print(len(cycles))
    cycles = refine_cycles(cycles)
    # print(len(cycles))
    return cycles


def convert_graph(graph):
    num_v = len(graph.keys())
    v_mapping = dict(zip(range(1, num_v + 1), graph.keys()))
    v_mapping_r = dict(zip(graph.keys(), range(1, num_v + 1)))

    converted_graph = dict()
    for v in graph.keys():
        for s in graph[v]:
            converted_graph.setdefault(v_mapping_r[v], []).append(v_mapping_r[s])

    return converted_graph, v_mapping


def convert_cycles(cycles, v_mapping):
    cycle_sets = list()
    non_rep_cycles = list()
    for cycle in cycles:
        cycle_set = set(sorted(cycle))
        add_flag = True
        for other in cycle_sets:
            if other == cycle_set:
                add_flag = False
                break
            else:
                pass
        if add_flag:
            cycle_sets.append(cycle_set)
            non_rep_cycles.append(cycle)

    # print(non_rep_cycles)
    converted_cycles = list()
    for cycle in non_rep_cycles:
        converted_cycle = list()
        for v in cycle:
            converted_cycle.append(v_mapping[v])
        converted_cycles.append(converted_cycle)

    return converted_cycles


def refine_cycles(cycles):
    delete_indices = list()
    print('refine cycles', len(cycles))

    areas = [_calc_area(cycle) for cycle in cycles]

    for i, cycle in enumerate(cycles):
        for j, other in enumerate(cycles):
            if j in delete_indices:
                continue
            if j == i:
                continue
            if _is_inside(cycle, other, i, j, areas):
                delete_indices.append(j)

    refined_cycles = list()
    # print('delete indices ', delete_indices)

    for i, cycle in enumerate(cycles):
        if i not in delete_indices:
            refined_cycles.append(cycle)

    return refined_cycles


def _calc_area(cycle):
    area = cycle[0][1] * (cycle[-1][0] - cycle[1][0])
    for i in range(1, len(cycle)):
        area += cycle[i][1] * (cycle[i-1][0] - cycle[(i+1) % len(cycle)][0])
    return abs(area / 2)


def _is_inside(cycle, other, i, j, areas):
    min_x_1, max_x_1, min_y_1, max_y_1 = _get_bound(cycle)
    min_x_2, max_x_2, min_y_2, max_y_2 = _get_bound(other)

    if abs(min_x_1 - min_x_2) <= EPSILON and abs(max_x_1 - max_x_2) <= EPSILON and abs(min_y_1 - min_y_2) <= EPSILON \
            and abs(max_y_1 - max_y_2) <= EPSILON:
        if _calc_area(cycle) < _calc_area(other):
            return True
        else:
            return False

    elif min_x_1 >= min_x_2 and max_x_1 <= max_x_2 and min_y_1 >= min_y_2 and max_y_1 <= max_y_2:
        # print('0')
        # if max_x_2 - max_x_1 <= EPSILON:
            # print('1')
            # cyc_list = _get_ys(cycle, max_x_1)
            # other_list = _get_ys(other, max_x_2)
            # for i in cyc_list:
            #     if i not in other_list:
            #         return False
            # return True
            # if _check_in(areas[i] + areas[j], areas):
            #     return False
            # else:
            #     return True
        # elif min_x_1 - min_x_2 <= EPSILON:
            # print('2')
            # cyc_list = _get_ys(cycle, min_x_1)
            # other_list = _get_ys(cycle, min_x_2)
            # for i in cyc_list:
            #     if i not in other_list:
            #         return False
            # return True
            # if _check_in(areas[i] + areas[j], areas):
            #     return False
            # else:
            #     return True
        # elif min_y_1 - min_y_2 <= EPSILON:
            # print('3')
            # cyc_list = _get_xs(cycle, min_y_1)
            # other_list = _get_xs(other, min_y_2)
            # for i in cyc_list:
            #     if i not in other_list:
            #         return False
            # return True
            # if _check_in(areas[i] + areas[j], areas):
            #     return False
            # else:
            #     return True
        # elif max_y_2 - max_y_1 <= EPSILON:
            # print('4')
            # cyc_list = _get_xs(cycle, max_y_1)
            # other_list = _get_xs(other, max_y_2)
            # for i in cyc_list:
            #     if i not in other_list:
            #         return False
            # return True
            # if _check_in(areas[i] + areas[j], areas):
            #     return False
            # else:
            #     return True
        if _totally_inside(cycle, other):
            # if _is_outside(cycle, other):

            return False
        else:
            if _check_in(areas[i] + areas[j], areas):
                return False
            else:
                return True
    else:
        return False


def _check_in(item, l):
    for i in l:
        if abs(item-i) <= EPSILON:
            return True
    return False


def _totally_inside(cycle, other):
    for pt1 in cycle:
        for pt2 in other:
            if abs(pt1[0] - pt2[0]) <= EPSILON and abs(pt1[1]-pt2[1]) <= EPSILON:
                return False
    return True


def _get_ys(cycle, x_value):
    ys = list()
    for pt in cycle:
        if abs(pt[0] - x_value) <= 0.5:
            ys.append(pt[1])
    return ys


def _get_xs(cycle, y_value):
    xs = list()
    for pt in cycle:
        if (pt[1] - y_value) <= 0.5:
            xs.append(pt[0])
    return xs


def _get_bound(cycle):
    sorted_x = sorted(cycle, key=lambda x: x[0])
    sorted_y = sorted(cycle, key=lambda x: x[1])
    return sorted_x[0][0], sorted_x[-1][0], sorted_y[0][1], sorted_y[-1][1]


def _get_cycle(stack):
    # need to make a copy of stack, otherwise the saved cycle will change dynamically with
    # stack
    return copy.deepcopy(stack)


def _next_adj_v(graph, current_v, next_v):
    if next_v == 0:
        if len(graph[current_v]) > 0:
            return graph[current_v][0]
    else:
        idx = graph[current_v].index(next_v)
        if idx + 1 < len(graph[current_v]):
            return graph[current_v][idx + 1]
    return -1


def output_polygons(polys, base_dir='./'):
    poly_filepath = osp.join(base_dir, 'annotations-1.txt')
    type_filepath = osp.join(base_dir, 'annotations-2.txt')

    with open(poly_filepath, 'w') as f1, open(type_filepath, 'w') as f2:
        for idx, poly in enumerate(polys):
            line_poly = list()
            line_type = list()
            line_poly.append(str(idx))
            line_type.append(str(idx))
            line_type.append('Flat')
            line_poly += ['0', '1']
            for pt in poly:
                line_poly += [str(pt[0]), str(pt[1])]
            line_poly_str = ' '.join(line_poly) + '\n'
            line_type_str = ' '.join(line_type) + '\n'
            f1.write(line_poly_str)
            f2.write(line_type_str)


if __name__ == '__main__':
    all_cycles = list()

    for filename in sorted(os.listdir(SOURCE_DIR)):
        file_path = osp.join(SOURCE_DIR, filename)
        # if filename != 'annot- 1524966663.5.npy':
        #     continue
        print(file_path)
        annot = np.load(file_path)[()]
        graph = annot['graph']
        cycles = get_graph_cycles(graph)
        # there is no repetitive cycles in different samples, so just concat
        print(len(cycles))
        for cycle in cycles:
            print(len(cycle))
        all_cycles += cycles

    output_polygons(all_cycles)

