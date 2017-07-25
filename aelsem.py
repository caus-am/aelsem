# Collection of functions for dealing with linear structural equation models.

from __future__ import print_function

import math
import collections
import heapq
import networkx as nx
import itertools
import numpy as np
import pickle
import time
from pads.UnionFind import UnionFind

import sys

import matplotlib.pyplot as plt
##import pygraphviz

#from mpl_toolkits.mplot3d import Axes3D

#from scipy.optimize import fmin_bfgs

#import cProfile
#import re

def empty_model(d):
    mB = np.zeros((d,d), dtype=bool)
    mO = np.eye(d, dtype=bool)
    return (mB, mO)

def add_edge(model, v, w, edge):
    # Same format for edges as used by print_model and related functions:
    # -->, <--: directed edge
    # <->: bidirected edge
    # ==>, <==: bow
    # =-=: 2-cycle
    # ===: 2-cycle and bidirected edge
    if v == w:
        return model
    (mB, mO) = model
    error = False
    if edge == "<->":
        mO[v,w] = mO[w,v] = True
    else:
        if edge[1] == '=':
            mO[v,w] = mO[w,v] = True
        elif edge[1] != '-':
            error = True

        if edge[0] == '=' and edge[2] == '=':
            mB[v,w] = mB[w,v] = True
        else:
            if (edge[0] == '<') == (edge[2] == '>'):
                error = True

            if edge[0] == '<':
                mB[v,w] = True
            elif edge[0] != edge[1]:
                error = True

            if edge[2] == '>':
                mB[w,v] = True
            elif edge[2] != edge[1]:
                error = True
    if error:
        raise ValueError('Unknown edge type: ' + edge)
    return (mB, mO)


def num_nodes(model):
    (mB, mO) = get_first_model(model)
    return mB.shape[0]

def num_edges(model):
    (mB, mO) = get_first_model(model)
    d = mB.shape[0]
    dim = np.count_nonzero(mB) + (np.count_nonzero(mO) - d) / 2
    return dim

def num_directed_edges(mB):
    return np.count_nonzero(mB)

def num_bidirected_edges(mO):
    d = mO.shape[0]
    return (np.count_nonzero(mO) - d) / 2

def num_bows(model):
    (mB, mO) = get_first_model(model)
    return np.count_nonzero(np.logical_and(mB, mO))

def num_adjacencies(model):
    (mB, mO) = get_first_model(model)
    d = mB.shape[0]
    return (np.count_nonzero(np.logical_or(np.logical_or(mB, mB.T), mO))
            - d) / 2

def is_bow_free(model):
    (mB, mO) = get_first_model(model)
    return not np.any(np.logical_and(mB, mO))

def transitive_reflexive_closure(mB):
    d = mB.shape[0]
    trans_refl_closure = np.logical_or(np.eye(d, dtype=bool), mB)
    prev_num = np.count_nonzero(trans_refl_closure)
    # O(log n) calls to numpy matrix multiplication probably faster than
    # O(n^3) loop in Python
    while True:
        #print(trans_refl_closure)
        trans_refl_closure = np.linalg.matrix_power(trans_refl_closure, 2)
        num = np.count_nonzero(trans_refl_closure)
        if num == prev_num:
            break
        prev_num = num
    return trans_refl_closure

def is_maximal_ancestral(model):
    (mB_, mO_) = model
    mB = mB_.astype(bool)
    mO = mO_.astype(bool)
    d = mB.shape[0]
    if not is_bow_free(model) or has_cycles(mB):
        return False
    mB_trans = transitive_reflexive_closure(mB)
    for v in range(d):
        for w in range(v+1, d):
            if mO[v,w] and (mB_trans[v,w] or mB_trans[w,v]):
                # almost-directed cycle: graph is not ancestral
                #print("almost-directed cycle involving {0} <-> {1}"
                #      .format(v, w))
                return False
    skel = np.logical_or(mO, np.logical_or(mB, mB.T))
    for v in range(d):
        for w in range(v+1, d):
            if not skel[v,w]:
                ancestors = np.logical_or(mB_trans[v,:], mB_trans[w,:])
                mO_ancestors = mO.copy()
                mO_ancestors[np.logical_not(ancestors),:] = False
                mO_ancestors[:,np.logical_not(ancestors)] = False
                mO_ancestors_trans = transitive_reflexive_closure(mO_ancestors)
                start = mB[:,v]
                start[v] = True
                finish = mB[:,w]
                finish[w] = True
                if np.any(mO_ancestors_trans[start,:][:,finish]):
                    # primitive inducing path between nonadjacent nodes:
                    # graph is not *maximal* ancestral
                    #print("primitive inducing path between {0} and {1}"
                    #      .format(v, w))
                    return False
    #print("model is a MAG")
    return True

def marginalize_node(model, v):
    # (Modifies model in place)
    (mB, mO) = model
    d = mB.shape[0]
    parents = mB[v,:]
    children = mB[:,v]
    siblings = mO[:,v]
    # Add edges between nodes adjacent to v
    mB[np.outer(children, parents)] = True
    mO[np.outer(children, children)] = True
    mO[np.outer(siblings, children)] = True
    mO[np.outer(children, siblings)] = True
    # Remove edges incident to v
    mB[v,:] = False
    mB[:,v] = False
    mO[v,:] = False
    mO[:,v] = False
    mO[v,v] = True
    # For cyclic models, the above may result in the addition of self-loops.
    # Remove those. (TODO: is this the appropriate solution?)
    mB = np.logical_and(mB, np.logical_not(np.eye(d, dtype=bool)))

def marginalized_model(model, observed_mask):
    (mB, mO) = get_first_model(model)
    mB_marg = mB.copy()
    mO_marg = mO.copy()
    for v in range(len(observed_mask)):
        if not observed_mask[v]:
            marginalize_node((mB_marg, mO_marg), v)
    return mB_marg, mO_marg

def has_cycles(mB):
    # test for cycles of length 2 or more
    d = mB.shape[0]
    trans_edges = np.logical_xor(transitive_reflexive_closure(mB),
                                 np.eye(d, dtype=bool))
    return np.any(np.logical_and(trans_edges, trans_edges.T))

def hash_model((mB, mO)):
    # (Can be used for models, or other pairs of matrices)
    return mB.tostring() + mO.tostring()


def flatten_list(ll):
    # Input: list of lists of items; output: list of items
    # (Does not recurse: if items are themselves lists, they are left alone)
    return [item for sublist in ll for item in sublist]

def get_first_model(models):
    if type(models) is list:
        return get_first_model(models[0])
    return models

def iter_models(models):
    if type(models) is list:
        for sub in models:
            for result in iter_models(sub):
                yield result
    else:
        yield models

def count_models(models):
    if type(models) is list:
        return sum(map(count_models, models))
    return 1

def seek_model(models, query_model):
    for m, model_class in enumerate(models):
        for model in iter_models(model_class):
            if np.all(model[0] == query_model[0]) and np.all(model[1] == query_model[1]):
                print("Found: index =", m)
                #print_models(model_class)
                return m
    return None

def print_model((mB, mO), indent=0):
    d = mB.shape[0]
    for i in range(d):
        print(' '*4*indent, end='')
        for j in range(d):
            if i == j:
                print(" O ", end=' ')
            elif not mO[i,j]:
                if mB[i,j] and mB[j,i]:
                    print("=-=", end=' ')
                elif mB[i,j]:
                    print("<--", end=' ')
                elif mB[j,i]:
                    print("-->", end=' ')
                else:
                    print(" . ", end=' ')
            else:
                if mB[i,j] and mB[j,i]:
                    print("===", end=' ')
                elif mB[i,j]:
                    print("<==", end=' ')
                elif mB[j,i]:
                    print("==>", end=' ')
                else:
                    print("<->", end=' ')
        print()

def model_class_summary(models):
    # TODO To distinguish presence/absence of eg. v-structures, maybe have a
    # mark for "can form a collider, but only with proper subset of marks
    # that can form a collider here"?
    if not models:
        d = 1
    else:
        d = get_first_model(models)[0].shape[0]
    any_heads = np.zeros((d,d), dtype=bool) # some model has just heads here
    any_tails = np.zeros((d,d), dtype=bool) # some model has just a tail here
    any_both = np.zeros((d,d), dtype=bool) # some model has head and tail here
    any_edges = np.zeros((d,d), dtype=bool)
    all_edges = np.ones((d,d), dtype=bool)
    any_bows = np.zeros((d,d), dtype=bool)
    all_bows = np.ones((d,d), dtype=bool)
    all_multi = np.ones((d,d), dtype=bool)
    #any_colliders = np.zeros((d,d), dtype=bool)
    #any_pure_coll = np.zeros((d,d), dtype=bool)
    #any_part_coll = np.zeros((d,d), dtype=bool)
    any_coll = np.zeros((d,d), dtype=bool)
    all_part_coll = np.ones((d,d), dtype=bool)
    for (mB, mO) in iter_models(models):
        dir_edges = np.logical_or(mB, mB.T)
        edges = np.logical_or(mO, dir_edges)
        any_edges = np.logical_or(any_edges, edges)
        all_edges = np.logical_and(all_edges, edges)
        any_heads = np.logical_or(any_heads,
                                  np.logical_and(np.logical_or(mB.T, mO),
                                                 np.logical_not(mB)))
        any_tails = np.logical_or(any_tails,
                                  np.logical_and(np.logical_and(mB, np.logical_not(mO)), np.logical_not(mB.T)))
        both = np.logical_and(mB, np.logical_or(mO, mB.T))
        any_both = np.logical_or(any_both, both)
        bows = np.logical_and(dir_edges, mO)
        any_bows = np.logical_or(any_bows, bows)
        all_bows = np.logical_and(all_bows, bows)
        two_cycles = np.logical_and(mB, mB.T)
        all_multi = np.logical_and(all_multi, np.logical_or(bows, two_cycles))

        head_or_both = np.logical_or(mB.T, mO)
        indeg = np.sum(head_or_both, axis=0, keepdims=True) - 1
        coll = head_or_both * (indeg >= 2) # arrowheads forming colliders
        both_at_node = np.any(both, axis=0, keepdims=True)
        part_coll = coll * both_at_node
        any_coll = np.logical_or(any_coll, coll)
        all_part_coll = np.logical_and(all_part_coll, part_coll)
    return (any_heads, any_tails, any_both,
            any_edges, all_edges, any_bows, all_bows, all_multi,
            any_coll, all_part_coll)

def model_class_marks_from_summary((any_heads, any_tails, any_both,
                                    any_edges, all_edges,
                                    any_bows, all_bows, all_multi,
                                    any_coll, all_part_coll)):
    d = any_heads.shape[0]
    marks = np.empty((d,d), dtype='S3')
    #mark_legend0 = '   ---===+#]<<<+o]{{]+*]' # detailed marks
    #mark_legend2 = '   ---===+#[>>>+o[}}[+*['
    mark_legend0 = '   ---===+*]<<<+*]+*]+*]' # less detail, for use in paper
    mark_legend2 = '   ---===+*[>>>+*[+*[+*['
    for i in range(d):
        for j in range(d):
            m0 = ' '
            m2 = ' '
            if i == j:
                m1 = 'O'
            elif not any_edges[i,j]:
                m1 = '.'
            else:
                m0 = mark_legend0[12*any_heads[j,i]
                                  + 6*any_both[j,i] + 3*any_tails[j,i]
                                  + any_coll[j,i] + all_part_coll[j,i]]
                m2 = mark_legend2[12*any_heads[i,j]
                                  + 6*any_both[i,j] + 3*any_tails[i,j]
                                  + any_coll[i,j] + all_part_coll[i,j]]

                m1 = '-'
                if m0 == '=' and m2 == '=':
                    # equivalently: there is guaranteed to be a 2-cycle
                    if all_bows[i,j]:
                        m1 = '='
                    elif any_bows[i,j]:
                        m1 = 'x'
                elif all_multi[i,j]:
                    m1 = '='
                if not all_edges[i,j]:
                    if m1 == '=':
                        m1 = ':'
                    else:
                        m1 = '?'

            marks[i][j] = "{0}{1}{2}".format(m0, m1, m2)
    return marks

def combine_summaries((any_heads1, any_tails1, any_both1,
                       any_edges1, all_edges1, any_bows1, all_bows1, all_multi1,
                       any_coll1, all_part_coll1),
                      (any_heads2, any_tails2, any_both2,
                       any_edges2, all_edges2, any_bows2, all_bows2, all_multi2,
                       any_coll2, all_part_coll2)):
    any_heads3 = np.logical_or(any_heads1, any_heads2)
    any_tails3 = np.logical_or(any_tails1, any_tails2)
    any_both3 = np.logical_or(any_both1, any_both2)
    any_edges3 = np.logical_or(any_edges1, any_edges2)
    all_edges3 = np.logical_and(all_edges1, all_edges2)
    any_bows3 = np.logical_or(any_bows1, any_bows2)
    all_bows3 = np.logical_and(all_bows1, all_bows2)
    all_multi3 = np.logical_and(all_multi1, all_multi2)
    any_coll3 = np.logical_or(any_coll1, any_coll2)
    all_part_coll3 = np.logical_and(all_part_coll1, all_part_coll2)
    return (any_heads3, any_tails3, any_both3,
            any_edges3, all_edges3, any_bows3, all_bows3, all_multi3,
            any_coll3, all_part_coll3)

def is_descriptional_submodel_of(summary1, summary2):
    # Would the model class description of the second argument change
    # if the models from the first argument were added to it?
    summary3 = combine_summaries(summary1, summary2)
    marks2 = model_class_marks_from_summary(summary2)
    marks3 = model_class_marks_from_summary(summary3)
    return np.all(marks2 == marks3)

def model_class_marks(models):
    # Find the graph pattern corresponding to a set of path diagrams
    summary = model_class_summary(models)
    marks = model_class_marks_from_summary(summary)
    return marks

def print_model_class(models, indent=0):
    d = get_first_model(models)[0].shape[0]
    marks = model_class_marks(models)
    for i in range(d):
        print(' '*4*indent, end='')
        for j in range(d):
            print(marks[i,j], end=' ')
        print()

def print_models(models, nesting=float('inf'), show_subclasses=False,
                 check_isomorphisms=False, tex_file=None,
                 show_numbering=True, annotation=None,
                 xpos=None, ypos=None, node_name=None, label_dir=None,
                 indent=0, num_cols=None):
    # Smart printing function for nested lists of models.
    # - nesting: largest number of nested ['s allowed (required for tex output);
    # - show_subclasses: before each nontrivial list, show contents as class;
    # - check_isomorphisms: False, True, or a list of permutations to check
    #         (typically the automorphism group of models as a whole)
    # Internal parameters that should be set only in recursive calls:
    # - indent: indent level for plain output / column in which this call draws
    #         (0-base with show_subclasses; 1-base without)
    # - num_cols: number of table columns for tex output

    # TODO FIXME: if annotations are attached to internal nodes of printed tree,
    # function tries passing single characters to recursive calls

    # Note: for tex output, this function assumes that the indentation
    # (i.e. skipping columns) has already been arranged by the caller (which is
    # this function, calling itself recursively)
    print_begin_and_end_table = False
    if indent == 0 and tex_file and num_cols is None:
        print_begin_and_end_table = True
        #if num_cols is None:
        num_tikz_cols = nesting + show_subclasses
        if annotation is None:
            num_annot = 0
        else:
            tmp = annotation
            while type(tmp) is list:
                tmp = tmp[0]
            if isinstance(tmp, basestring):
                num_annot = 1
            else:
                # tuple of strings
                num_annot = len(tmp)
        num_cols = show_numbering + num_tikz_cols + num_annot
        column_spec = 'r'*show_numbering + 'c'*num_tikz_cols + 'l'*num_annot
        print(tex_table_begin(column_spec), file=tex_file)
        print("  \\hline", file=tex_file)

    is_leaf = False
    if not type(models) is list:
        print_model(models, indent=indent)
        is_leaf = True
        if tex_file:
            print(tikz_diagram(models, xpos=xpos, ypos=ypos,
                               node_name=node_name, label_dir=label_dir),
                  file=tex_file)
    else:
        #print("print_models recursive call")
        #print(check_isomorphisms)

        # Decide whether to collapse the nesting
        if nesting > 0 and len(models) == 1 and type(models[0]) is list:
            # If only one model (class) will end up being shown, collapse the
            # nesting and just show it now
            submodels = models
            i = 1
            while i < nesting:
                submodels = submodels[0]
                # we already checked that submodels is a list
                if len(submodels) > 1:
                    # found multiple models on a displayable level: keep
                    # the nesting
                    break
                if not type(submodels[0]) is list:
                    # found only one model: collapse the nesting if we don't
                    # like [[model]]
                    #nesting = 0
                    break
                i += 1
            else:
                # (executed when i == nesting, but not if break hit)
                # input contains more levels of nested lists than we can
                # display, and all displayable levels had only one element
                nesting = 0

        if nesting <= 0 or (show_subclasses and len(models) > 1):
            # print a summary of all models in this list as a single diagram
            print(' '*4*indent, end='{\n')
            print_model_class(models, indent + 1)
            print(' '*4*indent, end='}\n')
            if nesting > 0:
                print(' '*4*indent, end='=\n')
            else:
                is_leaf = True
        if tex_file:
            if nesting <= 0 or show_subclasses:
                if xpos is None:
                    marks = model_class_marks(models)
                    skel = (marks != np.asarray(' . ', dtype='S3'))
                    xpos, ypos, label_dir = node_placement(skel)
                print(tikz_diagram(models, xpos=xpos, ypos=ypos,
                                   node_name=node_name, label_dir=label_dir),
                      file=tex_file)
                if nesting > 0:
                    print("  &", file=tex_file)
            elif indent > 0:
                # For tex output without show_subclasses, print a bullet
                # to show the hierarchical structure (except for the root)
                print("\bullet &", file=tex_file)

        if nesting > 0:
            # recursively print all elements of the list
            if check_isomorphisms:
                if type(check_isomorphisms) is not list:
                    d = get_first_model(models)[0].shape[0]
                    check_isomorphisms = all_permutations(d)
                iso, aut = find_isomorphisms(models, check_isomorphisms)
            char = '[' # will be set to ',' by first item
            for i, inner in enumerate(models):
                if check_isomorphisms and iso[i][0] != i:
                    continue
                indices_to_visit = [i]
                if check_isomorphisms:
                    full_class = (len(iso[i]) * len(aut[i])
                                  == len(check_isomorphisms))
                    aut_apparent_inner = find_apparent_automorphisms(inner)
                    if not full_class:
                        indices_to_visit = iso[i]
                for ordinal, index in enumerate(indices_to_visit, 1):
                    # Preparations before printing each element to tex
                    if tex_file:
                        column = (indent + show_subclasses + show_numbering) # of model to be printed by recursive call; 0-based
                        # print horizontal lines between models
                        if char == ',':
                            if indent + show_subclasses > 0: # was indent > 0, but didn't work for print_model_clusters
                                # cline{col,col} [1-based] for partial line;
                                # displays only once at page breaks though
                                print("  \\tabucline{{{0},{1}}}"
                                      .format(column+1, num_cols),
                                      file=tex_file)
                            else:
                                print("  \\hline", file=tex_file)
                        # print numbering (indices) for outer list
                        if indent == 0 and show_numbering:
                            print("  {{[{0}]}} &".format(i), file=tex_file)
                            #print("  \\bullet &", file=tex_file)
                            column -= 1
                        # leave table cells empty if required
                        if char == ',' and column > 0:
                            print(" ", " &"*column,
                                  sep="", file=tex_file)
                    if check_isomorphisms:
                        print(' '*4*indent, end=char)
                        print(end=' ')
                    else:
                        print(' '*4*indent, char, sep='')
                    char = ','

                    print_just_summary = False
                    if check_isomorphisms:
                        # Print isomorphism label (e.g. "24x")
                        if full_class:
                            # All expected isomorphisms present: print just one
                            # representative and the size of this class
                            print(len(iso[i]), 'x', sep='')
                            if tex_file:
                                print("${0} \\times$".format(len(iso[i])),
                                      file=tex_file)
                        elif ordinal == 1:
                            # Some expected isomorphisms are missing: print them
                            # all, grouped together
                            num_expected = len(check_isomorphisms) / len(aut[i])
                            print("1/", len(iso[i]), " (expected ",
                                  num_expected, ")", sep='')
                            if tex_file:
                                # (Alternative for resizing: scalebox)
                                print("$1 / {0} {{\\scriptscriptstyle (/ {1})}}$".format(len(iso[i]),
                                                                 num_expected),
                                      file=tex_file)
                        else:
                            # subsequent members of incomplete set above
                            print(ordinal, "/", len(iso[i]), sep='')
                            if tex_file:
                                print("${0} / {1}$".format(ordinal,
                                                           len(iso[i])),
                                      file=tex_file)
                            if len(aut[i]) == len(aut_apparent_inner):
                                print_just_summary = True

                    if print_just_summary:
                        print_models(models[index], nesting=0,
                                     indent=indent + 1,
                                     tex_file=tex_file,
                                     show_numbering=show_numbering,
                                     annotation=(annotation[index]
                                                 if annotation else None),
                                     xpos=xpos, ypos=ypos,
                                     node_name=node_name, label_dir=label_dir,
                                     num_cols=num_cols)
                    else:
                        if not check_isomorphisms:
                            aut_apparent_here = False
                        elif index == i:
                            aut_apparent_here = aut_apparent_inner
                        else:
                            aut_apparent_here = find_apparent_automorphisms(models[index])
                        print_models(models[index], # WAS inner
                                     nesting=nesting - 1,
                                     show_subclasses=show_subclasses,
                                     indent=indent + 1,
                                     check_isomorphisms=aut_apparent_here,
                                     tex_file=tex_file,
                                     show_numbering=show_numbering,
                                     annotation=(annotation[index]
                                                 if annotation else None), # WAS [i]
                                     xpos=xpos, ypos=ypos,
                                     node_name=node_name, label_dir=label_dir,
                                     num_cols=num_cols)
            if char == '[':
                # We just tried printing an empty list
                print(' '*4*indent, end='[]\n')
                if tex_file:
                    print("  \\varnothing", file=tex_file) # or \emptyset
                    is_leaf = True
            else:
                print(' '*4*indent, end=']\n')

    if is_leaf and tex_file:
        column = (indent + show_subclasses + show_numbering) # of leaf just printed; 1-based
        # print annotations to tex_file
        while type(annotation) is list and len(annotation) == 1: # was if
            annotation = annotation[0]
        if isinstance(annotation, basestring): # Python 3: (., str)
            print(" ", " &"*(num_cols - column - 1),
                  sep="", file=tex_file)
            print('&', annotation, file=tex_file)
        elif annotation is not None:
            # tuple of strings
            print(" ", " &"*(num_cols - column - len(annotation)),
                  sep="", file=tex_file)
            for annot in annotation:
                print('  &\n', annot, file=tex_file)
        print("  \\\\", file=tex_file)

    if print_begin_and_end_table:
        print(tex_table_end(), file=tex_file)

def print_model_clusters(clusters, nesting=1,
                         check_isomorphisms=True, tex_file=None,
                         annotation=None):
    if check_isomorphisms:
        if type(check_isomorphisms) is not list:
            d = get_first_model(clusters)[0].shape[0]
            check_isomorphisms = all_permutations(d)
        iso, aut = find_isomorphisms(clusters, check_isomorphisms)

    if tex_file:
        # TODO determine actual number of columns (and use throughout function)
        print(tex_table_begin("cccc"), file=tex_file)

    #print_models(clusters)
    print_level = [2 for _ in clusters]
    for i, model_class_by_skeleton in enumerate(clusters):
        # Adjust print_level of isomorphic clusters
        if check_isomorphisms and iso[i][0] == i:
            print_level_others = 0
            if len(iso[i]) * len(aut[i]) < len(check_isomorphisms):
                # Not *all* possible isomorphic classes actually appear, so
                # the output must show which ones do exist
                print_level_others = 1
            for j in iso[i][1:]:
                print_level[j] = print_level_others

        # Header: cluster number, isomorphism information
        if not check_isomorphisms:
            #print("Cluster", i)
            header = "Cluster {0}".format(i)
        elif print_level[i] < 2:
            #print("Cluster", i, "is isomorphic to cluster", iso[i][0])
            header = ("Cluster {0} is isomorphic to cluster {1}"
                      .format(i, iso[i][0]))
        else:
            if len(iso[i]) > 1:
                #print("Cluster", i, "is first of", len(iso[i]),
                #      "isomorphic clusters:",
                #      ", ".join(map(str, iso[i])), end='')
                header = ("Cluster {0} is first of {1} isomorphic clusters: {2}"
                          .format(i, len(iso[i]), ", ".join(map(str, iso[i]))))
            else:
                #print("Cluster", i,
                #      "forms a singleton isomorphism class", end='')
                header = ("Cluster {0} forms a singleton isomorphism class"
                          .format(i))
            if len(iso[i]) * len(aut[i]) < len(check_isomorphisms):
                #print(" (expected ", len(check_isomorphisms) / len(aut[i]),
                #      " clusters)", sep='')
                header += (" (expected {0} clusters)"
                           .format(len(check_isomorphisms) / len(aut[i])))
            #else:
                #print()
        print(header)
        if tex_file:
            if print_level[i] == 2:
                print("  \\hline", file=tex_file)
                print("  \\multicolumn{{4}}{{l}}{{{0}}}\\\\*"
                      .format(header), file=tex_file)

        # Display the contents of the cluster by skeleton
        if print_level[i] >= 1:
            app_aut = find_apparent_automorphisms(model_class_by_skeleton)
            if tex_file and print_level[i] == 2:
                print_models(model_class_by_skeleton,
                             nesting=nesting, show_subclasses=True,
                             check_isomorphisms=app_aut, tex_file=tex_file,
                             show_numbering=False,
                             #annotation=annot,
                             annotation=(annotation[i] if annotation else None),
                             indent=0, num_cols=4)
            else:
                print_models(model_class_by_skeleton,
                             nesting=nesting, show_subclasses=True,
                             check_isomorphisms=app_aut)
                             #check_isomorphisms=aut[i])
        #print(aut[i])

    if tex_file:
        print(tex_table_end(), file=tex_file)


def tex_file_begin():
    return """\\documentclass[a4paper]{article}

\\usepackage[left=1.2cm,right=1.2cm,top=2cm,bottom=2.5cm]{geometry}
\\usepackage{longtable,tabu,amsmath,amssymb}
\\usepackage{tikz}
\\usetikzlibrary{arrows.meta,decorations.pathmorphing}
\\newcommand{\\sortkey}[1]{}

\\begin{document}
"""

def tex_file_end():
    return """\\end{document}"""

def tex_table_begin(column_spec):
    return """\\setlength{\\tabulinesep}{3pt}
\\begin{longtabu}{""" + column_spec + "}"

def tex_table_end():
    return """  \\hline
\\end{longtabu}
"""

def tikz_diagram(model_class, xpos=None, ypos=None,
                 node_name=None, label_dir=None):
    # Similar to print_model_class, but outputs tex code that creates
    # a tikzpicture visualizing the model class
    d = get_first_model(model_class)[0].shape[0]
    marks = model_class_marks(model_class)
    use_labels = True #(label_dir is not None)
    # node names and positions
    if node_name is None:
        node_name = list(map(chr, range(97, 97 + d)))
    # intended for models with one missing adjacency:
    #node_name = list(map(chr, range(97, 97 + d - 2))) + ['x', 'y']
    if xpos is None:
        skel = (marks != np.asarray(' . ', dtype='S3'))
        xpos, ypos, label_dir = node_placement(skel)
    elif label_dir is None:
        # This will probably look bad
        label_dir = d*[90]

    s = "  \\begin{tikzpicture}\n"
    # draw nodes
    for v in range(d):
        if use_labels:
            # Removing mathstrut will reduce distance between labels and nodes,
            # but may lead to labels not being aligned with each other in some
            # situations
            label_spec = ("[label={0}:$\\mathstrut {1}$] "
                          .format(math.floor(label_dir[v] + .5), node_name[v]))
        else:
            label_spec = ""
        s += "    \\node [circle,fill=black,inner sep=1pt] ({0}) at ({1},{2}) {3}{{}};\n".format(node_name[v], xpos[v], ypos[v], label_spec)
    # draw edges
    for v in range(d):
        for w in range(v+1, d):
            m0 = marks[v,w][0]
            m1 = marks[v,w][1]
            m2 = marks[v,w][2]
            if m1 == '.':
                continue

            # Determine colour and line style (snake; solid, dotted, dashed;
            # double)
            style = ''
            if m0 == '=' and m2 == '=':
                style += 'decorate,decoration={snake,amplitude=1,segment length=5,pre length=1ex,post length=1ex},' # lenghts were 5 (no unit)
            if m1 == '?' or m1 == ':':
                style += 'gray,dotted'
            elif (marks[v,w] == "-->" or marks[v,w] == "<--"
                  or marks[v,w] == "=-="):
                # one possibility, involving only directed arrows
                style += 'blue'
            elif marks[v,w] == "<->":
                # one possibility, namely a bidirected arrow
                style += 'red,dashed'
            elif (marks[v,w] == "==>" or marks[v,w] == "<=="
                  or marks[v,w] == '==='):
                # one possibility, involving directed and bidirected edges
                style += 'magenta!80!black'
            else:
                # multiple possibilities
                style += 'green!67!black'
            #if ((m1 == '=' or m1 == 'x' or m1 == ':')
            #    and (m0 == '=' or m2 == '=')):
                # part after 'and' added because otherwise all green arrows
                # in the clustering are double (apparently)
            if m1 == '=' or m1 == ':':
                # Let model_class_marks decide whether arrow should be double,
                # so that this function doesn't cause information to be lost;
                # '=x=' could also be double, but doesn't need to be.
                style += ',double'
            # double looks really bad with any kind of harpoons, so I have to
            # choose for one or the other

            # sep below leaves some distance between node and arrow tip.
            # Another good arrow tip would be Diamond[open].

            if m2 == '>':
                rtip = 'Stealth[sep,length=1ex]'
            elif m2 == '+':
                rtip = 'Bar[sep,width=1ex] __'
            elif m2 == 'o':
                rtip = 'Circle[open,sep,length=.9ex]'
            elif m2 == '[':
                rtip = 'Bracket[reversed,sep,length=.4ex,width=.8ex]'
            elif m2 == '}':
                rtip = 'Arc Barb[sep,length=.45ex]'
            elif m2 == '#':
                rtip = 'Square[open,sep,length=.8ex]'
            elif m2 == '*':
                rtip = 'Rays[n=6,sep,length=.9ex]'
            #elif m2 == '-' and m1 == '=':
            #    rtip = 'Rectangle[sep,length=1ex,width=.4pt]' # double>single
            #elif m2 == '=':
            #    rtip = 'Stealth[right,sep,length=1ex]' # harpoon
            #elif m0 == '=' and m2 == '=':
            #    # used iff line is wiggly
            #    rtip = 'Stealth[open,sep,length=1ex]'
            elif m2 == '=':
                #rtip = 'Stealth[open,reversed,sep,length=1ex]' # not perfect with double
                #rtip = 'Straight Barb[reversed,sep,length=.4ex,width=.8ex]' # ~ inset part of Stealth . For double lines, line should extend a bit more. Also, visually too similar to ] at a glance.
                rtip = 'Stealth[reversed,sep,length=.64ex,inset=.25ex,width=.8ex]' # mimic Straight Barb, but with pointy ends; length and inset chosen so that bisecting line corresponds to Straight Barb, and size of that angle similar to other Stealth arrows (and not too many decimals)
            elif m2 == '-': # or m2 == '=':
                # for '=': used iff line is double and not wiggly (check)
                rtip = '_'
            else:
                rtip = 'Square[sep,length=.8ex]' # unrecognized kind of arrow

            if m0 == '<':
                ltip = 'Stealth[sep,length=1ex]'
            elif m0 == '+':
                ltip = '__ Bar[sep,width=1ex]'
            elif m0 == 'o':
                ltip = 'Circle[open,sep,length=.9ex]'
            elif m0 == ']':
                ltip = 'Bracket[reversed,sep,length=.4ex,width=.8ex]'
            elif m0 == '{':
                ltip = 'Arc Barb[sep,length=.45ex]'
            elif m0 == '#':
                ltip = 'Square[open,sep,length=.8ex]'
            elif m0 == '*':
                ltip = 'Rays[n=6,sep,length=.9ex]'
            #elif m0 == '-' and m1 == '=':
            #    ltip = 'Rectangle[sep,length=1ex,width=.4pt]' # double>single
            #elif m0 == '=':
            #    ltip = 'Stealth[right,sep,length=1ex]' # harpoon
            #elif m0 == '=' and m2 == '=':
            #    ltip = 'Stealth[open,sep,length=1ex]'
            elif m0 == '=':
                ltip = 'Stealth[reversed,sep,length=.64ex,inset=.25ex,width=.8ex]'
            elif m0 == '-': # or m0 == '=':
                ltip = '_'
            else:
                ltip = 'Square[sep,length=.8ex]' # unrecognized kind of arrow

            s += '    \\draw [{0},arrows=\n'.format(style)
            s += '      {{{0}-{1}}}]\n'.format(ltip, rtip)
            s += '      ({0}) -- ({1});\n'.format(node_name[v], node_name[w])
    s += "  \\end{tikzpicture}"
    return s

class GraphShape:
    Default, Path, Cycle, Star, TriangleWithTail, OneMissingEdge = range(6)

def graph_shape(skel):
    # Recognizes the graph shapes for which special node placements are
    # implemented -- only if skel's node ordering corresponds to that placement.
    # Assumes skel is connected, and reflexive and symmetric like mO.
    ret = GraphShape.Default
    d = skel.shape[0]
    m = num_bidirected_edges(skel)
    if m == d - 1:
        is_path = True
        for i in range(d - 1):
            if not skel[i, i+1]:
                is_path = False
        if is_path:
            ret = GraphShape.Path
        elif d > 3 and all(skel[0, :]):
            ret = GraphShape.Star
    elif m == d:
        is_cycle = True
        for i in range(d - 1):
            if not skel[i, i+1]:
                is_cycle = False
        if is_cycle and skel[d-1, 0]:
            ret = GraphShape.Cycle
        elif d == 4 and not skel[0,3] and not skel[1,3]:
            ret = GraphShape.TriangleWithTail
    elif m == d * (d-1) / 2 - 1:
        if (d == 4 or d == 5) and not skel[d-2, d-1]:
            ret = GraphShape.OneMissingEdge
    return ret

def node_placement_component(skel):
    d = skel.shape[0]
    shape = graph_shape(skel)
    if shape == GraphShape.Path:
        xpos = list(range(d))
        ypos = d*[0]
        label_dir = d*[270]
    elif shape == GraphShape.Star:
        xpos = [0] + (d-1)*[1]
        #ypos = [0] + list(np.arange(d-1) - (d-2.0)/2)
        ypos = [0] + list(np.arange(d-1, 0, -1) - 0.5 * d)
        label_dir = [180] + (d-1)*[0]
    elif shape == GraphShape.TriangleWithTail:
        # vertically:
        #xpos = [0, 1, .5, .5]
        #ypos = [math.sqrt(3)/2, math.sqrt(3)/2, 0, -1]
        # horizontally:
        xpos = [-math.sqrt(3)/2, -math.sqrt(3)/2, 0, 1]
        ypos = [1, 0, .5, .5]
        label_dir = [120, 240, 285, 270]
    elif shape == GraphShape.OneMissingEdge and d <= 5:
        if d == 2: # (not used: is a Path)
            xpos = [0, 1]
            ypos = [0, 0]
            label_dir = [270, 270]
        elif d == 3: # (not used: prefer Path)
            xpos = [1, 0, 2]
            ypos = [0, 0, 0]
            label_dir = [270, 270, 270]
        elif d == 4:
            xpos = [0, 0, -math.sqrt(3)/2, math.sqrt(3)/2]
            ypos = [1, 0, .5, .5]
            label_dir = [90, 270, 180, 0]
        elif d == 5: # TODO rather arbitrary positions
            xpos = [1.5, 1, 2, 0, 3]
            ypos = [1.5, 0, 0, 1, 1]
            label_dir = [90, 270, 270, 180, 0]
    else: # Default & Cycle
        # general approach: arrange nodes in regular polygon with one side
        # at the bottom, numbered clockwise from the top
        #inds = range(int(d / 2), int(d / 2) + d)
        inds = np.arange(d) + int(d / 2)
        # angle = 0 is at the bottom; then goes clockwise
        angles = 2 * math.pi * (inds + .5) / float(d)
        #radius = d / math.pi / 2 # d / math.pi / 2 makes circumference = d
        radius = 1 / (2 * math.sin(math.pi / d)) # edges have length 1
        xpos = radius * -np.sin(angles)
        ypos = radius * -np.cos(angles)
        label_dir = 270 - 360 * angles / (2 * math.pi)
    return (np.asarray(xpos), np.asarray(ypos), label_dir)

def node_placement(skel, trans_closure=None):
    #print(skel)
    d = skel.shape[0]
    if trans_closure is None:
        trans_closure = transitive_reflexive_closure(skel)
    vis = np.zeros(d, dtype=bool)
    xpos = np.zeros(d)
    ypos = np.zeros(d)
    label_dir = np.zeros(d)
    space_between_comps = 1 # math.sqrt(2)/2
    xmax_prev = -space_between_comps
    for j in range(d):
        if not vis[j]:
            comp_mask = trans_closure[j,:]
            #print(comp_mask)
            vis = np.logical_or(vis, comp_mask)
            comp_skel = skel[comp_mask,:][:,comp_mask]
            comp_xpos, comp_ypos, comp_dir = node_placement_component(comp_skel)
            #print(comp_xpos)
            #print(comp_ypos)
            xmin = np.min(comp_xpos)
            comp_xpos = comp_xpos + (xmax_prev - xmin + space_between_comps)
            ymin = np.min(comp_ypos)
            comp_ypos -= ymin
            xmax_prev = np.max(comp_xpos)
            xpos[comp_mask] = comp_xpos
            ypos[comp_mask] = comp_ypos
            label_dir[comp_mask] = comp_dir
    #print("Overall:")
    #print(xpos)
    #print(ypos)
    return (xpos, ypos, label_dir)

def sort_models(models):
    keys = [[] for _ in models]
    for i, submodels in enumerate(models):
        # compute the elements that make up the sort key
        d = get_first_model(submodels)[0].shape[0]
        min_num_edges = d*d
        superskel = np.eye(d)
        # From a number of isomorphic model sets, one with the smallest
        # iso_penalty will get displayed
        iso_penalty = 0
        for model in iter_models(submodels):
            mB, mO = model
            min_num_edges = min(min_num_edges, num_edges(model))
            skel = np.logical_or(mO, np.logical_or(mB, mB.T))
            superskel = np.logical_or(superskel, skel)

        component_sizes = []
        trans_closure = transitive_reflexive_closure(superskel)
        vis = np.zeros(d, dtype=bool)
        for j in range(d):
            if not vis[j]:
                comp = trans_closure[j,:]
                component_sizes.append(np.sum(comp))
                vis = np.logical_or(vis, comp)
                if graph_shape(superskel[comp,:][:,comp]) == GraphShape.Default:
                    # prefer isomorphisms with implemented node placements
                    iso_penalty += 1
        if not all(component_sizes[i] >= component_sizes[i+1]
                   for i in range(len(component_sizes) - 1)):
            component_sizes.sort(reverse=True)
            iso_penalty += 1 # components should be largest-first

        degree_sequence = list(np.sum(superskel, axis=0))
        degree_sequence.sort(reverse=True)

        num_adj = (np.sum(superskel) - d) / 2
        if num_adj == min_num_edges:
            num_adj_penalty = 0
        else:
            # num_adj could be smaller than min_num_edges due to bows, or
            # larger due to optional edges
            num_adj_penalty = num_adj

        for j in range(d-1):
            if not trans_closure[j,j+1]:
                iso_penalty += 1 # components should be contiguous

        keys[i] = [component_sizes,
                   min_num_edges,        # not a function of superskel
                   num_adj_penalty,
                   degree_sequence,
                   iso_penalty,          # above elts are isomorphism-invariant
                   superskel.tostring()] # finally, keep same skels together
    models = [model for (key, model) in sorted(zip(keys, models),
                                               key=lambda pair: pair[0])]
    return models


def all_permutations(d):
    ret = []
    for pi in itertools.permutations(range(d)):
        P = np.zeros(shape=(d,d), dtype=bool)
        for i in range(d):
            P[i, pi[i]] = True
        ret.append(P)
    return ret

def permuted_models(models, P):
    # return an ordinary model or list of models, preserving structure and order
    # of input
    if not type(models) is list:
        mB, mO = models
        return P.dot(mB).dot(P.T), P.dot(mO).dot(P.T)
    else:
        return [permuted_models(submodel, P) for submodel in models]

def permuted_models_as_set(models, P):
    # returns a set: flattens any list hierarchy and removes order
    return {hash_model(permuted_models(submodel, P))
            for submodel in iter_models(models)}

def find_isomorphisms(models, permutations=None, return_mappings=False):
    # TODO: could also compute `weak isomorphisms': group two classes together
    # if they contain an isomorphic model. Would this be useful? (e.g. for
    # ordering them together in output)
    # Function input/output:
    # - assumes models is a list (possibly nested);
    # - assumes permutations forms a group (only used for computing
    #   model2's automorphisms?)
    # - output:
    #   - isomorphic_models: for each element of models, a list of indices
    #     of models isomorphic to it;
    #   - model_automorphisms: for each element of models, a list of
    #     permutations that are automorphisms of that model
    d = get_first_model(models)[0].shape[0]
    if permutations is None:
        permutations = all_permutations(d)
    isomorphic_models = [[] for _ in models]
    model_automorphisms = [[] for _ in models]
    if return_mappings:
        mappings = [[] for _ in models]
    model_sizes = np.zeros(len(models)) # number of distinct elements
    occurs_in = {} # maps hashed model to list of indices of models
    for i, member in enumerate(models):
        for model in iter_models(member):
            hashed = hash_model(model)
            if hashed in occurs_in:
                if occurs_in[hashed][-1] != i:
                    model_sizes[i] += 1
                    occurs_in[hashed].append(i)
            else:
                model_sizes[i] += 1
                occurs_in[hashed] = [i]
    for i1, member1 in enumerate(models): # was model1
        if isomorphic_models[i1]:
            continue
        #print("i1 =", i1)
        isomorphism_class = [i1]
        isomorphic_models[i1] = isomorphism_class # (use mutability of lists)
        if model_sizes[i1] == 0:
            # ignore empty models (don't mark them as isomorphic to each other)
            continue
        for P in permutations:
            # Possible further optimization: if nontrivial automorphisms have
            # been found, use them to ignore part of the remaining permutations
            equal_indices = None
            for model1 in iter_models(member1):
                permuted_model1 = permuted_models(model1, P)
                hashed = hash_model(permuted_model1)
                if hashed not in occurs_in:
                    equal_indices = []
                    break
                occ_inds = occurs_in[hashed]
                #print(occ_inds)
                if equal_indices is None:
                    # (performed during first iteration) Final value of
                    # equal_indices will be sublist of list computed here,
                    # so this is a good place to already get rid of models
                    # whose size doesn't match.
                    equal_indices = [ind for ind in occ_inds
                                     if model_sizes[ind] == model_sizes[i1]]
                elif occ_inds == equal_indices:
                    continue
                else:
                    # compute intersection by merge (assumes sortedness)
                    intersection = []
                    j1 = 0
                    j2 = 0
                    while j1 < len(equal_indices) and j2 < len(occ_inds):
                        if equal_indices[j1] == occ_inds[j2]:
                            intersection.append(equal_indices[j1])
                            j1 += 1
                            j2 += 1
                        elif equal_indices[j1] < occ_inds[j2]:
                            j1 += 1
                        else:
                            j2 += 1
                    equal_indices = intersection
                if equal_indices == []:
                    break
            #print("Equal_indices:", equal_indices)
            # equal_indices completely computed (typically just one element)
            for i2 in equal_indices:
                if i2 == i1:
                    # this permutation is an automorphism of member1
                    model_automorphisms[i1].append(P)
                    if return_mappings and mappings[i1] == []:
                        mappings[i1] = P
                elif not isomorphic_models[i2]:
                    # member1 and models[i2] are isomorphic
                    isomorphic_models[i2] = isomorphism_class
                    isomorphism_class.append(i2)
                    model_automorphisms[i2] = P # temporary: replaced at end
                    if return_mappings:
                        mappings[i2] = P
        isomorphism_class.sort()
        # compute all automorphisms of other models
        #print(i1, isomorphism_class)
        for i2 in isomorphism_class[1:]:
            P_iso = model_automorphisms[i2]
            model_automorphisms[i2] = [P_iso.dot(P_aut).dot(P_iso.T)
                                       for P_aut in model_automorphisms[i1]]
    if return_mappings:
        return isomorphic_models, model_automorphisms, mappings
    else:
        return isomorphic_models, model_automorphisms

def find_apparent_automorphisms(model_class, permutations=None):
    # Find the automorphisms of the output of model class marks. In general,
    # the automorphisms found this way are a superset of the automorphisms
    # of the two model classes when examined as sets of models.
    d = get_first_model(model_class)[0].shape[0]
    if permutations is None:
        permutations = all_permutations(d)
    summary = model_class_summary(model_class)
    marks = model_class_marks_from_summary(summary)
    m12 = np.zeros((d, d))
    #m2 = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            # use only 2nd and 3rd character: 1st matches 3rd of transpose
            m12[i,j] = ord(marks[i,j][1]) << 8 + ord(marks[i,j][2])
            #m2[i,j] = ord(marks[i,j][2])
    ret = []
    #print(m1)
    #print(m2)
    for P in permutations:
        permuted_m12 = P.dot(m12).dot(P.T)
        #permuted_m2 = P.dot(m2).dot(P.T)
        if np.all(permuted_m12 == m12): # and np.all(permuted_m2 == m2):
            ret.append(P)
    #print("find_apparent_automorphisms found", len(ret))
    return ret

def disambiguate_skeleton(include_models, exclude_models, aut):
    # Split models in include_models into sublists so the models in
    # exclude_models do not fit the sublists' descriptions.
    # TODO Ideas for making it faster: (but profile first!)
    # - take as input the nonambiguous lists of summaries previously computed,
    #   and somehow organize things to take the most advantage of this;
    # - start testing validities with longer sublists (somehow...);
    # - ?
    # DONE: Include models in multiple sublists, if they fit the descriptions,
    # so that print_models recognizes the isomorphisms [If this turns out
    # to degrade the output, only allow duplication when a sublist is created
    # as an isomorphism of another]
    n = count_models(include_models)
    ret = []
    used = np.zeros(n, dtype=bool)
    #print("Disambiguating skeleton:")
    #print_model_class(include_models)
    print("{0} models included, {1} excluded".format(n, len(exclude_models)))
    summaries_excl = [model_class_summary(model) for model in exclude_models]
    are_marks_safe = {}
    if len(aut) > 1:
        include_models_dict = {}
        for i, model in enumerate(iter_models(include_models)):
            include_models_dict[hash_model(model)] = i
    while not np.all(used):
        sublist = []
        used_here = np.zeros(n, dtype=bool)
        summary_so_far = None
        for i, model in enumerate(iter_models(include_models)):
            if used[i]:
                continue
            summary_this = model_class_summary(model)
            safe_to_add = True
            if summary_so_far is None:
                summary_new = summary_this
            else:
                summary_new = combine_summaries(summary_so_far, summary_this)
                marks_new = model_class_marks_from_summary(summary_new).tostring()
                if marks_new in are_marks_safe:
                    safe_to_add = are_marks_safe[marks_new]
                else:
                    # determine if this description respects the excluded models
                    for summary_excl in summaries_excl:
                        if is_descriptional_submodel_of(summary_excl,
                                                        summary_new):
                            safe_to_add = False
                            break
                    are_marks_safe[marks_new] = safe_to_add
            if safe_to_add:
                sublist.append(model)
                summary_so_far = summary_new
                used_here[i] = True
        # Now that an unambiguous sub-desciption has been obtained, add models
        # from previous sublists that fit the description (TODO: or even ones
        # that extend it to a known valid description?)
        marks_chosen = model_class_marks_from_summary(summary_so_far).tostring()
        for i, model in enumerate(iter_models(include_models)):
            if not used[i]:
                continue
            summary_this = model_class_summary(model)
            summary_new = combine_summaries(summary_so_far, summary_this)
            marks_new = model_class_marks_from_summary(summary_new).tostring()
            #if marks_new in are_marks_safe and are_marks_safe[marks_new]:
            if marks_new == marks_chosen:
                sublist.append(model)
                summary_so_far = summary_new
                marks_chosen = model_class_marks_from_summary(summary_so_far).tostring() # in case summary changed but marks didn't (unlikely, but just in case)
                used_here[i] = True
        ret.append(sublist)
        used = np.logical_or(used, used_here)

        # Also add isomorphic copies of sublist
        if len(aut) > 1:
            for P in aut: # not sure if it's safe to assume that aut[0] = I
                permuted_sublist = permuted_models(sublist, P)
                any_new = False
                for model in permuted_sublist:
                    model_i = include_models_dict[hash_model(model)]
                    if not used[model_i]:
                        any_new = True
                        used[model_i] = True
                # not any_new happens when sublist has nontrivial automorphisms
                if any_new:
                    ret.append(permuted_sublist)

        #print("created sublist of {0} models".format(len(sublist)))
        #print_model_class(sublist)
    print("Done; split into {0} sublists".format(len(ret)))
    return ret

def disambiguate_clusters(clusters):
    # Assumes the clustering into the outer list is what's of interest, and
    # splits up clusters where necessary so that the subsets' class descriptions
    # unambiguously specify in which cluster each model belongs.

    # Assumes that each sublist of clusters contains models with the same
    # skeleton

    print("Disambiguating...")

    # clusters x skeletons:
    subclass_skeleton_table = tabulate_models(clusters, superskeleton_grouper)
    # skeletons x clusters:
    subclasses_by_skeleton = [list(x) for x in zip(*subclass_skeleton_table)]

    n_clusters = len(clusters)
    n_skels = len(subclasses_by_skeleton)
    split = [[] for _ in range(n_clusters * n_skels)]

    flat_list = flatten_list(subclasses_by_skeleton)
    iso, aut, mappings = find_isomorphisms(flat_list, return_mappings=True)

    for i_skel, skeleton in enumerate(subclasses_by_skeleton):
        # skeleton contains a list of subclasses for a single skeleton
        summaries = [model_class_summary(skeletal_subclass)
                     for skeletal_subclass in skeleton]
        for i_corr, correct_subclass in enumerate(skeleton):
            # skip empty subclasses
            if not correct_subclass:
                continue
            i_flat = i_skel * n_clusters + i_corr
            if iso[i_flat][0] != i_flat:
                i_orig_flat = iso[i_flat][0]
                # Found an isomorphic subclass
                split[i_flat] = permuted_models(split[i_orig_flat],
                                                mappings[i_flat])
                #print("Isomorphic subclass:")
                #print_model_class(correct_subclass)
                #print("  summary of split =")
                #print_model_class(split[i_flat])
                continue
            # Get list of models that fit correct_subclass's description, but
            # belong in another subclass
            exclude_models = []
            for i_other, other_subclass in enumerate(skeleton):
                if i_other == i_corr or not other_subclass:
                    continue
                # Maybe first check subclass relation between the two models?
                for model in iter_models(other_subclass):
                    model_summary = model_class_summary(model)
                    if is_descriptional_submodel_of(model_summary,
                                                    summaries[i_corr]):
                        exclude_models.append(model)
            if not exclude_models:
                #split_clusters[i_corr].extend([correct_subclass])
                split[i_flat] = [correct_subclass]
                continue
            #split_clusters[i_corr].extend(disambiguate_skeleton(correct_subclass,
            #                                                    exclude_models))
            split[i_flat] = disambiguate_skeleton(correct_subclass,
                                                  exclude_models,
                                                  aut[i_flat])

    split_clusters = [[] for _ in clusters]
    for i_corr in range(n_clusters):
        for i_skel in range(n_skels):
            split_clusters[i_corr].extend(split[i_skel * n_clusters + i_corr])
    return split_clusters

def generate_all_models(d):
    # allow bows and cycles
    models = []
    n = d * (d-1) / 2
    N = 1 << (3 * n)
    if N >= 131071:
        print("Total number of masks:", N)
    for mask in range(N):
        if (mask & 65535) == 65535:
            print("current mask:", mask)
        mB = np.zeros((d,d))
        mO = np.eye(d)
        for i in range(1, d):
            for j in range(0, i):
                if mask & 1:
                    mB[i,j] = 1
                if mask & 2:
                    mB[j,i] = 1
                if mask & 4:
                    mO[i,j] = 1
                    mO[j,i] = 1
                mask = mask >> 3
        models.append((mB, mO))
    print("Generated", len(models), "models")
    return models

def generate_all_ADMG_models(d):
    # allow bows, but not cycles
    models = []
    n = d * (d-1) / 2
    N = 6 ** n
    if N >= 131071:
        print("Total number of masks:", N)
    for mask in range(6 ** n):
        if (mask & 65535) == 65535:
            print("current mask:", mask)
        mB = np.zeros((d,d))
        mO = np.eye(d)
        for i in range(1, d):
            for j in range(0, i):
                mask, curmask = divmod(mask, 6)
                bidir, arrow = divmod(curmask, 3)
                if arrow == 1:
                    mB[i,j] = 1
                elif arrow == 2:
                    mB[j,i] = 1
                if bidir == 1:
                    mO[i,j] = 1
                    mO[j,i] = 1
        if not has_cycles(mB):
            models.append((mB, mO))
    print("Generated", len(models), "models")
    return models

def generate_ADMG_models_with_skeleton(skel):
    # allow bows, but not cycles
    d = skel.shape[0]
    models = []
    skel = np.logical_or(skel, skel.T)
    n = np.count_nonzero(np.tril(skel, -1)) # count below main diagonal
    for mask in range(5 ** n):
        mB = np.zeros((d,d))
        mO = np.eye(d)
        for i in range(1, d):
            for j in range(0, i):
                if skel[i,j]:
                    mask, curmask = divmod(mask, 5)
                    bidir, arrow = divmod(curmask + 1, 3)
                    if arrow == 1:
                        mB[i,j] = 1
                    elif arrow == 2:
                        mB[j,i] = 1
                    if bidir == 1:
                        mO[i,j] = 1
                        mO[j,i] = 1
        if not has_cycles(mB):
            models.append((mB, mO))
    print("Generated", len(models), "models")
    return models

def generate_all_BAP_models(d):
    models = []
    n = d * (d-1) / 2
    for mask in range(1 << (2*n)):
        mB = np.zeros((d,d))
        mO = np.eye(d)
        for i in range(1, d):
            for j in range(0, i):
                curmask = mask & 3
                if curmask == 1:
                    mB[i,j] = 1
                elif curmask == 2:
                    mB[j,i] = 1
                elif curmask == 3:
                    mO[i,j] = 1
                    mO[j,i] = 1
                mask = mask >> 2
        if not has_cycles(mB):
            models.append((mB, mO))
    print("Generated", len(models), "models")
    return models

def generate_BAP_models_with_skeleton(skel):
    d = skel.shape[0]
    models = []
    skel = np.logical_or(skel, skel.T)
    n = np.count_nonzero(np.tril(skel, -1)) # count below main diagonal
    for mask in range(3 ** n):
        mB = np.zeros((d,d))
        mO = np.eye(d)
        for i in range(1, d):
            for j in range(0, i):
                if skel[i,j]:
                    mask, curmask = divmod(mask, 3)
                    if curmask == 0:
                        mB[i,j] = 1
                    elif curmask == 1:
                        mB[j,i] = 1
                    elif curmask == 2:
                        mO[i,j] = 1
                        mO[j,i] = 1
        if not has_cycles(mB):
            models.append((mB, mO))
    print("Generated", len(models), "models")
    return models

def generate_ADMG_models_with_max_one_bow(d):
    # similar to generate_all_BAP_models, but may add a bow afterwards
    models = []
    n = d * (d-1) / 2
    for mask in range(1 << (2*n)):
        mB = np.zeros((d,d))
        mO = np.eye(d)
        for i in range(1, d):
            for j in range(0, i):
                curmask = mask & 3
                if curmask == 1:
                    mB[i,j] = 1
                elif curmask == 2:
                    mB[j,i] = 1
                elif curmask == 3:
                    mO[i,j] = 1
                    mO[j,i] = 1
                mask = mask >> 2
        if not has_cycles(mB):
            models.append((mB, mO))
            for i in range(0, d):
                for j in range(0, d):
                    if mB[i,j]:
                        mO_bow = mO.copy()
                        mO_bow[i,j] = 1
                        mO_bow[j,i] = 1
                        models.append((mB.copy(), mO_bow))
    print("Generated", len(models), "models")
    return models

def generate_ADMG_models_with_max_one_bow_and_skeleton(skel):
    # similar to generate_BAP_models_with_skeleton, but may add a bow afterwards
    d = skel.shape[0]
    models = []
    skel = np.logical_or(skel, skel.T)
    n = np.count_nonzero(np.tril(skel, -1)) # count below main diagonal
    for mask in range(3 ** n):
        mB = np.zeros((d,d))
        mO = np.eye(d)
        for i in range(1, d):
            for j in range(0, i):
                if skel[i,j]:
                    mask, curmask = divmod(mask, 3)
                    if curmask == 0:
                        mB[i,j] = 1
                    elif curmask == 1:
                        mB[j,i] = 1
                    elif curmask == 2:
                        mO[i,j] = 1
                        mO[j,i] = 1
        if not has_cycles(mB):
            models.append((mB, mO))
            for i in range(0, d):
                for j in range(0, d):
                    if mB[i,j]:
                        mO_bow = mO.copy()
                        mO_bow[i,j] = 1
                        mO_bow[j,i] = 1
                        models.append((mB.copy(), mO_bow))
    print("Generated", len(models), "models")
    return models


def find_d_separating_set(model, v, w):
    # Simple check: only gives answer for certain "canonical" models.
    # Possible return values:
    # - bool array: conditioning set which gives d-separation;
    # - None: no conditioning set exists;
    # - '?': could not determine for this model.
    mB, mO = model
    d = mB.shape[0]
    if mO[v,w] or mB[v,w] or mB[w,v]:
        return None
    if not is_bow_free(model):
        #print("bow")
        return '?'
    cond = np.zeros(d, dtype=bool)

    edges = np.logical_or(mO, np.logical_or(mB, mB.T))
    heads = np.logical_and(np.logical_or(mB.T, mO), np.logical_not(mB))
    tails = np.logical_and(np.logical_and(mB, np.logical_not(mO)),
                           np.logical_not(mB.T))
    for x in range(d):
        if x == v or x == w:
            continue
        if not edges[v,x] or not edges[x,w]:
            #print("missing 2-path")
            return '?'
        if heads[v,x] and heads[w,x]:
            # v-structure v *-> x <-* w
            if np.any(tails[:,x]):
                # such a tail might be the start of a directed path to v or w,
                # which would make v-x-w an inducing path
                #print("fail at collider")
                return '?'
            # don't condition on x
        else:
            # no v-structure v *-+ x +-* w
            if heads[w,x]:
                # by disallowing any such heads, we can rule out inducing paths
                # through the conditioning set
                #print("fail at non-collider")
                return '?'
            # condition on x
            cond[x] = True
    return cond

def is_t_separated(model, A, B):
    # Check if there are sets (C_A, C_B) with #C_A + #C_B < min(#A, #B) that
    # t-separate A and B. (Larger sets C_A, C_B would satisfy the definition
    # of t-separation, but don't lead to conclusions from the theorem, so
    # we don't want to consider those.)
    mB, mO = model
    d = mB.shape[0]
    minAB = min(np.sum(A), np.sum(B))
    if minAB == 0:
        return False
    # construct a directed graph G such that the directed paths in G correspond
    # to the treks in our graph
    G = nx.DiGraph()
    s = 4 * d
    t = s + 1
    # Other nodes:
    # [0,d), [d,2d): backward part of treks, pre- & post-capacity
    # [2d,3d), [3d,4d): forward part of treks, pre- & post-capacity
    for v in range(d):
        G.add_edge(v, d+v, capacity=1)
        G.add_edge(2*d+v, 3*d+v, capacity=1)
        if A[v]:
            G.add_edge(s, v, capacity=1)
        if B[v]:
            G.add_edge(3*d+v, t, capacity=1)
        for w in range(d):
            if mB[w,v]:
                # v --> w
                G.add_edge(d+w, v, capacity=1)
                G.add_edge(3*d+v, 2*d+w, capacity=1)
            if mO[v,w]:
                # including if v == w
                G.add_edge(d+v, 2*d+w, capacity=1)
    max_flow = nx.maximum_flow_value(G, s, t)
    if max_flow < minAB:
        # there is a cut of size smaller than minAB, so we have a useful t-sep
        return True
    return False

def find_t_separation(model):
    # Find sets A, B that are t-separated by some sets (C_A, C_B) small enough
    # to satisfy the t-separation theorem for mixed graphs (Thm2.17 in
    # SullivantTalaskaDraisma2010). We limit ourselves to sets A, B with:
    # - #A = #B (if <, t-sep also holds for all subsets of B of size #A);
    # - #(A\B) [= #(B\A)] >= 2 (otherwise C must include A cap B, so it's just
    # d-separation);
    # - A lexicographically before B (because the cases are symmetric).
    # Currently, stops if some (A,B) is found; could be modified to find all.
    mB, mO = model
    d = mB.shape[0]
    # d < 4: no calls to is_t_separated
    # d = 4: 3 calls (1 * 1 * 3 * 1 * 1)
    # d = 5: 30 calls (2 * 1 * (4*3 + 3*1) * 1)
    # d = 6: 190 calls (1*[1*(5*6+4*3+3*1)*1 + 1*10*1*1] + (1)*1*(5*6+4*3+3*1)*(2+1)) (I think)
    #print("Find t-sep for model:")
    #print_model(model)
    for num_both in range(d-3): # from 0 up to and including d-4
        #print("num_both =", num_both)
        for num_only in range(2, int((d-num_both) / 2)+1):
            #print("num_only =", num_only)
            for A_only in itertools.combinations(range(d), num_only):
                #print("A_only =", A_only)
                A_only_array = np.zeros(d, dtype=bool)
                A_only_array[list(A_only)] = True
                # A <_lex B iff A_only[0] < B_only[0]
                B_only_candidates = [v for v in range(A_only[0]+1, d)
                                     if not A_only_array[v]]
                for B_only in itertools.combinations(B_only_candidates,
                                                     num_only):
                    #print("B_only =", B_only)
                    B_only_array = np.zeros(d, dtype=bool)
                    B_only_array[list(B_only)] = True
                    both_candidates = [v for v in range(d)
                                       if not (A_only_array[v]
                                               or B_only_array[v])]
                    for both in itertools.combinations(both_candidates,
                                                       num_both):
                        #print("both =", both)
                        both_array = np.zeros(d, dtype=bool)
                        both_array[list(both)] = True
                        A = np.logical_or(A_only_array, both_array)
                        B = np.logical_or(B_only_array, both_array)
                        if is_t_separated(model, A, B):
                            return (A, B)
    return None


def check_htc(model, allowed_nodes, v):
    # Test whether a system of half-treks with no sided interaction exists
    # from (subset of) allowed nodes to parents of v. If so, return such a
    # subset as a list (or True if it's the empty subset); otherwise False.
    mB, mO = get_first_model(model)
    if np.sum(mB[v,:]) == 0:
        return True # no parents, so Y=[] works
    if not np.any(allowed_nodes):
        return False
    d = mB.shape[0]
    G = nx.DiGraph()
    ns = np.sum(allowed_nodes)
    s = ns + 2*d # supersource
    t = s + 1 # supersink
    # node [source number]: a source node
    # node ns+[node number]: pre-capacity part of a main node
    # node ns+d+[node number]: post-capacity " (some of these are sinks)
    s_counter = 0 # grows to ns during graph construction
    s_orig_node = []
    for w in range(d):
        if allowed_nodes[w]:
            G.add_edge(s, s_counter, capacity=1)
            #print(s, "-1>", s_counter)
            G.add_edge(s_counter, ns+w) # half-trek is directed path
            #print(s_counter, "-->", ns+w)
            for x in range(d):
                if x != w and mO[w,x]:
                    G.add_edge(s_counter, ns+x) # " start with bidir edge
                    #print(s_counter, "-->", ns+x)
            s_counter += 1
            s_orig_node.append(w)
        G.add_edge(ns+w, ns+d+w, capacity=1)
        #print(ns+w, "-1>", ns+d+w)
        for x in range(d):
            if mB[x,w]:
                G.add_edge(ns+d+w, ns+x)
                #print(ns+d+w, "-->", ns+x)
        if mB[v,w]:
            # w is parent of v, so it is a sink
            G.add_edge(ns+d+w, t)
            #print(ns+d+w, "-->", t)
    max_flow, flow_dict = nx.maximum_flow(G, s, t)
    if False:
        print("node", v)
        print("sources:", ns)
        print("flow:", max_flow)
        #print(nx.maximum_flow(G, s, t))
    if max_flow == np.sum(mB[v,:]):
        Y = []
        flow_from_s = flow_dict[s]
        for to, amount in flow_from_s.items():
            if amount > 0:
                Y.append(s_orig_node[to])
        return Y
    return False

def htc_Yv(model, v):
    # Return a list of bool arrays describing all potential
    # choices of Y_v (i.e. all sets of nodes that satisfy the halftrek
    # criterion for v; this doesn't look at the htc ordering of all nodes in
    # the graph)
    mB, mO = get_first_model(model)
    d = mB.shape[0]
    num_pa = int(np.sum(mB[v,:]))
    if num_pa == 0:
        return [np.zeros(d, dtype=bool)]
    reachable = transitive_reflexive_closure(mB).T
    # Determine matrix of half-trek reachability;
    # unlike in paper, also define v and its siblings as htr from v
    htr = np.dot(mO, reachable) # (gives ints)
    available_nodes = np.logical_and(htr[:,v], np.logical_not(mO[v,:]))
    available_nodes[v] = False
    available_nodes_inds = np.where(available_nodes)[0]
    choices = []
    for try_inds in itertools.combinations(available_nodes_inds, num_pa):
        # Convert list of indices back to mask; itertools yiels tuples, which
        # numpy indexing treats differently, so convert to np.array first
        try_mask = np.zeros(d, dtype=bool)
        try_mask[np.array(try_inds)] = True
        if check_htc(model, try_mask, v):
            choices.append(try_mask)
    return choices

def htc_Yv_tex(model, v, node_names):
    # Return a tex formula string (excluding $'s) describing all potential
    # choices of Y_v (i.e. all sets of nodes that satisfy the halftrek
    # criterion for v; this doesn't look at the htc ordering of all nodes in
    # the graph)
    choices = htc_Yv(model, v)
    if len(choices) == 0:
        return "\\sortkey{! } \\text{none}"
    if len(choices) == 1 and not np.any(choices[0]):
        #return "\\varnothing"
        return "\\sortkey{0- } \\varnothing"
    mB, mO = get_first_model(model)
    d = mB.shape[0]
    reachable = transitive_reflexive_closure(mB).T
    # Determine matrix of half-trek reachability;
    # unlike in paper, also define v and its siblings as htr from v
    htr = np.dot(mO, reachable) # (gives ints)
    s = ""
    sortkey = str(num_pa) # for sorting based on set size [was ""]
    for i, Yv in enumerate(choices):
        if i > 0:
            s += " \\,|\\, "
            sortkey += "/"
        for ind in range(d):
            if Yv[ind]:
                s_node = node_names[ind]
                sortkey += node_names[ind]
                if htr[v, ind]:
                    # mark mutually ht-reachable nodes
                    s_node = "\\underline{" + s_node + "}"
                    sortkey += "_"
                else:
                    sortkey += " "
                s += s_node
    return "\\sortkey{{{0} }} {1}".format(sortkey, s)

def htc_Yvs_annotator(model):
    d = model[0].shape[0]
    node_names = list(map(chr, range(97, 97 + d))) # TODO: duplicated
    annot = "  $\\begin{aligned}\n"
    for v in range(d):
        annot += ("    Y_{{{0}}} &= {1}\\\\\n"
                  .format(node_names[v], htc_Yv_tex(model, v, node_names)))
    annot += "  \\end{aligned}$"
    return annot

def Jacobian_annotator(model):
    # Output the matrix for the htc-nonidentifiability proof (the model is
    # generically infinite-to-one iff this matrix is generically not of full
    # column rank).
    (mB,mO) = model
    d = model[0].shape[0]
    node_names = list(map(chr, range(97, 97 + d))) # TODO: duplicated
    nnsp = d*(d-1)/2 - num_bidirected_edges(mO)
    ndir = num_directed_edges(mB)
    row_v = np.zeros(nnsp, dtype=int)
    row_w = np.zeros(nnsp, dtype=int)
    col_u = np.zeros(ndir, dtype=int)
    col_v = np.zeros(ndir, dtype=int)
    i = 0
    j = 0
    for v in range(d):
        for w in range(d):
            if v < w and not mO[v][w]:
                row_v[i] = v
                row_w[i] = w
                i += 1
            if mB[v][w]:
                col_u[j] = w
                col_v[j] = v
                j += 1
    matrix = np.empty((nnsp,ndir), dtype=object)
    for i in range(nnsp):
        for j in range(ndir):
            w = None
            if row_v[i] == col_v[j]:
                w = row_w[i]
            elif row_w[i] == col_v[j]:
                w = row_v[i]
            if w is None:
                matrix[i][j] = '0'
            else:
                matrix[i][j] = '{0}{1}'.format(node_names[w],
                                               node_names[col_u[j]])

    annot = "  $\\begin{{array}}{{{0}}}\n".format('c'*nnsp)
    for i in range(nnsp):
        for j in range(ndir):
            if j == 0:
                annot += '    '
            else:
                annot += ' & '
            annot += matrix[i,j]
        if i < nnsp - 1:
            annot += '\\\\'
        annot += '\n'
    annot += "  \\end{array}$"
    return annot

def raw_constraints_tex(model, cY_matrix, htr, node_names, v=None, w=None):
    (mB,mO) = model
    d = mB.shape[0]
    underlined_cY_matrix = np.logical_and(cY_matrix, htr)

    # Enumerate the pairs of nodes between which a constraint is imposed
    num_constraints = 0
    constraints_s = ""
    used_pairs = np.logical_or(mO, np.logical_or(cY_matrix, cY_matrix.T))
    nodes_in_constraints = np.zeros(d, dtype=bool)
    for v2 in ([v] if v is not None else range(d)):
        for w2 in ([w] if w is not None else range(v2+1, d)):
            if not used_pairs[v2,w2]:
                if num_constraints > 0:
                    constraints_s += ", "
                constraints_s += "${0}{1}$".format(node_names[v2],
                                                   node_names[w2])
                num_constraints += 1
                nodes_in_constraints[v2] = True
                nodes_in_constraints[w2] = True
    if num_constraints == 0:
        desc = "No constraints"
        return desc
    elif num_constraints == 1:
        desc = "Constraint on {0} with ".format(constraints_s)
    else:
        desc = "Constraints on {0} with ".format(constraints_s)

    # List the Lambda's that appear in the constraints: for a node v that
    # appears in the constraints, include v, and also Y_v \cap htr(v),
    # recursively.
    num_lambdas = 0
    lambda_appears = np.dot(nodes_in_constraints,
                            transitive_reflexive_closure(underlined_cY_matrix))
    for v in range(d):
        if lambda_appears[v]:
            if num_lambdas > 0:
                desc += ", "
            num_lambdas += 1
            Yv_s = ""
            if not np.any(cY_matrix[v,:]):
                Yv_s = "\\varnothing"
            else:
                for y in range(d):
                    if cY_matrix[v,y]:
                        s_node = node_names[y]
                        if htr[v, y]:
                            # mark mutually ht-reachable nodes
                            s_node = "\\underline{" + s_node + "}"
                        Yv_s += s_node
            if np.all(mB[v,:] == cY_matrix[v,:]):
                # special form for case Y_v = pa(v): Lambda_{Yv}
                desc += "$\\Lambda_{{{0},{1}}}$".format(Yv_s,
                                                        node_names[v])
            else:
                pa_s = ""
                for p in range(d):
                    if mB[v,p]:
                        pa_s += node_names[p]
                # standard form: Lambda_{Pv:Y}
                desc += "$\\Lambda_{{{2},{1}:{0}}}$".format(Yv_s,
                                                            node_names[v],
                                                            pa_s)
    return desc

def raw_constraints_list_tex(models, one_by_one=False):
    # Return list of Theorem-1-form sets of constraints, for each model and
    # each choice of cY. By default, each element is a string describing
    # those constraints; if one_by_one is True, it is a list of strings, each
    # describing one constraint.
    descriptions = []
    for model in iter_models(models):
        if not is_htc_identifiable(model):
            #return "not htc-identifiable"
            continue
        (mB,mO) = model
        d = mB.shape[0]
        if num_edges(model) == d*(d-1)/2:
            # Optimization: avoid spending time on (typically large) clusters
            # that impose no constraints
            desc = "No constraints"
            descriptions.append(desc)
            continue
        node_names = list(map(chr, range(97, 97 + d)))
        Yv_choices = [htc_Yv(model, v) for v in range(d)] # list of lists of masks
        reachable = transitive_reflexive_closure(mB).T
        # Determine matrix of half-trek reachability;
        # unlike in paper, also define v and its siblings as htr from v
        htr = np.dot(mO, reachable) # (gives ints)
        for cY in itertools.product(*Yv_choices):
            # cY: tuple of masks
            cY_matrix = np.zeros((d,d), dtype=bool)
            for v in range(d):
                cY_matrix[v,:] = cY[v]

            # Verify that cY is legal for htc-identifiability
            underlined_cY_matrix = np.logical_and(cY_matrix, htr)
            if has_cycles(underlined_cY_matrix):
                # no htc ordering exists
                continue

            if one_by_one:
                desc_list = []
                for v in range(d):
                    for w in range(v+1, d):
                        desc = raw_constraints_tex(model, cY_matrix, htr,
                                                   node_names, v=v, w=w)
                        if desc != "No constraints":
                            desc_list.append(desc)
                descriptions.append(desc_list)
            else:
                desc = raw_constraints_tex(model, cY_matrix, htr, node_names)
                descriptions.append(desc)
    return descriptions

def create_constraint_dictionary(clusters):
    # For each cluster imposing exactly one equality constraint, but all forms
    # of that Theorem-1-type constraint in a dictionary, mapping them to a
    # recognizable name
    constraint_names = {}
    d = get_first_model(clusters)[0].shape[0]
    node_names = list(map(chr, range(97, 97 + d)))
    for cluster_ind, cluster in enumerate(clusters):
        min_num_edges = d*d
        for model in iter_models(cluster):
            min_num_edges = min(min_num_edges, num_edges(model))
        if min_num_edges != d*(d-1)/2 - 1:
            continue

        name = "C{0}".format(cluster_ind)
        # More informative names for vanishing (partial) correlations:
        # first, check for d-separation between some pair of nodes
        special_name = False
        for v in range(d):
            for w in range(v+1, d):
                for model in iter_models(cluster):
                    cond = find_d_separating_set(model, v, w)
                    #print_model(model)
                    #print(v,w,cond)
                    if cond is '?':
                        continue
                    elif cond is None:
                        break # no need to try other models
                    else:
                        cond_names = ""
                        if np.any(cond):
                            cond_names = "."
                        for i in range(d):
                            if cond[i]:
                                cond_names += node_names[i]
                        name = ("$\\rho_{{{0}{1}{2}}} = 0$"
                                .format(node_names[v], node_names[w],
                                        cond_names))
                        special_name = True
                        break
        if not special_name:
            t_sep = find_t_separation(get_first_model(cluster))
            if t_sep is not None:
                (A, B) = t_sep
                A_names = ""
                B_names = ""
                for i in range(d):
                    if A[i]:
                        A_names += node_names[i]
                    if B[i]:
                        B_names += node_names[i]
                name = ("$\\lvert \\Sigma_{{{0},{1}}} \\rvert = 0$"
                        .format(A_names, B_names))
                special_name = True

        # get rid of duplicate strings describing this cluster, then add
        # unique strings to dictionary
        descriptions = set()
        for desc in raw_constraints_list_tex(cluster):
            descriptions.add(desc)
        for desc in descriptions:
            if desc not in constraint_names:
                constraint_names[desc] = name
            else:
                constraint_names[desc] += "&" + name
                print("Warning: Constraint occurring in different one-constraint clusters:", constraint_names[desc], desc, sep='\n')
    return constraint_names

def all_constraints_annotator(models, constraint_names=None):
    use_constraint_names = (constraint_names is not None)
    descriptions = set()
    annot = '\\begin{tabular}{@{}l@{}}'
    constraints_list = raw_constraints_list_tex(models,
                                                one_by_one=use_constraint_names)
    if (not use_constraint_names
        or not constraints_list or len(constraints_list[0]) == 1):
        for desc in constraints_list:
            if type(desc) is list:
                desc = desc[0]
            if desc not in descriptions:
                if descriptions:
                    annot += ' \\\\ '
                annot += desc
                descriptions.add(desc)
    else:
        # Sets of multiple constraints each, to be translated by
        # constraint_names
        for raw_desc_list in constraints_list:
            desc = ""
            if raw_desc_list == "No constraints":
                desc = raw_desc_list
            else:
                for raw_desc in raw_desc_list:
                    if desc:
                        desc += ", "
                    if raw_desc in constraint_names:
                        desc += ("(${0}{1}$) {2}"
                                 .format(raw_desc[15], raw_desc[16],
                                         constraint_names[raw_desc]))
                    else:
                        desc += "unrecognized: {0}".format(raw_desc)
            if desc not in descriptions:
                if descriptions:
                    annot += ' \\\\ '
                annot += desc
                descriptions.add(desc)
    annot += '\\end{tabular}'
    return annot

def annotate(models, annotator):
    return [annotator(model) for model in models]

def annotate_recursive(models, annotator):
    if type(models) is list:
        return [annotate_recursive(model, annotator) for model in models]
    return annotator(models)

def annotate_sort(models, annotator):
    if len(models) == 0:
        return [], []
    annotations_models = [(annotator(model), model)
                          for model in models]
    annotations_models.sort(key=lambda model: model[0])
    annotations, models = zip(*annotations_models) # returns two tuples
    annotations = list(annotations)
    models = list(models)
    return annotations, models

def annotate_standardize_sort(models, annotator):
    # For each model, consider all its permutations and take the one with the
    # smallest annotation key. Then sort the models.
    # (Only works if models is non-nested. Assumes only one representative of
    # each isomorphism class is present.)
    n = len(models)
    if n == 0:
        return [], []
    d = models[0][0].shape[0]
    permutations = all_permutations(d)
    annotations_models = n * [('~', None)] # this comes last
    for i, model in enumerate(models):
        for P in permutations:
            permuted_model = permuted_models(model, P)
            annot = annotator(permuted_model)
            if annot < annotations_models[i][0]:
                annotations_models[i] = (annot, permuted_model)
    # Note: without key, sort throws value-error: array of bools in ambiguous
    annotations_models.sort(key=lambda model: model[0])
    #return zip(*annotations_models)
    annotations, models = zip(*annotations_models) # returns two tuples
    annotations = list(annotations)
    models = list(models)
    return annotations, models

def recursive_zip(annotations):
    # recursively convert a tuple/list of lists of (lists of ...)
    # to a list (of lists ...) of tuples
    #print("Received", annotations)
    if len(annotations) == 1:
        #print("len == 1")
        return annotations
    elif type(annotations[0]) is not list:
        #print("no lists:", annotations[0])
        return annotations
    else:
        zipped = zip(*annotations)
        #print(zipped)
        return [recursive_zip(elt) for elt in zipped]

def is_htc_identifiable(model, return_ordering=False):
    # Test if model is htc-identifiable using the algorithm in section 6.1
    # of FoygelDraismaDrton2012.
    # If the model is htc-identifiable, return True (or a valid ordering of the
    # nodes) as a list; otherwise, return False.
    mB, mO = get_first_model(model)
    d = mB.shape[0]
    # Optimization for bowfree models
    if not return_ordering and is_bow_free(model) and not has_cycles(mB):
        return True
    reachable = transitive_reflexive_closure(mB).T
    # Determine matrix of half-trek reachability;
    # unlike in paper, also define v and its siblings as htr from v
    htr = np.dot(mO, reachable) # (gives ints)
    solved_nodes = np.logical_not(np.any(mB, axis=1))
    ordering = list(np.nonzero(solved_nodes)[0])
    keepgoing = True
    while keepgoing and not np.all(solved_nodes):
        keepgoing = False
        for v in range(d):
            if not solved_nodes[v]:
                allowed_nodes = np.logical_and(
                    np.logical_or(solved_nodes, np.logical_not(htr[v,:])),
                    np.logical_not(mO[v,:]))
                if check_htc(model, allowed_nodes, v):
                    solved_nodes[v] = True
                    ordering.append(v)
                    keepgoing = True
    if np.all(solved_nodes):
        return ordering
    return False

def is_htc_unidentifiable(model):
    # Test if model is htc-unidentifiable using the algorithm in section 6.2
    # of FoygelDraismaDrton2012
    mB, mO = get_first_model(model)
    d = mB.shape[0]
    if num_edges((mB, mO)) > d*(d-1)/2:
        return True
    if num_directed_edges(mB) == 0:
        # no bows or cycles, so identifiable
        # (must be special case, or nx complains that sink is not in graph)
        return False
    #if num_bidirected_edges(mO) == d*(d-1)/2:
        # complete bidirected (and at least one bow): infinite-to-one
        # (must be special case, or nx complains that source is not in graph)
        #return True
    G = nx.DiGraph()
    s = 2*d*d # supersource
    t = s + 1 # supersink
    # node [2*(d*a+b)]: pre-capacity part of node Ra(b) [0-based]
    # node [" + 1]: post-capacity "
    l_counter = s + 2 # grows during graph construction
    for v in range(0, d):
        for w in range(0, v):
            if not mO[v,w]:
                # construct node L{v,w} and its edges
                G.add_edge(s, l_counter, capacity=1)
                G.add_edge(l_counter, 2*(d*v+w))
                G.add_edge(l_counter, 2*(d*w+v))
                for u in range(0, d):
                    if w != u and mO[w,u]:
                        G.add_edge(l_counter, 2*(d*v+u))
                    if v != u and mO[v,u]:
                        G.add_edge(l_counter, 2*(d*w+u))
                l_counter += 1
    for i in range(d*d):
        G.add_edge(2*i, 2*i+1, capacity=1)
    for v in range(d):
        for w in range(d):
            if mB[v,w]: # w --> v
                for a in range(d):
                    G.add_edge(2*(d*a+w)+1, 2*(d*a+v))
                G.add_edge(2*(d*v+w)+1, t)
    max_flow = nx.maximum_flow_value(G, s, t)
    return max_flow < num_directed_edges(mB)

def is_htc_identifiable_after_decomposition(model):
    #print_model(model)
    mB, mO = get_first_model(model)
    d = mB.shape[0]
    if has_cycles(mB):
        # c-decomposition only applies to acyclic graphs.
        return is_htc_identifiable(model)
    mO_trans = transitive_reflexive_closure(mO)
    visited = np.zeros(d, dtype=bool)
    for v in range(d):
        if not visited[v]:
            current_component = mO_trans[v, :]
            visited = np.logical_or(visited, current_component)
            mB_comp = np.logical_and(mB.T, current_component).T
            mO_comp = np.logical_or(np.logical_and(mO, current_component),
                                    np.eye(d, dtype=bool))
            #print_model((mB_comp, mO_comp))
            if not is_htc_identifiable((mB_comp, mO_comp)):
                #print("Failed")
                return False
    # All components are htc-identifiable
    #print("Identified")
    return True

def skeleton_collider_bows_normalizer(model):
    # Such models can be shown to be distributionally equivalent, even in
    # case with bows: proof from NowzohourMaathuisBuhlman carries over.
    mB, mO = get_first_model(model)
    head = np.logical_or(mB, mO)
    indeg = np.sum(head, axis=1, keepdims=True) - 1
    tail = np.logical_or(mB.T, head * (indeg < 2)) # tail or non-coll head
    head = head * (indeg >= 2) # arrowheads forming colliders
    # (exact representation of model, except at noncolliding nodes,
    # where head and head+tail are replaced by tail)
    return (head, tail)

def superskeleton_grouper(models):
    # looks at the entire model class instead of at just one representative
    d = get_first_model(models)[0].shape[0]
    any_edges = np.zeros((d,d), dtype=bool)
    for (mB, mO) in iter_models(models):
        dir_edges = np.logical_or(mB, mB.T)
        edges = np.logical_or(mO, dir_edges)
        any_edges = np.logical_or(any_edges, edges)
    return (any_edges, np.zeros(1))


def preferred_models_first(models, verbose=False):
    # In a list of models, find one that RICF seems most likely to work well on,
    # and move it to the front.
    # (If models contains nested lists, this function doesn't reorder those: it
    # assumes that the first element in each sublist is the preferred one.)
    pref = []
    #d = get_first_model(models)[0].shape[0]
    for i in range(len(models)):
        (mB, mO) = get_first_model(models[i])
        skel = np.logical_or(mB, np.logical_or(mB.T, mO))
        tiebreaker = (num_edges((mB, mO)), num_bidirected_edges(mO))
        # Prefer models with larger inclusionwise skeletons. If different
        # skeletons are inclusionwise maximal, put *one* model for each
        # skeleton in front. This model should have the fewest edges (equiv:
        # fewest bows), and secondarily the fewest bidirected edges among
        # models with that skeleton.
        add_i = True
        keep = np.ones(len(pref), dtype=bool)
        if verbose:
            print_model((mB, mO))
        for j, (i_pref, skel_pref, tiebreaker_pref) in enumerate(pref):
            if np.all(skel == skel_pref):
                # same skeleton
                if verbose:
                    print("same skel")
                if tiebreaker < tiebreaker_pref:
                    # replace the model in pref
                    keep[j] = False
                else:
                    add_i = False
            elif np.all(np.logical_or(skel_pref, np.logical_not(skel))):
                # old skeleton is strictly larger
                add_i = False
            elif np.all(np.logical_or(skel, np.logical_not(skel_pref))):
                # new skeleton is strictly larger
                if verbose:
                    print("new skel is larger")
                keep[j] = False
            else:
                # skeletons are not inclusionwise comparable
                if verbose:
                    print("skels are incomparable")
                pass
        if not np.all(keep):
            pref = [pref_elt for j, pref_elt in enumerate(pref) if keep[j]]
        if add_i:
            pref.append((i, skel, tiebreaker))
        if verbose:
            print(keep)
            print(add_i)
            print(len(pref))
    models_pref = []
    models_nonpref = []
    j = 0
    if verbose:
        print("Preferring models:", pref)
    for i, model in enumerate(models):
        if j < len(pref) and i == pref[j][0]:
            models_pref.append(model)
            j += 1
        else:
            models_nonpref.append(model)
    if verbose:
        print(j)
        print(models_pref)
    return models_pref + models_nonpref

def partition_models(models, normalizer, verbose=False):
    classes = {}
    ret = []
    for model in models:
        which_class = normalizer(model)
        if verbose >= 5:
            print_model(get_first_model(model))
            print(which_class)
        if hash_model(which_class) in classes:
            classes[hash_model(which_class)].append(model)
        else:
            classes[hash_model(which_class)] = [model]
            ret.append(which_class)
    if verbose:
        print("Identified", len(ret), "distinct classes using",
              normalizer.__name__)
    for i, which_class in enumerate(ret):
        ret[i] = preferred_models_first(classes[hash_model(which_class)])
    return ret

def tabulate_models(model_classes, normalizer, verbose=False):
    # Input: list of list of models (this outer division into elements will be
    # preserved);
    # Ouput: list of list of list of models; outer list is same, second level
    # is result of normalizer; all second level lists will have same length
    # (but some elements will be [])
    columns = {} # maps normalizer to column number
    num_columns = 0
    ret = [[] for _ in model_classes]
    for class_no, models in enumerate(model_classes):
        # Start new row with same number of columns as previous row
        if num_columns > 0:
            ret[class_no] = [[] for _ in range(num_columns)]
        #print("Row", class_no, "starting at", len(ret[class_no]), "elements")
        for model in models:
            which_column = normalizer(model)
            if verbose >= 5:
                print_model(get_first_model(model))
                print(which_column)
            hash = hash_model(which_column)
            if hash in columns:
                column_no = columns[hash]
                ret[class_no][column_no].append(model)
            else:
                ret[class_no].append([model])
                columns[hash] = num_columns
                num_columns += 1
                #print("Column added")
    # Early rows may be shorter than later ones: grow them to the same size
    for row in ret:
        if len(row) < num_columns:
            row.extend([[] for _ in range(num_columns - len(row))])
    if verbose:
        print("Identified", num_columns, "distinct classes using",
              normalizer.__name__)
    return ret

def theoretical_clusters(models):
    # Determine algebraic equivalence classes using Theorem 2 in UAI2017
    # paper. This theorem gives a sufficient condition for equivalence; some
    # clusters in the output may actually be algebraically equivalent to each
    # other.
    # Assumes each model comes after its submodels (else raises error).
    d = get_first_model(models)[0].shape[0]
    models_dict = {} # map graph hash to index
    n = len(models)
    possibly_identifiable = np.ones(n, dtype=bool)
    possibly_in_between = np.ones(n, dtype=bool)
    possibly_infinite_to_one = np.ones(n, dtype=bool)
    rank_deficiency_known = np.zeros(n, dtype=bool)
    rank_deficiency = np.zeros(n)
    num_inconclusive = 0
    num_inf_unassigned = 0
    #clusters_list = [[] for _ in range(n)] # list of sorted lists of model indices (empty for members beyond the first)
    clusters = UnionFind()
    for ind, model in enumerate(models):
        models_dict[hash_model(model)] = ind

        if is_htc_identifiable(model):
            possibly_infinite_to_one[ind] = possibly_in_between[ind] = False
        elif is_htc_unidentifiable(model):
            possibly_identifiable[ind] = possibly_in_between[ind] = False
        elif num_nodes(model) == 4 and not has_cycles(model[0]):
            # known from htc paper to be finite-to-one but not identifiable
            possibly_identifiable[ind] = possibly_infinite_to_one[ind] = False

        if possibly_identifiable[ind] or possibly_in_between[ind]:
            # model is either finite-to-one (i.e. identifiable or in between),
            # or inconclusive using above criteria
            if not possibly_infinite_to_one[ind]:
                rank_deficiency_known[ind] = True
                rank_deficiency[ind] = 0
            else:
                #print("inconclusive model encountered:")
                #print_model(model)
                if num_inconclusive == 0:
                    print("warning: encountered an inconclusive models")
                num_inconclusive += 1
            #clusters_list[ind] = [ind]
        else:
            # Model is definitely infinite-to-one.
            # Apply theorem: try removing each edge. If any non-infinite-to-one
            # models are obtained this way, those are equivalent to each other
            # and to the current model. (If only infinite-to-one models are
            # obtained, look at the models with the smallest rank deficiency
            # instead.) Leave inconclusive models isolated; if inconclusive
            # models are present and everything else is infinite-to-one, we
            # can't decide what to do -- print a warning.
            (mB, mO) = model
            submodels = []
            min_rank_deficiency = 2*d*d
            any_unknown = False
            for i in range(d):
                for j in range(d):
                    for edge_type in range(2): # directed / bidirected
                        if ((edge_type == 0 and mB[i][j]) or
                            (edge_type == 1 and i < j and mO[i][j])):
                            (mBred, mOred) = np.copy(model)
                            if edge_type == 0:
                                mBred[i][j] = False
                            else:
                                mOred[i][j] = mOred[j][i] = False
                            red_hash = hash_model((mBred, mOred))
                            if red_hash not in models_dict:
                                raise ValueError("Models not properly ordered")
                            ind_sub = models_dict[red_hash]
                            if False: #not rank_deficiency_known[ind_sub]:
                                any_unknown = True
                            else:
                                submodels.append(ind_sub)
                                min_rank_deficiency = min(min_rank_deficiency,
                                                          rank_deficiency[ind_sub])
            if True: #not any_unknown or min_rank_deficiency == 0:
                #rank_deficiency_known[ind] = True
                rank_deficiency[ind] = min_rank_deficiency + 1
                # group current model with submodels of minimum rank deficiency
                equiv = [ind_sub for ind_sub in submodels
                         if rank_deficiency[ind_sub] == min_rank_deficiency
                         and rank_deficiency_known[ind_sub]]
                if equiv:
                    # Of the child models of minimum rank deficiency, at least
                    # one has exactly that rank deficiency, so that we know
                    # this model's rank deficiency exactly.
                    rank_deficiency_known[ind] = True
                    equiv.append(ind)
                    clusters.union(*equiv)
                else:
                    # not sure what to group together: leave isolated and warn
                    #clusters_list[ind] = [ind]
                    #print("theoretical_clusters doesn't know what to do for this infinite-to-one model:")
                    #print_model(model)
                    num_inf_unassigned += 1

    clusters_list = [[] for _ in range(n)]
    for i in range(n):
        # Omit models with unknown rank deficiency (i.e. unknown dimension)
        # from output (they are all singletons anyway)
        if rank_deficiency_known[i]:
            clusters_list[clusters[i]].append(i)
    #ret = [[models[ind] for ind in cluster]
    #       for cluster in clusters_list if cluster]
    ret = [preferred_models_first([models[ind] for ind in cluster])
           for cluster in clusters_list if cluster]

    if num_inconclusive:
        print("encountered", num_inconclusive, "inconclusive models")
        print("encountered", num_inf_unassigned,
              "infinite-to-one models that couldn't be assigned")
    return ret

def count_class_sizes(model_classes):
    #print("#(models per class), #(classes with that size):")
    c = collections.Counter()
    for models in model_classes:
        c[len(models)] += 1
    for key, value in sorted(c.iteritems()):
        print(value, "models of size", key)
    return c


def correlation_matrix(S):
    # Rescale a covariance matrix into a correlation matrix
    scale_vector = np.power(np.diag(S), -.5)
    Corr = scale_vector * S * scale_vector.reshape((-1, 1))
    return Corr

def draw_coefficient_matrix(mB, var=1.0):
    #B = mB * np.random.uniform(0, 1, (d, d)) # limited to positive effects
    #B = mB * np.random.uniform(-1, 1, mB.shape)
    B = math.sqrt(var) * mB * np.random.standard_normal(mB.shape)
    return B

def draw_covariance_matrix(d, var):
    pos_def = False
    while not pos_def:
        S = np.random.randn(d,d)
        S = math.sqrt(var) * np.dot(S, S.T)
        if np.amin(np.linalg.eigvals(S)) > 0:
            pos_def = True
    return S

def draw_masked_covariance_matrix(mO, var):
    pos_def = False
    while not pos_def:
        O = np.random.standard_normal(mO.shape) # same as randn, but takes tuple as argument
        O = math.sqrt(var) * mO * np.dot(O, O.T)
        if np.amin(np.linalg.eigvals(O)) > 0:
            pos_def = True
    return O

def draw_multivariate_data(true_B, true_O, N):
    d = true_B.shape[0]
    m = np.zeros(d) # vector of means
    # S = (I - B)^{-1} true_O (I - B)^{-T} --- covariance matrix of data (paper Def 1)
    S = np.dot((np.linalg.inv(np.eye(d) - true_B)), np.dot(true_O, (np.linalg.inv(np.eye(d) - true_B)).T))
    Y = np.random.multivariate_normal(m, S, N) # Nxd
    return Y


def sample_covariance_matrix(X):
    # Input: X\in R^{Nxd}; Output: sample covariance matrix S
    return np.cov(X, rowvar=0, ddof=0)

def model2cov(model, N):
    (true_mB, true_mO) = get_first_model(model)
    true_B = draw_coefficient_matrix(true_mB, 1.0)
    true_O = draw_masked_covariance_matrix(true_mO, .1)
    Y = draw_multivariate_data(true_B, true_O, N)
    return sample_covariance_matrix(Y)

def dim2cov(d, N):
    # Generate data from saturated model on d variables (using complete graph
    # of bidirected edges with parameters drawn according to var=.1)
    true_B = np.zeros((d,d))
    true_O = draw_covariance_matrix(d, .1)
    Y = draw_multivariate_data(true_B, true_O, N)
    return sample_covariance_matrix(Y)

def random_permutation_matrix(d):
    pi = np.random.permutation(d)
    P = np.zeros(shape=(d,d), dtype=bool)
    P[np.arange(d), pi] = True
    return P


def param_map(B, O):
    d = B.shape[0]
    # Sigma is the covariance matrix that (B,O) is mapped to by the
    # parameterization map; this matrix equals the inverse of
    # (I-B)' O^{-1} (I-B)
    Sigma = np.dot(np.linalg.pinv((np.eye(d) - B)),
                   np.dot(O, np.linalg.pinv((np.eye(d) - B).T)))
    return Sigma

def log_likelihood_ADMG(B, O, S, N):
    # Input: model parameters B and O, empirical covariance S and sample size N;
    # Output: the log likelihood.

    d = B.shape[0]
    Sigma = param_map(B, O)
    loglik = -0.5 * N * (np.log(np.linalg.det(O))
                         + np.trace(np.dot(np.linalg.pinv(Sigma), S)))
    # Loglik can be nan, -inf, or inf in weird cases!
    if not np.isfinite(loglik):
        loglik = -np.inf
    elif np.amin(np.linalg.eigvals(O)) < -1e-5:
        print("WARNING: log_likelihood_ADMG computed a finite value, but O parameter was not positive (semi)definite! [replaced return value by -inf]")
        loglik = -np.inf
    return loglik

def model_dimension(model):
    # Also counts variance parameters. Whether the dimension should be counted
    # like this for infinite-to-one models seems to be an question, but we can
    # avoid it because each of our model classes contains also finite-to-one
    # models.
    (mB, mO) = model
    dim = num_edges(model) + mB.shape[0]
    return dim

def BIC_score(loglik, dim, N):
    # Smaller is better
    bic = -2 * loglik + dim * np.log(N)
    return bic

def ic2prob(ic):
    # Convert vector of (B)IC scores (-2 log probs) to a probability
    # distribution in a numerically accurate way
    lo = min(ic)
    w = np.exp(-0.5 * (ic - lo))
    return w / np.sum(w)

def partial_correlation(S, a, b, i):
    # currently for one 'given'
    Corr = correlation_matrix(S)
    num = Corr[a,b] - Corr[a,i] * Corr[b,i]
    denom = math.sqrt((1 - Corr[a,i] ** 2) * (1 - Corr[b,i] ** 2))
    return num / denom

def report_solution(B, O, mB, mO, S, N, always=False):
    d = mB.shape[0]
    # gradient: Drton Prop 8
    Oinv = np.linalg.pinv(O)
    grB = mB * Oinv.dot(np.eye(d) - B).dot(S)
    grO = mO * (Oinv - Oinv.dot(np.eye(d) - B).dot(S)
                .dot((np.eye(d) - B).T).dot(Oinv))
    flat_gradient = (np.amax(np.fabs(grB)) < 1e-4
                     and np.amax(np.fabs(grO)) < 1e-4)

    # Hessian: Drton Prop 10 (excluding incorrect factor 1/2 there)
    m_dir = num_directed_edges(mB)
    m_bidir = num_bidirected_edges(mO)
    m = m_dir + m_bidir + d
    P = np.zeros((d*d, m_dir))
    Q = np.zeros((d*d, m_bidir + d))
    k1 = 0
    k2 = 0
    for i in range(d):
        for j in range(d):
            if mB[i,j]:
                P[d*i+j, k1] = 1
                k1 += 1
            if mO[i,j] and i <= j:
                Q[d*i+j, k2] = 1
                k2 += 1
    H = np.zeros((m, m))
    Sigma = param_map(B, O)
    H[0:m_dir, 0:m_dir] = (P.T).dot(np.kron(Sigma, Oinv,)).dot(P)
    H[0:m_dir, m_dir:m] = (P.T).dot(np.kron(np.linalg.inv(np.eye(d) - B),
                                            Oinv)).dot(Q)
    H[m_dir:m, 0:m_dir] = H[0:m_dir, m_dir:m].T
    #H[m_dir:m, m_dir:m] = .5 * (Q.T).dot(np.kron(Oinv, Oinv)).dot(Q)
    # Factor .5 is error in Drton (compare to proof)
    H[m_dir:m, m_dir:m] = (Q.T).dot(np.kron(Oinv, Oinv)).dot(Q)
    #eigs = list(np.linalg.eigvals(H))
    #eigs.sort()
    eigs, eigvecs = np.linalg.eigh(H) # eigv are sorted in ascending order

    if (always or not flat_gradient or eigs[0] < 1e-5):
        print_model((mB, mO))
        print("B:")
        print(B)
        print("O:")
        print(O)
        Corr = correlation_matrix(O)
        print("rho:")
        print(Corr)
        O_eigs = list(np.linalg.eigvals(O))
        O_eigs.sort()
        if O_eigs[0] < 1e-5:
            if O_eigs[0] < -1e-5:
                print("O is not positive definite or even semidefinite!")
            else:
                print("O is not positive definite (it is close to singular)")
            print("  eigenvalues of O:", O_eigs)
        print("log-likelihood:", log_likelihood_ADMG(B, O, S, N))
        if not flat_gradient:
            if np.amax(np.fabs(grB)) > 1e-1 or np.amax(np.fabs(grO)) > 1e-1:
                print("Gradient is clearly not flat!")
            else:
                print("Gradient is not entirely flat")
            print("  Max absolute value in gradients (of B and O):",
                  "{0:.3g}".format(np.amax(np.fabs(grB))),
                  "{0:.3g}".format(np.amax(np.fabs(grO))))
        if eigs[0] < 1e-5:
            if eigs[0] < -1e-1:
                print("Hessian has clearly negative eigenvalue(s)!")
            elif eigs[0] < -1e-5:
                print("Hessian has negative eigenvalue(s)")
            else:
                print("Hessian has eigenvalue(s) close to 0: inconclusive as to minimum/saddle point")
            print("  Sorted eigenvalues of Hessian:", eigs)
        if flat_gradient and eigs[0] < -1e-5:
            delta = 1e-3
            B_step = P.dot(eigvecs[0:m_dir, 0])
            O_step = Q.dot(eigvecs[m_dir:m, 0])
            B_step = B_step.reshape((d,d))
            O_step = O_step.reshape((d,d))
            print("Solution in direction of first eigenvector:")
            print("B_step:")
            print(B_step)
            print("O_step:")
            print(O_step)
            print("log-likelihood:",
                  log_likelihood_ADMG(B + delta*B_step, O + delta*O_step,
                                      S, N))
            print("log-likelihood in opposite direction:",
                  log_likelihood_ADMG(B - delta*B_step, O - delta*O_step,
                                      S, N))
            xs = np.concatenate((-(10**np.arange(0.0, -6.0, -0.25)),
                                np.array([0.0]),
                                10**np.arange(-6.0, 0.0, 0.25)))
            ys = np.empty_like(xs)
            for i, x in enumerate(xs):
                ys[i] = log_likelihood_ADMG(B - x*B_step, O - x*O_step, S, N)
            f1 = plt.figure()
            af1 = f1.add_subplot(111) # 1x1
            af1.plot(xs, ys, '-b')
            plt.show(block=False)


def RICF(model, Y=None, S=None, N=None,
         epsilon=1e-6, conv_ref=None, iteration_limit=5000, damping=1.0,
         random_init=False, B_init=None, O_init=None,
         verbose=0):
    """Find the maximum likelihood parameters of a BAP (bow-free ADMG) model.

    Implementation of the Residual Iterative Conditional Fitting algorithm
    (Drton, Eichler and Richardson 2009).

    Input: Binary model matrices mB and mO (mB[i,j] means j-->i; mO[i,j] means
    i<->j), and data vector Y\in R^{N x d}; instead of Y, sample covariance
    matrix S and sample size N can be passed. If an empty list bound to some
    name is passed as conv_ref, it will be changed into a one-element list
    (containing the number of iterations) iff the algorithm converged.
    Output: matrices B, O containing the parameters of the model.
    """

    # TODO: further optimization mentioned in RICF paper (top of P2342):
    # consider just dis(i) and its parents when inverting Omega

    model = get_first_model(model)
    (mB, mO) = model
    if has_cycles(mB):
        raise ValueError("RICF() called for cyclic model")
    if S is None:
        #print("Determining S from Y")
        S = sample_covariance_matrix(Y)
    if N is None:
        #print("Determining N from Y")
        N = Y.shape[1]
    if conv_ref is None:
        conv_ref = []

    d = mB.shape[0]
    if B_init is not None:
        B = B_init
    elif not random_init:
        # as in Drton's R code
        B = np.zeros((d,d))
    else:
        B = mB * np.random.randn(d, d)
    if O_init is not None:
        O = O_init
    elif not random_init:
        # as in Drton's R code
        O = np.diag(np.diag(S))
    else:
        O = draw_masked_covariance_matrix(mO, .1)

    if verbose:
        print() # don't print first message to the right of 'run #, model #'
        likelihood = []
        param_dist = []

    B_old = B.copy()
    O_old = O.copy()

    num_reported_Oinvs = 0

    conv = False
    iteration = 0

    pa_list = [[] for _ in range(d)]
    sp_list = [[] for _ in range(d)]
    mO_no_diag = np.logical_and(mO, np.logical_not(np.eye(d, dtype=bool)))
    for v in range(d):
        # * np.where returns list of arrays, one per dimension, containing
        #   the indices where condition is true
        #   (Here, there is only one dimension)
        # * with [0]: an array of just the major-dimension indices
        pa_list[v] = np.where(mB[v,:])[0]
        sp_list[v] = np.where(mO_no_diag[v,:])[0]

    visit_nodes = range(d)
    visit_nodes_subsequent_iterations = np.where(np.any(mO_no_diag, axis=0))[0]
    while not conv and iteration < iteration_limit:
        if iteration == 1:
            visit_nodes = visit_nodes_subsequent_iterations
        for v in visit_nodes:
            pa = pa_list[v]
            sp = sp_list[v]
            num_pa = len(pa)
            num_sp = len(sp)
            num_param = num_pa + num_sp

            report_iteration = verbose >= 2 and (iteration < 3) # or iteration % 1000 == 999)
            if report_iteration and v == 0:
                print()
                print("ITERATION", iteration)
                print(B)
                print(O)

            Oinv = O.copy()
            Oinv[v,:] = 0
            Oinv[:,v] = 0
            Oinv[v,v] = 1 # make it into a block matrix: blocks indep. inv.ble
            try:
                Oinv = np.linalg.inv(Oinv)
            except np.linalg.linalg.LinAlgError:
                if report_iteration:
                    print("inv failed, using pinv")
                try:
                    Oinv = np.linalg.pinv(Oinv)
                except np.linalg.linalg.LinAlgError:
                    if num_reported_Oinvs == 0:
                        print("pinv(Oinv) failed in iteration", iteration,
                              "; Oinv =")
                        print(Oinv)
                    num_reported_Oinvs += 1
            Oinv[v,v] = 0
            Zmult = Oinv[sp,:].dot(np.eye(d) - B)

            if (num_pa + num_sp) > 0:
                XX = np.empty((num_param, num_param))
                XX[:num_pa, :num_pa] = S[np.ix_(pa,pa)]
                XX[:num_pa, num_pa:] = np.dot(S[pa,:], Zmult.T)
                XX[num_pa:, :num_pa] = XX[:num_pa, num_pa:].T
                XX[num_pa:, num_pa:] = Zmult.dot(S).dot(Zmult.T)
                YX = np.empty((num_param, 1))
                YX[:num_pa, 0] = S[v, pa]
                YX[num_pa:, 0] = np.dot(S[v, :], Zmult.T)
                if report_iteration:
                    print("Oinv:")
                    print(Oinv)
                    print("Zmult:")
                    print(Zmult)
                    print("XX:")
                    print(XX)
                    print("YX:")
                    print(YX)
                try:
                    beta_hat = np.linalg.lstsq(XX, YX, rcond=1e-6)[0]
                except np.linalg.linalg.LinAlgError:
                    if verbose:
                        print("Exception caught: Singular matrix that even lstsq couldn't deal with.")
                        print("Model:")
                        print_model((mB, mO))
                        print("Covariance matrix:")
                        print(S)
                        print("Number of samples:", N)
                        print("Returning from RICF with unconverged result.")
                    return (B, O)

                # Updating the parameters
                B[v, pa] = beta_hat[:num_pa, 0]
                O[v, sp] = beta_hat[num_pa:, 0]
                O[sp, v] = beta_hat[num_pa:, 0]
                O[v,v] = (S[v,v] - (beta_hat.T).dot(YX)
                          + O[v,sp].dot(Oinv[np.ix_(sp,sp)]).dot(O[sp,v]))
            else:
                # v has no parents or spouses
                O[v,v] = S[v,v] # needed when using random_init

            if report_iteration:
                print("Visited node", v)
                print(B)
                print(O)

        dist = np.linalg.norm(B_old - B) + np.linalg.norm(O_old - O)
        B_old = B.copy()
        O_old = O.copy()

        if verbose and iteration >= iteration_limit - 2:
            likelihood.append(log_likelihood_ADMG(B, O, S, N))
            param_dist.append(dist)

        if (iteration > 0):
            if (dist < epsilon):
                conv=True
                conv_ref.append(iteration)
        iteration += 1

    if verbose:
        if not conv:
            print("RICF FAILED TO CONVERGE after", iteration, "iterations")
            print("Model:")
            print_model((mB, mO))
            #print("Covariance matrix:")
            #print(S)
            #print("Number of samples:", N)
            print("Log-likelihood was growing by",
                  likelihood[-1] - likelihood[-2], "in last iteration")
            print("Parameters were changing by",
                  param_dist[-1], "in last iteration")
        else:
            print("RICF converged after", iteration, "iterations")
        report_solution(B, O, mB, mO, S, N, verbose>=2)

    return (B, O)

def solve_parameters(model, S):
    # Attempts to find values for the parameters that give exactly the matrix S.
    # Follows proof of Thm 1 in FoygelDraismaDrton2012.
    # Found parameters can be used to initialize RICF.
    (mB, mO) = get_first_model(model)
    d = mB.shape[0]
    ordering = is_htc_identifiable(model, return_ordering=True)
    if not ordering:
        bows = np.logical_and(mB, mO)
        mO = np.logical_and(mO, np.logical_not(bows))
        model = (mB, mO)
        ordering = is_htc_identifiable(model, return_ordering=True)
    reachable = transitive_reflexive_closure(mB).T
    htr = np.dot(mO, reachable).astype(bool)
    solved_nodes = np.zeros(d, dtype=bool)

    #print_model(get_first_model(model))
    #print("ordering:", ordering)

    B = np.zeros((d,d)) # Note: Lambda in article is transpose of B here
    for i, v in enumerate(ordering):
        pa_v = np.where(mB[v,:])[0]
        if len(pa_v) > 0:
            allowed_nodes = np.logical_and(
                np.logical_or(solved_nodes, np.logical_not(htr[v,:])),
                np.logical_not(mO[v,:]))
            y_v = check_htc(model, allowed_nodes, v)
            #print("v =", v, "; pa_v =", pa_v, "; y_v =", y_v)
            # for computation of M, zero out rows of B not in htr(v)
            M = (np.eye(d) - B * htr[v,:].reshape((d,1)))[y_v,:]
            #print("htr(v) =", list(htr[v,:]))
            #print("M =")
            #print(M)
            A = np.dot(M, S[:,pa_v])
            b = np.dot(M, S[:,v])
            x = np.linalg.lstsq(A, b, rcond=1e-6)[0]
            B[v,pa_v] = x
            #print("B =")
            #print(B)
        solved_nodes[v] = True
    O = mO * (np.eye(d) - B).dot(S).dot(np.eye(d) - B.T)
    return (B, O)

def RICF_wrapper(mode, model, Y=None, S=None, N=None,
                 epsilon=1e-6, conv_ref=None, iteration_limit=5000, damping=1.0,
                 verbose=0):
    # Run RICF with some predefined choice of B_init, O_init and random_init.
    # Modes 0-3 are special; rest is random restarts.
    (mB, mO) = get_first_model(model)
    if S is None:
        S = sample_covariance_matrix(Y)
    if N is None:
        N = Y.shape[1]
    if mode == 1:
        # call RICF without special settings
        (B, O) = RICF((mB, mO), S=S, N=N, epsilon=epsilon, conv_ref=conv_ref,
                      iteration_limit=iteration_limit, damping=damping,
                      verbose=verbose)
    elif mode < 4:
        if mode == 0:
            # Try to solve for parameter values exactly, and initialize RICF
            # with those values
            (B_init, O_init) = solve_parameters((mB, mO), S)
        # initialize with simpler, bow-free model
        elif mode == 2:
            # replace bows by just directed edges
            bows = np.logical_and(mB, mO)
            mOred = np.logical_and(mO, np.logical_not(np.logical_or(bows, bows.T)))
            (B_init, O_init) = RICF((mB, mOred), S=S, N=N, epsilon=epsilon,
                                    iteration_limit=iteration_limit,
                                    damping=damping)
        elif mode == 3:
            # replace bows by just bidirected edges
            bows = np.logical_and(mB, mO)
            mBred = np.logical_and(mB, np.logical_not(bows))
            (B_init, O_init) = RICF((mBred, mO), S=S, N=N, epsilon=epsilon,
                                    iteration_limit=iteration_limit,
                                    damping=damping)
        B, O = RICF((mB, mO), S=S, N=N,
                    epsilon=epsilon, conv_ref=conv_ref,
                    iteration_limit=iteration_limit, damping=damping,
                    B_init=B_init, O_init=O_init,
                    verbose=verbose)
    else:
        # random restart
        B, O = RICF((mB, mO), S=S, N=N,
                      epsilon=epsilon, conv_ref=conv_ref,
                      iteration_limit=iteration_limit, damping=damping,
                      random_init=True,
                      verbose=verbose)
    return (B, O)

def fit_and_compute_loglik(model_class, S, N, epsilon, verbose=0):
    # Works best of model_class is ordered by preferred_models_first.
    # Always use initialization mode 2 (i.e. replace bows by directed
    # edges); try different graphs until one converges (approach based
    # on empirical results)
    for model in iter_models(model_class):
        conv = []
        (B_hat, O_hat) = RICF_wrapper(2, model, S=S, N=N,
                                      epsilon=epsilon, conv_ref=conv,
                                      verbose=verbose)
        if conv:
            loglik = log_likelihood_ADMG(B_hat, O_hat, S, N)
            if np.isfinite(loglik):
                break # no need to try other models in this model class
            else:
                conv = False
        if not conv:
            loglik = np.NAN
    return loglik

def classify_prepare_compute(filename, d, allow_cycles=False):
    if allow_cycles:
        models = generate_all_models(d)
    else:
        models = generate_all_ADMG_models(d)
    clusters = theoretical_clusters(models)
    clusters = sort_models(clusters)
    if d == 4 and not allow_cycles:
        # Manually merge the classes not found by the theorem
        # TODO: this code will completely break if the numbering changes for
        # any reason
        clusters[379] += clusters[397] + clusters[399]
        clusters[380] += clusters[403] + clusters[405]
        clusters[381] += clusters[391] + clusters[407]
        clusters[382] += clusters[406] + clusters[412]
        clusters[383] += clusters[400] + clusters[410]
        clusters[384] += clusters[396] + clusters[402]
        clusters[385] += clusters[398] + clusters[409]
        clusters[386] += clusters[394] + clusters[401]
        clusters[387] += clusters[408] + clusters[414]
        clusters[388] += clusters[393] + clusters[395]
        clusters[389] += clusters[404] + clusters[411]
        clusters[390] += clusters[392] + clusters[413]
        del clusters[391:415]
        clusters[247] += clusters[253]
        clusters[248] += clusters[257]
        clusters[249] += clusters[250]
        clusters[250] = clusters[254] + clusters[256]
        clusters[251] += clusters[258]
        clusters[252] += clusters[255]
        #clusters[247], clusters[248] = clusters[248], clusters[247]
        del clusters[253:259]
        clusters = sort_models(clusters) # moves C247 to front of its iso's
    iso, aut = find_isomorphisms(clusters)
    clusters = [preferred_models_first(cluster) for cluster in clusters]
    data = dict(model_classes=clusters, iso=iso, aut=aut)

    # To view the first model in each cluster:
    #clusters = [get_first_model(cluster) for cluster in clusters]
    #with open("tex/table.tex", "w") as tex_file:
    #    print(tex_file_begin(), file=tex_file)
    #    print_model_clusters(clusters, nesting=0, check_isomorphisms=True,
    #                         tex_file=tex_file)
    #    print(tex_file_end(), file=tex_file)

    model_index = {}
    for i, model_class in enumerate(clusters):
        for model in iter_models(model_class):
            # TODO: conversion to bool should happen earlier
            (mB, mO) = model
            model_bool = (mB.astype(bool), mO.astype(bool))
            model_index[hash_model(model_bool)] = i
            #model_index[hash_model(model)] = i
    data['model_index'] = model_index

    n = len(clusters)
    model_type = np.zeros(n, dtype=int)
    for i, model_class in enumerate(clusters):
        is_DAG = False
        is_MAG = False
        is_BAP = False
        is_ADMG = False
        for model in iter_models(model_class):
            (mB, mO) = model
            if not has_cycles(mB):
                is_ADMG = True
                if num_bidirected_edges(mO) == 0:
                    is_DAG = True
                    break
                if is_maximal_ancestral(model):
                    is_MAG = True
                if is_bow_free(model):
                    is_BAP = True
        if is_DAG:
            model_type[i] = 0
        elif is_MAG:
            model_type[i] = 1
        elif is_BAP:
            model_type[i] = 2
        elif is_ADMG:
            model_type[i] = 3
        else:
            model_type[i] = 4
        data['model_type'] = model_type

    with open(filename, "wb") as pickle_file:
        pickle.dump(data, pickle_file, pickle.HIGHEST_PROTOCOL)
    return data

def classify_prepare(d, allow_cycles=False):
    # Compute the data structures that classify needs (in particular, the
    # graphs organized into equivalence classes). (Currently just clusters, but
    # may become more later.)
    filename = ("model_classes_{0}_{1}.pickle"
                .format(d, 'cyc' if allow_cycles else 'acyc'))
    try:
        pickle_file = open(filename, "rb")
        data = pickle.load(pickle_file)
    except IOError:
        print("Computing classify_data; will store it in {0}".format(filename))
        data = classify_prepare_compute(filename, d, allow_cycles)
    else:
        pickle_file.close()

    return data

def classify(classify_data,
             Y=None, S=None, N=None, epsilon=1e-6,
             get_scores_by_type=False, smooth=False, verbose=0):
    # Advanced implementation, taking into account model equivalences and
    # RICF convergence behaviour.
    # Input:
    # - classify_data should come from classify_prepare;
    # - Y,S,N: data; epsilon: for convergence criterion of RICF;
    # - get_scores_by_type: don't just determine best model, but best
    #   DAG, MAG, BAP, ADMG, and best model overall
    # - smooth: don't just determine best model, but an approximate posterior
    #   over the models (obtained by converting BIC scores to probabilities).
    #
    # - Initialize graphs with bows using the parameters for the graph with
    #   bows replaced by directed edges. For 4-node acyclic graphs, these
    #   reduced models are always DAGs and RICF is very fast on these, so the
    #   parameters are just recomputed rather than stored and reused.
    # - Use the saturated model as a lower bound on the minus log-likelihood
    #   of other models, so that we can decide their BIC score will be worse
    #   without fitting them. (Assumption: model_classes[-1] is saturated.)
    # - Not doing much against local minima: hopefully they mostly affect
    #   models that wouldn't fit well anyway. TODO investigate this further.
    if S is None:
        #print("Determining S from Y")
        S = sample_covariance_matrix(Y)
    if N is None:
        #print("Determining N from Y")
        N = Y.shape[1]

    model_classes = classify_data['model_classes']
    if smooth:
        print("classify(): smooth overrides get_scores_by_type")
        get_scores_by_type = False
    if get_scores_by_type:
        class_types = classify_data['model_type']
    num_c = len(model_classes)
    d = get_first_model(model_classes)[0].shape[0]
    loglik = np.empty(num_c)
    loglik[:] = np.NAN
    bic = loglik.copy()

    loglik[-1] = fit_and_compute_loglik(model_classes[-1], S, N, epsilon)
    bic[-1] = BIC_score(loglik[-1],
                        model_dimension(get_first_model(model_classes[-1])), N)
    loglik_upper_bound = loglik[-1]
    min_bic = bic[-1]
    if not get_scores_by_type:
        best_class = num_c - 1
    else:
        # still use min_bic to decide when search is done
        min_bic_by_type = min_bic * np.ones(5)
        best_class_by_type = (num_c - 1) * np.ones(5, dtype=int)
    for est_c, model_class in enumerate(model_classes):
        if np.isfinite(loglik[est_c]):
            continue
        dim = model_dimension(get_first_model(model_class))
        if not smooth:
            # If we only need the index(/indices) of the best model, we can
            # sometimes terminate the search early.
            # (TODO: there are various possibilities for doing something like
            # this even if smooth is True)
            optimistic_bic = BIC_score(loglik_upper_bound, dim, N)
            if optimistic_bic > min_bic:
                # this model (and the models after it, which have the same or
                # larger BIC penalty) can't possibly improve the best BIC score
                break
            if get_scores_by_type:
                class_type = class_types[est_c]
                if optimistic_bic > min_bic_by_type[class_type]:
                    # this model is of a type more general than just the DAGs,
                    # so has stronger competition, and improvement isn't
                    # possible
                    continue
        if verbose >= 2:
            print("\rmodel class =", est_c,
                  end='   ', file=sys.stderr)
        loglik[est_c] = fit_and_compute_loglik(model_class, S, N, epsilon)
        bic[est_c] = BIC_score(loglik[est_c], dim, N)
        if not get_scores_by_type:
            if bic[est_c] < min_bic:
                min_bic = bic[est_c]
                best_class = est_c
        else:
            for i in range(class_type, 5):
                if bic[est_c] < min_bic_by_type[i]:
                    min_bic_by_type[i] = bic[est_c]
                    best_class_by_type[i] = est_c
                    if i == 0:
                        min_bic = bic[est_c]
    if smooth:
        return ic2prob(bic)
    elif not get_scores_by_type:
        if verbose >= 3:
            print("Best is", best_class)
        return best_class
    else:
        if verbose >= 3:
            print("Best are", best_class_by_type)
        return best_class_by_type

def list_top_models(models, bic_prob):
    num_m = len(models)
    inds = range(num_m)
    inds.sort(key=lambda ind: bic_prob[ind], reverse=True)
    rem_prob = 1.0
    top_prob = bic_prob[inds[0]]
    for ii in range(num_m):
        i = inds[ii]
        this_prob = bic_prob[i]
        if this_prob < .01 * top_prob:
            print("Other models have total posterior", rem_prob)
            break
        rem_prob -= this_prob
        print("Model class", i, "has posterior", this_prob)
        print_model_class(models[i])

def print_loglik_comparison_table(cluster_loglik):
    #num_clusters = len(clusters)
    num_clusters = len(cluster_loglik)
    num_runs = len(cluster_loglik[0])
    print("Comparison of cluster log-likelihoods:")
    print("   ", end='')
    for c2 in range(num_clusters):
        print(" {0:>3}".format(c2), end='')
    print()
    for c1 in range(num_clusters):
        print("{0:>3}".format(c1), end='')
        for c2 in range(num_clusters):
            # TODO: split out function for converting loglik-vector to
            # string showing comparison symbols
            num_lt = 0
            num_gt = 0
            num_eq = 0
            num_nan = 0
            for i in range(num_runs):
                if (not np.isnan(cluster_loglik[c1][i])
                    and not np.isnan(cluster_loglik[c2][i])):
                    d = cluster_loglik[c1][i] - cluster_loglik[c2][i]
                    if d < -.1:
                        num_lt += 1
                    elif d > .1:
                        num_gt += 1
                    else:
                        num_eq += 1
                else:
                    num_nan += 1
            if c1 == c2:
                print("  = ", end='')
            elif num_lt > num_eq and num_gt > num_eq:
                print("  . ", end='')
            elif num_lt >= num_runs * 4 / 5 and num_gt == 0:
                print("  < ", end='')
            elif num_gt >= num_runs * 4 / 5 and num_lt == 0:
                print("  > ", end='')
            else:
                # try to show the most useful information in 2 or 3 symbols
                symbols = " ~<=>? "
                vec = [0, 0, num_lt, num_eq, num_gt, num_nan, 0]
                num_max = max(vec)
                for i, val in enumerate(vec):
                    if val >= max(num_max * 4 / 5, 1):
                        vec[i] = 2 # usually (the majority, or close to it)
                    elif val >= max(num_max / 3, 1):
                        vec[i] = 1 # occasionaly
                    else:
                        vec[i] = 0 # rarely: ignored
                if vec[2] > 0 and vec[4] == vec[2]:
                    # replace (< and >) by ~
                    # (it's ok if this leaves just one symbol: no confusion)
                    vec[1] = vec[2]
                    vec[2] = 0
                    vec[4] = 0
                # Leave out the symbols whose frequency is 'occasionally';
                # leave out ?'s first
                if sum(vec) > 3 and vec[5] == 1:
                    vec[5] = 0
                if sum(vec) > 3:
                    for i in range(5):
                        if vec[i] == 1:
                            vec[i] -= 1
                # Display 'usually' symbols once instead of twice, but always
                # display at least two symbols to prevent confusion with
                # one-symbol cases;
                # again start reducing ?'s
                if sum(vec) > 3 and vec[5] == 2:
                    vec[5] = 1
                if sum(vec) > 3:
                    for i in range(5):
                        if vec[i] == 2:
                            vec[i] -= 1
                # sum should be 2 or 3 now; add spaces to make it 4 symbols
                if sum(vec) == 1:
                    vec[6] = 1
                vec[0] = 4 - sum(vec)
                s = ""
                for i in range(len(vec)):
                    s += vec[i] * symbols[i]
                print(s, end='')
        print()

def auto_cluster_iterative(models, num_runs, loglik, N, epsilon, verbose=False):
    assign_all = True # False: models with too many NaNs get left out
    max_restarts = 6 #10
    matches_required = num_runs * 2 / 3 + 1 # quite conservative
    #matches_required = num_runs # for testing purposes
    loglik_tolerance = .1

    num_m = len(models)
    d = get_first_model(models)[0].shape[0]
    num_convs = np.zeros(num_m, dtype=int)
    data_S = num_runs*[None]
    for run in range(num_runs):
        #gen_m = num_m - 1
        #S = model2cov(get_first_model(models[gen_m]), N)
        data_S[run] = dim2cov(d, N)
    store_fits = False
    if store_fits:
        fit = num_m*[num_runs*[None]]
    for run in range(num_runs):
        for est_m, est_model in enumerate(models):
            print("\rrun =", run, "\tmodel =", est_m,
                  end='   ', file=sys.stderr)
            conv = []
            (B_hat, O_hat) = RICF_wrapper(0, est_model, S=data_S[run], N=N,
                                          epsilon=epsilon, conv_ref=conv,
                                          verbose=verbose)
            if store_fits:
                fit[est_m][run] = (B_hat, O_hat)
            if conv:
                loglik[est_m, run] = log_likelihood_ADMG(B_hat, O_hat,
                                                         data_S[run], N)
                if not np.isfinite(loglik[est_m, run]):
                    conv = False
            if conv:
                num_convs[est_m] += 1
            else:
                loglik[est_m, run] = np.NAN
                # could also try to use inequality information in loglik
            #print('\r', end='', file=sys.stderr)
    print()

    clusters = []
    cluster_loglik = []
    cluster_reliable = []
    num_runs_matched_counter = collections.Counter()
    num_restarts = np.zeros((num_m, num_runs), dtype=int)
    loglik_before_restarts = loglik.copy()
    # Assign best convergers first, to know each cluster's logliks sooner
    #inds = range(num_m)
    #inds.sort(key=lambda ind: -num_convs[ind])
    model_queue = [(-num_convs[m], m) for m in range(num_m)]
    heapq.heapify(model_queue) # min heap: prefers more valid runs
    #for m in inds:
    while model_queue:
        m = heapq.heappop(model_queue)[1]
        assigned = False
        assignable = True
        reliable = True # set to False if very close mismatch encountered
        almost_match_reported = False
        if (not assign_all and max_restarts == 0
            and num_convs[m] < matches_required):
            # Don't bother looking (and give just one message)
            # if nothing will match
            print("Model", m, "has only", num_convs[m],
                  "valid runs and can't be matched")
            assignable = False
        else:
            for c, cluster in enumerate(clusters):
                mc = cluster[0] # first model in cluster
                while True:
                    num_runs_matched = 0
                    num_runs_mismatched = 0
                    for i in range(num_runs):
                        if (not np.isnan(loglik[m,i])
                            and not np.isnan(cluster_loglik[c][i])):
                            delta = abs(loglik[m,i] - cluster_loglik[c][i])
                            if delta > loglik_tolerance:
                                num_runs_mismatched += 1
                            else:
                                num_runs_matched += 1
                    if (num_runs_mismatched == 0
                        and num_runs_matched >= matches_required):
                        # declare positive match
                        assign_here = True
                        break
                    if num_runs_mismatched > (num_runs - 1) / 5 + 1:
                        # too many differences: declare negative match
                        assign_here = False
                        break
                    # close match; might be resolved by restarts of RICF
                    best_score = (0, 0) # (type, tiebreaker);
                    # type 0 = nothing
                    best_action = None # (model, run) to do restart on
                    num_definite_nans = 0
                    for i in range(num_runs):
                        # first determine type of difference here (if any),
                        # then check if it is worth restarts
                        score_here = (0, 0)
                        nan_m_priority = 2
                        if num_convs[m] <= (num_runs - 1) / 5 + 1:
                            # If model m has very few non-nan results, first
                            # try to get more of those, otherwise the model
                            # will try to match with all other clusters.
                            # (must use optimistic tiebreaker here)
                            nan_m_priority = 4
                        # Optimistic version: hope to quickly find convergence
                        # somewhere
                        tiebreaker = (-num_restarts[m,i], -num_restarts[mc,i])
                        #tiebreaker = -num_restarts[mc,i] OR [m,i] [as act_here]
                        # Pessimistic version: hope to quickly exhaust the
                        # possibilities
                        #tiebreaker = (-i, -i)
                        if (np.isnan(loglik[m,i])
                            and np.isnan(cluster_loglik[c][i])):
                            # if one of the two will remain NaN, don't bother
                            if (num_restarts[m,i] < max_restarts
                                and num_restarts[mc,i] < max_restarts):
                                # type 1 = two NaNs
                                score_here = (1, tiebreaker[1])
                                action_here = (mc, i)
                            else:
                                num_definite_nans += 1
                        elif np.isnan(loglik[m,i]):
                            if num_restarts[m,i] < max_restarts:
                                # type 2(/4) = one NaN
                                score_here = (nan_m_priority, tiebreaker[0])
                                action_here = (m, i)
                            else:
                                num_definite_nans += 1
                        elif np.isnan(cluster_loglik[c][i]):
                            if num_restarts[mc,i] < max_restarts:
                                score_here = (2, tiebreaker[1])
                                action_here = (mc, i)
                            else:
                                num_definite_nans += 1
                        elif (loglik[m,i]
                              < cluster_loglik[c][i] - loglik_tolerance):
                            if num_restarts[m,i] < max_restarts:
                                # type 3 = different non-Nan values
                                # (always use 'pessimistic' tiebreaker here)
                                score_here = (3, -i)
                                action_here = (m, i)
                            else:
                                # type 5 = unresolvable difference
                                score_here = (5, 0)
                                action_here = None
                        elif (cluster_loglik[c][i]
                              < loglik[m,i] - loglik_tolerance):
                            if num_restarts[mc,i] < max_restarts:
                                score_here = (3, -i)
                                action_here = (mc, i)
                            else:
                                score_here = (5, 0)
                                action_here = None
                        # found a better difference?
                        if score_here > best_score:
                            best_score = score_here
                            best_action = action_here
                    if (not best_action
                        or num_runs - num_definite_nans < matches_required):
                        # no restarts available that can make this a match:
                        # give up
                        assign_here = False
                        break
                    # Try a restart
                    restart_m = best_action[0]
                    restart_run = best_action[1]
                    mode = num_restarts[restart_m, restart_run] + 1
                    num_restarts[restart_m, restart_run] = mode
                    print("(assigned = {0}/{1})"
                          .format(num_m - len(model_queue) - 1, num_m), end=' ')
                    print("Restart for model {0}, run {1} (try {2})"
                          .format(restart_m, restart_run,
                                  num_restarts[restart_m, restart_run]))
                    conv2 = []
                    (B2, O2) = RICF_wrapper(mode, models[restart_m],
                                            S=data_S[restart_run], N=N,
                                            epsilon=epsilon, conv_ref=conv2,
                                            verbose=verbose)
                    if conv2:
                        loglik2 = log_likelihood_ADMG(B2, O2,
                                                      data_S[restart_run], N)
                        if not np.isfinite(loglik2):
                            conv2 = False
                        elif loglik2 < loglik[restart_m,restart_run]:
                            # Treat worsened loglik the same as nonconvergence
                            conv2 = False
                    if conv2:
                        # RICF really converged and improved the loglik
                        if (np.isnan(loglik[restart_m,restart_run])
                            or loglik2 > loglik[restart_m,restart_run] + 1e-3):
                            print("Restart gave improvement in loglik, from",
                                  loglik[restart_m,restart_run], "to", loglik2)
                        if np.isnan(loglik[restart_m,restart_run]):
                            num_convs[restart_m] += 1
                        loglik[restart_m,restart_run] = loglik2
                        if store_fits:
                            fit[restart_m][restart_run] = (B2, O2)
                    if (restart_m == mc and len(cluster) > 1
                        and not np.isnan(loglik[mc,restart_run])
                        and not np.isnan(cluster_loglik[c][restart_run])
                        and abs(loglik[mc,restart_run]
                                - cluster_loglik[c][restart_run])
                        > loglik_tolerance):
                        # The restart changed the loglik of one member of
                        # a multi-model cluster: break the cluster up
                        # Note: this messes up num_runs_matched_counter
                        print("Breaking up cluster", c)
                        for m2 in cluster:
                            if m2 != mc:
                                heapq.heappush(model_queue,
                                               (-num_convs[m2], m2))
                        clusters[c] = [mc] # So clusters actually changes
                        cluster = clusters[c] # So later write to cluster works
                        cluster_loglik[c] = loglik[mc,:].tolist()
                    elif (restart_m == mc
                          and not np.isnan(loglik[mc,restart_run])
                          and (np.isnan(cluster_loglik[c][restart_run])
                               or len(cluster) == 1)):
                        # A change in loglik that doesn't require breaking up
                        # is recorded in cluster_loglik unless it's possible
                        # that cluster_loglik was determined by another model
                        cluster_loglik[c][restart_run] = loglik[mc,restart_run]
                    # Now return to top of while loop to see if things changed

                if False and not assign_here and num_runs_mismatched <= num_runs_matched and cluster_reliable[c]:
                    # OLD; TODO: maybe do this at cluster level, afterwards?
                    reliable = False
                    almost_match_reported = True
                    print("Not assigning model", m, "to cluster", c,
                          "based on", num_runs_mismatched, "mismatches out of",
                          num_runs_mismatched + num_runs_matched)
                    print_model(get_first_model(models[m]))
                    for i in range(num_runs):
                        if (not np.isnan(loglik[m,i])
                            and not np.isnan(cluster_loglik[c][i])):
                            delta = abs(loglik[m,i] - cluster_loglik[c][i])
                            if delta > loglik_tolerance:
                                print("Fitted parameter values in mismatched run ", i, ":", sep='')
                                print(fit[m][i][0])
                                print(fit[m][i][1])
                                print("det(O) =", np.linalg.det(fit[m][i][1]))
                if assign_here:
                    if num_runs_matched < matches_required:
                        # OLD: won't happen with new adaptive restart code
                        print("Model", m, "might belong to cluster", c,
                              "but I have only", num_runs_matched, "runs")
                        if not assign_all:
                            assignable = False
                        else:
                            almost_match_reported = True
                            # reliable = False # TODO: is this the right way to limit "might-belong" spam?
                    else:
                        cluster.append(m) # add to this cluster
                        if i in range(num_runs):
                            if (np.isnan(cluster_loglik[c][i])
                                and not np.isnan(loglik[m,i])):
                                cluster_loglik[c][i] = loglik[m,i]
                        num_runs_matched_counter[num_runs_matched] += 1
                        assigned = True
                        if almost_match_reported:
                            print("Assigned model", m, "to cluster", c)
                        break # don't try other clusters
        if not assigned:
            if assignable:
                clusters.append([m]) # create new cluster
                cluster_loglik.append(loglik[m,:].tolist())
                cluster_reliable.append(reliable)
                if almost_match_reported:
                    print("Model", m, "started new cluster", len(clusters) - 1)
            else:
                # Do not assign this model to any cluster, not even to a new one
                # (happens if assign_all is False and there is a lot of
                # uncertainty about which cluster m might belong in)
                print_model(get_first_model(models[m]))

    for key, value in sorted(num_runs_matched_counter.iteritems()):
        print(value, "matches made based on", key, "runs")

    if True:
        data = dict(models=models, data_S=data_S, clusters=clusters,
                    loglik=loglik, num_restarts=num_restarts)
        if store_fits:
            data['fit'] = fit
        with open("clusters.pickle", "wb") as pickle_file:
            pickle.dump(data, pickle_file, pickle.HIGHEST_PROTOCOL)
            # Load with:
            #data = pickle.load(pickle_file)

    print("Cluster log-likelihoods:")
    print(np.array(cluster_loglik).T)

    print_loglik_comparison_table(cluster_loglik)

    #loglik_max = np.nanmax(loglik, axis=0)
    loglik_max = np.zeros(num_runs)
    for i in range(num_runs):
        # log-likelihood of saturated model with all bidirected edges
        loglik_max[i] = log_likelihood_ADMG(np.zeros((d,d)), data_S[i],
                                            data_S[i], N)
    #print(loglik_max)
    print("Clustr:\tModel:\tOld relation to max - #restarts - new relation, per run")
    for c, cluster in enumerate(clusters):
        print(c, end='')
        for m in cluster:
            num_nan = 0
            num_ineq = 0
            print("\t{0}\t".format(m), end='')
            for i in range(num_runs):
                #print(loglik_max[i] - loglik_before_restarts[m,i])
                if np.isnan(loglik_before_restarts[m,i]):
                    sym_before = '?'
                elif abs(loglik_max[i] - loglik_before_restarts[m,i]) < 1e-3:
                    sym_before = '='
                elif (loglik_max[i] - loglik_before_restarts[m,i]
                    < loglik_tolerance):
                    sym_before = '~'
                else:
                    sym_before = '<'
                if np.isnan(loglik[m,i]):
                    sym_now = '?'
                    num_nan += 1
                elif abs(loglik_max[i] - loglik[m,i]) < 1e-3:
                    sym_now = '='
                elif loglik_max[i] - loglik[m,i] < loglik_tolerance:
                    sym_now = '~'
                else:
                    sym_now = '<'
                    num_ineq += 1
                print("{0}{1}{2} "
                      .format(sym_before, num_restarts[m,i], sym_now),
                      end='')
            print("({0}x<, {1}x?)".format(num_ineq, num_nan))

    # TODO: also cluster together previously determined 'clusters' that compare
    # as '==', '==?', '<==', '==>', or '~==' (including transitively)
    return clusters

def auto_cluster(models, num_runs=None, N=10000, epsilon=1e-6, verbose=False):
    if num_runs is None:
        num_runs = 10 # TODO larger default for large number of models (O(log)?)
    num_m = len(models)
    loglik = np.empty((num_m, num_runs))
    loglik[:] = np.NAN
    indices = range(num_m) # Python 3: list(range(num_m))
    clusters = auto_cluster_iterative(models, num_runs, loglik,
                                      N, epsilon, verbose=verbose)
    for i, cluster in enumerate(clusters):
        num_c = len(cluster)
        if num_c == 1:
            continue
        loglik_cluster = loglik[cluster,:]
        loglik_cluster -= np.mean(loglik_cluster, axis=0, keepdims=True)
        sstd = np.std(loglik_cluster, ddof=num_runs)
        if sstd > 1e-3:
            print("Warning: log-likelihoods in cluster", i, "have sample std dev", sstd)
    #print(clusters)
    print("Empirically grouped models into", len(clusters), "clusters")
    return [[models[ind] for ind in cluster] for cluster in clusters]


def test_RICF_initialization_modes(models, num_runs=None, num_restarts=None,
                                   N=10000, epsilon=1e-6, verbose=False):
    print("testing RICF with different initialization modes for the following {0} models:".format(len(models)))
    print_models(models, show_subclasses=True)
    if num_runs is None:
        num_runs = 10 # TODO larger default for large number of models (O(log)?)
    num_m = len(models)
    if num_restarts is None:
        num_restarts = 10 # First four are special, rest is random restarts
    loglik = np.empty((num_m, num_runs, num_restarts))
    loglik[:] = np.NAN
    time_used = np.zeros((num_m, num_runs, num_restarts))
    loglik_max = np.empty(num_runs)
    loglik_max[:] = np.NINF
    loglik_tolerance = .1
    d = get_first_model(models)[0].shape[0]
    data_S = num_runs*[None]
    for run in range(num_runs):
        #gen_m = num_m - 1
        #S = model2cov(get_first_model(models[gen_m]), N)
        data_S[run] = dim2cov(d, N)
    for run in range(num_runs):
        for est_m, est_model in enumerate(models):
            for mode in range(num_restarts):
                print("\rrun =", run, "\tmodel =", est_m,
                      "\tmode =", mode,
                      end='   ', file=sys.stderr)
                conv = []
                time_before = time.clock()
                (B_hat, O_hat) = RICF_wrapper(mode, est_model,
                                              S=data_S[run], N=N,
                                              epsilon=epsilon, conv_ref=conv,
                                              verbose=verbose)
                time_after = time.clock()
                time_used[est_m, run, mode] = time_after - time_before
                if conv:
                    loglik[est_m, run, mode] = log_likelihood_ADMG(B_hat, O_hat,
                                                                   data_S[run], N)
                    if np.isfinite(loglik[est_m, run, mode]):
                        loglik_max[run] = max(loglik[est_m, run, mode],
                                              loglik_max[run])
                    else:
                        conv = False
                if not conv:
                    loglik[est_m, run, mode] = np.NAN
    print()

    print("model:\tmode:\tresult per run:")
    total_ineq = 0
    total_approx = 0
    for m, model in enumerate(models):
        print(m, end='')
        for mode in range(num_restarts):
            if mode <= 4:
                num_nan = 0
                num_ineq = 0
                total_time_used = 0
                num_times = 0
                total_time_used_eqs = 0
                num_eq = 0
            print("\t{0}\t".format(mode if mode < 4 else 'R'), end='')
            for i in range(num_runs):
                if np.isnan(loglik[m,i,mode]):
                    sym = '?'
                    num_nan += 1
                elif abs(loglik_max[i] - loglik[m,i,mode]) < 1e-3:
                    sym = '='
                    total_time_used_eqs += time_used[m,i,mode]
                    num_eq += 1
                elif loglik_max[i] - loglik[m,i,mode] < loglik_tolerance:
                    sym = '~'
                    total_approx += 1
                else:
                    sym = '<'
                    num_ineq += 1
                    total_ineq += 1
                total_time_used += time_used[m,i,mode]
                num_times += 1
                print("{0} ".format(sym), end='')
            if mode < 4 or mode == num_restarts - 1:
                avg_time_used = total_time_used / num_times
                avg_time_used_eqs = total_time_used_eqs / num_eq
                print("({0}x<, {1}x?; avg. time {2:.3f}, for =s {3:.3f})"
                      .format(num_ineq, num_nan,
                              avg_time_used, avg_time_used_eqs))
            else:
                print() # only final 'R' shows aggregated info
        print()
    if total_ineq > 0 or total_approx > 0:
        print("Detected evidence of multimodal likelihood: {0}x<, {1}x~"
              .format(total_ineq, total_approx))


def generate_models():
    # Generate lists of models (edit this code to change which). Also outputs
    # list of automorphisms, if nontrivial (otherwise aut == True).
    skel = None
    add_triangle = False
    add_allbidir = False
    use_filter = False
    allow_bows = True # True: any number; 1: at most one; False: none
    allow_cycles = False
    d = 4 #5

    #skel = np.zeros((d, d))
    ### Constructions for any number of nodes:
    #skel[0,:] = True # star
    #skel = np.ones((d, d)) # complete graph
    #skel = np.ones((d, d)); skel[-2,-1] = skel[-1,-2] = False # Kd-e
    ### Three-node skeletons:
    #skel[1,:] = add_triangle = True # path
    ### Four-node skeletons:
    #skel[0,1] = skel[1,2] = skel[2,3] = True # path
    #skel[0,1] = skel[1,2] = skel[2,3] = skel[3,0] = True # square
    #skel[0,1] = skel[1,2] = skel[2,0] = skel[2,3] = True # C3 + edge

    if allow_cycles:
        # ignore all other settings for now
        models = generate_all_models(d)
        aut = True
    elif skel is not None:
        if not allow_bows:
            models = generate_BAP_models_with_skeleton(skel)
        elif type(allow_bows) == int and allow_bows == 1:
            models = generate_ADMG_models_with_max_one_bow_and_skeleton(skel)
        else:
            models = generate_ADMG_models_with_skeleton(skel)
        skel = np.logical_or(skel, skel.T)
        aut = find_isomorphisms([(skel, skel)])[1][0]
    else:
        if not allow_bows:
            models = generate_all_BAP_models(d)
        elif type(allow_bows) == int and allow_bows == 1:
            models = generate_ADMG_models_with_max_one_bow(d)
        else:
            models = generate_all_ADMG_models(d)
        aut = True
    if add_triangle:
        mB = np.zeros((d, d))#, dtype=bool)
        mO = np.eye(d)#, dtype=bool)
        mB[1,0] = mB[2,1] = mB[2,0] = 1
        models.append((mB, mO))
    if add_allbidir:
        mB = np.zeros((d, d))
        mO = np.ones((d, d))
        models.append((mB, mO))

    #print(count_models(models))

    # Filter models
    if use_filter:
        models2 = []
        for (mB, mO) in models:
            #if mB[0,1] or mB[2,1] or mB[2,3] or mB[0,3]: # square
            #if not mB[0,1] and not mB[1,0] and not mO[0,1]: # require adj
            #if mB[0,1] and mO[0,1]: # forbid bow
            #if num_edges((mB, mO)) == 5:
            head = np.logical_or(mO, mB)
            #print_model((mB, mO))
            #print(list(np.logical_and(head[0:-2,-2], head[0:-2,-1])))
            if (list(np.logical_and(head[0:-2,-2], head[0:-2,-1]))
                == [False, False, False]): # v-structure pattern in Kd-e
                #and mB[1,0] and mB[4,1] and mO[0,4] and not mB[2,0] and not mB[3,1]): # extra test for 1 v
                models2.append((mB, mO))
        models = models2
        print(len(models), "models remained after applying filter")

    return (models, aut)

def main_test_RICF_convergence(clusters):
    # To compare RICF results for different (equiv.) models & modes:
    #test_cluster = clusters[247] + clusters[253] # tetrad & independence
    #test_cluster = clusters[415] # tetrad
    #test_cluster = clusters[380] + clusters[403] + clusters[405] # strange constraint
    #test_cluster = clusters[34] # saturated on three nodes
    #test_cluster = clusters[109] # seemingly unrelated regression: multimodal
    #test_cluster = [model for model in test_cluster if num_bows(model) <= 1]
    #test_cluster = [(mB,mO) for (mB,mO) in test_cluster if num_adjacencies((mB,mO)) <= 3 and num_bows((mB,mO)) <= 2]
    #test_cluster = [(mB,mO) for (mB,mO) in test_cluster if num_edges((mB,mO)) == 4]
    ###test_RICF_initialization_modes([clusters[109][0]], num_runs=1, num_restarts=6)
    ###return
    iso, aut = find_isomorphisms(clusters)
    for i, test_cluster in enumerate(clusters):
        if iso[i][0] != i:
            continue
        req_bidir = True
        req_bows = True
        for (mB, mO) in test_cluster:
            if num_bidirected_edges(mO) == 0:
                req_bidir = False
            if is_bow_free((mB, mO)):
                req_bows = False
        if req_bidir and not req_bows:
            #print("Will test:")
            #print_models(test_cluster, nesting=0)
            test_RICF_initialization_modes(test_cluster)
            #test_RICF_initialization_modes(test_cluster, num_runs=40, num_restarts=20)
    return

def main_sandbox():
    # Sandbox function try out various things

    models, aut = generate_models()
    if False:
        # Group models together that have the same skeleton and colliders:
        clusters = partition_models(models, skeleton_collider_bows_normalizer,
                                    verbose=True)
    else:
        # To group models together using Theorem 2:
        clusters = theoretical_clusters(models)

    # To further group clusters together if they show the same RICF results:
    #clusters = auto_cluster(clusters) #, verbose=True)

    #print_model_clusters(clusters, check_isomorphisms=aut)
    with open("tex/table.tex", "w") as tex_file:
        print(tex_file_begin(), file=tex_file)
        print_model_clusters(clusters, nesting=0, check_isomorphisms=aut,
                             tex_file=tex_file)
        print(tex_file_end(), file=tex_file)


def main():
    data = classify_prepare(4)
    clusters = data['model_classes']
    iso = data['iso']
    aut = True # print_model_clusters expects the automorphisms of the set as a whole, while data contains the automorphisms per list element

    # Filter: require that some nodes are intervention variables
    if False:
        num_interv = 1
        modified_clusters = []
        for cluster in clusters:
            modified_cluster = []
            for model in cluster:
                (mB, mO) = model
                if np.all(mO[0:num_interv,:][:,0:num_interv]):
                    if (not np.any(mB[0:num_interv,:])
                        and not np.any(mO[0:num_interv,:][:,num_interv:])):
                        modified_cluster.append(model)
            if modified_cluster:
                modified_clusters.append(modified_cluster)
        clusters = modified_clusters

    show_constraints = False # either show constraints or disambiguations
    show_disambiguations = True
    nest = 1 if show_disambiguations else 0
    #clusters = sort_models(clusters)

    if show_disambiguations:
        clusters = disambiguate_clusters(clusters)
        clusters = [sort_models(cluster) for cluster in clusters]

    annotation = None
    if show_constraints:
        # Precompute dictionary for translating Theorem-1-form constraints to
        # shorter constraint names
        constraint_names = create_constraint_dictionary(clusters)
        # Each line in the annotation is a set of algebraic constraints
        # describing the equivalence class. There may be multiple lines, for
        # different models in the class and for different choices of
        # HTC-identifying sets cY.
        annotation = [all_constraints_annotator(model, constraint_names)
                      for model in clusters]
    if not show_constraints and not show_disambiguations:
        annotation = ["  {0}".format(data['model_type'][i])
                      for i, model_class in enumerate(clusters)]

    with open("tex/table.tex", "w") as tex_file:
        print(tex_file_begin(), file=tex_file)
        print_model_clusters(clusters, nesting=nest, check_isomorphisms=aut,
                             tex_file=tex_file, annotation=annotation)
        print(tex_file_end(), file=tex_file)

    return

if __name__ == "__main__":
    main()
    print()
