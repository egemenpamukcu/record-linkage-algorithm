'''
Linking restaurant records in Zagat and Fodor's list using restaurant
names, cities, and street addresses.

Egemen Pamukcu

'''
import csv
import jellyfish
import pandas as pd

import util

def find_matches(output_filename, mu, lambda_, block_on_city=False):
    '''
    Put it all together: read the data and apply the record linkage
    algorithm to classify the potential matches.

    Inputs:
      output_filename (string): the name of the output file,
      mu (float) : the maximum false positive rate,
      lambda_ (float): the maximum false negative rate,
      block_on_city (boolean): indicates whether to block on the city or not.
    '''

    # Hard-coded filename
    zagat_filename = "data/zagat.csv"
    fodors_filename = "data/fodors.csv"
    known_links_filename = "data/known_links.csv"
    zagat = pd.read_csv(zagat_filename, index_col=0, dtype={'restaurant name': str,
                                                            'city': str,
                                                            'street address': str})
    fodors = pd.read_csv(fodors_filename, index_col=0, dtype={'restaurant name': str,
                                                             'city': str,
                                                             'street address': str})
    match = pd.read_csv(known_links_filename, names=['zagat', 'fodors'])
    unmatch = pd.read_csv('data/unmatch_pairs.csv', names=['zagat', 'fodors'])

    match_prob = compute_probabilities(match, zagat, fodors)
    unmatch_prob = compute_probabilities(unmatch, zagat,fodors)
    op = ordered_probabilities(match_prob, unmatch_prob)
    label_dict = put_labels(op, mu, lambda_)
    with open(output_filename, 'w') as csvfile:
        writer = csv.writer(csvfile)
        for i in zagat.index:
            for j in fodors.index:
                if block_on_city and zagat.loc[i, 'city'] != fodors.loc[j, 'city']:
                     continue
                writer.writerow((i, j, label_dict.get(gen_prob_tuple(i, j, zagat, fodors),
                                                      'possible match')))


def compute_probabilities(df, zagat, fodors):
    '''
    Computes relative frequency for each probability tuple generated using
    matched and unmatched training data.

    Inputs:
      df (DataFrame): has two columns corresponding to indices of each dataset,
      zagat (DataFrame) : dataset associated with the first column of df,
      fodors (DataFrame): dataset associated with the second column of df,

    Output:
      Dictionary with probability tuples as key and relative frequencies as value.
    '''
    probs = {}
    for i in range(len(df)):
        tpl = []
        z_i, f_i = df.iloc[i, :]
        tpl = gen_prob_tuple(z_i, f_i, zagat, fodors)
        probs[tpl] = probs.get(tpl, 0) + (1 / len(df))
    return probs

def ordered_probabilities(mp, up):
    '''
    Combines two probability dictionaries and orders them in a list by the
    probability of match.

    Inputs:
      mp (dict): the first probability dictionary,
      up (dict) : the second probability dictionary,

    Output:
      List of tuples with probability tuple, match probability and unmatch
      probability.
    '''
    sim_tpls = list((set(mp.keys()) | set(up.keys())))
    prob_tuples = [(sim, mp.get(sim, 0),
                    up.get(sim, 0)) for sim in sim_tpls]
    return(util.sort_prob_tuples(prob_tuples))


def put_labels(ordered_probs, mu, lambda_):
    '''
    Generates 'match', 'umatch' and 'possible match' labels for each
    probability tuple.

    Inputs:
      ordered_probs (list): list of tuples with probability tuple, match
        probability and unmatch probability.
      mu (float) : the maximum false positive rate,
      lambda_ (float): the maximum false negative rate,
    Output:
      Dictionary matching probability tuples with their likely match/unmatch
      conditions.
    '''
    u = 0
    m = 0
    labels = {}
    for i in range(len(ordered_probs)):
        if m + ordered_probs[-i - 1][1] <= lambda_:
            m += ordered_probs[-i - 1][1]
            if labels.get(ordered_probs[-i - 1][0]) != 'match':
                labels[ordered_probs[-i - 1][0]] = 'unmatch'
        else:
            m = lambda_
        if u + ordered_probs[i][2] <= mu:
            u += ordered_probs[i][2]
            labels[ordered_probs[i][0]] = 'match'
        else:
            u = mu
            labels[ordered_probs[i][0]] = labels.get(ordered_probs[i][0], 'possible match')
    return labels


def gen_prob_tuple(z_i, f_i, zagat, fodors):
    '''
    Creates a probability tuple such as (high, medium, low) given two indices
    and dataframes.

    Inputs:
      z_i (int): index of the first data frame,
      f_i (int): index of the second data frame,
      zagat (DataFrame): first data frame to be indexed,
      fodors (DataFrame): second data frame to be indexed

    Outputs:
      Tuple with the likelihood of match for each column of the given data frames.
    '''
    z = tuple(zagat.loc[z_i, :])
    f = tuple(fodors.loc[f_i, :])
    tpl = []
    for i in range(len(z)):
        sim = jellyfish.jaro_winkler_similarity(z[i], f[i])
        tpl.append(util.get_jw_category(sim))
    return(tuple(tpl))