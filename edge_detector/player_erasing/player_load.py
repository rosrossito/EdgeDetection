import numpy as numpy

from util import tools_IO


def load_players_labels_in_frame(filename_data):
    data_pos = tools_IO.load_mat(filename_data, numpy.chararray, ' ')
    pos = data_pos[:, 2:]
    return pos


def load_mat(filename, dtype=numpy.chararray, delim='\t', lines=None):
    mat = numpy.genfromtxt(filename, dtype=dtype, delimiter=delim)
    return mat
