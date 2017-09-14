import site
site.addsitedir('/Home/Desktop/mag/word_embeddings-master/word_embeddings/')
from src.disorder_parser import parse
import numpy as np
import keras.preprocessing.sequence as ker

# using reduced db for development purposes

# returns max length of a protein sequence from db
def get_max_size():
    max = 0
    for seq in parse(open('../disorder/trainDM').read()):
        tmp = seq.prot_size
        if(tmp > max):
            max = tmp
    return max

def representation_from_DM():
    representation = []
    for seq in parse(open('../disorder/trainDM').read()):
        representation.append([x[0] for x in seq.get_representation_and_disorder()])
    representation_array = np.asarray(representation)

    # padding used for achieving equal lengths for each sequence
    representation_array = ker.pad_sequences(representation_array, maxlen=get_max_size()-2, dtype = 'int32', padding='post', value = 10.)
    return representation_array

# returns disordered values in np array with dimension = max_size, 3
def disorder_from_DM():
    disorder = []
    dim1 = 0
    dim2 = 0
    for seq in parse(open('../disorder/trainDM').read()):
        disorder.append(np.zeros(shape=[get_max_size()-2,3]))
        for x in seq.get_representation_and_disorder():
            for dim3 in range(3):
                disorder[dim1][dim2,dim3] = x[1]
            dim2=dim2+1
        dim1=dim1+1
        dim2 = 0
    return np.asarray(disorder)

# same for test db
def representation_from_DM_test():
    representation = []
    for seq in parse(open('../disorder/testDM').read()):
        representation.append([x[0] for x in seq.get_representation_and_disorder()])
    representation_array = np.asarray(representation)
    representation_array = ker.pad_sequences(representation_array, maxlen=get_max_size()-2, dtype='int32', padding='post', value=10.)
    return representation_array

def disorder_from_DM_test():
    disorder = []
    dim1 = 0
    dim2 = 0
    for seq in parse(open('../disorder/testDM').read()):
        disorder.append(np.zeros(shape=[get_max_size()-2,3]))
        for x in seq.get_representation_and_disorder():
            for dim3 in range(3):
                disorder[dim1][dim2,dim3] = x[1]
            dim2=dim2+1
        dim1=dim1+1
        dim2 = 0
    return np.asarray(disorder)

