import cPickle
import gzip
import os
import sys
sys.setrecursionlimit(6000)
import time
from word2embeddings.nn.util import zero_value, random_value_normal
import numpy as np
import theano
import theano.tensor as T
import random

from logistic_sgd import LogisticRegression

from theano.tensor.signal import downsample
from random import shuffle
from preprocess import rel_idmatrix_to_word2vec_init, rel_idlist_to_word2vec_init, load_das_train, load_das_test, compute_map
from preprocess_socher_guu import load_guu_data, load_all_triples_inIDs, neg_entity_tensor, load_guu_data_v2, neg_entity_tensor_v2
from common_functions import store_model_to_file, load_model_from_file, create_conv_para, cosine_tensor3_tensor4, rmsprop, cosine_tensors, Adam, GRU_Batch_Tensor_Input_with_Mask_with_MatrixInit, Conv_with_input_para, LSTM_Batch_Tensor_Input_with_Mask, create_ensemble_para, L2norm_paraList, Diversify_Reg, create_GRU_para, GRU_Batch_Tensor_Input_with_Mask, create_LSTM_para, load_word2vec
def evaluate_lenet5(learning_rate=0.1, n_epochs=2000, L2_weight=0.00001, max_performance=0.513333333334, Div_reg=0.1, rel_emb_size=300, margin=0.5, ent_emb_size=300, batch_size=100, train_maxSentLen=12,test_maxSentLen=1, neg_size=20,
                    test_folders_size=2):
    model_options = locals().copy()
    print "model options", model_options

    rng = np.random.RandomState(1234)    #random seed, control the model generates the same results
    rootPath='/mounts/data/proj/wenpeng/Dataset/UMASS_relation_pred/release_with_entities_akbc/wenpeng/'
    #corpus,rel_id2wordlist,ent_str2id, relation_str2id, tuple2tailset, rel2tailset, ent2relset, ent2relset_maxSetSize  
    train_set,rel_id2wordlist,ent_str2id, relation_str2id, tuple2tailset, rel2tailset, ent2relset, ent2relset_maxSetSize =load_das_train(maxPathLen=train_maxSentLen, example_limit=10000)  #minlen, include one label, at least one word in the sentence
#     tuple2tailset=load_all_triples_inIDs(ent_str2id, relation_str2id, tuple2tailset)

    train_paths_store=train_set[0]
    train_masks_store=train_set[1]
    train_ents_store=train_set[2]
    train_size=len(train_paths_store)
    
    
    folders_paths_store, folders_ents_store, folders_masks_store, folders_labels_store=load_das_test(relation_str2id, ent_str2id, rel_id2wordlist, folders_limit=test_folders_size)

#     test_set=corpus[1]
#     test_paths_store=test_set[0]
#     test_masks_store=test_set[1]
#     test_ents_store=test_set[2]
#     test_size=len(test_paths_store)



    rel_vocab_size=  len(relation_str2id)+1 # add one zero pad index
    ent_vocab_size=len(ent_str2id)
#     rel_rand_values=rng.normal(0.0, 0.01, (rel_vocab_size, rel_emb_size))   #generate a matrix by Gaussian distribution
    rel_rand_values=random_value_normal((rel_vocab_size, rel_emb_size), theano.config.floatX, np.random.RandomState(1234))
    rel_embeddings=theano.shared(value=np.array(rel_rand_values,dtype=theano.config.floatX), borrow=True)

#     ent_rand_values=rng.normal(0.0, 0.01, (ent_vocab_size, ent_emb_size))   #generate a matrix by Gaussian distribution
    ent_rand_values=random_value_normal((ent_vocab_size, ent_emb_size), theano.config.floatX, np.random.RandomState(1234))
    ent_embeddings=theano.shared(value=np.array(ent_rand_values,dtype=theano.config.floatX), borrow=True)

    word2vec=load_word2vec()
    #now, start to build the input form of the model
    init_heads=T.ivector()
    path_id_matrix=T.imatrix('path_id_matrix')
    path_w2v_tensor3=T.ftensor3('path_w2v_tensor3') #(batch, len, emb_size)
    path_mask=T.fmatrix('path_mask')
    target_entities=T.imatrix()
    test_target_entities=T.ivector()
    neg_entities_tensor=T.itensor3() #in the beginning (batch, len, #neg)
    maxSentLen=T.iscalar()
#     entity_vocab=T.ivector() #vocab size
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'
    #para
    U1, W1, b1=create_GRU_para(rng, rel_emb_size, ent_emb_size)
    NN_para=[U1, W1, b1]     #U1 includes 3 matrices, W1 also includes 3 matrices b1 is bias
    params_to_store= [rel_embeddings, ent_embeddings]+NN_para
#     load_model_from_file(rootPath+'Best_Paras_v1_0.513333333333', params_to_store)
    
    
    
    neg_entities_tensor=neg_entities_tensor.dimshuffle(2, 0, 1)

    paths_input=rel_embeddings[path_id_matrix.flatten()].reshape((path_id_matrix.shape[0],maxSentLen, rel_emb_size)).dimshuffle(0,2,1) # (batch, hidden, len)
    ensemble_path_input=paths_input+path_w2v_tensor3.dimshuffle(0,2,1)  #(batch, hidden ,len)
    init_heads_input=ent_embeddings[init_heads].reshape((path_id_matrix.shape[0], ent_emb_size))
    ground_truth_entities=ent_embeddings[target_entities.flatten()].reshape((path_id_matrix.shape[0], maxSentLen, ent_emb_size)).dimshuffle(0,2,1)# (batch, hidden, len)
    neg_entities=ent_embeddings[neg_entities_tensor.flatten()].reshape((neg_size, path_id_matrix.shape[0], maxSentLen, ent_emb_size)).dimshuffle(0,1,3,2) #(#neg, batch, hidden ,len)
#     vocab_inputs=ent_embeddings.T #(hidden, vocab_size)
    test_ground_truth_entities=ent_embeddings[test_target_entities].reshape((path_id_matrix.shape[0], ent_emb_size)) #(batch, hidden)

    #GRU

    gru_input = ensemble_path_input   #gru requires input (batch_size, emb_size, maxSentLen)
    gru_layer=GRU_Batch_Tensor_Input_with_Mask_with_MatrixInit(gru_input, path_mask,  init_heads_input, ent_emb_size, U1, W1, b1)
    pred_ent_embs=gru_layer.output_tensor  # (batch_size, hidden_size, len)
    pred_last_ents=gru_layer.output_sent_rep #(batch, hidden)

    #cosine with ground truth
    simi_grounds=cosine_tensors(pred_ent_embs, ground_truth_entities) #(batch, len)
    simi_negs=cosine_tensor3_tensor4(pred_ent_embs, neg_entities) #(#neg, batch, len)
    raw_loss=T.maximum(0.0, margin+simi_negs-simi_grounds)
#     loss=T.sum(raw_loss*path_mask.dimshuffle('x', 0,1))
    valid_indice_list=T.repeat(path_mask.dimshuffle('x', 0,1), neg_size, axis=0).flatten().nonzero()[0]
    loss=T.sum(raw_loss.flatten()[valid_indice_list])



    #loss for testing
    dot_prod=T.sum(pred_last_ents*test_ground_truth_entities, axis=1) #(batch, hidden) * (hidden, vocab_size) == (batch, vocab_size)
    norm_ents=T.sqrt(T.sum(pred_last_ents**2, axis=1))#.reshape((batch_size, 1)) #(batch, 1)
    norm_vocab=T.sqrt(T.sum(test_ground_truth_entities**2, axis=1))#.reshape((1, ent_vocab_size))  # vocab
    cosines_test_vector=dot_prod/(norm_ents*norm_vocab+1e-8) #(batch, vocab_size)


    params = [rel_embeddings, ent_embeddings]+NN_para   # put all model parameters together
    
    L2_reg =L2norm_paraList([rel_embeddings, U1, W1]) #ent_embeddings, 
#     diversify_reg= Diversify_Reg(U_a.T)+Diversify_Reg(conv_W_into_matrix)

    cost=loss#+L2_weight*L2_reg

    grads = T.grad(cost, params)    # create a list of gradients for all model parameters
    accumulator=[]
    for para_i in params:
        eps_p=np.zeros_like(para_i.get_value(borrow=True),dtype=theano.config.floatX)
        accumulator.append(theano.shared(eps_p, borrow=True))
    updates = []
    for param_i, grad_i, acc_i in zip(params, grads, accumulator):
        acc = acc_i + T.sqr(grad_i)
        updates.append((param_i, param_i - learning_rate * grad_i / (T.sqrt(acc)+1e-8)))   #1e-8 is add to get rid of zero division
        updates.append((acc_i, acc))

#     updates=Adam(cost=cost, params=params, lr=learning_rate)

#     grads = T.grad(cost, params)
#     opt = rmsprop(params)
#     updates = opt.updates(params, grads, np.float32(0.01) / np.cast['float32'](batch_size), np.float32(0.9))

    train_model = theano.function([init_heads,path_id_matrix, path_w2v_tensor3, path_mask, target_entities, neg_entities_tensor, maxSentLen], cost, updates=updates,on_unused_input='ignore')

    test_model = theano.function([init_heads,path_id_matrix, path_w2v_tensor3, path_mask, test_target_entities, maxSentLen], cosines_test_vector, on_unused_input='ignore')


    ###############
    # TRAIN MODEL #
    ###############
    print '... training, train_size:', train_size
    # early-stopping parameters
    patience = 50000000000  # look as this many examples regardless
    start_time = time.time()
    mid_time = start_time
    past_time= mid_time
    epoch = 0
    done_looping = False

    '''
    split training/test sets into a list of mini-batches, each batch contains batch_size of sentences
    usually there remain some sentences that are fewer than a normal batch, we can start from the "train_size-batch_size" to the last sentence to form a mini-batch
    or cource this means a few sentences will be trained more times than normal, but doesn't matter
    '''
    n_train_batches=train_size/batch_size
    train_batch_start=list(np.arange(n_train_batches)*batch_size)+[train_size-batch_size]
#     n_test_batches=test_size/batch_size
#     test_batch_start=list(np.arange(n_test_batches)*batch_size)+[test_size-batch_size]


    max_acc=0.0
    combined=range(train_size)
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        random.shuffle(combined) #shuffle training set for each new epoch, is supposed to promote performance, but not garrenteed
        iter_accu=0
        cost_i=0.0
        for batch_id in train_batch_start: #for each batch
            # iter means how many batches have been run, taking into loop
            iter = (epoch - 1) * n_train_batches + iter_accu +1
            iter_accu+=1

            batch_indices=combined[batch_id:batch_id+batch_size]
            #init_heads,path_id_matrix, path_w2v_tensor3, path_mask, target_entities, neg_entities_tensor
            #path_id_matrix, path_w2v_tensor3, path_mask, taret_rel_idlist, target_w2v_matrix, labels


            ent_idmatrix=np.asarray([train_ents_store[id] for id in batch_indices], dtype='int32')
            rel_idmatrix=[train_paths_store[id] for id in batch_indices]
            ent_vocab_set=set(range(ent_vocab_size))
            cost_i+= train_model(
                                 ent_idmatrix[:,0],
                                    np.asarray(rel_idmatrix, dtype='int32'),
                                      np.asarray(rel_idmatrix_to_word2vec_init(rel_idmatrix, rel_id2wordlist, word2vec, 300), dtype=theano.config.floatX),
                                      np.asarray([train_masks_store[id] for id in batch_indices],dtype=theano.config.floatX),
                                      ent_idmatrix[:,1:],
                                      neg_entity_tensor_v2(ent_idmatrix, rel_idmatrix, tuple2tailset, rel2tailset, neg_size, ent_vocab_set),
                                      train_maxSentLen)
#                                       neg_entity_tensor(ent_idmatrix, rel_idmatrix, tuple2tailset, neg_size, ent_vocab_set))

            #after each 1000 batches, we test the performance of the model on all test data
            if iter%100==0:
                print 'Epoch ', epoch, 'iter '+str(iter)+' average cost: '+str(cost_i/iter_accu), 'uses ', (time.time()-past_time)/60.0, 'min'
                print 'Testing...'
                past_time = time.time()

                error_sum=0.0
                succ=0
                aver_map=0.0
                for folder_id in range(test_folders_size):
                    print '\t\t folder ', folder_id+1
                    #folders_paths_store, folders_ents_store, folders_masks_store, folders_labels_store
                    test_paths_store=folders_paths_store[folder_id]
                    test_masks_store=folders_masks_store[folder_id]
                    test_ents_store=folders_ents_store[folder_id]
                    test_labels_store=folders_labels_store[folder_id]
                    test_size=len(test_labels_store)      
                    
                    n_test_batches=test_size/batch_size
                    n_test_remains=test_size%batch_size
                    test_batch_start=list(np.arange(n_test_batches)*batch_size)+[test_size-n_test_remains]      
                    overall_simi=[]    
                    for idd, test_batch_id in enumerate(test_batch_start): # for each test batch
    
                        #init_heads,path_id_matrix, path_w2v_tensor3, path_mask
                        test_ent_idmatrix=np.asarray(test_ents_store[test_batch_id:test_batch_id+batch_size], dtype='int32')
                        test_rel_idmatrix=test_paths_store[test_batch_id:test_batch_id+batch_size]
                        test_ground_ent_idlist=test_ent_idmatrix[:,-1]
                        test_masks_matrix=np.asarray(test_masks_store[test_batch_id:test_batch_id+batch_size],dtype=theano.config.floatX)
                        cosine_batch_vector=test_model(
                                           test_ent_idmatrix[:,0],
                                        np.asarray(test_rel_idmatrix, dtype='int32'),
                                          np.asarray(rel_idmatrix_to_word2vec_init(test_rel_idmatrix, rel_id2wordlist, word2vec, 300), dtype=theano.config.floatX),
                                          test_masks_matrix,
                                          test_ground_ent_idlist,
                                          test_maxSentLen)
                        overall_simi+=list(cosine_batch_vector)
                    if len(overall_simi) !=test_size:
                        print 'len(overall_simi) !=test_size:', len(overall_simi), test_size
                        exit(0)
                        
                    map=compute_map(overall_simi, test_labels_store)  
                    aver_map+=map
                aver_map/=test_folders_size
                if aver_map > max_acc:
                    max_acc=aver_map
#                     if max_acc > max_performance:
#                         store_model_to_file(rootPath+'Best_Paras_v1_'+str(max_acc), params_to_store)
#                         print 'Finished storing best  params at:', max_acc  
                print 'current map:', aver_map, '\t\t\t\t\tmax map:', max_acc
                  




            if patience <= iter:
                done_looping = True
                break

        print 'Epoch ', epoch, 'uses ', (time.time()-mid_time)/60.0, 'min'
        mid_time = time.time()

        #print 'Batch_size: ', update_freq
    end_time = time.time()
    print('Optimization complete.')

    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))





if __name__ == '__main__':
    evaluate_lenet5()
