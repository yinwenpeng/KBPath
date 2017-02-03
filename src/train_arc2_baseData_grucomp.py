import cPickle
import gzip
import os
import sys
sys.setrecursionlimit(6000)
import time
from collections import defaultdict
from word2embeddings.nn.util import zero_value, random_value_normal
import numpy as np
import theano
import theano.tensor as T
import random

from logistic_sgd import LogisticRegression

from theano.tensor.signal import downsample
from random import shuffle

# from load_data import load_guu_data_4_CompTransE
# from preprocess import rel_idmatrix_to_word2vec_init, rel_idlist_to_word2vec_init, load_word2vec_to_init_rels
from preprocess_socher_guu import neg_entity_tensor_v2, load_guu_data_4_CompTransE
from common_functions import cosine_matrix1_matrix2_rowwise, GRU_OneStep_Matrix_Input, cosine_tensors, Conv_with_input_para, LSTM_Batch_Tensor_Input_with_Mask, create_conv_para,create_GRU_para, GRU_Batch_Tensor_Input_with_Mask, create_LSTM_para

'''


'''
#def evaluate_lenet5(learning_rate=0.001, n_epochs=2000, L2_weight=0.00001, max_performance=0.45, Div_reg=0.1, rel_emb_size=300, margin=0.07, ent_emb_size=300, batch_size=20, maxSentLen=5, neg_size=10):
def evaluate_lenet5(learning_rate=0.01, n_epochs=6, nn='GRU', rel_emb_size=300, margin=0.3, batch_size=50, maxSentLen=5, neg_size=20, filter_size=3):
    ent_emb_size=rel_emb_size
    model_options = locals().copy()
    print "model options", model_options

    rng = np.random.RandomState(1234)    #random seed, control the model generates the same results
    rootPath='/mounts/data/proj/wenpeng/Dataset/FB_socher/path/'
    #corpus,rel_id2wordlist,ent_str2id, relation_str2id, tuple2tailset, rel2tailset, ent2relset, ent2relset_maxSetSize  
    corpus,rel_id2wordlist,ent_str2id, relation_str2id, tuple2tailset, rel2tailset, ent2relset, ent2relset_maxSetSize, rel_id2inid =load_guu_data_4_CompTransE(maxPathLen=maxSentLen)  #minlen, include one label, at least one word in the sentence
#     tuple2tailset=load_all_triples_inIDs(ent_str2id, relation_str2id, tuple2tailset)
    train_set=corpus[0]
    train_paths_store=np.asarray(train_set[0], dtype='int32')
    train_masks_store=np.asarray(train_set[1], dtype=theano.config.floatX)
    train_ents_store=np.asarray(train_set[2], dtype='int32')
    train_size=len(train_paths_store)
    
#     dev_set=corpus[1]
#     dev_paths_store=np.asarray(dev_set[0], dtype='int32')
#     dev_masks_store=np.asarray(dev_set[1], dtype=theano.config.floatX)
#     dev_ents_store=np.asarray(dev_set[2], dtype='int32')
#     dev_size=len(dev_paths_store)

    test_set=corpus[1]
    test_paths_store=np.asarray(test_set[0], dtype='int32')
    test_masks_store=np.asarray(test_set[1], dtype=theano.config.floatX)
    test_ents_store=np.asarray(test_set[2], dtype='int32')
    test_size=1000#len(test_paths_store)



    rel_vocab_size=  len(relation_str2id)+1 # add one zero pad index
    ent_vocab_size=len(ent_str2id)
#     rel_rand_values=rng.normal(0.0, 0.01, (rel_vocab_size, rel_emb_size))   #generate a matrix by Gaussian distribution
    rel_rand_values=random_value_normal((rel_vocab_size, rel_emb_size), theano.config.floatX, rng)
    rel_embeddings=theano.shared(value=np.array(rel_rand_values,dtype=theano.config.floatX), borrow=True)

    ent_rand_values=random_value_normal((ent_vocab_size, ent_emb_size), theano.config.floatX, rng)
    ent_embeddings=theano.shared(value=np.array(ent_rand_values,dtype=theano.config.floatX), borrow=True)

    #now, start to build the input form of the model
    init_heads=T.ivector()
    path_id_matrix=T.imatrix('path_id_matrix')
    path_mask=T.fmatrix('path_mask')
    target_tails=T.ivector() #batch
    neg_tails=T.imatrix() #in the beginning (batch,  #neg)

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'
    
    path_common_input = rel_embeddings[path_id_matrix.flatten()].reshape((batch_size,maxSentLen, rel_emb_size))
    init_heads_input=ent_embeddings[init_heads].reshape((batch_size, ent_emb_size))
    ground_truth_entities=ent_embeddings[target_tails].reshape((batch_size, ent_emb_size))#.dimshuffle(0,2,1)# (batch, hidden, len)
    neg_entities=ent_embeddings[neg_tails.flatten()].reshape((batch_size, neg_size, ent_emb_size)).dimshuffle(0,2,1) #(batch, hidden,#neg)
    vocab_inputs=ent_embeddings.T #(hidden, vocab_size)
    
    #cnn
    if nn=='CNN':
        conv_input = path_common_input.dimshuffle((0,'x', 2,1)) #(batch_size, 1, emb_size, maxsenlen)
        conv_W, conv_b=create_conv_para(rng, filter_shape=(ent_emb_size, 1, rel_emb_size, filter_size))
        NN_para=[conv_W, conv_b]
        conv_model = Conv_with_input_para(rng, input=conv_input,
                 image_shape=(batch_size, 1, rel_emb_size, maxSentLen),
                 filter_shape=(ent_emb_size, 1, rel_emb_size, filter_size), W=conv_W, b=conv_b)
        conv_output=conv_model.narrow_conv_out #(batch, 1, hidden_size, maxsenlen-filter_size+1)    
        conv_output_into_tensor3=conv_output.reshape((batch_size, ent_emb_size, maxSentLen-filter_size+1))
        mask_for_conv_output=T.repeat(path_mask[:,filter_size-1:].reshape((batch_size, 1, maxSentLen-filter_size+1)), ent_emb_size, axis=1) #(batch_size, emb_size, maxSentLen-filter_size+1)
        masked_conv_output=conv_output_into_tensor3*mask_for_conv_output      #mutiple mask with the conv_out to set the features by UNK to zero
        pred_ent_embs=T.max(masked_conv_output, axis=2) #(batch_size, hidden_size) # each sentence then have an embedding of length hidden_size
        
    #GRU
    if nn=='GRU':
        U1, W1, b1=create_GRU_para(rng, rel_emb_size, ent_emb_size)
        U_c, W_c, b_c=create_GRU_para(rng, ent_emb_size, ent_emb_size)
        NN_para=[U1, W1, b1, U_c, W_c, b_c]     #U1 includes 3 matrices, W1 also includes 3 matrices b1 is bias
        gru_input = path_common_input.dimshuffle(0,2,1) # (batch, hidden,len)
        gru_layer=GRU_Batch_Tensor_Input_with_Mask(gru_input, path_mask, ent_emb_size, U1, W1, b1)
        pred_ent_embs=gru_layer.output_sent_rep  # (batch_size, ent_emb_size)
    
    #LSTM
    if nn=='LSTM':
        LSTM_para_dict=create_LSTM_para(rng, rel_emb_size, ent_emb_size)
        NN_para=LSTM_para_dict.values() # .values returns a list of parameters
        lstm_input = path_common_input.dimshuffle((0,2,1)) #LSTM has the same inpur format with GRU
        lstm_layer=LSTM_Batch_Tensor_Input_with_Mask(lstm_input, path_mask,  ent_emb_size, LSTM_para_dict)
        pred_ent_embs=lstm_layer.output_sent_rep  # (batch_size, hidden_size)      
    
    
#     pred_last_ents=pred_ent_embs + init_heads_input #(batch, emb_size)
    compose_layer = GRU_OneStep_Matrix_Input( pred_ent_embs, init_heads_input, ent_emb_size, U_c, W_c, b_c) 
    pred_last_ents = compose_layer.matrix#(batch, emb_size)
    #cosine with ground truth
    simi_grounds=cosine_matrix1_matrix2_rowwise(pred_last_ents, ground_truth_entities) #batch
    #cosine with nega
    simi_negs=cosine_tensors(pred_last_ents.dimshuffle(0,1,'x'), neg_entities) #(batch, #neg)
    raw_loss=T.maximum(0.0, margin+simi_negs-simi_grounds.dimshuffle(0,'x')) #(batch, #neg)
    loss=T.mean(T.sum(raw_loss, axis=1))

    #loss for testing
    dot_prod=T.dot(pred_last_ents, vocab_inputs) #(batch, hidden) * (hidden, vocab_size) == (batch, vocab_size)
    norm_ents=T.sqrt(T.sum(pred_last_ents**2, axis=1)).reshape((batch_size, 1)) #(batch, 1)
    norm_vocab=T.sqrt(T.sum(vocab_inputs**2, axis=0)).reshape((1, ent_vocab_size))  # vocab
    cosines_test_matrix=dot_prod/(T.dot(norm_ents, norm_vocab)+1e-8) #(batch, vocab_size)


    params= [rel_embeddings, ent_embeddings]+NN_para
    


    cost=loss

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



    train_model = theano.function([init_heads,path_id_matrix, path_mask, target_tails, neg_tails ], cost, updates=updates,on_unused_input='ignore')
#     dev_model = theano.function([init_heads,path_id_matrix, path_mask], cosines_test_matrix, on_unused_input='ignore')
    test_model = theano.function([init_heads,path_id_matrix, path_mask], cosines_test_matrix, on_unused_input='ignore')


    ###############
    # TRAIN MODEL #
    ###############
    print '... training, train_size:', train_size, 'test_size:', test_size
    # early-stopping parameters
    patience = 50000000000  # look as this many examples regardless
    start_time = time.time()
    mid_time = start_time
    past_time= mid_time
    epoch = 0
    done_looping = False


    n_train_batches=train_size/batch_size
    train_batch_start=list(np.arange(n_train_batches)*batch_size)+[train_size-batch_size]
#     n_dev_batches=dev_size/batch_size
#     dev_batch_start=list(np.arange(n_dev_batches)*batch_size)+[dev_size-batch_size]
    n_test_batches=test_size/batch_size
    test_batch_start=list(np.arange(n_test_batches)*batch_size)+[test_size-batch_size]


    max_acc_dev=0.0
    max_acc_test=0.0
    combined=range(train_size)
    while epoch < n_epochs:
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


            ent_idmatrix=train_ents_store[batch_indices]
            rel_idmatrix=train_paths_store[batch_indices]
#             new_idmatrix, mask_matrix = recover_inverse_rel(rel_idmatrix, rel_id2inid)
            ent_vocab_set=set(range(ent_vocab_size))
            cost_i+= train_model(
                                 ent_idmatrix[:,0],
                                 rel_idmatrix,
                                 train_masks_store[batch_indices],
                                 ent_idmatrix[:,-1],
                                 neg_entity_tensor_v2(ent_idmatrix[:,-2:], rel_idmatrix[:,-1:], tuple2tailset, rel2tailset, neg_size, ent_vocab_set).reshape((batch_size, neg_size))
                                )
#                                       neg_entity_tensor(ent_idmatrix, rel_idmatrix, tuple2tailset, neg_size, ent_vocab_set))

            #after each 1000 batches, we test the performance of the model on all test data
            if epoch <5 and iter%60000==0:
                print 'Epoch ', epoch, 'iter '+str(iter)+' average cost: '+str(cost_i/iter_accu), 'uses ', (time.time()-past_time)/60.0, 'min'
                past_time = time.time()
            if epoch >=5 and iter%25000==0:

                print 'Epoch ', epoch, 'iter '+str(iter)+' average cost: '+str(cost_i/iter_accu), 'uses ', (time.time()-past_time)/60.0, 'min'
                past_time = time.time()


                #test
                succ=0
                for idd, test_batch_id in enumerate(test_batch_start): # for each test batch

                    #init_heads,path_id_matrix, path_w2v_tensor3, path_mask
                    test_ent_idmatrix=test_ents_store[test_batch_id:test_batch_id+batch_size]
                    test_rel_idmatrix=test_paths_store[test_batch_id:test_batch_id+batch_size]
                    test_ground_ent_idlist=test_ent_idmatrix[:,-1]
                    test_masks_matrix=test_masks_store[test_batch_id:test_batch_id+batch_size]
                    cosine_batch_vocab=test_model(
                                       test_ent_idmatrix[:,0],
                                       test_rel_idmatrix,
                                       test_masks_matrix)

                    sort_id_matrix=np.argsort(cosine_batch_vocab, axis=1)
                    succ_batch=0
                    for i in range(batch_size):
                        ground_id=test_ground_ent_idlist[i]
                        mask_sum=np.sum(test_masks_matrix[i])

                        if mask_sum==1.0:
                            head_id=test_ent_idmatrix[i][0]
                        else:
                            head_id=test_ent_idmatrix[i][-2]
                        pair=(head_id, test_rel_idmatrix[i][-1])
                        filted_idset=tuple2tailset.get(pair)
                        focus_idset=rel2tailset.get(test_rel_idmatrix[i][-1])
                        if filted_idset is None:
                            print pair, 'is not in the training set'
                            print idd*batch_size+i
                            print test_ent_idmatrix[i], test_rel_idmatrix[i]
                            exit(0)
                        filted_idset=filted_idset-set([ground_id])
                        valid_co=0
                        id_list=sort_id_matrix[i]
                        for j in range(ent_vocab_size-1, -1, -1):
                            if id_list[j] in filted_idset or id_list[j] not in focus_idset:
                                continue
                            else:
                                if id_list[j] != ground_id:
                                    valid_co+=1
                                    if valid_co ==10:
                                        break
                                else:
                                    succ_batch+=1
                                    break
                    succ_batch=succ_batch*1.0/batch_size
                    succ+=succ_batch


                test_hit10=succ/len(test_batch_start)
                if test_hit10 > max_acc_test:
                    max_acc_test=test_hit10 
                print '\t\tcurrent test_hit10:', test_hit10, '\t\t\t\t\tmax test_hit10:', max_acc_test










        print 'Epoch ', epoch, 'uses ', (time.time()-mid_time)/60.0, 'min'
        mid_time = time.time()

        #print 'Batch_size: ', update_freq
    end_time = time.time()
    print('Optimization complete.')

    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
    return max_acc_test         

def recover_inverse_rel(rel_idmatrix, rel_id2inid):
    rows=len(rel_idmatrix)
    cols=len(rel_idmatrix[0])
    mask=[]
    new_idmatrix=[]
    for row in range(rows):
        for col in range(cols):
            old_id=rel_idmatrix[row][col]
            inID=rel_id2inid.get(old_id)
            if inID is None:
                mask.append(1.0)
                new_idmatrix.append(old_id)
            else:
                mask.append(-1.0)
                new_idmatrix.append(inID)
    mask_matrix=np.asarray(mask, dtype=theano.config.floatX).reshape((rows, cols))
    new_idmatrix=np.asarray(new_idmatrix, dtype='int32').reshape((rows, cols))
    return     new_idmatrix, mask_matrix,            
            


if __name__ == '__main__':
    evaluate_lenet5()
# (learning_rate=0.1, n_epochs=6, nn='LSTM', rel_emb_size=30, margin=0.3, batch_size=50, maxSentLen=5, neg_size=10, filter_size=3)
#     lr_list=[0.1,0.01,0.001]
#     r_emb_list=[30,50,100,150,200,250,300]#,250,300]
# #     e_emb_list=[50,100,150,200,250,300]
#     batch_list=[50,100,200]
# #     neg_list=[5,10,15,20]
#     margin_list=[0.3,0.4,0.5]#,0.2,0.1]
#      
#     best_acc=0.0
#     best_lr=0.1
#     for lr in lr_list:
#         acc_test= evaluate_lenet5(learning_rate=lr)
#         if acc_test>best_acc:
#             best_lr=lr
#             best_acc=acc_test
#         print '\t\t\t\tcurrent best_acc:', best_acc
#      
#     best_r_emb=30
#     for r_emb in r_emb_list:
#         acc_test= evaluate_lenet5(learning_rate=best_lr, rel_emb_size=r_emb)
#         if acc_test>best_acc:
#             best_r_emb=r_emb
#             best_acc=acc_test
#         print '\t\t\t\tcurrent best_acc:', best_acc
# 
# #     best_e_emb=30
# #     for e_emb in e_emb_list:
# #         acc_test= evaluate_lenet5(learning_rate=best_lr, rel_emb_size=best_r_emb, ent_emb_size=e_emb)
# #         if acc_test>best_acc:
# #             best_e_emb=e_emb
# #             best_acc=acc_test
# #         print '\t\t\t\tcurrent best_acc:', best_acc
#              
#     best_batch=50
#     for batch in batch_list:
#         acc_test= evaluate_lenet5(learning_rate=best_lr, rel_emb_size=best_r_emb, batch_size=batch)
#         if acc_test>best_acc:
#             best_batch=batch
#             best_acc=acc_test
#         print '\t\t\t\tcurrent best_acc:', best_acc
#         
# #     best_neg=20
# #     for neg in neg_list:
# #         acc_test= evaluate_lenet5(learning_rate=best_lr, rel_emb_size=best_r_emb, ent_emb_size=best_e_emb, batch_size=best_batch, neg_size=neg)
# #         if acc_test>best_acc:
# #             best_neg=neg
# #             best_acc=acc_test
# #         print '\t\t\t\tcurrent best_acc:', best_acc
#                              
#     best_margin=0.3        
#     for marg in margin_list:
#         acc_test= evaluate_lenet5(learning_rate=best_lr, rel_emb_size=best_r_emb, batch_size=best_batch, margin=marg)
#         if acc_test>best_acc:
#             best_margin=marg
#             best_acc=acc_test
#         print '\t\t\t\tcurrent best_acc:', best_acc
#     print 'Hyper tune finished, best test acc: ', best_acc, ' by  lr: ', best_lr, ' r_emb: ', best_r_emb, ' batch: ', best_batch, ' margin: ', best_margin