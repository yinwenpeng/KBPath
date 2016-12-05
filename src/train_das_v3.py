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
from preprocess import rel_idmatrix_to_word2vec_init, rel_idlist_to_word2vec_init, load_das_v2, compute_map, rel_id_to_word2vec_init
from preprocess_socher_guu import load_guu_data, load_all_triples_inIDs, neg_entity_tensor, load_guu_data_v2, neg_entity_tensor_v2
from common_functions import store_model_to_file, load_model_from_file, create_conv_para, cosine_tensor3_tensor4, rmsprop, cosine_tensors, Adam, GRU_Batch_Tensor_Input_with_Mask_with_MatrixInit, Conv_with_input_para, LSTM_Batch_Tensor_Input_with_Mask, create_ensemble_para, L2norm_paraList, Diversify_Reg, create_GRU_para, GRU_Batch_Tensor_Input_with_Mask, create_LSTM_para, load_word2vec
def evaluate_lenet5(learning_rate=0.1, n_epochs=2000, L2_weight=0.00001, max_performance=50.0,rel_emb_size=200, margin1=0.5, margin2=0.5, ent_emb_size=30, batch_size=20, maxPathLen=8, neg_size=4, path_size=50):
    model_options = locals().copy()
    print "model options", model_options

    rng = np.random.RandomState(1234)    #random seed, control the model generates the same results
    rootPath='/mounts/data/proj/wenpeng/Dataset/UMASS_relation_pred/release_with_entities_akbc/'

    corpus,target_rels_list, relation_str2id, ent_str2id, rel_id2wordlist, tuple2tailset, rel2tailset, ent2relset, ent2relset_maxSetSize =load_das_v2(maxPathLen, path_size)  #minlen, include one label, at least one word in the sentence

    train_pos=corpus[0]
#     print len(train_pos)
#     print train_pos[0][0][0], train_pos[1][0][0], train_pos[2][0][0]
    train_pos_rels_matrix_folders=train_pos[0]
    train_pos_masks_matrix_folders=train_pos[1]
    train_pos_ents_matrix_folders=train_pos[2]
    train_pos_path_masks_folders=train_pos[3]
#     
#     print train_pos_rels_matrix_folders[0][:100]
#     exit(0)
    train_neg=corpus[1]
    train_neg_rels_matrix_folders=train_neg[0]
    train_neg_masks_matrix_folders=train_neg[1]
    train_neg_ents_matrix_folders=train_neg[2]
    train_neg_path_masks_folders=train_neg[3]

    test_pos=corpus[2]
    test_pos_rels_matrix_folders=test_pos[0]
    test_pos_masks_matrix_folders=test_pos[1]
    test_pos_ents_matrix_folders=test_pos[2]
    test_pos_path_masks_folders=test_pos[3]
#     print len(test_pos_masks_matrix_folders[0]), len(test_pos_path_masks_folders[0])
#     exit(0)

    test_neg=corpus[3]
    test_neg_rels_matrix_folders=test_neg[0]
    test_neg_masks_matrix_folders=test_neg[1]
#     test_neg_ents_matrix_folders=test_neg[2]
    test_neg_path_masks_folders=test_neg[3]

    folder_size=len(train_pos_rels_matrix_folders)
    print 'True folder_size:', folder_size



    rel_vocab_size=  len(relation_str2id)+1 # add one zero pad index
    ent_vocab_size=len(ent_str2id)
    print 'entity size:', ent_vocab_size, 'relation size:', rel_vocab_size
#     rel_rand_values=rng.normal(0.0, 0.01, (rel_vocab_size, rel_emb_size))   #generate a matrix by Gaussian distribution
    rel_rand_values=random_value_normal((rel_vocab_size, rel_emb_size), theano.config.floatX, np.random.RandomState(1234))
    rel_embeddings=theano.shared(value=np.array(rel_rand_values,dtype=theano.config.floatX), borrow=True)

#     ent_rand_values=rng.normal(0.0, 0.01, (ent_vocab_size, ent_emb_size))   #generate a matrix by Gaussian distribution
#     ent_rand_values=random_value_normal((ent_vocab_size, ent_emb_size), theano.config.floatX, np.random.RandomState(1234))
#     ent_embeddings=theano.shared(value=np.array(ent_rand_values,dtype=theano.config.floatX), borrow=True)
# 
#     word2vec=load_word2vec()
    #now, start to build the input form of the model
#     pos_init_heads=T.ivector() #batch*#p
    pos_path_id_matrix=T.imatrix()#batch*#p * len
#     pos_path_w2v_tensor3=T.ftensor3() #(batch*#p, len, emb_size)
    pos_path_mask=T.fmatrix()   #(batch*#p, len)
#     pos_target_entities=T.imatrix() #(batch*#p, len)
#     pos_neg_entities_tensor=T.itensor3() #in the beginning (batch*#p, len, #neg)
    pos_pathPad_mask=T.fvector() #batch*#path

#     neg_init_heads=T.ivector() #batch*#p
    neg_path_id_matrix=T.imatrix()#batch*#p * len
#     neg_path_w2v_tensor3=T.ftensor3() #(batch*#p, len, emb_size)
    neg_path_mask=T.fmatrix()   #(batch*#p, len)
#     neg_target_entities=T.imatrix() #(batch*#p, len)
#     neg_neg_entities_tensor=T.itensor3() #in the beginning (batch*#p, len, #neg)
    neg_pathPad_mask=T.fvector() #batch*#path

    target_rel_id=T.iscalar()
#     target_rel_w2v_emb=T.fvector()
#     entity_vocab=T.ivector() #vocab size
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'
    #para
    U1, W1, b1=create_GRU_para(rng, rel_emb_size, rel_emb_size)
    NN_para=[U1, W1, b1]     #U1 includes 3 matrices, W1 also includes 3 matrices b1 is bias
#     params_to_store= [rel_embeddings, ent_embeddings]+NN_para
#     load_model_from_file(rootPath+'Best_Paras_v1_0.513333333333', params_to_store)

    input_batch_size=pos_path_id_matrix.shape[0]/path_size

    target_rel_emb=rel_embeddings[target_rel_id]#+target_rel_w2v_emb
    target_rel_reps_tensor3=target_rel_emb.dimshuffle('x','x', 0) #(1,1, rel_emb)

#     pos_neg_entities_tensor=pos_neg_entities_tensor.dimshuffle(2, 0, 1)  #( #neg, batch*#p, len)

    pos_paths_input=rel_embeddings[pos_path_id_matrix.flatten()].reshape((pos_path_id_matrix.shape[0],maxPathLen, rel_emb_size)).dimshuffle(0,2,1) # (batch*#path, hidden, len)
    pos_ensemble_path_input=pos_paths_input#+pos_path_w2v_tensor3.dimshuffle(0,2,1)  #(batch, hidden ,len)
#     pos_init_heads_input=ent_embeddings[pos_init_heads].reshape((pos_path_id_matrix.shape[0], ent_emb_size))  # (batch*#path, ent_emb)
#     pos_ground_truth_entities=ent_embeddings[pos_target_entities.flatten()].reshape((pos_path_id_matrix.shape[0], maxPathLen, ent_emb_size)).dimshuffle(0,2,1)# (batch*#path, hidden, len)
#     pos_neg_entities=ent_embeddings[pos_neg_entities_tensor.flatten()].reshape((neg_size, pos_path_id_matrix.shape[0], maxPathLen, ent_emb_size)).dimshuffle(0,1,3,2) #(#neg, batch*#path, hidden ,len)

    #GRU
    pos_gru_input = pos_ensemble_path_input   #gru requires input (batch_size, emb_size, maxSentLen)
    pos_gru_layer=GRU_Batch_Tensor_Input_with_Mask(pos_gru_input, pos_path_mask, rel_emb_size, U1, W1, b1)
#     pos_pred_ent_embs=pos_gru_layer.output_tensor  # (batch_size*#p, hidden_size, len)
    pos_pred_last_ents=pos_gru_layer.output_sent_rep #(batch*#p, hidden)

    #cosine with ground truth
#     pos_simi_grounds=cosine_tensors(pos_pred_ent_embs, pos_ground_truth_entities) #(batch, len)
#     pos_simi_negs=cosine_tensor3_tensor4(pos_pred_ent_embs, pos_neg_entities) #(#neg, batch, len)
#     pos_raw_loss=T.maximum(0.0, margin1+pos_simi_negs-pos_simi_grounds)
#     pos_valid_indice_list=T.repeat(pos_path_mask.dimshuffle('x', 0,1), neg_size, axis=0).flatten().nonzero()[0]
#     pos_loss_seq2seq=T.mean(pos_raw_loss.flatten()[pos_valid_indice_list])

    #cosine with target rela
    pos_path_reps_tensor3=pos_pred_last_ents.reshape((input_batch_size, path_size, rel_emb_size))
    pos_cos_rel2path=cosine_tensors(pos_path_reps_tensor3.dimshuffle(0,2,1), target_rel_reps_tensor3.dimshuffle(0,2,1)) #(batch, len)
    mask_pos_cos_rel2path=T.exp(pos_cos_rel2path)#pos_pathPad_mask.reshape((input_batch_size, path_size))*
    pos_batch_cosine=T.nnet.sigmoid(T.log(T.sum(mask_pos_cos_rel2path, axis=1))) # (batch, 1))


    #neg

#     neg_neg_entities_tensor=neg_neg_entities_tensor.dimshuffle(2, 0, 1)  #( #neg, batch*#p, len)

    neg_paths_input=rel_embeddings[neg_path_id_matrix.flatten()].reshape((neg_path_id_matrix.shape[0],maxPathLen, rel_emb_size)).dimshuffle(0,2,1) # (batch, hidden, len)
    neg_ensemble_path_input=neg_paths_input#+neg_path_w2v_tensor3.dimshuffle(0,2,1)  #(batch, hidden ,len)
#     neg_init_heads_input=ent_embeddings[neg_init_heads].reshape((neg_path_id_matrix.shape[0], ent_emb_size))
#     neg_ground_truth_entities=ent_embeddings[neg_target_entities.flatten()].reshape((neg_path_id_matrix.shape[0], maxPathLen, ent_emb_size)).dimshuffle(0,2,1)# (batch, hidden, len)
#     neg_neg_entities=ent_embeddings[neg_neg_entities_tensor.flatten()].reshape((neg_size, neg_path_id_matrix.shape[0], maxPathLen, ent_emb_size)).dimshuffle(0,1,3,2) #(#neg, batch, hidden ,len)

    #GRU
    neg_gru_input = neg_ensemble_path_input   #gru requires input (batch_size, emb_size, maxSentLen)
    neg_gru_layer=GRU_Batch_Tensor_Input_with_Mask(neg_gru_input, neg_path_mask, rel_emb_size, U1, W1, b1)
#     neg_pred_ent_embs=neg_gru_layer.output_tensor  # (batch_size, hidden_size, len)
    neg_pred_last_ents=neg_gru_layer.output_sent_rep #(batch*#p, hidden)

    #cosine with ground truth
#     neg_simi_grounds=cosine_tensors(neg_pred_ent_embs, neg_ground_truth_entities) #(batch, len)
#     neg_simi_negs=cosine_tensor3_tensor4(neg_pred_ent_embs, neg_neg_entities) #(#neg, batch, len)
#     neg_raw_loss=T.maximum(0.0, margin1+neg_simi_negs-neg_simi_grounds)
#     neg_valid_indice_list=T.repeat(neg_path_mask.dimshuffle('x', 0,1), neg_size, axis=0).flatten().nonzero()[0]
#     neg_loss_seq2seq=T.mean(neg_raw_loss.flatten()[neg_valid_indice_list])

    #cosine with target rela
    neg_path_reps_tensor3=neg_pred_last_ents.reshape((input_batch_size, path_size, rel_emb_size))
    neg_cos_rel2path=cosine_tensors(neg_path_reps_tensor3.dimshuffle(0,2,1), target_rel_reps_tensor3.dimshuffle(0,2,1)) #(batch, len)
    mask_neg_cos_rel2path=T.exp(neg_cos_rel2path) #neg_pathPad_mask.reshape((input_batch_size, path_size))*
    neg_batch_cosine=T.nnet.sigmoid(T.log(T.sum(mask_neg_cos_rel2path, axis=1)))

    pos_batch_cosine_matrix=T.repeat(pos_batch_cosine.dimshuffle(0,'x'), input_batch_size, axis=1)  #(batch, batch)
    neg_batch_cosine_matrix=T.repeat(neg_batch_cosine.dimshuffle('x', 0), input_batch_size, axis=0)  #(batch, batch)

    rank_loss=T.mean(T.maximum(0.0, margin2+neg_batch_cosine_matrix-pos_batch_cosine_matrix))


    loss=rank_loss#rank_loss(pos_loss_seq2seq+neg_loss_seq2seq)*0.0+rank_loss


    params = [rel_embeddings]+NN_para   # put all model parameters together

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




    train_model = theano.function([pos_path_id_matrix,pos_path_mask,pos_pathPad_mask,
                                   neg_path_id_matrix,neg_path_mask,neg_pathPad_mask,
                                   target_rel_id], cost, updates=updates,on_unused_input='ignore')

    test_model = theano.function([pos_path_id_matrix,pos_path_mask,pos_pathPad_mask,#pos_target_entities,pos_neg_entities_tensor,
                                target_rel_id], pos_batch_cosine, on_unused_input='ignore')


    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    # early-stopping parameters
    patience = 50000000000  # look as this many examples regardless
    start_time = time.time()
    mid_time = start_time
    past_time= mid_time
    epoch = 0
    done_looping = False


    max_acc=0.0
#     combined=range(train_size)
#     ent_vocab_set=set(range(ent_vocab_size))#set(range(ent_vocab_size))

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
#         random.shuffle(combined) #shuffle training set for each new epoch, is supposed to promote performance, but not garrenteed
        iter_accu=0
        cost_i=0.0

        for folder_id in range(folder_size):

            target_rel_id=target_rels_list[folder_id]

            train_pos_rels_matrix=train_pos_rels_matrix_folders[folder_id]
            train_pos_masks_matrix=train_pos_masks_matrix_folders[folder_id]
#             train_pos_ents_matrix=train_pos_ents_matrix_folders[folder_id]
            train_pos_pathPad_list=train_pos_path_masks_folders[folder_id]
#             print 'train_pos_rels_matrix_folders:', train_pos_rels_matrix_folders
#             print 'train_pos_ents_matrix_folders:', train_pos_ents_matrix_folders
#             print 'train_pos_ents_matrix:', train_pos_ents_matrix
#             exit(0)

            train_neg_rels_matrix=train_neg_rels_matrix_folders[folder_id]
            train_neg_masks_matrix=train_neg_masks_matrix_folders[folder_id]
#             train_neg_ents_matrix=train_neg_ents_matrix_folders[folder_id]
            train_neg_pathPad_list=train_neg_path_masks_folders[folder_id]
            
            train_pos_pair_size=len(train_pos_rels_matrix)/path_size
            if len(train_pos_rels_matrix)%path_size!=0:
                print 'len(train_pos_rels_matrix)%path_size!=0:', len(train_pos_rels_matrix), path_size
                exit(0)
            train_neg_pair_size=len(train_neg_rels_matrix)/path_size
            if len(train_neg_rels_matrix)%path_size!=0:
                print 'len(train_neg_rels_matrix)%path_size!=0:', len(train_neg_rels_matrix), path_size
                exit(0)
            n_train_batches_pos=train_pos_pair_size/batch_size
            train_batch_start_pos=list(np.arange(n_train_batches_pos)*batch_size)+[train_pos_pair_size-batch_size]

            n_train_batches_neg=train_neg_pair_size/batch_size
            train_batch_start_neg=list(np.arange(n_train_batches_neg)*batch_size)+[train_neg_pair_size-batch_size]
            for train_pos_iter in train_batch_start_pos:
                for train_neg_iter in train_batch_start_neg:
                    iter = (epoch - 1) * n_train_batches_pos + iter_accu +1
                    iter_accu+=1
#                     random_sample_vocab=set(random.sample(ent_vocab_set, 1000))



#                     ent_idmatrix_pos=np.asarray(train_pos_ents_matrix[train_pos_iter*path_size:(train_pos_iter+batch_size)*path_size], dtype='int32')
                    rel_idmatrix_pos=np.asarray(train_pos_rels_matrix[train_pos_iter*path_size:(train_pos_iter+batch_size)*path_size], dtype='int32')
                    mask_matrix_pos=np.asarray(train_pos_masks_matrix[train_pos_iter*path_size:(train_pos_iter+batch_size)*path_size], dtype=theano.config.floatX)
                    path_mask_pos=np.asarray(train_pos_pathPad_list[train_pos_iter*path_size:(train_pos_iter+batch_size)*path_size], dtype=theano.config.floatX)
                    
#                     ent_idmatrix_neg=np.asarray(train_neg_ents_matrix[train_neg_iter*path_size:(train_neg_iter+batch_size)*path_size], dtype='int32')
                    rel_idmatrix_neg=np.asarray(train_neg_rels_matrix[train_neg_iter*path_size:(train_neg_iter+batch_size)*path_size], dtype='int32')
                    mask_matrix_neg=np.asarray(train_neg_masks_matrix[train_neg_iter*path_size:(train_neg_iter+batch_size)*path_size], dtype=theano.config.floatX)
                    path_mask_neg=np.asarray(train_neg_pathPad_list[train_neg_iter*path_size:(train_neg_iter+batch_size)*path_size], dtype=theano.config.floatX)

#                     print 'ent_idmatrix_neg:'
#                     print ent_idmatrix_neg
#                     exit(0)
#                     print 'ent_idmatrix_pos[:,0]', ent_idmatrix_pos[:,0]
                    cost_i+=train_model(rel_idmatrix_pos,
#                                                    np.asarray(rel_idmatrix_to_word2vec_init(rel_idmatrix_pos, rel_id2wordlist, word2vec, 300), dtype=theano.config.floatX),
                                                   mask_matrix_pos,
#                                                    ent_idmatrix_pos[:,1:],
#                                                    neg_entity_tensor_v2(ent_idmatrix_pos, rel_idmatrix_pos, tuple2tailset, rel2tailset, neg_size, random_sample_vocab),
                                                   path_mask_pos,
#                                                     ent_idmatrix_neg[:,0],
                                                   rel_idmatrix_neg,
#                                                    np.asarray(rel_idmatrix_to_word2vec_init(rel_idmatrix_neg, rel_id2wordlist, word2vec, 300), dtype=theano.config.floatX),
                                                   mask_matrix_neg,
#                                                    ent_idmatrix_neg[:,1:],
#                                                    neg_entity_tensor_v2(ent_idmatrix_neg, rel_idmatrix_neg, tuple2tailset, rel2tailset, neg_size, random_sample_vocab),
                                                   path_mask_neg,
                                                   target_rel_id
#                                                    np.asarray(rel_id_to_word2vec_init(target_rel_id, rel_id2wordlist, word2vec, 300), dtype=theano.config.floatX),
                                                   )



                    #after each 1000 batches, we test the performance of the model on all test data
                    if iter%100==0:
                        print 'Epoch ', epoch, 'iter '+str(iter)+' average cost: '+str(cost_i/iter_accu), 'uses ', (time.time()-past_time)/60.0, 'min'

                        past_time = time.time()
                        aver_map=0.0
                        for folder_id in range(folder_size):
#                             print 'Testing...folder', folder_id
                            target_rel_id=target_rels_list[folder_id]

                            test_pos_rels_matrix=test_pos_rels_matrix_folders[folder_id]
                            test_pos_masks_matrix=test_pos_masks_matrix_folders[folder_id]
#                             test_pos_ents_matrix=test_pos_ents_matrix_folders[folder_id]
                            test_pos_pathPad_list=test_pos_path_masks_folders[folder_id]

                            test_neg_rels_matrix=test_neg_rels_matrix_folders[folder_id]
                            test_neg_masks_matrix=test_neg_masks_matrix_folders[folder_id]
#                             test_neg_ents_matrix=test_neg_ents_matrix_folders[folder_id]
                            test_neg_pathPad_list=test_neg_path_masks_folders[folder_id]

                            test_pos_path_size=len(test_pos_rels_matrix)
                            test_pos_pair_size=test_pos_path_size/path_size
                            test_pos_pair_remain=test_pos_pair_size%batch_size
                            if test_pos_path_size%path_size!=0:
                                print 'test_pos_path_size%path_size=0:', test_pos_path_size, path_size
                                exit(0)

                            test_neg_path_size=len(test_neg_rels_matrix)
                            test_neg_pair_size=test_neg_path_size/path_size
                            test_neg_pair_remain=test_neg_pair_size%batch_size
                            if test_neg_path_size%path_size!=0:
                                print 'test_neg_path_size%path_size!=0:', test_neg_path_size, path_size
                                exit(0)

                            # print 'test pair size:',test_pos_pair_size, test_neg_pair_size
                            # exit(0)
                            n_test_batches_pos=test_pos_pair_size/batch_size
                            test_batch_start_pos=list(np.arange(n_test_batches_pos)*batch_size)+[test_pos_pair_size-test_pos_pair_remain]

                            n_test_batches_neg=test_neg_pair_size/batch_size
                            test_batch_start_neg=list(np.arange(n_test_batches_neg)*batch_size)+[test_neg_pair_size-test_neg_pair_remain]
                            cosine_pos_list=[]
                            for test_pos_iter in test_batch_start_pos:



#                                     ent_idmatrix_pos=np.asarray(test_pos_ents_matrix[test_pos_iter*path_size:(test_pos_iter+batch_size)*path_size], dtype='int32')
                                    rel_idmatrix_pos=np.asarray(test_pos_rels_matrix[test_pos_iter*path_size:(test_pos_iter+batch_size)*path_size], dtype='int32')
                                    mask_matrix_pos=np.asarray(test_pos_masks_matrix[test_pos_iter*path_size:(test_pos_iter+batch_size)*path_size], dtype=theano.config.floatX)
                                    path_mask_pos=np.asarray(test_pos_pathPad_list[test_pos_iter*path_size:(test_pos_iter+batch_size)*path_size], dtype=theano.config.floatX)
                                    
#                                     if path_mask_pos.shape[0]!=ent_idmatrix_pos.shape[0]:
#                                         print 'path_mask_pos.shape[0]!=ent_idmatrix_pos.shape[0]:', path_mask_pos.shape[0],ent_idmatrix_pos.shape[0]
#                                         exit(0)
                                    
                                    cosine_batch_pos=test_model(rel_idmatrix_pos,
#                                                                    np.asarray(rel_idmatrix_to_word2vec_init(rel_idmatrix_pos, rel_id2wordlist, word2vec, 300), dtype=theano.config.floatX),
                                                                    mask_matrix_pos,
                                                                   path_mask_pos,
                                                                   target_rel_id
#                                                                    np.asarray(rel_id_to_word2vec_init(target_rel_id, rel_id2wordlist, word2vec, 300), dtype=theano.config.floatX)
                                                                    )
                                    cosine_pos_list+=list(cosine_batch_pos)

                            cosine_neg_list=[]
                            for   test_neg_iter in test_batch_start_neg:
#                                 ent_idmatrix_neg=np.asarray(test_neg_ents_matrix[test_neg_iter*path_size:(test_neg_iter+batch_size)*path_size], dtype='int32')
                                rel_idmatrix_neg=np.asarray(test_neg_rels_matrix[test_neg_iter*path_size:(test_neg_iter+batch_size)*path_size], dtype='int32')
                                mask_matrix_neg=np.asarray(test_neg_masks_matrix[test_neg_iter*path_size:(test_neg_iter+batch_size)*path_size], dtype=theano.config.floatX)
                                path_mask_neg=np.asarray(test_neg_pathPad_list[test_neg_iter*path_size:(test_neg_iter+batch_size)*path_size], dtype=theano.config.floatX)
                                
                                cosine_batch_neg=test_model(rel_idmatrix_neg,
#                                                                np.asarray(rel_idmatrix_to_word2vec_init(rel_idmatrix_neg, rel_id2wordlist, word2vec, 300), dtype=theano.config.floatX),
                                                               mask_matrix_neg,
                                                               path_mask_neg,
                                                               target_rel_id
#                                                                np.asarray(rel_id_to_word2vec_init(target_rel_id, rel_id2wordlist, word2vec, 300), dtype=theano.config.floatX)
                                                               )
                                cosine_neg_list+=list(cosine_batch_neg)

                            if len(cosine_pos_list) !=test_pos_pair_size or len(cosine_neg_list) !=test_neg_pair_size:
                                print 'len(cosine_pos_list) !=test_pos_pair_size or len(cosine_neg_list) !=test_neg_pair_size'
                                print len(cosine_pos_list),test_pos_pair_size,len(cosine_neg_list),test_neg_pair_size
                                exit(0)


                            map=compute_map(cosine_pos_list+cosine_neg_list, [1]*test_pos_pair_size+[0]*test_neg_pair_size)*100
#                             print '\t\t\t temp map:', map
                            aver_map+=map
                        aver_map/=folder_size
                        if aver_map > max_acc:
                            max_acc=aver_map
                            if max_acc > max_performance:
                                store_model_to_file(rootPath+'Best_Paras_v2_'+str(max_acc), params)
                                print 'Finished storing best  params at:', max_acc
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
