import json
from pprint import pprint
import codecs
import re
import numpy
import operator
import string
import random


# def generate_paths_from_socher_length1():
def find_entity_listlist(tuple2tailset, entity, target, rel_list, i):
    tuple=(entity, rel_list[i])
    tailset=tuple2tailset.get(tuple)
    if tailset is None:
#         print entity, rel_list[i], 'has no tailset'
        return [], False
    else:
        if target in tailset and i==len(rel_list)-1:
            return [entity], True
        elif target not in tailset and i==len(rel_list)-1:
            return [], False
        else:
            tail_list=list(tailset)
            random.shuffle(tail_list)
            for tail_candidate in tail_list:
                sublist, flag=find_entity_listlist(tuple2tailset, tail_candidate, target, rel_list, i+1)
                if flag is True:
                    return [entity]+sublist, True
#             print entity, rel_list[i], 'all tailset failed'
            return [], False  # is all sub entiti failed to return True, then return False
def check_path_soundness(entity_list, rel_list, tuple2tailset):
    size=len(rel_list)
    flag=True
    for i in range(size):
        tailset=tuple2tailset.get((entity_list[i], rel_list[i]))
        if entity_list[i+1] not in tailset:
            flag=False
            break
    return flag

def recover_entities_for_guu_paths():
    tuple2tailset=load_all_triples()
#     print tuple2tailset.get(('saladin', 'place_of_birth'))
    rootpath='/mounts/data/proj/wenpeng/Dataset/FB_socher/path/'
    files=['train', 'dev']
    for file in files:
        readfile=open(rootpath+file, 'r')
        print 'recovering file', rootpath+file, '...'
        writefile=open(rootpath+file+'_ent_recovered.txt', 'w')
        line_co=0
        for line in readfile:
#             print line
            parts=line.strip().split('\t')
            head=parts[0]
            tail=parts[-1]
            path=parts[1]
            rel_list=path.split(',')
            entity_list, flag=find_entity_listlist(tuple2tailset, head, tail, rel_list, 0)
#             correct=check_path_soundness(entity_list+[tail], rel_list, tuple2tailset)
#             if correct is False:
#                 print 'correct is False'
#                 print line
#                 exit(0)
            if flag is False:
                print 'correct is False'
                print line
                exit(0)
            for i in range(len(rel_list)):
                writefile.write(entity_list[i]+'\t'+rel_list[i]+'\t')
            writefile.write(tail+'\n')
            line_co+=1
            if line_co % 1000==0:
                print line_co, '....',file
        readfile.close()
        writefile.close()
        print '\t\t\t\t ..........over'

def load_all_triples_inIDs(ent_str2id, relation_str2id, tuple2tailset):
    rootpath=    '/mounts/data/proj/wenpeng/Dataset/FB_socher/length_1/'
    files=['train.txt', 'dev.txt', 'test.txt']
#     triple2id={}
#     tuple2tailset={}

    for file_id, file in enumerate(files):
        readfile=open(rootpath+file, 'r')
        add_sum=0
        for line in readfile:
            parts=line.strip().split()
            length=len(parts)
            if length ==3 or (length ==4 and parts[3]=='1'):
                head_id=ent_str2id.get(parts[0])
                rel_id=relation_str2id.get(parts[1])
                r_rel_id=relation_str2id.get('_'+parts[1])
                tail_id=ent_str2id.get(parts[2])

                if head_id is not None and rel_id is not None and tail_id is not None:
                    tuple=(head_id, rel_id)
                    exist_tailset=tuple2tailset.get(tuple)
                    if exist_tailset is not None and tail_id not in exist_tailset:
                        exist_tailset.add(tail_id)
                        tuple2tailset[tuple]=exist_tailset
                        add_sum+=1

#                 if tail_id is not None and r_rel_id is not None and head_id is not None:
#                     tuple=(tail_id, r_rel_id)
#                     exist_tailset=tuple2tailset.get(tuple)
#                     if exist_tailset is not None and head_id not in exist_tailset:
#                         exist_tailset.add(head_id)
#                         tuple2tailset[tuple]=exist_tailset
#                         add_sum+=1



        readfile.close()
        print rootpath+file, '... load over, size', len(tuple2tailset), 'add tuple2tailset size:', add_sum
    return tuple2tailset

def load_all_triples():
    rootpath=    '/mounts/data/proj/wenpeng/Dataset/FB_socher/length_1/'
    files=['train.txt', 'dev.txt', 'test.txt']
#     triple2id={}
    tuple2tailset={}
    for file_id, file in enumerate(files):
        readfile=open(rootpath+file, 'r')
        for line in readfile:
            parts=line.strip().split()
            length=len(parts)
            if length ==3 or (length ==4 and parts[3]=='1'):
                tuple=(parts[0], parts[1])
                tail=parts[2]
                r_tuple=(parts[2], '**'+parts[1])
                r_tail=parts[0]
#                 triple=(parts[0], parts[1], parts[2])
                exist_tailset=tuple2tailset.get(tuple)
                if exist_tailset is  None:
                    exist_tailset=set()
                exist_tailset.add(tail)
                tuple2tailset[tuple]=exist_tailset

                r_exist_tailset=tuple2tailset.get(r_tuple)
                if r_exist_tailset is  None:
                    r_exist_tailset=set()
                r_exist_tailset.add(r_tail)
                tuple2tailset[r_tuple]=r_exist_tailset


        readfile.close()
        print rootpath+file, '... load over, size', len(tuple2tailset)
    return tuple2tailset


def keylist_2_valuelist(keylist, dic, start_index=0):
    value_list=[]
    for key in keylist:
        value=dic.get(key)
        if value is None:
            value=len(dic)+start_index
            dic[key]=value
        value_list.append(value)
    return value_list

def add_tuple2tailset(ent_path, one_path, tuple2tailset):
    size=len(one_path)
    if len(ent_path)!=size+1:
        print 'len(ent_path)!=len(one_path)+1:', len(ent_path),size
        exit(0)
    for i in range(size):
        tuple=(ent_path[i], one_path[i])
        tail=ent_path[i+1]
        tailset=tuple2tailset.get(tuple)
        if tailset is None:
            tailset=set()
        if tail not in tailset:
            tailset.add(tail)
            tuple2tailset[tuple]=tailset
def add_rel2tailset(ent_path, one_path, rel2tailset):
    size=len(one_path)
    if len(ent_path)!=size+1:
        print 'len(ent_path)!=len(one_path)+1:', len(ent_path),size
        exit(0)
    for i in range(size):
#         tuple=(ent_path[i], one_path[i])
        tail=ent_path[i+1]
        rel=one_path[i]
        tailset=rel2tailset.get(rel)
        if tailset is None:
            tailset=set()
        if tail not in tailset:
            tailset.add(tail)
            rel2tailset[rel]=tailset
def add_ent2relset(ent_path, one_path, ent2relset, maxSetSize):
    size=len(one_path)
    if len(ent_path)!=size+1:
        print 'len(ent_path)!=len(one_path)+1:', len(ent_path),size
        exit(0)
    for i in range(size):
        ent_id=ent_path[i+1]
        rel_id=one_path[i]
        relset=ent2relset.get(ent_id)
        if relset is None:
            relset=set()
        if rel_id not in relset:
            relset.add(rel_id)
            if len(relset) > maxSetSize:
                maxSetSize=len(relset)
            ent2relset[ent_id]=relset 
    return maxSetSize   

def load_guu_data(maxPathLen=20):
    rootPath='/mounts/data/proj/wenpeng/Dataset/FB_socher/path/'
    files=['train_ent_recovered.txt', 'test_ent_recovered.txt']
#     rootPath='/mounts/data/proj/wenpeng/Dataset/FB_socher/length_1/'
#     files=['/mounts/data/proj/wenpeng/Dataset/FB_socher/length_1/train.txt', '/mounts/data/proj/wenpeng/Dataset/FB_socher/path/train_ent_recovered.txt', '/mounts/data/proj/wenpeng/Dataset/FB_socher/path/test_ent_recovered.txt']
    relation_str2id={}
    relation_id2wordlist={}
    ent_str2id={}
    tuple2tailset={}
    train_paths_store=[]
    train_ents_store=[]

    train_masks_store=[]


#     dev_paths_store=[]
#     dev_targets_store=[]
#     dev_masks_store=[]
#     dev_heads_store=[]

    test_paths_store=[]
    test_ents_store=[]
    test_masks_store=[]

    max_path_len=0
    for file_id, fil in enumerate(files):

            filename=rootPath+fil
            print 'loading', filename, '...'
            readfile=open(filename, 'r')
            line_co=0
            for line in readfile:

                parts=line.strip().split('\t')
                ent_list=[]
                rel_list=[]
                for i in range(len(parts)):
                    if i%2==0:
                        ent_list.append(parts[i])
                    else:
                        rel_list.append(parts[i].replace('**', '_'))
                if len(ent_list)!=len(rel_list)+1:
                    print 'len(ent_list)!=len(rel_list)+1:', len(ent_list),len(rel_list)
                    print 'line:', line
                    exit(0)
                ent_path=keylist_2_valuelist(ent_list, ent_str2id, 0)
                one_path=[]
                for potential_relation in rel_list:

                    rel_id=relation_str2id.get(potential_relation)
                    if rel_id is None:
                        rel_id=len(relation_str2id)+1
                        relation_str2id[potential_relation]=rel_id
                    wordlist=potential_relation.split('_')
#                                 wordIdList=strs2ids(potential_relation.split(), word2id)
                    relation_id2wordlist[rel_id]=wordlist
                    one_path.append(rel_id)
                add_tuple2tailset(ent_path, one_path, tuple2tailset)

                #pad
                valid_size=len(one_path)
                if valid_size > max_path_len:
                    max_path_len=valid_size
                pad_size=maxPathLen-valid_size
                if pad_size > 0:
                    one_path=[0]*pad_size+one_path
                    # ent_path=ent_path[:pad_size]+ent_path
                    ent_path=ent_path[:1]+ent_path[1:2]*pad_size+ent_path[1:]
                    one_mask=[0.0]*pad_size+[1.0]*valid_size
                else:
                    one_path=one_path[-maxPathLen:]  # select the last max_len relations
                    ent_path=ent_path[:1]+ent_path[-maxPathLen:]
                    one_mask=[1.0]*maxPathLen

                if file_id < 1:
                    if len(ent_path)!=maxPathLen+1 or len(one_path) != maxPathLen:
                        print 'len(ent_path)!=5:',len(ent_path), len(one_path)
                        print 'line:', line
                        exit(0)
                    train_paths_store.append(one_path)
                    train_ents_store.append(ent_path)
                    train_masks_store.append(one_mask)
                else:
                    test_paths_store.append(one_path)
                    test_ents_store.append(ent_path)
                    test_masks_store.append(one_mask)

                # line_co+=1
                # if line_co==10000:#==0:
                #     #  print line_co
                #     break

            readfile.close()
            print '\t\t\t\tload over, overall ',    len(train_paths_store), ' train,', len(test_paths_store), ' test,', 'tuple2tailset size:', len(tuple2tailset),', max path len:', max_path_len

    return ((train_paths_store, train_masks_store, train_ents_store),
            (test_paths_store, test_masks_store, test_ents_store)) , relation_id2wordlist,ent_str2id, relation_str2id, tuple2tailset
def load_guu_data_v2(maxPathLen=20):
    rootPath='/mounts/data/proj/wenpeng/Dataset/FB_socher/path/'
    files=['train_ent_recovered.txt', 'test_ent_recovered.txt']
#     rootPath='/mounts/data/proj/wenpeng/Dataset/FB_socher/length_1/'
#     files=['/mounts/data/proj/wenpeng/Dataset/FB_socher/length_1/train.txt', '/mounts/data/proj/wenpeng/Dataset/FB_socher/path/train_ent_recovered.txt', '/mounts/data/proj/wenpeng/Dataset/FB_socher/path/test_ent_recovered.txt']
    relation_str2id={}
    relation_id2wordlist={}
    ent_str2id={}
    tuple2tailset={}
    rel2tailset={}
    train_paths_store=[]
    train_ents_store=[]
    ent2relset={}
    ent2relset_maxSetSize=0

    train_masks_store=[]


#     dev_paths_store=[]
#     dev_targets_store=[]
#     dev_masks_store=[]
#     dev_heads_store=[]

    test_paths_store=[]
    test_ents_store=[]
    test_masks_store=[]

    max_path_len=0
    for file_id, fil in enumerate(files):

            filename=rootPath+fil
            print 'loading', filename, '...'
            readfile=open(filename, 'r')
            line_co=0
            for line in readfile:

                parts=line.strip().split('\t')
                ent_list=[]
                rel_list=[]
                for i in range(len(parts)):
                    if i%2==0:
                        ent_list.append(parts[i])
                    else:
                        rel_list.append(parts[i].replace('**', '_'))
                if len(ent_list)!=len(rel_list)+1:
                    print 'len(ent_list)!=len(rel_list)+1:', len(ent_list),len(rel_list)
                    print 'line:', line
                    exit(0)
                ent_path=keylist_2_valuelist(ent_list, ent_str2id, 0)
                one_path=[]
                for potential_relation in rel_list:

                    rel_id=relation_str2id.get(potential_relation)
                    if rel_id is None:
                        rel_id=len(relation_str2id)+1
                        relation_str2id[potential_relation]=rel_id
                    wordlist=potential_relation.split('_')
#                                 wordIdList=strs2ids(potential_relation.split(), word2id)
                    relation_id2wordlist[rel_id]=wordlist
                    one_path.append(rel_id)
                add_tuple2tailset(ent_path, one_path, tuple2tailset)
                add_rel2tailset(ent_path, one_path, rel2tailset)
                ent2relset_maxSetSize=add_ent2relset(ent_path, one_path, ent2relset, ent2relset_maxSetSize)

                #pad
                valid_size=len(one_path)
                if valid_size > max_path_len:
                    max_path_len=valid_size
                pad_size=maxPathLen-valid_size
                if pad_size > 0:
                    one_path=[0]*pad_size+one_path
                    # ent_path=ent_path[:pad_size]+ent_path
                    ent_path=ent_path[:1]*(pad_size+1)+ent_path[1:]
                    one_mask=[0.0]*pad_size+[1.0]*valid_size
                else:
                    one_path=one_path[-maxPathLen:]  # select the last max_len relations
                    ent_path=ent_path[:1]+ent_path[-maxPathLen:]
                    one_mask=[1.0]*maxPathLen

                if file_id < 1:
                    if len(ent_path)!=maxPathLen+1 or len(one_path) != maxPathLen:
                        print 'len(ent_path)!=5:',len(ent_path), len(one_path)
                        print 'line:', line
                        exit(0)
                    train_paths_store.append(one_path)
                    train_ents_store.append(ent_path)
                    train_masks_store.append(one_mask)
                else:
                    test_paths_store.append(one_path)
                    test_ents_store.append(ent_path)
                    test_masks_store.append(one_mask)

                # line_co+=1
                # if line_co==10000:#==0:
                #     #  print line_co
                #     break

            readfile.close()
            print '\t\t\t\tload over, overall ',    len(train_paths_store), ' train,', len(test_paths_store), ' test,', 'tuple2tailset size:', len(tuple2tailset),', max path len:', max_path_len, 'max ent2relsetSize:', ent2relset_maxSetSize

    return ((train_paths_store, train_masks_store, train_ents_store),
            (test_paths_store, test_masks_store, test_ents_store)) , relation_id2wordlist,ent_str2id, relation_str2id, tuple2tailset, rel2tailset, ent2relset, ent2relset_maxSetSize

def neg_entity_tensor(ent_idmatrix, rel_idmatrix, pair2tailset, neg_size, ent_vocab_set):
    batch_size=len(ent_idmatrix)
    length=len(rel_idmatrix[0])
    if len(ent_idmatrix[0])!=length+1:
        print 'error in neg ent generation, len(ent_idlist)!=length+1:', len(ent_idmatrix[0]), length
        exit(0)
    negs=[]
    for ba in range(batch_size):
        for i in range(length):

            pair=(ent_idmatrix[ba][i], rel_idmatrix[ba][i])
            tailset=pair2tailset.get(pair)
            if tailset is None:
                tailset=set()
#             if ent_idmatrix[ba][i+1] not in tailset:
#                 print 'error in neg ent generation'
            neg_cand_set=ent_vocab_set-tailset
            neg_list=list(random.sample(neg_cand_set, neg_size))
            negs.append(neg_list)
    return numpy.asarray(negs, dtype='int32').reshape((batch_size, length, neg_size))

def neg_entity_tensor_v2(ent_idmatrix, rel_idmatrix, pair2tailset, rel2tailset, neg_size, ent_vocab_set):
    batch_size=len(ent_idmatrix)
    length=len(rel_idmatrix[0])
    if len(ent_idmatrix[0])!=length+1:
        print 'error in neg ent generation, len(ent_idlist)!=length+1:', len(ent_idmatrix[0]), length
        exit(0)
    negs=[]
    for ba in range(batch_size):
        for i in range(length):
            rel_id=rel_idmatrix[ba][i]
            pair=(ent_idmatrix[ba][i], rel_id)
            tailset=pair2tailset.get(pair)
            if tailset is None:
                tailset=set()
#             if ent_idmatrix[ba][i+1] not in tailset:
#                 print 'error in neg ent generation'
            rel_tailset=rel2tailset.get(rel_id)
            if rel_tailset is None:
                rel_tailset=set()
            key_neg_range_set=rel_tailset-tailset
            remain_size=neg_size-len(key_neg_range_set)
            if remain_size<=0:
                neg_list=random.sample(key_neg_range_set, neg_size)
            else:
                neg_cand_set=ent_vocab_set-key_neg_range_set
                neg_list=list(key_neg_range_set)+random.sample(neg_cand_set, remain_size)
            negs.append(neg_list)
    return numpy.asarray(negs, dtype='int32').reshape((batch_size, length, neg_size))

def load_guu_data_4_CompTransE(maxPathLen=20):
    rootPath='/mounts/data/proj/wenpeng/Dataset/FB_socher/path/'
    files=['train_ent_recovered.txt', 'test_ent_recovered.txt']
#     rootPath='/mounts/data/proj/wenpeng/Dataset/FB_socher/length_1/'
#     files=['/mounts/data/proj/wenpeng/Dataset/FB_socher/length_1/train.txt', '/mounts/data/proj/wenpeng/Dataset/FB_socher/path/train_ent_recovered.txt', '/mounts/data/proj/wenpeng/Dataset/FB_socher/path/test_ent_recovered.txt']
    relation_str2id={}
    relation_id2wordlist={}
    rel_id2inid={}
    ent_str2id={}
    tuple2tailset={}
    rel2tailset={}
    train_paths_store=[]
    train_ents_store=[]
    ent2relset={}
    ent2relset_maxSetSize=0

    train_masks_store=[]


#     dev_paths_store=[]
#     dev_targets_store=[]
#     dev_masks_store=[]
#     dev_heads_store=[]

    test_paths_store=[]
    test_ents_store=[]
    test_masks_store=[]

    max_path_len=0
    for file_id, fil in enumerate(files):

            filename=rootPath+fil
            print 'loading', filename, '...'
            readfile=open(filename, 'r')
            line_co=0
            for line in readfile:

                parts=line.strip().split('\t')
                ent_list=[]
                rel_list=[]
                for i in range(len(parts)):
                    if i%2==0:
                        ent_list.append(parts[i])
                    else:
                        rel_list.append(parts[i].replace('**', '_'))
                if len(ent_list)!=len(rel_list)+1:
                    print 'len(ent_list)!=len(rel_list)+1:', len(ent_list),len(rel_list)
                    print 'line:', line
                    exit(0)
                ent_path=keylist_2_valuelist(ent_list, ent_str2id, 0)
                one_path=[]
                for potential_relation in rel_list:

                    rel_id=relation_str2id.get(potential_relation)
                    if rel_id is None:
                        rel_id=len(relation_str2id)+1
                        relation_str2id[potential_relation]=rel_id
                    wordlist=potential_relation.split('_')
#                                 wordIdList=strs2ids(potential_relation.split(), word2id)
                    relation_id2wordlist[rel_id]=wordlist
                    one_path.append(rel_id)
                    if rel_id not in rel_id2inid and potential_relation[0]=='_':
                        inID=relation_str2id.get(potential_relation[1:])
                        if inID is not None:
                            rel_id2inid[rel_id]=inID
                add_tuple2tailset(ent_path, one_path, tuple2tailset)
                add_rel2tailset(ent_path, one_path, rel2tailset)
                ent2relset_maxSetSize=add_ent2relset(ent_path, one_path, ent2relset, ent2relset_maxSetSize)

                #pad
                valid_size=len(one_path)
                if valid_size > max_path_len:
                    max_path_len=valid_size
                pad_size=maxPathLen-valid_size
                if pad_size > 0:
                    one_path=[0]*pad_size+one_path
                    # ent_path=ent_path[:pad_size]+ent_path
                    ent_path=ent_path[:1]*(pad_size+1)+ent_path[1:]
                    one_mask=[0.0]*pad_size+[1.0]*valid_size
                else:
                    one_path=one_path[-maxPathLen:]  # select the last max_len relations
                    ent_path=ent_path[:1]+ent_path[-maxPathLen:]
                    one_mask=[1.0]*maxPathLen

                if file_id < 1:
                    if len(ent_path)!=maxPathLen+1 or len(one_path) != maxPathLen:
                        print 'len(ent_path)!=5:',len(ent_path), len(one_path)
                        print 'line:', line
                        exit(0)
                    train_paths_store.append(one_path)
                    train_ents_store.append(ent_path)
                    train_masks_store.append(one_mask)
                else:
                    test_paths_store.append(one_path)
                    test_ents_store.append(ent_path)
                    test_masks_store.append(one_mask)

                # line_co+=1
                # if line_co==10000:#==0:
                #     #  print line_co
                #     break

            readfile.close()
            print '\t\t\t\tload over, overall ',    len(train_paths_store), ' train,', len(test_paths_store), ' test,', 'tuple2tailset size:', len(tuple2tailset),', max path len:', max_path_len, 'max ent2relsetSize:', ent2relset_maxSetSize

    return ((train_paths_store, train_masks_store, train_ents_store),
            (test_paths_store, test_masks_store, test_ents_store)) , relation_id2wordlist,ent_str2id, relation_str2id, tuple2tailset, rel2tailset, ent2relset, ent2relset_maxSetSize, rel_id2inid


if __name__ == '__main__':
#     load_all_triples()
    recover_entities_for_guu_paths()
