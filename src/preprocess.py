import json
from pprint import pprint
import codecs
import re
import numpy
import operator
import string
from itertools import izip

def strs2ids(str_list, word2id):
    ids=[]
    for word in str_list:
        id=word2id.get(word)
        if id is None:
            id=len(word2id)   # start from 0
        word2id[word]=id
        ids.append(id)
    return ids
def replacePunctuationsInStrByUnderline(str_input):
    replace_punctuation = string.maketrans(string.punctuation, ' '*len(string.punctuation))
    pure_str= str_input.translate(replace_punctuation).strip()
    return '_'.join(pure_str.split())


def load_relationPath_and_labels(maxPathLen=20):
    rootPath='/mounts/data/proj/wenpeng/Dataset/UMASS_relation_pred/release_with_entities_akbc/'
    folders=['_architecture_structure_address',
             '_aviation_airport_serves',
             '_book_book_characters',
             '_book_literary_series_works_in_this_series',
             '_book_written_work_original_language',
             '_broadcast_content_artist',
             '_broadcast_content_genre',
             '_business_industry_companies',
             '_cvg_computer_videogame_cvg_genre',
             '_cvg_game_version_game',
             '_cvg_game_version_platform',
             '_education_educational_institution_campuses',
             '_education_educational_institution_school_type',
             '_film_film_cinematography',
             '_film_film_country',
             '_film_film_directed_by',
             '_film_film_film_festivals',
             '_film_film_language',
             '_film_film_music',
             '_film_film_rating',
             '_film_film_sequel',
             '_geography_river_cities',
             '_geography_river_mouth',
             '_location_location_contains',
             '_music_album_genre',
             '_music_artist_genre',
             '_music_artist_label',
             '_music_artist_origin',
             '_music_composition_composer',
             '_music_composition_lyricist',
             '_music_genre_albums',
             '_organization_organization_founders',
             '_organization_organization_locations',
             '_organization_organization_sectors',
             '_people_deceased_person_cause_of_death',
             '_people_deceased_person_place_of_death',
             '_people_ethnicity_people',
             '_people_family_members',
             '_people_person_nationality',
             '_people_person_place_of_birth',
             '_people_person_profession',
             '_people_person_religion',
             '_soccer_football_player_position_s',
             '_time_event_locations',
             '_tv_tv_program_country_of_origin',
             '_tv_tv_program_genre']
    files=['positive_matrix.tsv.translated','negative_matrix.tsv.translated','dev_matrix.tsv.translated','test_matrix.tsv.translated']
    relation_str2id={}
    relation_id2wordlist={}
    word2id={}
    train_paths_store=[]
    train_targets_store=[]
    train_masks_store=[]
    train_labels_store=[]

    dev_paths_store=[]
    dev_targets_store=[]
    dev_masks_store=[]
    dev_labels_store=[]

    test_paths_store=[]
    test_targets_store=[]
    test_masks_store=[]
    test_labels_store=[]
    for file_id, fil in enumerate(files):
#         file_paths_store=[]
#         file_targets_store=[]
#         file_masks_store=[]
#         file_labels_store=[]
        for folder in folders:
            target_rel=replacePunctuationsInStrByUnderline(folder).strip()
            target_rel_id=relation_str2id.get(target_rel)
            if target_rel_id is None:
                target_rel_id=len(relation_str2id)+1
                relation_str2id[target_rel]=target_rel_id
                wordlist=target_rel.split()
#                 wordIdList=strs2ids(wordlist, word2id)
                relation_id2wordlist[target_rel_id]=wordlist
#             folder_paths_store=[]
#             folder_targets_store=[]
#             folder_masks_store=[]
#             folder_labels_store=[]

            filename=rootPath+folder+'/'+fil
            print 'loading', folder+'/'+fil, '...'
            readfile=open(filename, 'r')

            for line in readfile:

                parts=line.strip().split('\t')
                if (len(parts) !=3 and file_id <2) or (len(parts) !=4 and file_id >=2):
                    print 'len(parts):', len(parts), 'file_id:', file_id
                    exit(0)

                relationPaths=parts[2]
                path_list=relationPaths.split('###')
                for relationPath in path_list:

                    one_path=[]
                    one_mask=[]
                    pathSplit=relationPath.split('-')
                    for potential_relation in pathSplit:
                        if potential_relation.find('/m/')<0: # is a relation
                            potential_relation=replacePunctuationsInStrByUnderline(potential_relation).strip()
                            rel_id=relation_str2id.get(potential_relation)
                            if rel_id is None:
                                rel_id=len(relation_str2id)+1
                                relation_str2id[potential_relation]=rel_id
                                wordlist=potential_relation.split()
#                                 wordIdList=strs2ids(potential_relation.split(), word2id)
                                relation_id2wordlist[rel_id]=wordlist
                            one_path.append(rel_id)
                    #pad
                    valid_size=len(one_path)
                    pad_size=maxPathLen-valid_size
                    if pad_size > 0:
                        one_path=[0]*pad_size+one_path
                        one_mask=[0.0]*pad_size+[1.0]*valid_size
                    else:
                        one_path=one_path[:maxPathLen]
                        one_mask=[1.0]*maxPathLen

                    if file_id <2:
                        train_paths_store.append(one_path)
                        train_masks_store.append(one_mask)
                        train_targets_store.append(target_rel_id)
                        if file_id ==0:#posi is 1, nega is 0
                            train_labels_store.append(1)
                        else:
                            train_labels_store.append(0)

                    if file_id ==2 :
                        dev_paths_store.append(one_path)
                        dev_masks_store.append(one_mask)
                        dev_targets_store.append(target_rel_id)

                        if parts[3]=='1':
                            dev_labels_store.append(1)
                        else:
                            dev_labels_store.append(0)

                    if file_id ==3:
                        test_paths_store.append(one_path)
                        test_masks_store.append(one_mask)
                        test_targets_store.append(target_rel_id)

                        if parts[3]=='1':
                            test_labels_store.append(1)
                        else:
                            test_labels_store.append(0)

            readfile.close()
            print '\t\t\t\tload over, overall ',    len(train_paths_store), ' train,', len(dev_paths_store), ' dev,', len(test_paths_store), ' test'

    return ((train_paths_store, train_targets_store, train_masks_store, train_labels_store),
            (dev_paths_store, dev_targets_store, dev_masks_store, dev_labels_store),
            (test_paths_store, test_targets_store, test_masks_store, test_labels_store)) , relation_id2wordlist,word2id


def rel_idmatrix_to_word2vec_init(idmatrix, id2wordlist, word2vec, dim):
    emb_tensor=[]
    for idlist in idmatrix:
        emb_matrix=rel_idlist_to_word2vec_init(idlist, id2wordlist, word2vec, dim)
        emb_tensor.append(emb_matrix)
    return emb_tensor   #(batch, len, emb_size)



def rel_idlist_to_word2vec_init(idlist, id2wordlist, word2vec, dim):
    zero_emb=list(numpy.zeros(dim))
    overall_emblist=[]
    for id in idlist:
        if id ==0:
            overall_emblist.append(zero_emb)
            continue
        else:
            emb_list=[]
            wordlist=id2wordlist.get(id)
            for word in wordlist:
                emb=word2vec.get(word)
                if emb is not None:
                    emb_list.append(emb)
            if len(emb_list)==0:
                emb_list.append(zero_emb)
            sum_emb=numpy.sum(numpy.asarray(emb_list), axis=0)
            overall_emblist.append(sum_emb)
    return overall_emblist #(len, emb_size)

def rel_id_to_word2vec_init(id, id2wordlist, word2vec, dim):
    zero_emb=list(numpy.zeros(dim))

    emb_list=[]
    wordlist=id2wordlist.get(id)
    if wordlist is None:
        print 'wordlist is None'
        exit(0)
    for word in wordlist:
        emb=word2vec.get(word)
        if emb is not None:
            emb_list.append(emb)
    if len(emb_list)==0:
        emb_list.append(zero_emb)
    sum_emb=numpy.sum(numpy.asarray(emb_list), axis=0)
            
    return sum_emb #(len, emb_size)

def ent2relSet_pad(ent_size, ent2relset, maxSetSize):
    idvector=[]
    maskvector=[]
    for i in range(ent_size):
        relset=ent2relset.get(i)
        if relset is None or len(relset)==0:
            idlist=[0]*maxSetSize
            mask=[0.0]*maxSetSize
#             idmatrix.append(idlist)
#             maskMatrix.append(mask)
        else:
            valid_list=list(relset)
            valid_size=len(valid_list)
            pad_size=maxSetSize-valid_size
            if pad_size > 0:
                idlist=valid_list+[0]*pad_size
                mask=[1.0]*valid_size+[0.0]*pad_size
            else:
                idlist=valid_list[:maxSetSize]
                mask=[1.0]*maxSetSize
        idvector+=idlist
        maskvector+=mask
    return idvector, maskvector


def deal_with_one_line(head, tail, paths):
    path_list=paths.split('###')
    overall_path=[]
    for relationPath in path_list:

        one_path=[]
        one_path.append(head)
        pathSplit=relationPath.split('-')
        valid_path=True
        for id, potential_relation in enumerate(pathSplit):
            if potential_relation.find('/m/')<0: # is a relation
                if id%2==0:
                    rel=replacePunctuationsInStrByUnderline(potential_relation)
                    one_path.append(rel)
                else:
                    valid_path=False
                    break
#                     print 'potential_relation.find(/m/)<0, but id%2==0:', potential_relation, id, pathSplit
#                     exit(0)
            else: # is a entity
                if id%2==1:
                    one_path.append(potential_relation)
                else:
                    valid_path=False
                    break
#                     print 'potential_relation.find(/m/)>0, but id%2==0:', potential_relation, id, pathSplit
#                     exit(0)
        if valid_path is True:
            one_path.append(tail)
            overall_path.append('\t'.join(one_path))
    return overall_path



def reformat_UMASS(maxPathLen=20):
    rootPath='/mounts/data/proj/wenpeng/Dataset/UMASS_relation_pred/release_with_entities_akbc/'
    folders=['_architecture_structure_address',
             '_aviation_airport_serves',
             '_book_book_characters',
             '_book_literary_series_works_in_this_series',
             '_book_written_work_original_language',
             '_broadcast_content_artist',
             '_broadcast_content_genre',
             '_business_industry_companies',
             '_cvg_computer_videogame_cvg_genre',
             '_cvg_game_version_game',
             '_cvg_game_version_platform',
             '_education_educational_institution_campuses',
             '_education_educational_institution_school_type',
             '_film_film_cinematography',
             '_film_film_country',
             '_film_film_directed_by',
             '_film_film_film_festivals',
             '_film_film_language',
             '_film_film_music',
             '_film_film_rating',
             '_film_film_sequel',
             '_geography_river_cities',
             '_geography_river_mouth',
             '_location_location_contains',
             '_music_album_genre',
             '_music_artist_genre',
             '_music_artist_label',
             '_music_artist_origin',
             '_music_composition_composer',
             '_music_composition_lyricist',
             '_music_genre_albums',
             '_organization_organization_founders',
             '_organization_organization_locations',
             '_organization_organization_sectors',
             '_people_deceased_person_cause_of_death',
             '_people_deceased_person_place_of_death',
             '_people_ethnicity_people',
             '_people_family_members',
             '_people_person_nationality',
             '_people_person_place_of_birth',
             '_people_person_profession',
             '_people_person_religion',
             '_soccer_football_player_position_s',
             '_time_event_locations',
             '_tv_tv_program_country_of_origin',
             '_tv_tv_program_genre']
    files=['positive_matrix.tsv.translated','negative_matrix.tsv.translated','dev_matrix.tsv.translated','test_matrix.tsv.translated']

    writetrain=open(rootPath+'wenpeng/train.txt', 'w')


    for file_id, fil in enumerate(files):
#         file_paths_store=[]
#         file_targets_store=[]
#         file_masks_store=[]
#         file_labels_store=[]
        for folder in folders:
            target_rel=replacePunctuationsInStrByUnderline(folder)

            filename=rootPath+folder+'/'+fil
            print 'reformatting', folder+'/'+fil, '...'
            readfile=open(filename, 'r')
            if file_id==3:
                writetest=open(rootPath+'wenpeng/test'+folder+'.txt', 'w')

            for line in readfile:

                parts=line.strip().split('\t')
                if (len(parts) !=3 and file_id <2) or (len(parts) !=4 and file_id >=2):
                    print 'len(parts):', len(parts), 'file_id:', file_id
                    exit(0)

                relationPaths=parts[2]
                head=parts[0]
                tail=parts[1]

                path_list=deal_with_one_line(head, tail, relationPaths)
                for path in path_list:
                    writetrain.write(path+'\n')




                if file_id ==0: #posi
                    writetrain.write(head+'\t'+target_rel+'\t'+tail+'\n')

                if file_id ==2 or file_id ==3 :
                    if parts[3]=='1':
                        writetrain.write(head+'\t'+target_rel+'\t'+tail+'\n')

                #creat test file
                if file_id ==3:
                    if parts[3]=='-1':
                        writetest.write(head+'\t'+target_rel+'\t'+tail+'\t0'+'\n')
                    else:
                        writetest.write(head+'\t'+target_rel+'\t'+tail+'\t1'+'\n')
            if file_id ==3:
                writetest.close()
            readfile.close()
    writetrain.close()
    print 'reformat over'

def load_das_train(maxPathLen=20, example_limit=10000):
    filename='/mounts/data/proj/wenpeng/Dataset/UMASS_relation_pred/release_with_entities_akbc/wenpeng/train.txt'

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



    max_path_len=0

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
                rel_list.append(parts[i])
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
            one_path=one_path[:maxPathLen]  # select the first max_len relations
            ent_path=ent_path[:maxPathLen+1]
            one_mask=[1.0]*maxPathLen


        if len(ent_path)!=maxPathLen+1 or len(one_path) != maxPathLen:
            print 'len(ent_path)!=maxPathLen:',len(ent_path), len(one_path)
            print 'line:', line
            exit(0)
        train_paths_store.append(one_path)
        train_ents_store.append(ent_path)
        train_masks_store.append(one_mask)


        line_co+=1
        if line_co==example_limit:#==0:
            #  print line_co
            break

    readfile.close()
    print '\t\t\t\tload over, overall ',    len(train_paths_store), ' train,', 'tuple2tailset size:', len(tuple2tailset),', max path len:', max_path_len, 'max ent2relsetSize:', ent2relset_maxSetSize

    return (train_paths_store, train_masks_store, train_ents_store), relation_id2wordlist,ent_str2id, relation_str2id, tuple2tailset, rel2tailset, ent2relset, ent2relset_maxSetSize

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

def load_das_test(rel2id, ent2id, relation_id2wordlist, folders_limit=10):
    rootpath='/mounts/data/proj/wenpeng/Dataset/UMASS_relation_pred/release_with_entities_akbc/wenpeng/'
    folders=['_architecture_structure_address',
             '_aviation_airport_serves',
             '_book_book_characters',
             '_book_literary_series_works_in_this_series',
             '_book_written_work_original_language',
             '_broadcast_content_artist',
             '_broadcast_content_genre',
             '_business_industry_companies',
             '_cvg_computer_videogame_cvg_genre',
             '_cvg_game_version_game',
             '_cvg_game_version_platform',
             '_education_educational_institution_campuses',
             '_education_educational_institution_school_type',
             '_film_film_cinematography',
             '_film_film_country',
             '_film_film_directed_by',
             '_film_film_film_festivals',
             '_film_film_language',
             '_film_film_music',
             '_film_film_rating',
             '_film_film_sequel',
             '_geography_river_cities',
             '_geography_river_mouth',
             '_location_location_contains',
             '_music_album_genre',
             '_music_artist_genre',
             '_music_artist_label',
             '_music_artist_origin',
             '_music_composition_composer',
             '_music_composition_lyricist',
             '_music_genre_albums',
             '_organization_organization_founders',
             '_organization_organization_locations',
             '_organization_organization_sectors',
             '_people_deceased_person_cause_of_death',
             '_people_deceased_person_place_of_death',
             '_people_ethnicity_people',
             '_people_family_members',
             '_people_person_nationality',
             '_people_person_place_of_birth',
             '_people_person_profession',
             '_people_person_religion',
             '_soccer_football_player_position_s',
             '_time_event_locations',
             '_tv_tv_program_country_of_origin',
             '_tv_tv_program_genre']
    folders_paths_store=[]
    folders_ents_store=[]
    folders_masks_store=[]
    folders_labels_store=[] # a list
    for i in range(folders_limit):
        readfile=open(rootpath+'test'+folders[i]+'.txt', 'r')
        folder_paths_store=[]
        folder_ents_store=[]
        folder_masks_store=[]
        folder_labels_store=[] # a list
#         line_co=0
        for line in readfile:

            parts=line.strip().split('\t')
            if len(parts) !=4:
                print 'len(parts) !=4:', line
                exit(0)
            ent_list=[parts[0], parts[2]]
            rel_str=parts[1]
#             rel_list=[rel_str]
            label=int(parts[3])

            ent_path=keylist_2_valuelist(ent_list, ent2id, 0)

            rel_id=rel2id.get(rel_str)
            if rel_id is None:
                rel_id=len(rel2id)+1
                rel2id[rel_str]=rel_id
            wordlist=rel_str.split('_')
#                                 wordIdList=strs2ids(potential_relation.split(), word2id)
            relation_id2wordlist[rel_id]=wordlist
            one_path=[rel_id]
            one_mask=[1.0]

#             add_tuple2tailset(ent_path, one_path, tuple2tailset)
#             add_rel2tailset(ent_path, one_path, rel2tailset)
#             ent2relset_maxSetSize=add_ent2relset(ent_path, one_path, ent2relset, ent2relset_maxSetSize)


            folder_paths_store.append(one_path)
            folder_ents_store.append(ent_path)
            folder_masks_store.append(one_mask)
            folder_labels_store.append(label)

        folders_paths_store.append(folder_paths_store)
        folders_ents_store.append(folder_ents_store)
        folders_masks_store.append(folder_masks_store)
        folders_labels_store.append(folder_labels_store)

        readfile.close()
    print '\t\t\t\tload test folders over'

    return folders_paths_store, folders_ents_store, folders_masks_store, folders_labels_store
def compute_map(sub_probs,sub_labels ):

    sub_dict = [(prob, label) for prob, label in izip(sub_probs, sub_labels)] # a list of tuple
    #sorted_probs=sorted(sub_probs, reverse = True)
    sorted_tuples=sorted(sub_dict,key=lambda tup: tup[0], reverse = True)

    map=0.0
#     find=False
    corr_no=0
    #MAP
    for index, (prob,label) in enumerate(sorted_tuples):
        if label==1:
            corr_no+=1 # the no of correct answers
#             all_corr_answer+=1
            map+=1.0*corr_no/(index+1)
#             find=True
#     #MRR
#     for index, (prob,label) in enumerate(sorted_tuples):
#         if label==1:
#             all_mrr+=1.0/(index+1)
#             break # only consider the first correct answer
#     if find is False:
#         print 'Did not find correct answers'
#         exit(0)
    map=map/corr_no
    return map

def load_das_v2(maxPathLen=20, path_size=10):
    rootPath='/mounts/data/proj/wenpeng/Dataset/UMASS_relation_pred/release_with_entities_akbc/'
    folders=['_architecture_structure_address',
             '_aviation_airport_serves',
             '_book_book_characters',
             '_book_literary_series_works_in_this_series',
             '_book_written_work_original_language',
             '_broadcast_content_artist',
             '_broadcast_content_genre',
             '_business_industry_companies',
             '_cvg_computer_videogame_cvg_genre',
             '_cvg_game_version_game',
             '_cvg_game_version_platform',
             '_education_educational_institution_campuses',
             '_education_educational_institution_school_type',
             '_film_film_cinematography',
             '_film_film_country',
             '_film_film_directed_by',
             '_film_film_film_festivals',
             '_film_film_language',
             '_film_film_music',
             '_film_film_rating',
             '_film_film_sequel',
             '_geography_river_cities',
             '_geography_river_mouth',
             '_location_location_contains',
             '_music_album_genre',
             '_music_artist_genre',
             '_music_artist_label',
             '_music_artist_origin',
             '_music_composition_composer',
             '_music_composition_lyricist',
             '_music_genre_albums',
             '_organization_organization_founders',
             '_organization_organization_locations',
             '_organization_organization_sectors',
             '_people_deceased_person_cause_of_death',
             '_people_deceased_person_place_of_death',
             '_people_ethnicity_people',
             '_people_family_members',
             '_people_person_nationality',
             '_people_person_place_of_birth',
             '_people_person_profession',
             '_people_person_religion',
             '_soccer_football_player_position_s',
             '_time_event_locations',
             '_tv_tv_program_country_of_origin',
             '_tv_tv_program_genre']
    files=['positive_matrix.tsv.translated','negative_matrix.tsv.translated','test_matrix.tsv.translated']
    relation_str2id={}
    entity_str2id={}
    relation_id2wordlist={}


    tuple2tailset={}
    rel2tailset={}
    ent2relset={}
    ent2relset_maxSetSize=0

    train_pos_rels_matrix_folders=[]
    train_pos_masks_matrix_folders=[]
    train_pos_ents_matrix_folders=[]

    train_neg_rels_matrix_folders=[]
    train_neg_masks_matrix_folders=[]
    train_neg_ents_matrix_folders=[]

    test_pos_rels_matrix_folders=[]
    test_pos_masks_matrix_folders=[]
    test_pos_ents_matrix_folders=[]

    test_neg_rels_matrix_folders=[]
    test_neg_masks_matrix_folders=[]
    test_neg_ents_matrix_folders=[]

    target_rels_list=[]

    for folder in folders[:2]:
        target_rel=replacePunctuationsInStrByUnderline(folder).strip()
        target_rel_id=relation_str2id.get(target_rel)
        if target_rel_id is None:
            target_rel_id=len(relation_str2id)+1
            relation_str2id[target_rel]=target_rel_id
            wordlist=target_rel.split('_')
            relation_id2wordlist[target_rel_id]=wordlist
        target_rels_list.append(target_rel_id)

        train_pos_rels_matrix=[]
        train_pos_masks_matrix=[]
        train_pos_ents_matrix=[]

        train_neg_rels_matrix=[]
        train_neg_masks_matrix=[]
        train_neg_ents_matrix=[]

        test_pos_rels_matrix=[]
        test_pos_masks_matrix=[]
        test_pos_ents_matrix=[]

        test_neg_rels_matrix=[]
        test_neg_masks_matrix=[]
        test_neg_ents_matrix=[]
        for file_id, fil in enumerate(files):



            filename=rootPath+folder+'/'+fil
            print 'loading', folder+'/'+fil, '...'
            readfile=open(filename, 'r')

            for line in readfile:

                parts=line.strip().split('\t')
                if (len(parts) !=3 and file_id <2) or (len(parts) !=4 and file_id >=2):
                    print 'len(parts):', len(parts), 'file_id:', file_id
                    exit(0)


                relationPaths=parts[2]
                head=parts[0]
                tail=parts[1]

                path_list=deal_with_one_line(head, tail, relationPaths)

                current_path_size=len(path_list)
                if current_path_size ==0:
#                     print line
#                     print path_list
#                     exit(0)
                    continue
                repeat_path_times=path_size/current_path_size
                remain_path_times=path_size%current_path_size
                if current_path_size < path_size:
                    path_list=path_list*repeat_path_times+path_list[:remain_path_times]
                else:
                    path_list=path_list[:path_size]


                for relationPath in path_list:

                    one_path=[]
                    one_mask=[]
                    one_ents=[]
                    pathSplit=relationPath.split() #ent and rel appear alternatively
                    for id, potential_relation in enumerate(pathSplit):
                        if id % 2 ==1: # is a relation

                            rel_id=relation_str2id.get(potential_relation)
                            if rel_id is None:
                                rel_id=len(relation_str2id)+1
                                relation_str2id[potential_relation]=rel_id
                                wordlist=potential_relation.split('_')
#                                 wordIdList=strs2ids(potential_relation.split(), word2id)
                                relation_id2wordlist[rel_id]=wordlist
                            one_path.append(rel_id)
                        else: # is a ent
                            ent_id=entity_str2id.get(potential_relation)
                            if ent_id is None:
                                ent_id=len(entity_str2id)
                                entity_str2id[potential_relation]=ent_id
                            one_ents.append(ent_id)
                    add_tuple2tailset(one_ents, one_path, tuple2tailset)
                    add_rel2tailset(one_ents, one_path, rel2tailset)
                    ent2relset_maxSetSize=add_ent2relset(one_ents, one_path, ent2relset, ent2relset_maxSetSize)
                    #pad
                    valid_size=len(one_path)
                    pad_size=maxPathLen-valid_size
                    if pad_size > 0:
                        one_path=[0]*pad_size+one_path
                        one_mask=[0.0]*pad_size+[1.0]*valid_size
                        one_ents=one_ents[:1]*(pad_size+1)+one_ents[1:]
                    else:
                        one_path=one_path[:maxPathLen]
                        one_mask=[1.0]*maxPathLen
                        one_ents=one_ents[:maxPathLen+1]
                    
#                     if len(one_path)!=maxPathLen or len(one_ents)!=maxPathLen+1:
#                         print 'len(one_path)!=maxPathLen or len(one_ents)!=maxPathLen+1:', len(one_path), len(one_ents), maxPathLen
#                         exit(0)
                    
                    
#                     print 'maxPathLen:', maxPathLen
#                     print 'one_path:', one_path
#                     print 'one_ents:', one_ents
#                     exit(0)
                    if file_id ==0: #train pos
                        train_pos_rels_matrix.append(one_path)
                        train_pos_masks_matrix.append(one_mask)
                        train_pos_ents_matrix.append(one_ents)

                    if file_id ==1: #train neg
                        train_neg_rels_matrix.append(one_path)
                        train_neg_masks_matrix.append(one_mask)
                        train_neg_ents_matrix.append(one_ents)

                    if file_id ==2 : #test file
                        if parts[3]=='1':
                            test_pos_rels_matrix.append(one_path)
                            test_pos_masks_matrix.append(one_mask)
                            test_pos_ents_matrix.append(one_ents)
                        else:
                            test_neg_rels_matrix.append(one_path)
                            test_neg_masks_matrix.append(one_mask)
                            test_neg_ents_matrix.append(one_ents)
            readfile.close()
        #store each folder
        train_pos_ents_matrix_folders.append(train_pos_ents_matrix)
        train_pos_rels_matrix_folders.append(train_pos_rels_matrix)
        train_pos_masks_matrix_folders.append(train_pos_masks_matrix)

        train_neg_ents_matrix_folders.append(train_neg_ents_matrix)
        train_neg_rels_matrix_folders.append(train_neg_rels_matrix)
        train_neg_masks_matrix_folders.append(train_neg_masks_matrix)

        test_pos_ents_matrix_folders.append(test_pos_ents_matrix)
        test_pos_rels_matrix_folders.append(test_pos_rels_matrix)
        test_pos_masks_matrix_folders.append(test_pos_masks_matrix)

        test_neg_ents_matrix_folders.append(test_neg_ents_matrix)
        test_neg_rels_matrix_folders.append(test_neg_rels_matrix)
        test_neg_masks_matrix_folders.append(test_neg_masks_matrix)

            # print '\t\t\t\tload over, overall ',    len(train_paths_store), ' train,', len(dev_paths_store), ' dev,', len(test_paths_store), ' test'
#     print train_pos_rels_matrix_folders[0]
#     exit(0)
    return     [[train_pos_rels_matrix_folders, train_pos_masks_matrix_folders,train_pos_ents_matrix_folders],
        [train_neg_rels_matrix_folders, train_neg_masks_matrix_folders, train_neg_ents_matrix_folders],
        [test_pos_rels_matrix_folders, test_pos_masks_matrix_folders, test_pos_ents_matrix_folders],
        [test_neg_rels_matrix_folders, test_neg_masks_matrix_folders, test_neg_ents_matrix_folders]], target_rels_list, relation_str2id, entity_str2id, relation_id2wordlist, tuple2tailset, rel2tailset, ent2relset, ent2relset_maxSetSize


if __name__ == '__main__':
    reformat_UMASS()
