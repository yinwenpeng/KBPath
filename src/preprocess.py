import json
from pprint import pprint
import codecs
import re
import numpy
import operator
import string

from common_functions import replacePunctuationsInStrBySpace, strs2ids




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
            target_rel=replacePunctuationsInStrBySpace(folder).strip()
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
                            potential_relation=replacePunctuationsInStrBySpace(potential_relation).strip()
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
        
    
    
    
    
