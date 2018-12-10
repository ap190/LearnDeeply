#!/usr/bin/env python3.6.5

'''
=========================

Script to preprocess the data.

=========================
'''

import json
import numpy as np
import utils
import sys

def main():
    json_data = utils.preprocess.json_data
    preproc_funct = utils.preprocess.preproc_functs
    add_data = utils.preprocess.add_model_data

    num_likes = []

    # image classification preprocessing
    image_class_data = {
        'detections': [],
        'probabilities': []
    }
    # metadata preprocessing
    meta_data = {
        'followers' : [],
        'following' : [],
        'num_posts' : [],
        'num_tags'  : [],
        'len_desc'  : [],
        'num_ments' : [],
        'avg_likes' : [],
        'tag_weight': [],
        'weekday'   : [],
        'hour'      : []
    }
    # nima preprocessing
    nima_data = {
        'image_path': sys.path[0] + '/data/',
        'images': []
    }

    print('preprocessing data . . .')
    for user in json_data:
        for image in user['images']:
            likes = utils.to_int(image['likes']) 

            if not likes > 0:
                continue

            # labels are all the same for the models, the number of likes
            num_likes.append(likes)

            # metadata preproocessing
            user_data = {'following': user['following'], 'followers': user['followers'], 'posts': user['posts'], 'avg_likes': user['avg_likes']}
            info_data = {'user': user_data, 'image': image}
            preproc_funct['metadata'](info_data, meta_data)

            # image classification preprocessing
            preproc_funct['image_class'](image, image_class_data)

            # nima preprocessing
            image_path = nima_data['image_path'] + image['picture']
            preproc_funct['nima'](image_path, nima_data)

    # add metadata
    print('                 . . . adding metadata')
    meta_data = {
        'followers' : np.column_stack(np.array(meta_data['followers']) / max(meta_data['followers'])).T,
        'following' : np.column_stack(np.array(meta_data['following']) / max(meta_data['following'])).T,
        'num_posts' : np.column_stack(np.array(meta_data['num_posts']) / max(meta_data['num_posts'])).T,
        'num_tags'  : np.column_stack(np.array(meta_data['num_tags'])  / max(meta_data['num_tags'])).T,
        'len_desc'  : np.column_stack(np.array(meta_data['len_desc'])  / max(meta_data['len_desc'])).T,
        'num_ments' : np.column_stack(np.array(meta_data['num_ments']) / max(meta_data['num_ments'])).T,
        'avg_likes' : np.column_stack(np.array(meta_data['avg_likes'])).T,
        'tag_weight': np.column_stack(np.array(meta_data['tag_weight']) / max(meta_data['tag_weight'])).T,
        'weekday'   : np.array(meta_data['weekday']),
        'hour'      : np.array(meta_data['hour'])
    }
    meta_data = np.concatenate((meta_data['followers'], meta_data['following'], meta_data['num_posts'], meta_data['num_tags'],
        meta_data['len_desc'], meta_data['num_ments'], meta_data['avg_likes'], meta_data['tag_weight'],
        meta_data['weekday'], meta_data['hour']), axis=1)
    meta_data = {
        'combined_meta': list(meta_data)
    }
    add_data('metadata', meta_data)

    # add image classification data
    print('                 . . . adding image classifications')
    image_class_data['detections'], image_class_data['vocab_size'], _ = utils.embed_vector(image_class_data['detections'])
    add_data('image_class', image_class_data)

    # add nima 
    print('                 . . . adding nima data')
    add_data('nima', nima_data)

    # add labels
    print('                 . . . adding labels')
    add_data('labels', utils.log(np.array(num_likes)))

    print('preprocessing finished.')
    stop

if __name__ == 'preprocess_data':
    main()

