import numpy as np
import tensorflow as tf

#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
import random
from dataloader import Gen_Data_loader, Dis_dataloader
from generator import Generator
from discriminator import Discriminator
from rollout import ROLLOUT
import collections
import datetime
import networkx as nx
import subprocess
import pandas as pd
#########################################################################################
#  Generator  Hyper-parameters
######################################################################################
EMB_DIM = 2 # embedding dimension
HIDDEN_DIM = 2 # hidden state dimension of lstm cell
SEQ_LENGTH = 10 # sequence length
PRE_EPOCH_NUM = 10 # supervise (maximum likelihood estimation) epochs
BATCH_SIZE = 64

#########################################################################################
#  Discriminator  Hyper-parameters
#########################################################################################
dis_embedding_dim = 2
# dis_filter_sizes = [1,2]
dis_filter_sizes = [3, 4]
# dis_num_filters = [10,10]
dis_num_filters = [10,10]
dis_dropout_keep_prob = 0.75
dis_l2_reg_lambda = 0.2
dis_batch_size = 64

#########################################################################################
#  Basic Training Parameters
#########################################################################################
vocab_size = 4181
TOTAL_BATCH = 50
network_file = r'data/PPI_dealed/HI-II-14(dealed).txt'
positive_file = r'save/PPI/real_data.txt'
negative_file = 'save/PPI/generator_sample.txt'
generated_num =60

def generate_samples(sess, trainable_model, batch_size, generated_num_total, output_file):
    # Generate Samples
    generated_samples = []
    real_start = np.loadtxt(positive_file)
    real_start = real_start[:,0]
    for _ in range(int(generated_num_total / batch_size)):
        start_token = np.random.randint(vocab_size, size=[BATCH_SIZE])
        generated_samples.extend(trainable_model.generate(sess,start_token))

    with open(output_file, 'w') as fout:
        for poem in generated_samples:
            buffer = ' '.join([str(x) for x in poem]) + '\n'
            fout.write(buffer)

def generate_samples_true(sess,network, batch_size, generated_num, output_file,seq_length):
    # Generate Samples
    samples = []
    for start in network.keys():
        for _ in range(generated_num):
            path = [start]
            while len(path) < seq_length:
                cur = path[-1]
                if len(network[cur]) > 0:
                    path.append(np.random.choice(network[cur]))
                else:
                    break
            samples.append(path)
    with open(output_file, 'w') as fout:
        for poem in samples:
            buffer = ' '.join([str(x) for x in poem]) + '\n'
            fout.write(buffer)


def pre_train_epoch(sess, trainable_model, data_loader):
    # Pre-train the generator using MLE for one epoch
    supervised_g_losses = []
    data_loader.reset_pointer()

    for it in range(data_loader.num_batch):
        batch = data_loader.next_batch()
        _, g_loss = trainable_model.pretrain_step(sess, batch, batch[:,0])
        supervised_g_losses.append(g_loss)
    return np.mean(supervised_g_losses)


def dis_ypred_for_auc(sess,discriminator,generator,short,graph_nx):
    start_token = np.random.randint(vocab_size, size=[BATCH_SIZE])
    samples = generator.generate(sess, start_token)
    feed = {discriminator.input_x:samples,
            discriminator.dropout_keep_prob:1.0}
    ypred = sess.run(discriminator.ypred_for_auc,feed_dict=feed)
   # length = get_length(samples,short,graph_nx)
    length=10
    return np.mean(ypred[:,1]),length


def read_graph_adjlist(file_name):
    graph = collections.defaultdict(list)
    with open(file_name,'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            line = line.split(' ')
            line = [int(x)-1 for x in line]
            for j in range(1,len(line)):
                graph[line[0]].append(line[j])
    return graph


def read_graph_edgelist(file_name):
    graph = collections.defaultdict(list)
    with open(file_name,'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            line = line.split(' ')
            line = [int(x) for x in line]
            graph[line[0]].append(line[1])
            graph[line[1]].append(line[0])
    return graph


def main():

    starttime = datetime.datetime.now()
    short = {}
    graph = read_graph_edgelist(network_file)
    #graph = read_graph_adjlist(network_file)
    graph_nx = nx.from_dict_of_lists(graph)
    gen_data_loader = Gen_Data_loader(BATCH_SIZE,SEQ_LENGTH)
    dis_data_loader = Dis_dataloader(BATCH_SIZE,SEQ_LENGTH)
    generator = Generator(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH)
    discriminator = Discriminator(sequence_length=SEQ_LENGTH, num_classes=2, vocab_size=vocab_size, embedding_size=dis_embedding_dim,
                                filter_sizes=dis_filter_sizes, num_filters=dis_num_filters, l2_reg_lambda=dis_l2_reg_lambda)
    #generated_num = np.loadtxt(positive_file).shape[0]
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    saver = tf.train.Saver(max_to_keep=12)
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    # First, generate true random walk path
    generate_samples_true(sess, graph, BATCH_SIZE, generated_num, positive_file, SEQ_LENGTH)
    gen_data_loader.create_batches(positive_file)
    #  pre-train generator
    print('Start pre-training...')
    for epoch in range(15):
        loss = pre_train_epoch(sess, generator, gen_data_loader)
        endtime = datetime.datetime.now()
        print('pre-train epoch:', epoch, ' test_loss ', loss, ' time:', (endtime - starttime).seconds)

    generated_num_total = np.loadtxt(positive_file).shape[0]
    #  pre-train discriminator
    print('Start pre-training discriminator...')
    for _ in range(3):
        generate_samples(sess, generator, BATCH_SIZE, generated_num_total, negative_file)
        dis_data_loader.load_train_data(positive_file, negative_file)
        for _ in range(3):
            d_loss_his = []
            dis_data_loader.reset_pointer()
            for it in range(dis_data_loader.num_batch):
                x_batch, y_batch = dis_data_loader.next_batch()
                feed = {
                    discriminator.input_x: x_batch,
                    discriminator.input_y: y_batch,
                    discriminator.dropout_keep_prob: dis_dropout_keep_prob
                }
                _,d_loss = sess.run([discriminator.train_op,discriminator.loss], feed)
                d_loss_his.append(d_loss)
        endtime = datetime.datetime.now()
        loss,length = dis_ypred_for_auc(sess, discriminator, generator,short,graph_nx)
        print('discriminator loss: ', np.mean(d_loss_his),'ypred:',loss,'length',length, 'time: ', (endtime - starttime).seconds)


    rollout = ROLLOUT(generator, 0.8)

    print('#########################################################################')
    print('Start Adversarial Training...')
    for total_batch in range(TOTAL_BATCH):
        if total_batch % 5 == 0:
            saver.save(sess, 'log/train.checkpoint', global_step=total_batch)
        # Train the generator for one step
        for it in range(20):
            generator.graph = graph
            start_token = np.random.randint(vocab_size, size=[BATCH_SIZE])
            samples = generator.generate(sess, start_token)
            rewards, sample_temp = rollout.get_reward(sess, samples, 16, discriminator)
            feed = {generator.x: samples, generator.rewards: rewards, generator.start_token: start_token}
            _, gen_loss = sess.run([generator.g_updates, generator.g_loss], feed_dict=feed)
            generator.graph = None
            loss,length = dis_ypred_for_auc(sess, discriminator, generator,short,graph_nx)
            endtime = datetime.datetime.now()

            print('before total_batch: ', total_batch, 'reward: ',
                  loss,'length:',length, 'test_loss: ', gen_loss, 'time: ', (endtime - starttime).seconds)
        rollout.update_params()
        loss,length = dis_ypred_for_auc(sess, discriminator, generator,short,graph_nx)
        print('after total_batch: ', total_batch, 'reward: ',
              loss,'length',length, 'time: ', (endtime - starttime).seconds)
        # Train the discriminator
        for _ in range(3):
            generator.graph = None
            generate_samples(sess, generator, BATCH_SIZE, generated_num_total, negative_file)
            dis_data_loader.load_train_data(positive_file, negative_file)
            d_loss_his = []
            for _ in range(1):
                dis_data_loader.reset_pointer()
                for it in range(dis_data_loader.num_batch):
                    x_batch, y_batch = dis_data_loader.next_batch()
                    feed = {
                        discriminator.input_x: x_batch,
                        discriminator.input_y: y_batch,
                        discriminator.dropout_keep_prob: dis_dropout_keep_prob
                    }
                    _, d_loss = sess.run([discriminator.train_op, discriminator.loss], feed)
                    d_loss_his.append(d_loss)
            endtime = datetime.datetime.now()
            loss,length = dis_ypred_for_auc(sess, discriminator, generator,short,graph_nx)
            print('discriminator loss: ', np.mean(d_loss_his), 'ypred:', loss,'length',length ,'time: ', (endtime - starttime).seconds)

        if total_batch %5 ==0:
            saver.save(sess, 'log/train.checkpoint', global_step=epoch)
    saver.save(sess, 'log/train.checkpoint', global_step=epoch+1)
    #generare final fake path
    generate_samples(sess, generator, BATCH_SIZE, generated_num_total, 'save/PPI/HI-II-14_final.txt')
   # command = 'deepwalk --input data/wiki/wiki.edgelist --outpu  save/wiki/wiki(extra).embeddings --extra save/wiki/wiki_final.txt'

  #  subprocess.call(command, shell=True)

if __name__ == '__main__':
    main()
