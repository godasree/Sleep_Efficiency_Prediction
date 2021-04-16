import tensorflow as tf



def input_to_value(step, input_vec, hidden_size):
    with tf.variable_scope("InputToVal{}".format(step)):
        input_w = tf.Variable(tf.truncated_normal([hidden_size[step], hidden_size[7]], stddev=0.1)) 
        input_b = tf.Variable(tf.constant(0., shape=[hidden_size[7]]))
        input_r = tf.nn.tanh(tf.matmul(input_vec, input_w) + input_b)
        return input_r

def phase2_trainorload(modelname, traindata, trainmetadata, trainY, testdata, testmetadata, testY,
                       batchsize, learningrate, hiddensize, metadatasize, metadatahiddensize, epochs_num, keep_prob,
                      ):
    MODELPATH = 'model'
    tf.reset_default_graph()
    input_list = []
    for step in range(7):
        input_list.append(tf.placeholder(tf.float32, [None, hiddensize[step]], name='X{}'.format(step + 1)))
    
    metadata = tf.placeholder(tf.float32, [None, metadatasize], name='metaX')
    keep_prob_ph = tf.placeholder(tf.float32, name='keep_prob_ph')
    Y = tf.placeholder(tf.float32, [None], name='Y')
    reduced_value_list = []
    for step in range(7):
        reduced_value_list.append(input_to_value(step, input_list[step], hiddensize))
    Weightvector1 = tf.Variable(tf.truncated_normal([metadatasize, metadatahiddensize[0]], stddev=0.1))
    biasvector1 = tf.Variable(tf.constant(0., shape=[metadatahiddensize[0]]))
    r1 = tf.nn.relu(tf.matmul(metadata, Weightvector1) + biasvector1)
    drop1 = tf.nn.dropout(r1, keep_prob_ph)
    Weightvector2 = tf.Variable(tf.truncated_normal([metadatahiddensize[0], metadatahiddensize[1]], stddev=0.1))
    biasvector2 = tf.Variable(tf.constant(0., shape=[metadatahiddensize[1]]))
    r2 = tf.nn.relu(tf.matmul(drop1, Weightvector2) + biasvector2)
    drop2 = tf.nn.dropout(r2, keep_prob_ph)
    Weightvector3 = tf.Variable(tf.truncated_normal([metadatahiddensize[1], hiddensize[7]], stddev=0.1))
    biasvector3 = tf.Variable(tf.constant(0., shape=[hiddensize[7]]))
    query = tf.nn.tanh(tf.matmul(drop2, Weightvector3) + biasvector3)
        
    # Attention
    X_transposed = tf.transpose(tf.stack(reduced_value_list), [1, 0, 2])  
    vu = tf.matmul(tf.nn.l2_normalize(X_transposed, dim = [1, 2]), tf.expand_dims(query, -1))
    alphas = tf.nn.softmax(tf.squeeze(vu), name='alphas')
    output = tf.reduce_sum(X_transposed * tf.expand_dims(alphas, -1), 1)
    
    outW = tf.Variable(tf.truncated_normal([hiddensize[7], 1], stddev=0.1))
    outb = tf.Variable(tf.constant(0., shape=[1]))
    y_hat = tf.squeeze(tf.nn.relu(tf.matmul(output, outW) + outb))

    # Cross-entropy loss and optimizer initialization
    loss = tf.reduce_mean(tf.square(y_hat - Y))
    optimizer = tf.train.AdamOptimizer(learning_rate = learningrate).minimize(loss)

    saver = tf.train.Saver()
    init = tf.global_variables_initializer()    
    log = open('log/log_{}.csv'.format(modelname), 'w')
        
    total_batch = int(traindata[0].shape[0] / batchsize)
    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(epochs_num):
            loss_train = 0
            for i in range(total_batch):
                batch_range = range(i * batchsize, (i + 1) * batchsize)
                loss_tr, _ = sess.run([loss, optimizer],
                                      feed_dict = make_feed_dict(traindata, trainmetadata, trainY, keep_prob, batch_range))
                loss_train += loss_tr
            loss_train /= total_batch

            if  epoch % 10 == 0:
                log.write("{} epoch: {}\t training loss: {:.6f}\n".format(modelname, epoch, loss_train))

            saver.save(sess, './{}/{}'.format(MODELPATH, modelname))
  
        test_loss, alphas, y_hat = sess.run([loss, alphas, y_hat], 
                                            feed_dict= make_feed_dict(testdata, testmetadata, testY, 1.0))
               

        log.write("{} final test_loss : {}".format(modelname, test_loss))
        log.close()
            
        return test_loss, alphas, y_hat


def make_feed_dict(X, metaX, Y, keepprob, batchrange=None):
    if batchrange == None:
        return {'X1:0': X[0],
                'X2:0': X[1],
                'X3:0': X[2],
                'X4:0': X[3],
                'X5:0': X[4],
                'X6:0': X[5],
                'X7:0': X[6],
                'metaX:0': metaX,
                'Y:0': Y,
                'keep_prob_ph:0': keepprob}

    return {'X1:0': X[0][batchrange, :],
            'X2:0': X[1][batchrange, :],
            'X3:0': X[2][batchrange, :],
            'X4:0': X[3][batchrange, :],
            'X5:0': X[4][batchrange, :],
            'X6:0': X[5][batchrange, :],
            'X7:0': X[6][batchrange, :],
            'metaX:0': metaX[batchrange],
            'Y:0': Y[batchrange],
            'keep_prob_ph:0': batchrange}