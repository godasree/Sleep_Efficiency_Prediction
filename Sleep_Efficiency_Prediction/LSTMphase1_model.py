import tensorflow as tf
def phase1train(modelname, trainingdata, testdata, lengthsequence, batchsize, ratelearning, hiddensize, epoch_num):
    tf.reset_default_graph()
    MODEL_PATH = 'model'
    X = tf.placeholder(tf.float32, [None, lengthsequence, 14], name='X')
    Y = tf.placeholder(tf.float32, [None], name='Y')
    keep_prob_ph = tf.placeholder(tf.float32, name='keep_prob_ph')
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hiddensize, forget_bias=1.0)
    rnn_outputs, _ = tf.nn.dynamic_rnn(lstm_cell, X, dtype=tf.float32)
    attention_output, alphas = attention(rnn_outputs, lengthsequence)
    drop = tf.nn.dropout(attention_output, keep_prob_ph)
    W2 = tf.Variable(tf.truncated_normal([hiddensize, 1], stddev=0.1))
    b2 = tf.Variable(tf.constant(0., shape=[1]))
    y_hat = tf.nn.relu(tf.matmul(drop, W2) + b2)
    y_hat = tf.squeeze(y_hat)
    loss = tf.reduce_mean(tf.square(y_hat - Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=ratelearning).minimize(loss)
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    total_batch = int(trainingdata.shape[0] / batchsize)
    log = open('log/log_{}.csv'.format(modelname), 'w')
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(epoch_num):
            loss_train = 0
            for i in range(total_batch):
                batch = trainingdata[i * batchsize: (i + 1) * batchsize]
                batch_xs = batch[:, 7 - lengthsequence:7, :]
                batch_ys = batch[:, 7, 3]
                loss_tr, _ = sess.run([loss, optimizer],
                                      feed_dict={X: batch_xs, Y: batch_ys, keep_prob_ph: 0.8})
                loss_train += loss_tr
            loss_train /= total_batch

            if epoch % 10 == 0:
                log.write("{} epoch: {}\t training loss: {:.6f}\n".format(modelname, epoch, loss_train))

        saver.save(sess, "./{}/{}".format(MODEL_PATH, modelname))

        test_loss, alphas, y_hat, test_phase2_h = sess.run([loss, alphas, y_hat, attention_output],
                                                           feed_dict={X: testdata[:, 7 - lengthsequence:7, :],
                                                                      Y: testdata[:, 7, 3],
                                                                      keep_prob_ph: 1.0})

        training_phase2_h = sess.run(attention_output,
                                     feed_dict={X: trainingdata[:, 7 - lengthsequence:7, :],
                                                Y: trainingdata[:, 7, 3],
                                                keep_prob_ph: 1.0})

        log.write("{} final test loss: {}".format(modelname, test_loss))
        log.close()

        return test_loss, alphas, y_hat, training_phase2_h, test_phase2_h


def attention(rnnoutput, lengthsequence):
    hidden_size = rnnoutput.shape[2].value

    weight_query = tf.Variable(tf.random_normal([lengthsequence * hidden_size, hidden_size], stddev=0.1))
    bias_query = tf.Variable(tf.random_normal([hidden_size], stddev=0.1))
    queryvector = tf.tanh(
        tf.tensordot(tf.reshape(rnnoutput, [-1, lengthsequence * hidden_size]), weight_query, axes=1) + bias_query)
    resized_inputs = tf.nn.l2_normalize(rnnoutput, dim=[1, 2])

    vu = tf.squeeze(tf.matmul(resized_inputs, tf.expand_dims(queryvector, -1)))
    alphas = tf.nn.softmax(vu, name='alphas')
    output = tf.reduce_sum(rnnoutput * tf.expand_dims(alphas, -1), 1)
    return output, alphas