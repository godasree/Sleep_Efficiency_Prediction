import pyprind
import numpy as np
import tensorflow as tf


def dataImputation_GAIN(mask_window,window_dataset,epouh):
    def generator(dataVector, noiseVector, maskVector):
        generatorInput = maskVector * dataVector + (1 - maskVector) * noiseVector
        InputMaskConcat = tf.concat(axis=1, values=[generatorInput, maskVector])
        Generator_hint1 = tf.nn.relu(tf.matmul(InputMaskConcat, Generator_weight1) + Generator_bias1)
        Generator_hint2 = tf.nn.relu(tf.matmul(Generator_hint1, Generator_weight2) + Generator_bias2)
        Generator_probability = tf.nn.sigmoid(tf.matmul(Generator_hint2, Generator_weight3) + Generator_bias3)
        return Generator_probability

    def discriminator(dataVector, maskVector, generator_sample, hintVector):
        discriminatorInput = maskVector * dataVector + (1 - maskVector) * generator_sample
        inputs = tf.concat(axis=1, values=[discriminatorInput, hintVector])
        Discriminator_hint1 = tf.nn.relu(tf.matmul(inputs, Discriminator_weight1) + Discriminator_bias1)
        Discriminator_hint2 = tf.nn.relu(tf.matmul(Discriminator_hint1, Discriminator_weight2) + Discriminator_bias2)
        Discriminator_logit = tf.matmul(Discriminator_hint2, Discriminator_weight3) + Discriminator_bias3
        Discriminator_probability = tf.nn.sigmoid(Discriminator_logit)
        return Discriminator_probability
    reshaped_window_dataset = window_dataset.reshape((-1,window_dataset.shape[1]*window_dataset.shape[2]))
    reshaped_mask_window = mask_window.reshape((-1,window_dataset.shape[1]*window_dataset.shape[2]))
    Dataset_dimensions = reshaped_window_dataset.shape[1]
    data_Sample_size = reshaped_window_dataset.shape[0]
    Test_Sample_No = 150
    Train_Sample_No = data_Sample_size-Test_Sample_No
    train_sample = reshaped_window_dataset[:Train_Sample_No]
    minibatch_size = 8
    missing_rate_probability = 0.6
    hint_rate_probability = 0.8
    alphaloss = 10
    train_mask_matrix = generateMaskHint(Train_Sample_No, Dataset_dimensions, missing_rate_probability)
    dataVector = tf.placeholder(tf.float32, shape = [None, Dataset_dimensions])
    maskVector = tf.placeholder(tf.float32, shape = [None, Dataset_dimensions])
    hintVector = tf.placeholder(tf.float32, shape = [None, Dataset_dimensions])
    noiseVector = tf.placeholder(tf.float32, shape = [None, Dataset_dimensions])
    Discriminator_weight1 = tf.Variable(initialize_xavier([Dataset_dimensions*2, 256]))
    Discriminator_bias1 = tf.Variable(tf.zeros(shape = [256]))
    Discriminator_weight2 = tf.Variable(initialize_xavier([256, 128]))
    Discriminator_bias2 = tf.Variable(tf.zeros(shape = [128]))
    Discriminator_weight3 = tf.Variable(initialize_xavier([128, Dataset_dimensions]))
    Discriminator_bias3 = tf.Variable(tf.zeros(shape = [Dataset_dimensions]))
    Generator_weight1 = tf.Variable(initialize_xavier([Dataset_dimensions*2, 256]))
    Generator_bias1 = tf.Variable(tf.zeros(shape = [256]))
    Generator_weight2 = tf.Variable(initialize_xavier([256, 128]))
    Generator_bias2 = tf.Variable(tf.zeros(shape = [128]))
    Generator_weight3 = tf.Variable(initialize_xavier([128, Dataset_dimensions]))
    Generator_bias3 = tf.Variable(tf.zeros(shape = [Dataset_dimensions]))
    discriminator_varlist = [Discriminator_weight1, Discriminator_weight2, Discriminator_weight3, Discriminator_bias1,
                           Discriminator_bias2, Discriminator_bias3]
    generator_varlist = [Generator_weight1, Generator_weight2, Generator_weight3, Generator_bias1, Generator_bias2, Generator_bias3]
    generator_sample = generator(dataVector,noiseVector,maskVector)
    discriminator_probability = discriminator(dataVector, maskVector, generator_sample, hintVector)
    discriminator_loss = -tf.reduce_mean(maskVector * tf.log(discriminator_probability + 1e-8) + (1-maskVector) * tf.log(1. - discriminator_probability + 1e-8)) * 2
    generator_loss1 = -tf.reduce_mean((1-maskVector) * tf.log(discriminator_probability + 1e-8)) / tf.reduce_mean(1-maskVector)
    train_loss_MSE = tf.reduce_mean((maskVector * dataVector - maskVector * generator_sample)**2) / tf.reduce_mean(maskVector)
    generator_loss = generator_loss1  + alphaloss * train_loss_MSE
    test_loss_MSE = tf.reduce_mean(((1-maskVector) * dataVector - (1-maskVector)*generator_sample)**2) / tf.reduce_mean(1-maskVector)
    discriminator_solver = tf.train.AdamOptimizer().minimize(discriminator_loss, var_list=discriminator_varlist)
    generator_solver = tf.train.AdamOptimizer().minimize(generator_loss, var_list=generator_varlist)
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    savertensorFlow = tf.train.Saver(var_list = generator_varlist)
    for iteration in pyprind.prog_bar(range(epouh)):
        minibatchIndex = minibatch_index(Train_Sample_No, minibatch_size)
        data_minibatch = train_sample[minibatchIndex,:]
        noise_minibatch = generateRandomNoise(minibatch_size, Dataset_dimensions)
        mask_minibatch = train_mask_matrix[minibatchIndex,:]
        Hint_minibatch1 = generateMaskHint(minibatch_size, Dataset_dimensions, 1-hint_rate_probability)
        Hint_minibatch = mask_minibatch * Hint_minibatch1
        
        New_data_minibatch = mask_minibatch * data_minibatch + (1-mask_minibatch) * noise_minibatch
        _, Discloss_curr_iter = session.run([discriminator_solver, discriminator_loss], feed_dict = {dataVector: data_minibatch, maskVector: mask_minibatch, noiseVector: New_data_minibatch, hintVector:Hint_minibatch})
        _, genloss_curr_iter, MSE_train_loss_curr, MSE_test_loss_curr = session.run([generator_solver, generator_loss1, train_loss_MSE, test_loss_MSE],
                                                                        feed_dict = {dataVector: data_minibatch, maskVector: mask_minibatch, noiseVector: New_data_minibatch, hintVector: Hint_minibatch})
        savertensorFlow.save(session, './checkpoint/gain.ckpt')

        if iteration % 100 == 0:
            print('Iter: {}'.format(iteration))
            print('Training_loss_MSE: {:.4}'.format(MSE_train_loss_curr))
            print('Testing_loss_MSE: {:.4}'.format(MSE_test_loss_curr))
            print()
        if iteration == 100:
            data_minibatch = reshaped_window_dataset
            mask_minibatch = reshaped_mask_window
            noise_minibatch= generateRandomNoise(data_Sample_size, Dataset_dimensions)
            New_data_minibatch = mask_minibatch * data_minibatch + (1-mask_minibatch) * noise_minibatch
            gen_samples = session.run(generator_sample,feed_dict = {dataVector: reshaped_window_dataset, maskVector: reshaped_mask_window, noiseVector: New_data_minibatch})
            gen_samples = mask_minibatch * data_minibatch + (1-mask_minibatch) * gen_samples
    saver = tf.train.Saver(var_list=generator_varlist)
    
    with tf.Session() as session:
        saver.restore(session, tf.train.latest_checkpoint('checkpoint'))
        data_minibatch = reshaped_window_dataset
        mask_minibatch = reshaped_mask_window
        noise_minibatch = generateRandomNoise(data_minibatch.shape[0], Dataset_dimensions)
        New_data_minibatch = mask_minibatch * data_minibatch + (1-mask_minibatch) * noise_minibatch
        gen_samples = session.run(generator_sample,feed_dict = {dataVector: reshaped_window_dataset, maskVector: reshaped_mask_window, noiseVector: New_data_minibatch})
        generate_window = mask_minibatch * data_minibatch + (1-mask_minibatch) * gen_samples
    return generate_window


def generateMaskHint(samplesize, totalsampledim, missing_rate):
    randomNumber = np.random.uniform(0., 1., size = [samplesize, totalsampledim])
    booleanrand = randomNumber > missing_rate
    maskvalue = 1.*booleanrand
    return maskvalue

def initialize_xavier(size):
    inputDimensions = size[0]
    xavier_standarddeviation = 1. / tf.sqrt(inputDimensions / 2.)
    return tf.random_normal(shape = size, stddev = xavier_standarddeviation)

def generateRandomNoise(samplesize, totalsampledim):
    return np.random.uniform(0., 1., size = [samplesize, totalsampledim])

def minibatch_index(totalsampledim,samplesize):
    permuted_matrix = np.random.permutation(totalsampledim)
    permuted_matrix_minibatch = permuted_matrix[:samplesize]
    return permuted_matrix_minibatch
