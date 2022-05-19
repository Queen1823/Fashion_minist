# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 19:29:16 2021

@author: Administrator
"""

#!/user/bin/env python
# -*- coding:utf-8 -*-
"""
created on
author:queen
"""
# 导入库函数
import pylab
import tensorflow as tf
print("TF version:", tf.__version__)
#检测Tensorflow是否支持GPU
print("GPU is", "available"if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")

import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time
import glob
from IPython import display

# 导入数据
# 我们使用fashion MNIST 进行GAN 的训练，生成器将生成类似于FashionMNIST的数据集
# 导入fashion mnist数据库
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# 数据处理
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5)/127.5    # 将图片标准化到[-1,1]区间内

buffer_size = 60000
batch_size = 256

# 批量化和打乱数据
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(buffer_size).batch(batch_size)

# 构建生成器模型
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256)    # 注意batch size没有限制

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape ==(None, 28, 28, 1)

    return model

# 查看生成器网络结构
generator = make_generator_model()
print(generator.summary())

# 使用（未训练的）生成器创建一张图片
noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)

plt.imshow(generated_image[0,:, :, 0],cmap='gray')
plt.show()

# 构建判别器
# 判别器可以视作一个CNN分类器
# 判别器的目的是尽量正确判别输入数据是真实数据还是来自生成器
def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                            input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2),padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

# 查看判别器网络结构
discriminator = make_discriminator_model()
print(discriminator.summary())

# 查看判别器的判别结果
decision = discriminator(generated_image)
print(decision)

# 定义损失函数和优化器
cross_entroy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# 判别器损失函数
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entroy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entroy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

# 生成器损失函数
def generator_loss(fake_output):
    return cross_entroy(tf.ones_like(fake_output), fake_output)

# 两者的优化器
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# 定义检查点
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer = generator_optimizer,
                                 discriminator_optimizer = discriminator_optimizer,
                                 generator = generator,
                                 discriminator = discriminator)

# 训练模型
# 定义超参数
epochs = 100
noise_dim = 100
num_examples_to_generate = 16

# 产生随机种子作为输入
# 后面将重复使用该种子（因此在动画GIF中更容易可视化进度）
seed = tf.random.normal([num_examples_to_generate, noise_dim])

# 定义单次训练过程
"""注意‘tf.function’的使用，该注解使函数被‘编译’为计算图模式"""
@tf.function
def train_step(images):
    noise = tf.random.normal([batch_size, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training = True)

        real_output = discriminator(images, training = True)    # 真实图像的判别结果
        fake_output = discriminator(generated_images, training = True)   # 生成图像的判别结果

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
"""训练在生成器接收到一个随机种子作为输入时开始，用于生产一张图片。判别器随后被用于区分真实图片（选自训练集）和伪造图片（由生成器生成）。
针对这里的每一个模型都计算损失函数，并且计算梯度用于更新生成器与判别器"""

# 生成和保存图片
# 注意‘training’设定为false
# 因此所有层都在推理模式下运行（batchnorm）
def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training = False)

    fig = plt.figure(figsize=(4,4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0]*127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()
    pylab.show()

# 定义循环训练过程
def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch)

            # 生成图片
            display.clear_output(wait=True)
            generate_and_save_images(generator, epoch + 1, seed)

            # 每5 epochs 进行一次存储
            if (epoch + 1 ) % 5 == 0:
                checkpoint.save(file_prefix = checkpoint_prefix)

            print("Time for epoch {} is {} sec".format(epoch + 1, time.time()-start))

# 进行训练
time     # %%time 将会给出cell的代码运行一次所花费的时间
train(train_dataset, epochs)

# 恢复最新的检查点
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

# 展示某一个 epoch 的生成图像
def display_image(epoch_no):
    return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))

print(display_image(epochs))

# 合成训练过程产生图像的GIf图
import imageio

anim_file = 'dcgan_gif'

with imageio.get_writer(anim_file, mode='I') as writer:
    filenames = glob.glob('image*.png')
    filenames = sorted(filenames)
    last = -1
    for i, filename in enumerate(filenames):
        frame = 2*(i**0.5)
        if round(frame) > round(last):
            last = frame
        else:
            continue

        image = imageio.imread(filename)
        writer.append_data(image)   # 可在文件夹中找到生成的GIF图

# 显示生成的gif动图
print(display.Image(filename=anim_file))


























































