# Get the transpose of x in the 0-dimension, as specified by the permutation order, and

from keras.models import Model, Sequential, load_model
from keras.layers import GlobalAveragePooling2D
from keras.applications.inception_v3 import InceptionV3
import utils
import tensorflow as tf
import numpy as np


class Kernel_Inception(object):
    def __init__(
        self, input_shape, gamma: float = 1.0, lin_coeff: float = 1.0, power: int = 3
    ):
        super().__init__()
        self.gamma = gamma
        self.coef = lin_coeff
        self.pow = power
        self.inception_model = Sequential(name="inception")
        self.inception_model.add(
            InceptionV3(input_shape=input_shape, include_top=False, weights="imagenet")
        )
        self.inception_model.add(GlobalAveragePooling2D())

    def poly_kernel(self, x, y, exclude_ii=False):
        """As per the paper:
        batched
        Binkowski, M., Sutherland, D.J., Arbel, M., & Gretton, A. (2018).
        Demystifying MMD GANS. ArXiv, abs/1801.01401.
        The kernel distance is defined as k(x, y) = ( (gamma/d) (xT)y + coef)**pow.
        Where the paper used pow=3 and g oeff = 1
        S::.param x:_...The InceptionV3 embedding tensor, or np array of shape( batch_sµe,	-embedding_length)
        :.".'param y: The InceptionV3 embedding tensor, or np array of shape (batch_ 'ze, 1.,..--em6edding_length)
        :return:
        """


        # Get the size of the embedding space
        print('x shape ', x.shape)
        dim = tf.shape(x)[1]
        dim = tf.cast(dim, dtype=tf.float32)
        # assuming that the o-dimension corresponds to the batch dimension.
        x_t = tf.transpose(x)
        # Compute the kernel
        kernel = (tf.linalg.matmul(y, x_t) * (self.gamma / dim) + self.coef) ** self.pow
        #print('kernel shape ', kernel.shape)

        # exclude i,i cross product
        if exclude_ii:
            ka= kernel.numpy()
            for i in range(kernel.shape[0]):
                ka[i,i]=0
            kernel=tf.constant(ka)
        #print(kernel)
        return kernel


    def calc_mmd(self, real_imgs, gen_imgs):
        ''' 
        jwang: is this based on equa (4), code seems off, should minus the cross term
        Calculates MMDwith:
        MMD=  1/(n*(n-l))*Sum_i_j@x_i,x_j)  +  1/(m*(m-l))*Sum_i_j{k(y_i,y_j)  +
        where x and y are the generated and real images respectively and n & m are the respective
        sample sizes.
        :param real_imgs: Real images as tensor of shape (batch_size, img_width, img_height)
        :param gen_imgs: Generated images as tensor of shape (batch_size, img_width, img_height). The image shapes
        must correspond to the input shape provided upon initialization of the class. The batch size are not required
        to be equal
        return:
        ''' 
# Get the batch sizes
        n = real_imgs.shape[0]
        m = gen_imgs.shape[0]
# Get the real and generated embedding by calling the Ince tf.reduce_sum(kernel_real * (1.0 - tf.eye(batch_size))) / (	batch_size_f * (batch_size. _f. - 1.0)ptionV3 model
        x = self.inception_model(real_imgs, training=False)
        y = self.inception_model(gen_imgs, training=False)
# Calculate the real-real, gen-gen, and real-gen kernels
        k_xx = self.poly_kernel(x, x, exclude_ii=True)
        k_yy = self.poly_kernel(y, y, exclude_ii=True)
        k_xy = self.poly_kernel(x, y)
# Calculate the mean of the kernel
        mean_x = tf.reduce_sum(k_xx) / (n * (n - 1))
        mean_y = tf.reduce_sum(k_yy) / (m * (m - 1))
        mean_xy = tf.reduce_sum(k_xy) / (m * n)
        ki_mmd = mean_x + mean_y - 2 * mean_xy
        return ki_mmd

if __name__ == '__main__':
    print('0000000000000000000000000000000000000000')
    input_shape = (75, 75, 3) #inception minimum shape, mnist not good
    gen_model = Kernel_Inception(input_shape=input_shape)
    images1, _ = utils.load_data(
    "/home/student/Documents/datasets/Office-31/amazon",
    input_shape=input_shape,
    )
    images2, _ = utils.load_data(
    "/home/student/Documents/datasets/Office-31/webcam",
    input_shape=input_shape,
    )
    print('images1 len, shape ', len(images1), images1[0].shape)
    print('images2 len, shape ', len(images2), images2[0].shape)
    in1 = np.array(images1)[:64, :, :, :]
    in2 = np.array(images2)[:64, :, :, :]
    mmd = gen_model.calc_mmd(in1, in1)
    print('real-real mmd = ', mmd)
    mmd = gen_model.calc_mmd(in2, in2)
    print('gen-gen mmd = ', mmd)
    mmd = gen_model.calc_mmd(in1, in2)
    print('real-gen mmd = ', mmd)
