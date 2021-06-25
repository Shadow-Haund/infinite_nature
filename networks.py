"""Определяет начальный кодировщик изображения и сеть уточнения."""
import config
import ops
import spade
import tensorflow as tf


def reparameterize(mu, logvar):
  """Выбрать (Sample) случайную переменную.

  Аргументы:
    mu: Среднее значение нормального шума для выборки
    logvar: log отклонение нормального шума от образца

  Вывод:
    Случайная гауссовская выборка из mu и logvar.
  """
  with tf.name_scope("reparameterization"): # к именам всех операций включенных в него добавляется префикс
    std = tf.exp(0.5 * logvar)
    eps = tf.random.normal(std.get_shape()) # выводит случайные значения из нормального распределения

    return eps * std + mu


def encoder(x, scope="spade_encoder"):
  """Энкодер выдающий глобальные N(mu, sig) параметры.

  Аргументы:
    x: [B, H, W, 4]  RGBD изображение который используется для выборки шума из распределения для
    подачи в сеть уточнения. Диапазон [0, 1].
    scope: (str) variable scope

  Вывод:
    (mu, logvar) это [B, 256] тензоры параметров, определяющих нормальное распределение для выборки.
  """

  x = 2 * x - 1
  num_channel = 16

  with tf.compat.v1.variable_scope(scope, reuse=tf.compat.v1.AUTO_REUSE): # tf.compat.v1.AUTO_REUSE говорит что
      # get_variable () должна создать запрошенную переменную, если она не существует, если же существует, то вернуть
    x = ops.sn_conv(x, num_channel, kernel_size=3, stride=2,
                    use_bias=True, use_spectral_norm=True, scope="conv_0")
    x = ops.instance_norm(x, scope="inst_norm_0")
    x = ops.leaky_relu(x, 0.2)

    x = ops.sn_conv(x, 2 * num_channel, kernel_size=3, stride=2,
                    use_bias=True, use_spectral_norm=True, scope="conv_1")
    x = ops.instance_norm(x, scope="inst_norm_1")
    x = ops.leaky_relu(x, 0.2)

    x = ops.sn_conv(x, 4 * num_channel, kernel_size=3, stride=2,
                    use_bias=True, use_spectral_norm=True, scope="conv_2")
    x = ops.instance_norm(x, scope="inst_norm_2")
    x = ops.leaky_relu(x, 0.2)

    x = ops.sn_conv(x, 8 * num_channel, kernel_size=3, stride=2,
                    use_bias=True, use_spectral_norm=True, scope="conv_3")
    x = ops.instance_norm(x, scope="inst_norm_3")
    x = ops.leaky_relu(x, 0.2)

    x = ops.sn_conv(x, 8 * num_channel, kernel_size=3, stride=2,
                    use_bias=True, use_spectral_norm=True, scope="conv_4")
    x = ops.instance_norm(x, scope="inst_norm_4")
    x = ops.leaky_relu(x, 0.2)

    x = ops.sn_conv(x, 8 * num_channel, kernel_size=3, stride=2,
                    use_bias=True, use_spectral_norm=True, scope="conv_5")
    x = ops.instance_norm(x, scope="inst_norm_5")
    x = ops.leaky_relu(x, 0.2)

    mu = ops.fully_connected(x, config.DIM_OF_STYLE_EMBEDDING,
                             scope="linear_mu")
    logvar = ops.fully_connected(x, config.DIM_OF_STYLE_EMBEDDING,
                                 scope="linear_logvar")
  return mu, logvar


def refinement_network(rgbd, mask, z, scope="spade_generator"):
  """Уточняет  rgbd, маска основана на шуме z.

  H, W должна быть разделена на 2 ** num_up_layers

  Аргументы:
    rgbd: [B, H, W, 4] отрендеренный кадр для уточнения
    mask: [B, H, W, 1] бинарная маска неизвестных областей. 1 где известно и 0 где неизвестно
    z: [B, D] вектор шума используется как шум для генератора
    scope: (str) variable scope

  Вывод:
    [B, H, W, 4] обновляет rgbd изображение.
  """
  img = 2 * rgbd - 1
  img = tf.concat([img, mask], axis=-1)

  num_channel = 32

  num_up_layers = 5
  out_channels = 4  # For RGBD

  batch_size, im_height, im_width, unused_c = rgbd.get_shape().as_list()

  init_h = im_height // (2 ** num_up_layers)
  init_w = im_width // (2 ** num_up_layers)

  with tf.compat.v1.variable_scope(scope, reuse=tf.compat.v1.AUTO_REUSE):
    x = ops.fully_connected(z, 16 * num_channel * init_h * init_w,
                            "fc_expand_z")
    x = tf.reshape(x, [batch_size, init_h, init_w, 16 * num_channel]) # # изменяет форму тензора не меняя его значения или их порядок
    x = spade.spade_resblock(
        x, img, 16 * num_channel,
        use_spectral_norm=config.USE_SPECTRAL_NORMALIZATION,
        scope="head")
    x = ops.double_size(x)
    x = spade.spade_resblock(
        x, img, 16 * num_channel,
        use_spectral_norm=config.USE_SPECTRAL_NORMALIZATION,
        scope="middle_0")
    x = spade.spade_resblock(
        x, img, 16 * num_channel,
        use_spectral_norm=config.USE_SPECTRAL_NORMALIZATION,
        scope="middle_1")
    x = ops.double_size(x)
    x = spade.spade_resblock(
        x, img, 8 * num_channel,
        use_spectral_norm=config.USE_SPECTRAL_NORMALIZATION,
        scope="up_0")
    x = ops.double_size(x)
    x = spade.spade_resblock(
        x, img, 4 * num_channel,
        use_spectral_norm=config.USE_SPECTRAL_NORMALIZATION,
        scope="up_1")
    x = ops.double_size(x)
    x = spade.spade_resblock(
        x, img, 2 * num_channel,
        use_spectral_norm=config.USE_SPECTRAL_NORMALIZATION,
        scope="up_2")
    x = ops.double_size(x)
    x = spade.spade_resblock(
        x, img, 1 * num_channel,
        use_spectral_norm=config.USE_SPECTRAL_NORMALIZATION,
        scope="up_3")
    x = ops.leaky_relu(x, 0.2)
    # Pre-trained checkpoint uses default conv scoping.
    x = ops.sn_conv(x, out_channels, kernel_size=3)
    x = tf.tanh(x) # вычисляет гиперболический тангенс каждого элемента в тензоре
    return 0.5 * (x + 1)