"""Библиотека для определения SPADE компонентов и функций."""
import numpy as np
import ops
import tensorflow as tf


def diff_resize_area(tensor, new_height_width):
  """Performs a resize op that passes gradients evenly.
   Выполняет операцию изменения размера, которая равномерно передает градиенты

  Тензор проходит через изменение размера и пул, где операции изменения размера и пула определяются наименьшим
  общим множителем. Поскольку изменение размера с помощью nearest_neighbors и avg_pool распределяет градиенты
  от выхода ко входу равномерно, меньше шансов ошибок обучения. Сначала мы изменяем размер до наименьшего общего
  кратного потом avg_pool к new_height_width. Эта операция изменения размера эффективна только в тех случаях,
  когда наименьшее общее кратное мало. Обычно это происходит при повышении или понижении дискретизации в 2 раза
  (например H = 0.5 * new_H).

  Аргументы:
    tensor: тензор формы [B, H, W, D]
    new_height_width: Кортеж длиной два, который указывает новую высоту и ширину соответственно

  Возвращает:
    Тензор измененной области [B, H_new, W_new, D].
    The resize area tensor [B, H_new, W_new, D].

  Ошибки:
    RuntimeError: Если наименьшее общее кратное больше 10 x new_height_width, возникает ошибка, выпадающая для
     предотвращения неэффективного использования памяти.
  """
  new_h, new_w = new_height_width
  unused_b, curr_h, curr_w, unused_d = tensor.shape.as_list()
  # Наименьшее общее кратное, используемое для определения промежуточной операции изменения размера.
  l_h = np.lcm(curr_h, new_h)
  l_w = np.lcm(curr_w, new_w)
  if l_h == curr_h and l_w == curr_w:
    im = tensor
  elif (l_h < (10 * new_h) and l_w < (10 * new_w)):
    im = tf.compat.v1.image.resize_bilinear(    # нет описания на сайте
        tensor, [l_h, l_w], half_pixel_centers=True)
  else:
    raise RuntimeError("DifferentiableResizeArea is memory inefficient"
                       "for resizing from (%d, %d) -> (%d, %d)" %
                       (curr_h, curr_w, new_h, new_w))
  lh_factor = l_h // new_h
  lw_factor = l_w // new_w
  if lh_factor == lw_factor == 1:
    return im
  return tf.nn.avg_pool2d( # выполняет average pooling на входных данных (возвращает среднее всех значений из
      # части изображения, покрываемой фильтром)
      im, [lh_factor, lw_factor], [lh_factor, lw_factor], padding="VALID")


def spade(x,
          condition,
          num_hidden=128,
          use_spectral_norm=False,
          scope="spade"):
  """реализация Spatially Adaptive Instance Norm.

  Для данного x применяется нормализация, обусловленная условием
  Given x, applies a normalization that is conditioned on condition.

  Аргументы:
    x: [B, H, W, C] Тензор для применения нормализации
    condition: [B, H', W', C'] Тензор, определяющий параметры нормализации
    num_hidden: (int) Количество промежуточных каналов для создания слоя SPADE
    use_spectral_norm: (bool) Еслт true, создает свертки со спектральной нормализацией, применяемой к его весам
    scope: (str) The variable scope

  Возвращает:
    Тензор, нормированный параметрами, оцененными с помощью условий.
  """
  channel = x.shape[-1]
  with tf.compat.v1.variable_scope(scope, reuse=tf.compat.v1.AUTO_REUSE):
    x_normed = ops.instance_norm(x)

    # Получение аффинных параметров из заданного условиями (conditioning) изображения.
    # First resize.
    height, width = x.get_shape().as_list()[1:3]

    condition = diff_resize_area(condition, [height, width])
    condition = ops.sn_conv(
        condition,
        num_hidden,
        kernel_size=3,
        use_spectral_norm=use_spectral_norm,
        scope="conv_cond")
    condition = tf.nn.relu(condition) # Вычисляет усеченное линейное преобразование как max(condition, 0)
    gamma = ops.sn_conv(condition, channel, kernel_size=3,
                        use_spectral_norm=use_spectral_norm, scope="gamma",
                        pad_type="CONSTANT")
    beta = ops.sn_conv(condition, channel, kernel_size=3,
                       use_spectral_norm=use_spectral_norm, scope="beta",
                       pad_type="CONSTANT")

    out = x_normed * (1 + gamma) + beta
    return out


def spade_resblock(tensor,
                   condition,
                   channel_out,
                   use_spectral_norm=False,
                   scope="spade_resblock"):
  """A SPADE resblock.

  Аргументы:
    tensor: [B, H, W, C] изображение, которое будет сгенерировано
    condition: [B, H, W, D] задание условий (conditioning) изображения для вычисления параметров аффинной нормализации.
    channel_out: (int) Количество каналов выходного тензора
    use_spectral_norm: (bool) Если true, использовать спектральную нормализацию в сверточных слоях
    scope: (str) The variable scope

  Возвращает:
    Выходное значение остаточного блока spade
  """

  channel_in = tensor.get_shape().as_list()[-1]
  channel_middle = min(channel_in, channel_out)

  with tf.compat.v1.variable_scope(scope, reuse=tf.compat.v1.AUTO_REUSE):
    x = spade(tensor, condition,
              use_spectral_norm=use_spectral_norm, scope="spade_0")
    x = ops.leaky_relu(x, 0.2)
    # Эта всегда использует спектральную норму.
    x = ops.sn_conv(x, channel_middle, kernel_size=3,
                    use_spectral_norm=True, scope="conv_0")

    x = spade(x, condition,
              use_spectral_norm=use_spectral_norm, scope="spade_1")
    x = ops.leaky_relu(x, 0.2)
    x = ops.sn_conv(x, channel_out, kernel_size=3,
                    use_spectral_norm=True, scope="conv_1")

    if channel_in != channel_out:
      x_in = spade(tensor, condition,
                   use_spectral_norm=use_spectral_norm, scope="shortcut_spade")
      x_in = ops.sn_conv(x_in, channel_out, kernel_size=1, stride=1,
                         use_bias=False, use_spectral_norm=True,
                         scope="shortcut_conv")
    else:
      x_in = tensor

    out = x_in + x

  return out
