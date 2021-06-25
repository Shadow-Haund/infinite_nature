"""Вспомогательные функции для определения сетевых компонентов."""
import math

import config
import numpy as np
import tensorflow as tf


def instance_norm(inputs, scope="instance_norm"):
  with tf.compat.v1.variable_scope(scope):
    beta = None
    gamma = None
    epsilon = 1e-05
    # All axes except first (batch) and last (channels).
    axes = list(range(1, inputs.shape.ndims - 1))
    mean, variance = tf.nn.moments(inputs, axes, keepdims=True) # Вычисляет среднее значение и дисперсию inputs по осям axes
    return tf.nn.batch_normalization(inputs, mean, variance, beta, gamma, epsilon) # Выполняет Batch normalization
    # или пакетная нормализация

def fully_connected(x, units, use_bias=True, scope="linear"):
  """Создает полностью связанный слой (layer).

  Аргументы:
    x: [B, ...] пакет векторов
    units: (int) Количество выходных функций (features)
    use_bias: (bool) Если истина, определяет (возможно инициализирует)  bias term
    scope: (str) variable scope
    bias term - смещение позволяет сместить функцию активации, добавив постоянную
  Вывод:
    [B, units] вывод полносвязного слоя по x
    output of the fully connected layer on x.
  """
  with tf.compat.v1.variable_scope(scope):
    x = tf.compat.v1.layers.flatten(x) # Сглаживает входной тензор не не изменяя ось
    x = tf.compat.v1.layers.dense( # функциональный интерфейс для полносвязного слоя
        x,
        units=units,
        use_bias=use_bias)

    return x


def double_size(image):
  """Удваивает размер изображения или пакета изображений.

  Она просто дублирует каждый пиксель в блок 2x2.
  Результат идентичен использованию tf.image.resize_area чтобы удвоить размер с с добавлением возможности
  взять градиент
  with the addition that we can take the gradient.

  Аргументы:
    image: [..., H, W, C] изображение для удвоения размера

  Вывод:
    [..., H*2, W*2, C] отмасштабированное.
  """
  image = tf.convert_to_tensor(image)
  shape = image.shape.as_list()
  multiples = [1] * (len(shape) - 2) + [2, 2]
  tiled = tf.tile(image, multiples)
  newshape = shape[:-3] + [shape[-3] * 2, shape[-2] * 2, shape[-1]]
  return tf.reshape(tiled, newshape) # изменяет форму тензора не меняя его значения или их порядок


def leaky_relu(x, alpha=0.01):
  return tf.nn.leaky_relu(x, alpha) # выщитывает функцию активации, тут функция Leaky ReLU


def spectral_norm(w, iteration=1, update_variable=False):
  """Применяет спектральную нормализацию к весовому тензору (weight tensor).

  (Степенной метод - итерационный алгоритм поиска собственного значения с максимальной
  абсолютной величиной и одного из соответствующих собственных векторов для произвольной матрицы)

  (Спектральная норма - это максимальное сингулярное значение матрицы.
  Интуитивно вы можете думать об этом как о максимальном «масштабе», по которому матрица может «растягивать» вектор)

  (Сингулярные значения - это диагональные элементы матрицы S, расположенные в порядке убывания. Особые значения
  всегда являются действительными числами)

  Когда update_variable равен True, обновляет u вектор спектральной нормализации
  его степенным методом (power-iteration method). Если спектральная норма вызывается несколько раз в
  одной и той же области (scope) (как в Infinite Nature), переменная нормализации
  u будет разделена между ними, и любые предыдущие операции присваивания к u
  будут выполнены перед текущей операцией (assign). Поскольку степенная степенной метод является сходящимся,
  имеет значения, происходит ли несколько обновлений за один прямой проход
  (in a single forward pass).

  Аргументы:
    w: (tensor) Весовой тензор (тензор весов) для применения спектральной нормализации
    iteration: (int) Сколько раз запускать степенной метод при вызове
    update_variable: (bool) Если True, обновитm переменную u.

  Вывод:
    Тензор той же формы, что w.
  """
  w_shape = w.shape.as_list()
  w = tf.reshape(w, [-1, w_shape[-1]])

  u = tf.compat.v1.get_variable( # берет существующую переменную с заданными параметрами или создает новую
      "u", [1, w_shape[-1]],
      initializer=tf.random_normal_initializer(),
      trainable=False)

  u_hat = u
  v_hat = None
  for _ in range(iteration):
    # степенной метод. Обычно достаточно одной итерации.
    v_ = tf.matmul(u_hat, tf.transpose(w))
    v_hat = tf.nn.l2_normalize(v_) # проводит нормализацию по оси применяя норму L2

    u_ = tf.matmul(v_hat, w)
    u_hat = tf.nn.l2_normalize(u_)

  u_hat = tf.stop_gradient(u_hat) # прекращает вычисление градиента
  v_hat = tf.stop_gradient(v_hat)

  sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

  if update_variable:
    # Заставить любые предыдущие assign_ops находящиеся выше (upstream) w назначить u_hat
    # чтобы предотвратить гонки.
    update_op = u.assign(u_hat)
  else:
    update_op = tf.no_op() # Ничего не делает. Используется как заглушка
  with tf.control_dependencies([update_op]): # диспетчер который указывает (specifies) управляющие зависимости (control dependencies)
    w_norm = w / sigma
    w_norm = tf.reshape(w_norm, w_shape)

  return w_norm


def sn_conv(tensor, channels, kernel_size=3, stride=1,
            use_bias=True, use_spectral_norm=True, scope="conv",
            pad_type="REFLECT"):
  """Сверточный слой с поддержкой заполнения (support for padding) и необязательная (optional)
  спектральной нормой.

  Аргументы:
    tensor: [B, H, W, C] Тензор для свертки (to perform a convolution on)
    channels: (int) Количество выходных каналов
    kernel_size: (int) Размер квадратного (a square) сверточного фильтра (convolutional filter)
    stride: (int) The stride to apply the convolution
    use_bias: (bool) Если true, добавляет a learned bias term
    use_spectral_norm: (bool) Если true, применяет спектральную нормализацию к весам
    scope: (str) scope of the variables
    pad_type: (str) The padding to use

  Вывод:
    Результат работы сверточного слоя на тензоре.
  """
  tensor_shape = tensor.shape
  with tf.compat.v1.variable_scope(scope):
    h, w = tensor_shape[1], tensor_shape[2]
    output_h, output_w = int(math.ceil(h / stride)), int(
        math.ceil(w / stride))

    p_h = (output_h) * stride + kernel_size - h - 1
    p_w = (output_w) * stride + kernel_size - w - 1

    pad_top = p_h // 2
    pad_bottom = p_h - pad_top
    pad_left = p_w // 2
    pad_right = p_w - pad_left
    tensor = tf.pad(
        tensor,
        [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]],
        mode=pad_type)
    if use_spectral_norm:
      w = tf.compat.v1.get_variable(
          "kernel",
          shape=[kernel_size, kernel_size, tensor_shape[-1], channels])
      x = tf.nn.conv2d( #
          tensor,
          spectral_norm(w, update_variable=config.is_training()),
          [1, stride, stride, 1],
          "VALID")
      if use_bias:
        bias = tf.compat.v1.get_variable(
            "bias", [channels], initializer=tf.constant_initializer(0.0))
        x = tf.nn.bias_add(x, bias) # добавляет bias к x где x - тензор и bias - 1D тензор

    else:
      x = tf.compat.v1.layers.conv2d( # Функциональный интерфейс для 2D сверточного слоя
          tensor,
          channels,
          kernel_size,
          strides=stride,
          use_bias=use_bias)

    return x
