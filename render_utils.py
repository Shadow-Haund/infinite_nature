"""Полезные функции для рендера."""
import numpy as np
import tensorflow as tf


def create_vertices_intrinsics(disparity, intrinsics):
  """3D mesh vertices from a given disparity and intrinsics.
  Вершины 3D сетки вычисляемые из внутренних свойств и несоответствия

  Аргументы:
     disparity: [B, H, W] инвертированная глубина
     intrinsics: [B, 4] ссылки на свойства

  Возвращает:
     [B, L, H*W, 3] координаты вершины.
  """
  # Фокусное расстояние
  fx = intrinsics[:, 0]
  fy = intrinsics[:, 1]
  fx = fx[Ellipsis, tf.newaxis, tf.newaxis]
  fy = fy[Ellipsis, tf.newaxis, tf.newaxis]

  # Центры
  cx = intrinsics[:, 2]
  cy = intrinsics[:, 3]
  cx = cx[Ellipsis, tf.newaxis]
  cy = cy[Ellipsis, tf.newaxis]

  batch_size, height, width = disparity.shape.as_list()
  vertex_count = height * width

  i, j = tf.meshgrid(tf.range(width), tf.range(height))
  i = tf.cast(i, tf.float32)
  j = tf.cast(j, tf.float32)
  width = tf.cast(width, tf.float32)
  height = tf.cast(height, tf.float32)
  # добаляем 0.5 чтобы получить позицию центральных пикселей (pixel centers).
  i = (i + 0.5) / width
  j = (j + 0.5) / height
  i = i[tf.newaxis]
  j = j[tf.newaxis]

  depths = 1.0 / tf.clip_by_value(disparity, 0.01, 1.0)
  mx = depths / fx
  my = depths / fy
  px = (i-cx) * mx
  py = (j-cy) * my

  vertices = tf.stack([px, py, depths], axis=-1)
  vertices = tf.reshape(vertices, (batch_size, vertex_count, 3))
  return vertices


def create_triangles(h, w):
  """Создает индексы треугольника сетки (mesh triangle, наверное полигоны)  из заданного размера сетки (grid) пикселей.

     Эта функция не является и не должна быть дифференцируемой поскольку индексы треугольников фиксированы.

  Аргументы:
    h: (int) обозначает высоту изображения.
    w: (int) обозначающий ширину изображения.

  Возвращает:
    triangles: 2D numpy массив индексов (int) формы (shape) (2(W-1)(H-1) x 3)
  """
  x, y = np.meshgrid(range(w - 1), range(h - 1))
  tl = y * w + x
  tr = y * w + x + 1
  bl = (y + 1) * w + x
  br = (y + 1) * w + x + 1
  triangles = np.array([tl, bl, tr, br, tr, bl])
  triangles = np.transpose(triangles, (1, 2, 0)).reshape(
      ((w - 1) * (h - 1) * 2, 3))
  return triangles


def perspective_from_intrinsics(intrinsics):
  """Вычисляет трехмерную проекцию (perspective matrix) из параметров (intrinsics) камеры .

  (пространство точек перед нормализацией)

  Матрица отображает пространство камеры (camera-space) в клип пространство (clip-space) (x, y, z, w)
  где (x/w, y/w, z/w) диарозоны от -1 до 1 на каждой оси. Это стандартная трехмерная проекция или матрица проекции
  вида OpenGL, только мы используем положительное значение Z для направления взгляда (вместо отрицательного)
  поэтому есть различия в знаках.

  Аргументы:
    intrinsics: [B, 4] Тензор внутренних характеристик исходной камеры
                       (Source camera intrinsics tensor) (f_x, f_y, c_x, c_y)

  Возвращает:
    A [B, 4, 4] float32 Tensor that maps from right-handed camera space
    to left-handed clip space.
  """
  intrinsics = tf.convert_to_tensor(intrinsics)
  focal_x = intrinsics[:, 0]
  focal_y = intrinsics[:, 1]
  principal_x = intrinsics[:, 2]
  principal_y = intrinsics[:, 3]
  zero = tf.zeros_like(focal_x)
  one = tf.ones_like(focal_x)
  near_z = 0.00001 * one
  far_z = 10000.0 * one

  a = (near_z + far_z) / (far_z - near_z)
  b = -2.0 * near_z * far_z / (far_z - near_z)

  matrix = [
      [2.0 * focal_x, zero, 2.0 * principal_x - 1.0, zero],
      [zero, 2.0 * focal_y, 2.0 * principal_y - 1.0, zero],
      [zero, zero, a, b],
      [zero, zero, one, zero]]
  return tf.stack([tf.stack(row, axis=-1) for row in matrix], axis=-2)
