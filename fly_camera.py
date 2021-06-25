"""Функции для эврестического полета камеры через сгенерированную сцену."""
import math

import geometry
import tensorflow as tf


def camera_with_look_direction(position, look_direction, down_direction):
  """Поза камеры определяемая тем где она и в каком напрвлении смотрит.

  Аргументы:
    position: [..., 3] позиция камеры в мире.
    look_direction: [..., 3] напрвление обзора (need not be normalised).
    down_direction: [..., 3] направление на ось Y смотрит вниз.
  Возвращаемое значение:
    [..., 3, 4] Camera pose.
  """
  # Вектора мира сконструированы чтобы согласовываться с камерой
  # look_direction это плоскость Z, down_direction это Y.
  # Y пересекающийся с Z это Х (по правилу правой руки)
  vector_z = tf.math.l2_normalize(look_direction, axis=-1) # нормализация вдоли плоскости l2 нормой
  vector_x = tf.math.l2_normalize(
      tf.linalg.cross(down_direction, vector_z), axis=-1) # вычислить попарное перекрестное произведение
  vector_y = tf.linalg.cross(vector_z, vector_x) #and b must be the same shape
  # С этими 3 векторами и позой можно собрать матрицу камеры:
  camera_to_world = tf.stack([vector_x, vector_y, vector_z, position], axis=-1) # Упаковывает список тензоров
                # в один тензор с рангом + 1, упаковывает вдоль оси
  return geometry.mat34_pose_inverse(camera_to_world)


def skyline_balance(disparity, horizon=0.3, near_fraction=0.2):
  """Вычисление параметров движения из изображения несоответствия

  Аргументы:
    disparity: [H, W, 1] изображение несоответствия.
    horizon: идельное расстояние вниз до горизонта.
    near_fraction: насколько "близко" должно быть изображение

  Возвращаемое значение:
    (x, y, h) где x и y то куда мы хотим смотреть в изображении (как координаты текстуры)
    и h насколько сильно мы хотим двигаться вверх.
  """
  # Эксперименты показали, что граница горизонта где-то между несоответствиями 0.05
  # и 0.1. Так обрежем и промасштабируеи чтобы получить мягкую маску неба.
  sky = tf.clip_by_value(20.0 * (0.1 - disparity), 0.0, 1.0) # Обрезает значения тензора до указанного минимума и максимума
  # возвращает тензор того же типа и формы, что и t, с его значениями, обрезанными до тут 0.0 и 1.0
  # значения меньше 0.0 устанавливаются в 0.0. и превышающие 1.0 в 1.0


  # Сколько на изображении неба?
  sky_fraction = tf.reduce_mean(sky) # Вычисляет среднее значение элементов по указанной плоскости и уменьшает тензор
  y = 0.5 + sky_fraction - horizon

  # Баланс неба на левой и правой стороне изображения.
  w2 = disparity.shape[-2] // 2
  sky_left = tf.reduce_mean(sky[Ellipsis, :w2, :])
  sky_right = tf.reduce_mean(sky[Ellipsis, w2:, :])
  # Turn away from mountain:
  epsilon = 1e-4
  x = (sky_right + epsilon) / (sky_left + sky_right + 2 * epsilon)

  # Определяем насколько мы "близко к земле", смотря на то
  # насколько большая часть изображения имеет несоответствие > 0.5 (возрастает до максимума при 0.6)
  ground = tf.clip_by_value(10.0 * (disparity - 0.5), 0.0, 1.0)
  ground_fraction = tf.reduce_mean(ground)
  h = horizon + (near_fraction - ground_fraction)
  return x, y, h


def fly_dynamic(
    intrinsics, initial_pose,
    speed=0.2, lerp=0.05, movelerp=0.05,
    horizon=0.3, near_fraction=0.2,
    meander_x_period=100, meander_x_magnitude=0.0,
    meander_y_period=100, meander_y_magnitude=0.0,
    turn_function=None):
  """Возвращает функцию для эвристического полета камеры

  Функция смотри на то, как меняется значение несоответствия и решает
  смотреть ли левее/правее, выше/ниже, а так же подняться или опуститься к земле

  Аргументы:
    intrinsics: [4] Параметры камеры (intrinsics).
    initial_pose: [3, 4] Начальная поза камеры.
    speed: Насколько далеко двигаться за шаг.
    lerp: Насколько быстро сводиться взгляд на цель (target). Я не нашел пояснения к тому что является target,
    поэтому предположу что это просто выравнивание как бы по центру сцены то есть нахождение оптимального положения
    movelerp: Скорость движения к цели.
    horizon: Часть изображения которая должна находиться над горизонтом
    near_fraction:
    meander_x_period: Количество фреймов для создания циклического изгиба в
      горизонтальном направлении
    meander_x_magnitude: Как далеко до горизонтального изгиба
    meander_y_period: Количество фреймов для создания циклического изгиба в
      вертикальном направлении
    meander_y_magnitude: Как далеко до вертикального изгиба
    turn_function: Возвращает позиции x и y для разворота

  Возвращает:
    функция fly_step которая берет rgbd изображение и возвращает позу для
    следующей камеры. Вызывай fly_step повторно для создания серии поз.
    Это функция отслеживает состояние и постоянно смотрит за позицией и скоростью
    камеры. Может работать только в режиме eager execution (вычисляет операции немедленно, без построения графов).
  """

  # Када смотрит камера и в каком направлении низ:
  camera_to_world = geometry.mat34_pose_inverse(initial_pose)
  look_dir = camera_to_world[:, 2]
  move_dir = look_dir  # Начать движением вперед.
  down = camera_to_world[:, 1]
  position = camera_to_world[:, 3]
  t = 0

  reverse = (speed < 0)

  def fly_step(rgbd):
    nonlocal camera_to_world
    nonlocal look_dir
    nonlocal move_dir
    nonlocal down
    nonlocal position
    nonlocal t

    if turn_function:
      (xoff, yoff) = turn_function(t)
    else:
      (xoff, yoff) = (0.0, 0.0)

    xoff += math.sin(t * 2.0 * math.pi/ meander_x_period) * meander_x_magnitude
    yoff += math.sin(t * 2.0 * math.pi/ meander_y_period) * meander_y_magnitude
    t = t + 1

    down = camera_to_world[:, 1]  # можно закоментировать чтобы зафиксировать низ
    disparity = rgbd[Ellipsis, 3:]
    x, y, h = skyline_balance(
        disparity, horizon=horizon, near_fraction=near_fraction)
    if reverse:
      h = 1.0 - h
      x = 1.0 - x
    look_uv = tf.stack([x + xoff, y + yoff])
    move_uv = tf.stack([0.5, h])
    uvs = tf.stack([look_uv, move_uv], axis=0)

    # Points in world
    # Точки в мире
    points = geometry.mat34_transform(
        camera_to_world,
        geometry.texture_to_camera_coordinates(uvs, intrinsics))
    new_look_dir = tf.math.l2_normalize(points[0] - position)
    new_move_dir = tf.math.l2_normalize(points[1] - position)

    # Простое сглаживание
    look_dir = look_dir * (1.0 - lerp) + new_look_dir * lerp
    move_dir = move_dir * (1.0 - movelerp) + new_move_dir * movelerp
    position = position + move_dir * speed

    # Следующая поза
    pose = camera_with_look_direction(position, look_dir, down)
    camera_to_world = geometry.mat34_pose_inverse(pose)
    return pose

  return fly_step
