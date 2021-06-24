"""Uses a heuristic to automatically navigate generated scenes.

Использует эвристику (heuristic) для автоматической навигации сгенерированных сцен

fly_camera.fly_dynamic сгенерирует позы, используя карты несоответствия которые не врежуться в блажайшую местность
"""
import pickle
import time

import config
import fly_camera
import imageio
import infinite_nature_lib
import numpy as np
import tensorflow as tf


tf.compat.v1.flags.DEFINE_string(           # создает (register так что вероятно резервирует, как константа для if условий)
    "output_folder", "autocruise_output",   # флаг, значением которого может быть любая строка
    "Folder to save autocruise results")
tf.compat.v1.flags.DEFINE_integer(      # создает (register)
    "num_steps", 500,                   # флаг, значение которого должно быть целым числом,
    "Number of steps to fly.")          # для числа можно задать диапазон

FLAGS = tf.compat.v1.flags.FLAGS        # Реестр объектов типа FLAG


def generate_autocruise(np_input_rgbd, checkpoint,
                        save_directory, num_steps, np_input_intrinsics=None):
  """Сохраняет num_steps фреймы infinite nature используя autocruise algorithm.

  Args:
    np_input_rgbd: [H, W, 4] numpy изображение и несоответствие, чтобы запустить
      Infinite Nature со значениями в диапазоне [0, 1]
    checkpoint: (str) путь к предварительно обученному чекпоинту
    save_directory: (str) папка для сохранения RGB изображений
    num_steps: (int) количество шагов для генерации
    np_input_intrinsics: [4] предполагаемые  характеристики (intrinsics). Если не указаны,
      делает предположение по FOV.

      FOV — это поле зрения
  """
  render_refine, style_encoding = infinite_nature_lib.load_model(checkpoint)
  if np_input_intrinsics is None:
    # 0.8 focal_x соответствует полю обзора ~64 градуса. Можно изменить
    # вручную если даны дополнительные предположения о входном изображении.
    h, w, unused_channel = np_input_rgbd.shape
    ratio = w / float(h)
    np_input_intrinsics = np.array([0.8, 0.8 * ratio, .5, .5], dtype=np.float32)

  np_input_rgbd = tf.image.resize(np_input_rgbd, [160, 256]) # изменяет размер изображения, можно указать метод
                                                             # изменения (bilinear для Билинейная интерполяция и т.д.)
  style_noise = style_encoding(np_input_rgbd)

  meander_x_period = 100
  meander_y_period = 100
  meander_x_magnitude = 0.0
  meander_y_magnitude = 0.0
  fly_speed = 0.2
  horizon = 0.3
  near_fraction = 0.2

  starting_pose = np.array(
      [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
      dtype=np.float32)

  # autocruise эвристическая функция
  # предположу, что autocruise предполагает автоматическое управление камерой алгоритмом без непосредственного
  # вмешательства
  fly_next_pose_function = fly_camera.fly_dynamic(
      np_input_intrinsics, starting_pose,
      speed=fly_speed,
      meander_x_period=meander_x_period,
      meander_x_magnitude=meander_x_magnitude,
      meander_y_period=meander_y_period,
      meander_y_magnitude=meander_y_magnitude,
      horizon=horizon,
      near_fraction=near_fraction)

  if not tf.io.gfile.exists(save_directory):    # если папки для сохранения не существует
    tf.io.gfile.makedirs(save_directory)        # создать ее

  curr_pose = starting_pose
  curr_rgbd = np_input_rgbd
  t0 = time.time()
  for i in range(num_steps - 1):
    next_pose = fly_next_pose_function(curr_rgbd)
    curr_rgbd = render_refine(
        curr_rgbd, style_noise, curr_pose, np_input_intrinsics,
        next_pose, np_input_intrinsics)

    # Обновить информацию о позе для просмотра.
    curr_pose = next_pose
    imageio.imsave("%s/%04d.png" % (save_directory, i),
                   (255 * curr_rgbd[:, :, :3]).astype(np.uint8))
    if i % 100 == 0:
      print("%d / %d frames generated" % (i, num_steps))
      print("time / step: %04f" % ((time.time() - t0) / (i + 1)))
      print()


def main(unused_arg):
  if len(unused_arg) > 1:
    raise tf.app.UsageError(    # тут функционал понятен из названия но, что странно, этой функции нет в документации
        "Too many command-line arguments.")
  config.set_training(False)
  model_path = "ckpt/model.ckpt-6935893"
  input_pkl = pickle.load(open("autocruise_input1.pkl", "rb"))
  generate_autocruise(input_pkl["input_rgbd"],
                      model_path,
                      FLAGS.output_folder,
                      FLAGS.num_steps)

if __name__ == "__main__":
  tf.compat.v1.enable_eager_execution()
  tf.compat.v1.app.run(main)
