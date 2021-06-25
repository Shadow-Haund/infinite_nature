"""Запускает Infinite Nature на изображении + несоответствие.
"""
import config
import networks
import render
import tensorflow as tf


def load_model(checkpoint):
  """Загружает обученную модель и возвращает функции для ее запуска.

  Этот код дает "eager-like (Нетерпеливое исполнение)" интерфейс для графа ниже
  Аргументы:
    checkpoint: Имя контрольной точки загрузки.

  Возвращает:
    pvsm_from_image: Функция которая берет [160, 256, 4] RGBD изображение
                     и [160, 256, 4] кодирование изображения,
                     и параметры камеры:
                     pose, pose_next [3, 4], intrinsics, intrinsics_next [4]
    и возвращает список из 3 изображений размера [H, W, 4], [predicted, render, mask]
    style_embedding_from_encoding: Функция берет [160, 256, 4] RGBD
                                   изображение и возвращает a встроенный стиль (style embedding) [256]
  """
  sess = tf.compat.v1.Session() # Класс для выполнения операций TensorFlow
  with sess.graph.as_default(): # Насколько я понял это создание (register) графа по умолчанию
    image_placeholder = tf.compat.v1.placeholder(tf.float32, [160, 256, 4]) # вставляет заглушку для тензора
    # (первый аргумент это тип данных для заглушки а второй это форма тензора)
    # Initial RGB_D to set the latent
    encoding_placeholder = tf.compat.v1.placeholder(tf.float32, [160, 256, 4])
    style_noise_placeholder = tf.compat.v1.placeholder(
        tf.float32, [config.DIM_OF_STYLE_EMBEDDING])
    intrinsic_placeholder = tf.compat.v1.placeholder(tf.float32, [4])
    intrinsic_next_placeholder = tf.compat.v1.placeholder(tf.float32, [4])
    pose_placeholder = tf.compat.v1.placeholder(tf.float32, [3, 4])
    pose_next_placeholder = tf.compat.v1.placeholder(tf.float32, [3, 4])

    # Add batch dimensions. Тут насколько я понимаю в окна для изображения у запущенного проекта вставляются заглушки
    image = image_placeholder[tf.newaxis]
    encoding = encoding_placeholder[tf.newaxis]
    style_noise = style_noise_placeholder[tf.newaxis]
    intrinsic = intrinsic_placeholder[tf.newaxis]
    intrinsic_next = intrinsic_next_placeholder[tf.newaxis]
    pose = pose_placeholder[tf.newaxis]
    pose_next = pose_next_placeholder[tf.newaxis]

    mulogvar = get_encoding_mu_logvar(encoding)
    if config.is_training():
      z = networks.reparameterize(mulogvar[0], mulogvar[1])
    else:
      z = mulogvar[0]

    z = z[0]

    refine_fn = create_refinement_network(style_noise)
    render_rgbd, mask = render.render(
        image, pose, intrinsic, pose_next, intrinsic_next)

    generated_image = refine_fn(render_rgbd, mask)

    refined_disparity = rescale_refined_disparity(render_rgbd[Ellipsis, 3:], mask,
                                                  generated_image[Ellipsis, 3:])
    generated_image = tf.concat([generated_image[Ellipsis, :3], refined_disparity],
                                axis=-1)[0]

    saver = tf.compat.v1.train.Saver()
    print("Restoring from %s" % checkpoint)
    saver.restore(sess, checkpoint)
    print("Model restored.")

  def as_numpy(x):
    if tf.is_tensor(x):
      return x.numpy()
    else:
      return x

  def render_refine(image, style_noise, pose, intrinsic,
                    pose_next, intrinsic_next):
    return sess.run(generated_image, feed_dict={    # Выполняет операции
        image_placeholder: as_numpy(image),
        style_noise_placeholder: as_numpy(style_noise),
        pose_placeholder: as_numpy(pose),
        intrinsic_placeholder: as_numpy(intrinsic),
        pose_next_placeholder: as_numpy(pose_next),
        intrinsic_next_placeholder: as_numpy(intrinsic_next),
    })

  def encoding_fn(encoding_image):
    return sess.run(z, feed_dict={
        encoding_placeholder: as_numpy(encoding_image)})

  return render_refine, encoding_fn


def rescale_refined_disparity(rendered_disparity, input_mask,
                              refined_disparity):
  """Изменяет масштаб уточненного несоответствия в соответствии с масштабом входных данных

  Это делается чтобы избежать дрейфа в несоответствии. Мы подбираем масштаб
  решая оптимизацию наименьших квадратов.

  Аргументы:
    rendered_disparity: [B, H, W, 1] несоответствие, созданное шагом рендеринга
    input_mask: [B, H, W, 1] маска в которой 1 обозначают области что видны через (through) ренеренг.
                              наверно предполагалось что они видны после рендеринга, но там было through
    refined_disparity: [B, H, W, 1] несоответствие на выходе уточняющей сети
  Возвращает:
    refined_disparity что было масштабировано и сдвинуто чтобы соответствовать статистике rendered_disparity.
  """
  log_refined = tf.math.log(tf.clip_by_value(refined_disparity, 0.01, 1)) # Вычисляет натуральный логарифм по элементно
  log_rendered = tf.math.log(tf.clip_by_value(rendered_disparity, 0.01, 1))
  log_scale = tf.reduce_sum(input_mask * (log_rendered - log_refined)) / (  # вычисляет сумму либо всех элемиментов тензора
      tf.reduce_sum(input_mask) + 1e-4)       # либо по пространствам или осям
  scale = tf.exp(log_scale)                   # Поэлементно вычисляет экспоненту
  scaled_refined_disparity = tf.clip_by_value(scale * refined_disparity, # # Обрезает значения тензора до указанного минимума и максимума
                                              0, 1)
  return scaled_refined_disparity


def create_refinement_network(noise_input):
  """Создает сеть уточнения с encoding_image's типом (style) шума.

  Аргументы:
    noise_input: [1, z_dim] шум взятый (sampled ) из нормального распределения
      параметризированный энкодером.
  Возвращает:
    Функция уточнения использующая encoding_image чтобы посять (seed) шум.
  """
  def fn(rgbd, mask):
    with tf.compat.v1.variable_scope("generator", reuse=tf.compat.v1.AUTO_REUSE): # контекстный менеджер для
      with tf.compat.v1.variable_scope("spade_network_noenc"):    # создания слоев или переменных
        return networks.refinement_network(rgbd, mask, noise_input)
  return fn


def get_encoding_mu_logvar(encoding_image):
  """Вычисляет encoding_image's style noise параметры.

  Аргументы:
    encoding_image: [B, H, W, 4] входное RGBD изображение для запуска Infinite Nature on
  Возвращает:
    кортеж тензоров ([B, z_dim], [B, z_dim]) соответствующих с mu и logvar
    параметрам (normal parameters).
  """

  with tf.compat.v1.variable_scope("generator", reuse=tf.compat.v1.AUTO_REUSE): # он проверяет что значения взяты из
    # одного и того же графа и проверяет, что граф является графом по умолчанию
    mu_logvar = networks.encoder(encoding_image)
  return mu_logvar
