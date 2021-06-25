"""Код библиотеки рендеринга для доступа к tf_mesh_renderer.
"""
import geometry
import ops
import render_utils
import tensorflow as tf
import tensorflow_addons as tfa
from tf_mesh_renderer.mesh_renderer import rasterize_triangles


def sobel_fg_alpha(idepth, beta=5.0):
  """Вычисляет foreground alpha с помощью оператора Собеля (sobel edges).

  (Оператор Собеля - дискретный дифференциальный оператор, вычисляющий приближённое значение
  градиента яркости изображения)

  Альфы будут низкими когда sobel edges велики.

  Args:
    idepth: [B, H, W, 1] тензор обратной глубины inverse depth tensor.
    beta: (float) Чем выше бета, тем выше чувствительность к краям глубины.

  Returns:
    alpha: [B, H, W, 1] alpha visibility.
  """
  # Вычисляет Sobel edges и их величину (magnitude).
  # Я не нащел понятия Sobel edges, есть Sobel operator и на сколько я понял он используется для определения
  # границ изображения то есть нахождения этих edges
  sobel_components = tf.image.sobel_edges(idepth) # Возвращает тензор, содержащий карты Sobel edges
  sobel_mag_components = sobel_components**2
  sobel_mag_square = tf.math.reduce_sum(sobel_mag_components, axis=-1) # вычисляет сумму значений для указанного измерения
  # и всех этих значений записывает эту сумму
  sobel_mag = tf.sqrt(sobel_mag_square + 1e-06) # Из значений тензора вычисляет поэлементно корень

  # Вычислить альфы из величин sobel edges.
  alpha = tf.exp(-1.0 * beta * sobel_mag)
  return alpha


def render(input_rgbd, input_pose, input_intrinsics,
           target_pose, target_intrinsics,
           alpha_threshold=0.3):
  """Рендерит rgbd в целевом кадре (to target view), также генерируя маску.

  Args:
    input_rgbd: [B, H, W, 4] an input RGBD (either the initial RGBD or output of a previous render_refine)
    input_pose: [B, 3, 4] pose of input_rgbd
    input_intrinsics: [B, 4] camera intrinsics of input_rgbd
    target_pose: [B, 3, 4] pose of the view to be generated
    target_intrinsics: [B, 4] camera intrinsics of the output view
  Returns:
    [...., height, width, 4] Rendered RGB-D image at the target view.
    [...., height, width, 1] Mask at the target view. The mask is 0 where holes
        were introduced by the renderer.
  """
  # Limit the range of disparity to avoid division by zero or negative values.
  min_disparity = 1e-6
  max_disparity = 1e5
  rgb = input_rgbd[Ellipsis, :-1]
  disparity = tf.clip_by_value(input_rgbd[Ellipsis, -1:], min_disparity, max_disparity)

  # This returns [B, H, W, 1]
  alpha = sobel_fg_alpha(disparity, beta=10.0)
  # Make the alpha hard.
  mask = tf.cast(tf.greater(alpha, alpha_threshold), dtype=tf.float32) # tf.greater поэлементно сравнивает
  # alpha > alpha_threshold и возвращает массив bool

  # Now we'll render RGB and mask from the target view:
  rgb_and_mask = tf.concat([rgb, mask], axis=-1)
  target_rgb_and_mask, target_disparity = render_channels(
      rgb_and_mask, disparity,
      input_pose, input_intrinsics,
      target_pose, target_intrinsics)

  # Multiply by mask.
  rgb, mask = tf.split(target_rgb_and_mask, [3, 1], axis=-1)
  rgbd = tf.concat([rgb, target_disparity], axis=-1)
  return rgbd * mask, mask * mask


def render_channels(
    channels, disparity,
    source_pose, source_intrinsics,
    target_pose, target_intrinsics):
  """Render channels from new target position, given disparity.

  Args:
    channels: [B, H, W, C] Channels to render
    disparity: [B, H, W, 1] Inverse depth
    source_pose: [B, 3, 4] эталонная поза камеры reference camera pose
    source_intrinsics: [B, 4] reference intrinsics
    target_pose: [B, 3, 4] target camera pose
    target_intrinsics: [B, 4] target intrinsics

  Returns:
    [B, H, W, C] Rendered channels at the target view.
    [B, H, W, 1] Rendered disparity at the target view.
  """
  (batch_size, height, width, channel_count) = channels.get_shape().as_list()

  # Относительная поза сопоставляет источник с целевой позой Relative pose maps source to target pose.
  relative_pose = geometry.mat34_product(
      target_pose, geometry.mat34_pose_inverse(source_pose))

  # Project source image into 3D mesh.
  vertices = render_utils.create_vertices_intrinsics(
      disparity[Ellipsis, 0], source_intrinsics)

  # Depth of each point from target camera.
  target_depths = geometry.mat34_transform(relative_pose, vertices)[Ellipsis, -1:]

  # Add target-view depths as an extra vertex attribute.
  attributes = tf.reshape(channels, (batch_size, width * height, channel_count))
  attributes = tf.concat([attributes, target_depths], -1)

  # Get triangles,
  triangles = render_utils.create_triangles(height, width)
  num_triangles = triangles.shape[0]
  triangles = tf.convert_to_tensor(triangles, tf.int32)

  # Матрицы камеры.
  target_perspective = render_utils.perspective_from_intrinsics(
      target_intrinsics)
  relative_pose = geometry.mat34_to_mat44(relative_pose)
  proj_matrix = tf.matmul(target_perspective, relative_pose)

  # Нулевое значение фона для каналов, большое значение фона для глубины.
  background = [0.0] * channel_count + [1000.0]

  # Рендеринг библиотекой  mesh_renderer
  output = rasterize_triangles.rasterize(
      vertices, attributes, triangles, proj_matrix, width, height, background)
  output_channels, output_depths = tf.split(output, [channel_count, 1], axis=-1)
  output_disparity = tf.math.divide_no_nan( # возвращает 0 если знаменатель 0
      1.0, tf.clip_by_value(output_depths, 1.0 / 100.0, 1.0 / 0.01))

  return (output_channels, output_disparity)
