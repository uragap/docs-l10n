# TensorFlow 2.x в TFX

[TensorFlow 2.0 был выпущен в 2019 году](https://blog.tensorflow.org/2019/09/tensorflow-20-is-now-available.html), [поддерживал тесную интеграцию с Keras](https://www.tensorflow.org/guide/keras/overview), [eager execution](https://www.tensorflow.org/guide/eager) по умолчанию, а также поддерживал [выполнение функций Python](https://www.tensorflow.org/guide/function) [среди прочего нового функционала и улучшений](https://www.tensorflow.org/guide/effective_tf2#a_brief_summary_of_major_changes).

В данном руководстве приводится всесторонний технический обзор TF 2.x в TFX.

## Какую версию стоит использовать?

TFX совместим с TensorFlow 2.x, и высокоуровневые API, которые существовали в TensorFlow 1.x (особенно Estimators), продолжают работать.

### Начать новые проекты в TensorFlow 2.x

Поскольку TensorFlow 2.x сохраняет высокоуровневые возможности TensorFlow 1.x, использование старой версии в новых проектах не дает никаких преимуществ, даже если вы не планируете использовать новые функции.

Поэтому, если вы начинаете новый проект TFX, мы рекомендуем вам использовать TensorFlow 2.x. Возможно, вы захотите обновить свой код позже, когда станет доступна полная поддержка Keras и других новых функций, и объем изменений будет гораздо более ограниченным, если вы начнете с TensorFlow 2.x, вместо того, чтобы пытаться обновить TensorFlow 1.x в будущее.

### Преобразование существующих проектов в TensorFlow 2.x

Код, написанный для TensorFlow 1.x, в значительной степени совместим с TensorFlow 2.x и будет продолжать работать в TFX.

Однако, если вы хотите воспользоваться преимуществами улучшений и новых функций по мере их появления в TF 2.x, вы можете следовать [инструкциям по переходу на TF 2.x.](https://www.tensorflow.org/guide/migrate)

## оценщик

Estimator API был сохранен в TensorFlow 2.x, но не является объектом новых функций и разработок. Код, написанный в TensorFlow 1.x или 2.x с использованием Оценщиков, будет продолжать работать, как и ожидалось, в TFX.

Вот пример сквозного TFX с использованием чистого Оценщика: [Пример Такси (Оценщик)](https://github.com/tensorflow/tfx/blob/r0.21/tfx/examples/chicago_taxi_pipeline/taxi_utils.py)

## Керас с `model_to_estimator`

Keras models can be wrapped with the `tf.keras.estimator.model_to_estimator` function, which allows them to work as if they were Estimators. To use this:

1. Построить модель Keras.
2. Передайте скомпилированную модель в `model_to_estimator` .
3. Используйте результат `model_to_estimator` в Trainer, как вы обычно используете Estimator.

```py
# Build a Keras model.
def _keras_model_builder():
  """Creates a Keras model."""
  ...

  model = tf.keras.Model(inputs=inputs, outputs=output)
  model.compile()

  return model


# Write a typical trainer function
def trainer_fn(trainer_fn_args, schema):
  """Build the estimator, using model_to_estimator."""
  ...

  # Model to estimator
  estimator = tf.keras.estimator.model_to_estimator(
      keras_model=_keras_model_builder(), config=run_config)

  return {
      'estimator': estimator,
      ...
  }
```

Кроме файла пользовательского модуля Trainer, остальная часть конвейера остается неизменной. Вот пример сквозного TFX с использованием Keras с model_to_estimator: [пример Iris (model_to_estimator)](https://github.com/tensorflow/tfx/blob/r0.21/tfx/examples/iris/iris_utils.py)

## Native Keras (i.e. Keras without `model_to_estimator`)

Примечание: полная поддержка всех функций в Keras выполняется, в большинстве случаев Keras в TFX будет работать так, как ожидается. Он пока не работает с разреженными функциями для FeatureColumns.

### Примеры и колаб

Вот несколько примеров с родными Keras:

- [Iris](https://github.com/tensorflow/tfx/blob/master/tfx/examples/iris/iris_pipeline_native_keras.py) ( [файл модуля](https://github.com/tensorflow/tfx/blob/master/tfx/examples/iris/iris_utils_native_keras.py) ): пример «конец света».
- [MNIST](https://github.com/tensorflow/tfx/blob/master/tfx/examples/mnist/mnist_pipeline_native_keras.py) ( [файл модуля](https://github.com/tensorflow/tfx/blob/master/tfx/examples/mnist/mnist_utils_native_keras.py) ): пример сквозного изображения и TFLite.
- [Такси](https://github.com/tensorflow/tfx/blob/master/tfx/examples/chicago_taxi_pipeline/taxi_pipeline_native_keras.py) ( [файл модуля](https://github.com/tensorflow/tfx/blob/master/tfx/examples/chicago_taxi_pipeline/taxi_utils_native_keras.py) ): сквозной пример с расширенным использованием Transform.

У нас также есть компонент [Keras Colab](https://www.tensorflow.org/tfx/tutorials/tfx/components_keras) .

### Компоненты TFX

В следующих разделах объясняется, как связанные компоненты TFX поддерживают собственные Keras.

#### преобразование

В настоящее время Transform имеет экспериментальную поддержку моделей Keras.

Сам компонент Transform может быть использован для собственных Keras без изменений. `preprocessing_fn` определение остается тем же самым , используя [TensorFlow](https://www.tensorflow.org/api_docs/python/tf) и [tf.Transform](https://www.tensorflow.org/tfx/transform/api_docs/python/tft) опа.

Функция обслуживания и функция eval изменены для собственных Keras. Детали будут обсуждаться в следующих разделах «Тренер» и «Оценщик».

Примечание. Преобразования в `preprocessing_fn` нельзя применить к метке для обучения или оценки.

#### тренер

To configure native Keras, the `GenericExecutor` needs to be set for Trainer component to replace the default Estimator based executor. For details, please check [here](trainer.md#configuring-the-trainer-component-to-use-the-genericexecutor).

##### Файл модуля Keras с помощью Transform

Файл обучающего модуля должен содержать `run_fn` который будет вызываться `GenericExecutor` , типичный `run_fn` будет выглядеть так:

```python
def run_fn(fn_args: TrainerFnArgs):
  """Train the model based on given args.

  Args:
    fn_args: Holds args used to train the model as name/value pairs.
  """
  tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

  # Train and eval files contains transformed examples.
  # _input_fn read dataset based on transformed feature_spec from tft.
  train_dataset = _input_fn(fn_args.train_files, tf_transform_output, 40)
  eval_dataset = _input_fn(fn_args.eval_files, tf_transform_output, 40)

  model = _build_keras_model()

  model.fit(
      train_dataset,
      steps_per_epoch=fn_args.train_steps,
      validation_data=eval_dataset,
      validation_steps=fn_args.eval_steps)

  signatures = {
      'serving_default':
          _get_serve_tf_examples_fn(model,
                                    tf_transform_output).get_concrete_function(
                                        tf.TensorSpec(
                                            shape=[None],
                                            dtype=tf.string,
                                            name='examples')),
  }
  model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)
```

In the `run_fn` above, a serving signature is needed when exporting the trained model so that model can take raw examples for prediction. A typical serving function would look like this:

```python
def _get_serve_tf_examples_fn(model, tf_transform_output):
  """Returns a function that parses a serialized tf.Example."""

  # the layer is added as an attribute to the model in order to make sure that
  # the model assets are handled correctly when exporting.
  model.tft_layer = tf_transform_output.transform_features_layer()

  @tf.function
  def serve_tf_examples_fn(serialized_tf_examples):
    """Returns the output to be used in the serving signature."""
    feature_spec = tf_transform_output.raw_feature_spec()
    feature_spec.pop(_LABEL_KEY)
    parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)

    transformed_features = model.tft_layer(parsed_features)

    return model(transformed_features)

  return serve_tf_examples_fn
```

В вышеупомянутой обслуживающей функции преобразования tf.Transform должны применяться к необработанным данным для вывода, используя слой [`tft.TransformFeaturesLayer`](https://github.com/tensorflow/transform/blob/master/docs/api_docs/python/tft/TransformFeaturesLayer.md) . Предыдущий `_serving_input_receiver_fn` который требовался для оценщиков, больше не нужен для Keras.

##### Файл модуля Keras без преобразования

Это похоже на файл модуля, показанный выше, но без преобразований:

```python
def _get_serve_tf_examples_fn(model, schema):

  @tf.function
  def serve_tf_examples_fn(serialized_tf_examples):
    feature_spec = _get_raw_feature_spec(schema)
    feature_spec.pop(_LABEL_KEY)
    parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
    return model(parsed_features)

  return serve_tf_examples_fn


def run_fn(fn_args: TrainerFnArgs):
  schema = io_utils.parse_pbtxt_file(fn_args.schema_file, schema_pb2.Schema())

  # Train and eval files contains raw examples.
  # _input_fn reads the dataset based on raw feature_spec from schema.
  train_dataset = _input_fn(fn_args.train_files, schema, 40)
  eval_dataset = _input_fn(fn_args.eval_files, schema, 40)

  model = _build_keras_model()

  model.fit(
      train_dataset,
      steps_per_epoch=fn_args.train_steps,
      validation_data=eval_dataset,
      validation_steps=fn_args.eval_steps)

  signatures = {
      'serving_default':
          _get_serve_tf_examples_fn(model, schema).get_concrete_function(
              tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')),
  }
  model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)
```

#####

[tf.distribute.Strategy](https://www.tensorflow.org/guide/distributed_training)

В настоящее время TFX поддерживает только стратегии одного работника (например, [MirroredStrategy](https://www.tensorflow.org/api_docs/python/tf/distribute/MirroredStrategy) , [OneDeviceStrategy](https://www.tensorflow.org/api_docs/python/tf/distribute/OneDeviceStrategy) ).

Чтобы использовать стратегию распространения, создайте соответствующий файл tf.distribute.Strategy и перенесите создание и компиляцию модели Keras в область действия стратегии.

Например, замените вышеупомянутую `model = _build_keras_model()` на:

```python
  mirrored_strategy = tf.distribute.MirroredStrategy()
  with mirrored_strategy.scope():
    model = _build_keras_model()

  # Rest of the code can be unchanged.
  model.fit(...)
```

Чтобы проверить устройство (CPU / GPU), используемое `MirroredStrategy` , включите ведение журнала тензор потока информационного уровня:

```python
import logging
logging.getLogger("tensorflow").setLevel(logging.INFO)
```

и вы должны быть в состоянии увидеть `Using MirroredStrategy with devices (...)` в журнале.

Note: The environment variable `TF_FORCE_GPU_ALLOW_GROWTH=true` might be needed for a GPU out of memory issue. For details, please refer to [tensorflow GPU guide](https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth).

#### оценщик

В TFMA v0.2x ModelValidator и Evaluator были объединены в один [новый компонент Evaluator](https://github.com/tensorflow/community/blob/master/rfcs/20200117-tfx-combining-model-validator-with-evaluator.md) . Новый компонент Evaluator может выполнять как оценку отдельной модели, так и проверять текущую модель по сравнению с предыдущими моделями. С этим изменением компонент Pusher теперь получает благословение от Evaluator, а не ModelValidator.

Новый Evaluator поддерживает модели Keras, а также модели Estimator. Сохраненные модели `_eval_input_receiver_fn` и eval, которые требовались ранее, больше не будут нужны с Keras, поскольку теперь Evaluator основан на той же `SavedModel` которая используется для обслуживания.

[Смотрите Evaluator для получения дополнительной информации](evaluator.md) .
