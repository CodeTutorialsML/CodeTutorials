"""Create and train a simple, custom, TensorFlow estimator"""
import tensorflow as tf


# Define the input function
def input_fn():
  # Make a dummy dataset
  dataset = tf.data.Dataset.range(10)
  dataset = dataset.map(lambda x: tf.cast(x, tf.float32))
  dataset = dataset.map(lambda x: {'x': x, 't': 0.5*x + 3})
  dataset = dataset.repeat(100)
  return dataset

# Define the model function
def model_fn(features, labels, mode, params):
  # Get the model inputs
  x = features['x']
  # Build the model
  a = tf.get_variable('a', 1)
  b = tf.get_variable('b', 1)
  y = a*x + b
  if mode == tf.estimator.ModeKeys.TRAIN:
      # Get the current iteration
    global_step = tf.train.get_or_create_global_step()
    # Get the model targets
    t = features['t']
    # Calculate the loss
    loss = (t - y)**2
    # Instantiate the optimizer
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    # Get the train operation
    train_op = optimizer.minimize(loss, global_step)
    # Return the estimator specification
    output_spec = tf.estimator.EstimatorSpec(
      mode=mode,
      loss=loss,
      train_op=train_op)

  return output_spec

# Print internal logs to console
tf.logging.set_verbosity(tf.logging.INFO)
# Instantiate the estimator
estimator = tf.estimator.Estimator(model_fn=model_fn)
# Train the estimator
estimator.train(input_fn=input_fn)
