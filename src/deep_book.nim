# This is just an example to get you started. A typical binary package
# uses this file as the main entry point of the application.
import sequtils
import strformat
import arraymancer
import ch1

let
  mnist = load_mnist(cache = true)
  # Training data is 60k 28x28 greyscale images from 0-255,
  # neural net prefers input rescaled to [0, 1] or [-1, 1]
  x_train = mnist.train_images.astype(F) / 255'f32
  
  # Change shape from [N, H, W] to [N, C, H, W], with C = 1 (unsqueeze). Convolution expect 4d tensors
  # And store in the context to track operations applied and build a NN graph
  # X_train = x_train.unsqueeze(1)
  
  # Labels are uint8, we must convert them to int
  y_train = mnist.train_labels.astype(int)
  
  # Item for testing data (10000 images)
  x_test = mnist.test_images.astype(F) / 255'f32
  #X_test = x_test.unsqueeze(1)
  y_test = mnist.test_labels.astype(int)

when isMainModule:
  let net = newNetwork([784, 30, 10])
  #let train = zip(x_train, y_train)
  #echo(repr(net))

  ## this will get 28x28 single image !!!
  ## echo x_train.shape[0]
  ## echo x_train.atAxisIndex(0,1).squeeze(0)

  #[
  var a = toSeq(1..6).toTensor().reshape(1,6)
  echo a
  var aa = a.reshape(6,1)
  echo fmt"reshaped {aa}"
  var b = toSeq(1..6).toTensor().reshape(1,6)
  echo aa * b
  ]#

  # for y we need vector of lenth 10 := WRONG ! WE decided to use Tensors everywhere, so it should be matrix [10,1]
  # for x we need single vector of length 784, not 28x28
  var t: seq[(Tensor[F],Tensor[F])]
  for i in 0 .. (x_train.shape[0]-1):
    var f = zeros[F]([10])
    f[y_train[i]] = 1.0
    f = f.reshape(f.shape[0],1)
    t.add( (x_train.atAxisIndex(0,i).squeeze(0).reshape(784) , f) )

  var e: seq[(Tensor[F],int)]
  for i in 0 .. (x_test.shape[0]-1):
    e.add( (x_test.atAxisIndex(0,i).squeeze(0).reshape(784) , y_test[i]) )
  
  ch1.SGD(net, t, 30, 10, 3.0, e)


  ## Inline iterator over an axis.
  ##
  ## Returns:
  ##   - A slice along the given axis at each iteration.
  ##
  ## Note: The slice dimension is not collapsed by default.
  ## You can use ``squeeze`` to collapse it.
  ##
  ## Usage:
  ##  .. code:: nim
  ##     for subtensor in t.axis(1):
  ##       # do stuff
