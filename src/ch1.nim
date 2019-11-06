import arraymancer
import sequtils, sugar
#import zero_functional
import strformat
import random

randomize()

type F* = float32

type
    Network* = ref object # of RootObj
      sizes*: seq[int]
      biases*: seq[Tensor[F]]
      weights*: seq[Tensor[F]]  # the * means that `name` is accessible from other modules
      num_layers*: int       # no * means that the field is hidden from other modules


proc newNetwork*(s: openarray[int]): Network =
    #var b = s[1 .. ^1].mapIt(randomNormalTensor(it, 0.F, 1.F))
    #b = b.reshape(b.shape[0],1)
    Network(
        sizes: s.toSeq,
        biases: s[1 .. ^1].mapIt(randomNormalTensor([it,1], 0.F, 1.F)),
        weights: zip(s[0 .. ^1], s[1 .. ^1]).mapIt(randomNormalTensor([it.b, it.a], 0.F, 1.F)),
        num_layers: s.len
    )

proc sigmoid_prime*(z: Tensor[F]): Tensor[F] =
    let s:Tensor[F] = sigmoid(z)
    let r:Tensor[F] = map_inline(s, x * (1.F - x))
    result = r
    #[
    let s = sig
    let t = map_inline(i):
        var s = 1 / (1 + exp(-x))  # can not call sigmoid here, som etemplate errors !!!
        s*(1-s)
    result = t[1]
    ]# 

proc feedforward*[T: SomeFloat](n: Network, a_in: Tensor[T]): Tensor[T] =
    #Return the output of the network if "a_in" is input.
    var a = a_in
    #zip(n.weights, n.biases).foldl( sigmoid( map3_inline(a, b.a, b.b): b.a * a + b.b), a_in)
    for (w,b) in zip(n.weights, n.biases):
        #echo "WEIGHT", w.shape
        #echo "BIAS", b.shape
        var t = w * a
        #echo "PRODUCT", t.shape
        t = t.reshape(t.shape[0],1)
        t = map2_inline( t, b, x + y )
        a = sigmoid(t)
    a = a.reshape(a.shape[0])
    #echo "RETURNING", a.shape
    result = a

proc shuffle[T](x: var openarray[T]) =
    for i in countdown(x.high, 0):
        let j = rand(i + 1)
        swap(x[i], x[j])

proc cost_derivative(net: Network, output_activations: Tensor[F], y_in: Tensor[F]): Tensor[F] =
    ## Return the vector of partial derivatives \partial C_x /
    ## \partial a for the output activations.
    result = map2_inline(output_activations, y_in, x-y)


proc backprop(net: Network, x: Tensor[F], y: Tensor[F]): (seq[Tensor[F]], seq[Tensor[F]]) =
    ## Return a tuple ``(nabla_b, nabla_w)`` representing the
    ## gradient for the cost function C_x.  ``nabla_b`` and
    ## ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
    ## to ``self.biases`` and ``self.weights``.
    
    var nabla_b = net.biases.mapIt(zeros_like(it))
    var nabla_w = net.weights.mapIt(zeros_like(it))

    # feedforward
    var activation = x
    var activations = @[x]
    var zs: seq[Tensor[F]] = @[]
    
    for (b, w) in zip(net.biases, net.weights):
        #echo "backprop w:"
        #echo w.shape
        #echo "======== b:"
        #echo b.shape
        ## echo "=========activation:"
        ## echo activation.shape
        var d = w * activation
        d = d.reshape(d.shape[0], 1)
        #echo "========== dot"
        #echo d.shape
        let z = map2_inline( d, b, x + y )
        #  Left-hand side has shape [30, 784] while right-hand side has shape [784] [ValueError]
        #let z = map3_inline( w, activation,b , (x * y) + z )
        #echo " 1", z.shape
        zs.add(z)
        activation = sigmoid(z)
        activations.add(activation)

    # backward pass
    #echo " 2 shapes:"
    #echo activations[^1].shape
    #echo y.shape
    #echo "---------"
    #echo net.cost_derivative(activations[^1], y).shape
    #echo type(sigmoid_prime(zs[^1]))
    ## echo "---------"
    ### cost_derivative = Tensor, sigmoid_prime = Tensor[system.float32]

    #### dot product is only with map2_inline !!!!!!!!!
    var delta: Tensor[F] = map2_inline(
                                        net.cost_derivative(activations[^1], y),
                                        sigmoid_prime(zs[^1]) ):
                                            x * y
    # delta SHAPE:[10]
    delta = delta.reshape(delta.shape[0],1)
    #echo "delta SHAPE:", delta.shape
    # delta SHAPE:[10,1]
    nabla_b[^1] = delta
    ### AHTUNG = we have not transposed it into correct VECTOR???
    ### https://mratsim.github.io/Arraymancer/tuto.linear_algebra.html
    nabla_w[^1] = delta * activations[^2].reshape(1, activations[^2].shape[0])
    ### NABLA LAST [10 30]
    ### BEFORE: delta [10 1]
    ### BEFORE: activations transposed [1 30]
    ### multiply two vectors
    # python xrange is not inclusive !!!
    # echo "NABLA FINAL:", nabla_w[^1].shape
    ### NABLA FINAL:[10, 30]
    for l in 2 .. (net.num_layers - 1):
        #echo fmt" index {l} zs.length: {zs.len} weights: {net.weights.len} biases: {net.biases.len}"
        let z = zs[^l]
        var sp = sigmoid_prime(z) # one number ???
        sp = sp.reshape(sp.shape[0],1)
        # sp.shape=[30]
        #echo "00000000--------------sp.shape=",  sp.shape

        # https://mratsim.github.io/Arraymancer/tuto.linear_algebra.html
        #let t = delta.reshape(delta.shape[0],1)
        #delta = map2_inline( net.weights[^(l-1)].transpose(), t):
        #       x * y
        #echo "============"
        #delta = delta * sp

        # AFTER TRANSPOSE [30 10]
        # RIGHT FOR DOT [10 1]
        # RESULT DELTA SHAPE [30 1]
        # DOT PRODUCT [30 1]
        # SP SHAPE FOR * [30 1]
        # Error: unhandled exception: Broadcasting error: non-singleton dimensions must be the same in both tensors. 
        # Tensors' shapes were: [30, 10] and [10] [ValueError]
        # should we convert vector delta [10] to [10,1] ????
        #echo "weights transpoised", net.weights[^(l-1)].transpose().shape
        # weights transpoised[30, 10]
        #echo "delta", delta.shape
        #echo "sp", sp.shape
        #delta[10, 1]
        #sp[30, 1]
        delta = (net.weights[^(l-1)].transpose() * delta) .* sp


        #echo "DELTA SHAPE: ", delta.shape # [30, 1]
        nabla_b[^l] = delta
        nabla_w[^l] = delta *  activations[^(l+1)].reshape(1, activations[^(l+1)].shape[0]);

    #echo "RETURN  2 shapes:"
    #echo nabla_b[0].shape
    #echo nabla_w[0].shape
    ### RETURN  2 shapes:
    ### [30, 1]
    ### [30, 784]
    ### совпадает с Clojure
    ### RETURN 0 nabla_b [30 1]
    ### RETURN 0 nabla_w [30 784]
    result = (nabla_b, nabla_w)

proc update_mini_batch(net: Network, mini_batch: openarray[(Tensor[F],Tensor[F])], eta:float32): float32 =
    ## Should return loss
    ## Update the network's weights and biases by applying
    ## gradient descent using backpropagation to a single mini batch.
    ## The "mini_batch" is a list of tuples "(x, y)", and "eta"
    ## is the learning rate.
    #echo "update_mini_batch: started"
    var nabla_b = net.biases.mapIt(zeros_like(it).reshape(it.shape[0],1))

    var nabla_w = net.weights.mapIt(zeros_like(it))
    #echo "update_mini_batch: init done"
    for t in mini_batch:
        #var delta_nabla_b, delta_nabla_w: seq[Tensor[F]]
        var (delta_nabla_b, delta_nabla_w) = net.backprop(t[0], t[1])
        nabla_b = zip(nabla_b, delta_nabla_b).mapIt(map2_inline(it.a, it.b, x + y))
        nabla_w = zip(nabla_w, delta_nabla_w).mapIt(map2_inline(it.a, it.b, x + y))
        # echo fmt"update_mini_batch: {nabla_b},{nabla_w}"
        # echo fmt"minibatch: {i}"
        # w, nw == x, y
    #echo "AFTER LOOP"
    net.weights = zip(net.weights, nabla_w).mapIt(map2_inline(it.a, it.b, x-(eta/len(mini_batch).float32)*y))
    #echo "weights"
    net.biases = zip(net.biases, nabla_b).mapIt(map2_inline(it.a, it.b, x-(eta/len(mini_batch).float32)*y ))
    #echo "qua"
    result = 0.8

proc evaluate(net: Network, test_data: openarray[(Tensor[F],int)]): float =
    for t in test_data:
        let net_result = net.feedforward(t[0]).argmax(0)
        echo fmt"Got from net: {net_result[0]} == {t[1]} ??"
        if net_result[0] == t[1]:
            result = result + 1

#openarray[array[0..1, Tensor[F]]]
proc SGD*(net: Network, 
          training_data: var openarray[(Tensor[F],Tensor[F])], 
          epochs: int, 
          mini_batch_size: int, 
          eta: float32, # is the learning rate, η
          test_data: openarray[(Tensor[F],int)] = @[]): void =
    ## Train the neural network using mini-batch stochastic
    ## gradient descent.  The "training_data" is a list of tuples
    ## "(x, y)" representing the training inputs and the desired
    ## outputs.  The other non-optional parameters are
    ## self-explanatory.  If "test_data" is provided then the
    ## network will be evaluated against the test data after each
    ## epoch, and partial progress printed out.  This is useful for
    ## tracking progress, but slows things down substantially.
    var n_test = test_data.len - 1
    var n = training_data.len - 1
    for j in 1 .. epochs:
        shuffle(training_data)
        echo fmt"Started epoch {j}"

        for k in countup(0, n, mini_batch_size):
            var mini_batch = training_data[k .. k+mini_batch_size-1]
            #echo mini_batch
            discard net.update_mini_batch(mini_batch, eta)

        #  discard training_data.toSeq.distribute(mini_batch_size).mapIt(net.update_mini_batch(it, eta))
        #  will get [1]    27079 killed     ./deep_book :-()
        # echo training_data.toSeq.distribute(mini_batch_size).mapIt(it.len)

        if n_test > 0:
            echo fmt"Epoch {j}: {net.evaluate(test_data) / test_data.len.toFloat}"
        else:
            echo fmt"Epoch {j} complete"

