import tensorflow as tf
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import nn_ops
import numpy as np

INF = 1e30


def dropout(args, keep_prob, is_train, mode="recurrent"):
    if keep_prob < 1.0:
        noise_shape = None
        scale = 1.0
        shape = tf.shape(args)
        if mode == "embedding":
            noise_shape = [shape[0], 1]
            scale = keep_prob
        if mode == "recurrent" and len(args.get_shape().as_list()) == 3:
            noise_shape = [shape[0], 1, shape[-1]]
        args = tf.cond(is_train, lambda: tf.nn.dropout(
            args, keep_prob, noise_shape=noise_shape) * scale, lambda: args)
    return args


def dot_attention(inputs, memory, mask, hidden, keep_prob=1.0, is_train=None, scope="dot_attention"):
    with tf.variable_scope(scope):
        d_inputs = dropout(inputs, keep_prob=keep_prob, is_train=is_train)
        d_memory = dropout(memory, keep_prob=keep_prob, is_train=is_train)
        JX = tf.shape(inputs)[1]

        inputs_ = tf.layers.dense(d_inputs, hidden, activation=tf.nn.relu, use_bias=False, name="inputs")
        memory_ = tf.layers.dense(d_memory, hidden, activation=tf.nn.relu, use_bias=False, name="memory")
        outputs = tf.matmul(inputs_, tf.transpose(memory_, [0, 2, 1])) / (hidden ** 0.5)
        mask = tf.tile(tf.expand_dims(mask, axis=1), [1, JX, 1])
        logits = tf.nn.softmax(softmax_mask(outputs, mask))
        outputs = tf.matmul(logits, memory)
        return outputs  # [B, JX, hidden]


def softmax_mask(val, mask):
    return -INF * (1 - tf.cast(mask, tf.float32)) + val


def dense(inputs, hidden, use_bias=True, scope="dense"):
    with tf.variable_scope(scope):
        shape = tf.shape(inputs)
        dim = inputs.get_shape().as_list()[-1]
        out_shape = [shape[idx] for idx in range(
            len(inputs.get_shape().as_list()) - 1)] + [hidden]
        flat_inputs = tf.reshape(inputs, [-1, dim])
        W = tf.get_variable("W", [dim, hidden])
        res = tf.matmul(flat_inputs, W)
        if use_bias:
            b = tf.get_variable(
                "b", [hidden], initializer=tf.constant_initializer(0.))
            res = tf.nn.bias_add(res, b)
        res = tf.reshape(res, out_shape)
        return res


# Note: this function is based on tf.contrib.legacy_seq2seq_attention_decoder, which is now outdated.
# In the future, it would make more sense to write variants on the attention mechanism using the new seq2seq library for tensorflow 1.0: https://www.tensorflow.org/api_guides/python/contrib.seq2seq#Attention
def attention_decoder(decoder_inputs,
                      initial_state,
                      encoder_states,
                      enc_padding_mask,
                      cell,
                      initial_state_attention=False,
                      attention_gate=None,
                      pointer_gen=True):
    """
  Args:
    decoder_inputs: A list of 2D Tensors [batch_size x input_size].
    initial_state: 2D Tensor [batch_size x cell.state_size].
    encoder_states: 3D Tensor [batch_size x attn_length x attn_size].
    cell: rnn_cell.RNNCell defining the cell function and size.
    initial_state_attention:
      Note that this attention decoder passes each decoder input through a linear layer with the previous step's context
      vector to get a modified version of the input. If initial_state_attention is False, on the first decoder step
      the "previous context vector" is just a zero vector. If initial_state_attention is True, we use initial_state to
      (re)calculate the previous step's context vector. We set this to False for train/eval mode (because we call
      attention_decoder once for all decoder steps) and True for decode mode (because we call attention_decoder once
      for each decoder step).
    pointer_gen: boolean. If True, calculate the generation probability p_gen for each decoder step.
    use_coverage: boolean. If True, use coverage mechanism.
    prev_coverage:
      If not None, a tensor with shape (batch_size, attn_length). The previous step's coverage vector. This is only not
      None in decode mode when using coverage.

  Returns:
    outputs: A list of the same length as decoder_inputs of 2D Tensors of
      shape [batch_size x cell.output_size]. The output vectors.
    state: The final state of the decoder. A tensor shape [batch_size x cell.state_size].
    attn_dists: A list containing tensors of shape (batch_size,attn_length).
      The attention distributions for each decoder step.
    p_gens: List of scalars. The values of p_gen for each decoder step. Empty list if pointer_gen=False.
    coverage: Coverage vector on the last step computed. None if use_coverage=False.
  """
    with tf.variable_scope("attention_decoder") as scope:
        batch_size = encoder_states.get_shape()[0].value
        atten_length = encoder_states.get_shape()[1].value
        attn_size = encoder_states.get_shape()[2].value
        assert batch_size is not None, "The batch size is None"
        assert atten_length is not None, "The atten length is None"
        assert attn_size is not None, "The atten size is None"

        # Reshape encoder_states (need to insert a dim)
        encoder_states = tf.expand_dims(encoder_states, axis=2)
        # now is shape (batch_size, attn_len, 1, attn_size)

        # To calculate attention, we calculate
        #   v^T tanh(W_h h_i + W_s s_t + b_attn)
        # where h_i is an encoder state, and s_t a decoder state.
        # attn_vec_size is the length of the vectors v, b_attn, (W_h h_i) and (W_s s_t).
        # We set it to be equal to the size of the encoder states.
        attention_vec_size = attn_size

        # Get the weight matrix W_h and apply it to each encoder state to get (W_h h_i), the encoder features
        W_h = tf.get_variable("W_h", [1, 1, attn_size, attention_vec_size])
        encoder_features = tf.nn.conv2d(encoder_states, W_h, [1, 1, 1, 1], "SAME")
        # shape (batch_size,attn_length,1,attention_vec_size)

        # Get the weight vectors v and w_c (w_c is for coverage)
        v = variable_scope.get_variable("v", [attention_vec_size])

        def attention(decoder_state):
            """Calculate the context vector and attention distribution from the decoder state.

            Args:
              decoder_state: state of the decoder

            Returns:
              context_vector: weighted sum of encoder_states
              attn_dist: attention distribution
            """
            with tf.variable_scope("Attention"):
                # Pass the decoder state through a linear layer (this is W_s s_t + b_attn in the paper)
                decoder_features = linear(decoder_state, attention_vec_size, True)
                # shape (batch_size, attention_vec_size)
                decoder_features = tf.expand_dims(tf.expand_dims(decoder_features, 1), 1)

                # reshape to (batch_size, 1, 1, attention_vec_size)

                def masked_attention(e):
                    """Take softmax of e then apply enc_padding_mask and re-normalize"""
                    attn_dist = tf.nn.softmax(e, axis=1)  # take softmax. shape (batch_size, attn_length)
                    attn_dist *= enc_padding_mask  # apply mask
                    masked_sums = tf.reduce_sum(attn_dist, axis=1)  # shape (batch_size)
                    return attn_dist / tf.reshape(masked_sums, [-1, 1])  # re-normalize

                # Calculate v^T tanh(W_h h_i + W_s s_t + b_attn)
                e = tf.reduce_sum(v * tf.tanh(encoder_features + decoder_features), [2, 3])
                # calculate e

                # Take softmax of e to get the attention distribution
                # attn_dist = nn_ops.softmax(e)
                if attention_gate is not None:
                    e *= attention_gate

                attn_dist = masked_attention(e)
                # shape (batch_size, attn_length)

                # Calculate the context vector from attn_dist and encoder_states
                context_vector = tf.reduce_sum(
                    tf.reshape(attn_dist, [batch_size, atten_length, 1, 1]) * encoder_states, [1, 2])
                context_vector = tf.reshape(context_vector, [-1, attn_size])
                # shape (batch_size, attn_size).

            return context_vector, attn_dist

        outputs = []
        attn_dists = []
        p_gens = []
        state = initial_state
        context_vector = tf.zeros([batch_size, attn_size], dtype=tf.float32)

        if initial_state_attention:  # true in decode mode
            # Re-calculate the context vector from the previous step so that we can pass it through a linear layer with this step's input to get a modified version of the input
            context_vector, _ = attention(initial_state)
            # in decode mode, this is what updates the coverage vector
        for i, inp in enumerate(decoder_inputs):
            if i > 0:
                tf.get_variable_scope().reuse_variables()

            # Merge input and previous attentions into one vector x of the same size as inp
            input_size = inp.get_shape().with_rank(2)[1]
            if input_size.value is None:
                raise ValueError("Could not infer input size from input: %s" % inp.name)

            x = linear([inp] + [context_vector], input_size, True)

            # Run the decoder RNN cell. cell_output = decoder state
            cell_output, state = cell(x, state)

            # Run the attention mechanism.
            if i == 0 and initial_state_attention:  # always true in decode mode
                with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                    # you need this because you've already run the initial attention(...) call
                    context_vector, attn_dist = attention(state)
                    # don't allow coverage to update
            else:
                context_vector, attn_dist = attention(state)
            attn_dists.append(attn_dist)

            # Calculate p_gen
            if pointer_gen:
                with tf.variable_scope('calculate_pgen'):
                    p_gen = linear([context_vector, state, x], 1, True)  # a scalar
                    p_gen = tf.sigmoid(p_gen)
                    p_gens.append(p_gen)

            # Concatenate the cell_output (= decoder state) and the context vector, and pass them through a linear layer
            # This is V[s_t, h*_t] + b in the paper
            with tf.variable_scope("AttnOutputProjection"):
                output = linear([cell_output] + [context_vector], cell.output_size, True)
            outputs.append(output)

        return outputs, state, attn_dists, p_gens


def attention_decoder_v2(decoder_inputs,
                         initial_state,
                         memory_enc,
                         memory_user,
                         mem_enc_padding_mask_1,
                         cell,
                         pointer_gen=True):
    """
  Args:
    decoder_inputs: A list of 2D Tensors [batch_size x input_size].
    initial_state: 2D Tensor [batch_size x cell.state_size].
    memory_1: 3D Tensor [batch_size x attn_length x attn_size].
    cell: rnn_cell.RNNCell defining the cell function and size.
    initial_state_attention:
      Note that this attention decoder passes each decoder input through a linear layer with the previous step's context
      vector to get a modified version of the input. If initial_state_attention is False, on the first decoder step
      the "previous context vector" is just a zero vector. If initial_state_attention is True, we use initial_state to
      (re)calculate the previous step's context vector. We set this to False for train/eval mode (because we call
      attention_decoder once for all decoder steps) and True for decode mode (because we call attention_decoder once
      for each decoder step).
    pointer_gen: boolean. If True, calculate the generation probability p_gen for each decoder step.
    use_coverage: boolean. If True, use coverage mechanism.
    prev_coverage:
      If not None, a tensor with shape (batch_size, attn_length). The previous step's coverage vector. This is only not
      None in decode mode when using coverage.

  Returns:
    outputs: A list of the same length as decoder_inputs of 2D Tensors of
      shape [batch_size x cell.output_size]. The output vectors.
    state: The final state of the decoder. A tensor shape [batch_size x cell.state_size].
    attn_dists: A list containing tensors of shape (batch_size,attn_length).
      The attention distributions for each decoder step.
    p_gens: List of scalars. The values of p_gen for each decoder step. Empty list if pointer_gen=False.
    coverage: Coverage vector on the last step computed. None if use_coverage=False.
  """
    with tf.variable_scope("attention_decoder") as scope:
        batch_size = memory_enc.get_shape()[0].value
        atten_length = memory_enc.get_shape()[1].value
        attn_size = memory_enc.get_shape()[2].value
        assert batch_size is not None, "The batch size is None"
        assert atten_length is not None, "The atten length is None"
        assert attn_size is not None, "The atten size is None"

        # Reshape encoder_states (need to insert a dim)
        memory_1 = tf.expand_dims(memory_enc, axis=2)

        atten_length_1 = memory_1.get_shape()[1].value
        # now is shape (batch_size, attn_len, 1, attn_size)

        # To calculate attention, we calculate
        #   v^T tanh(W_h h_i + W_s s_t + b_attn)
        # where h_i is an encoder state, and s_t a decoder state.
        # attn_vec_size is the length of the vectors v, b_attn, (W_h h_i) and (W_s s_t).
        # We set it to be equal to the size of the encoder states.
        attention_vec_size = attn_size

        # Get the weight matrix W_h and apply it to each encoder state to get (W_h h_i), the encoder features
        W_h_1 = tf.get_variable("W_h_1", [1, 1, attn_size, attention_vec_size])
        encoder_features_1 = tf.nn.conv2d(memory_1, W_h_1, [1, 1, 1, 1], "SAME")
        # shape (batch_size,attn_length,1,attention_vec_size)

        def attention(decoder_state, memory, atten_length, memory_mask, encoder_features, scope='Attention',
                      reuse=False):
            """Calculate the context vector and attention distribution from the decoder state.

            Args:
              decoder_state: state of the decoder

            Returns:
              context_vector: weighted sum of encoder_states
              attn_dist: attention distribution
            """
            with tf.variable_scope(scope, reuse=reuse):
                # Pass the decoder state through a linear layer (this is W_s s_t + b_attn in the paper)
                decoder_features = linear(decoder_state, attention_vec_size, True)
                # shape (batch_size, attention_vec_size)
                decoder_features = tf.expand_dims(tf.expand_dims(decoder_features, 1), 1)

                # reshape to (batch_size, 1, 1, attention_vec_size)

                def masked_attention(e):
                    """Take softmax of e then apply enc_padding_mask and re-normalize"""
                    attn_dist = tf.nn.softmax(e, axis=1)  # take softmax. shape (batch_size, attn_length)
                    attn_dist *= memory_mask  # apply mask
                    masked_sums = tf.reduce_sum(attn_dist, axis=1)  # shape (batch_size)
                    return attn_dist / (tf.reshape(masked_sums, [-1, 1])+1e-8)  # re-normalize

                # Calculate v^T tanh(W_h h_i + W_s s_t + b_attn)
                v = tf.get_variable('v', [attention_vec_size])
                e = tf.reduce_sum(v * tf.tanh(encoder_features + decoder_features), [2, 3])
                # calculate e

                # Take softmax of e to get the attention distribution
                # attn_dist = softmax(e)
                attn_dist = masked_attention(e)
                # shape (batch_size, attn_length)

                # Calculate the context vector from attn_dist and encoder_states
                context_vector = tf.reduce_sum(tf.reshape(attn_dist, [batch_size, atten_length, 1, 1]) * memory, [1, 2])
                context_vector = tf.reshape(context_vector, [-1, attn_size])
                # shape (batch_size, attn_size).

                return context_vector, attn_dist

        outputs = []
        attn_dists_1 = []
        p_gens = []
        state = initial_state

        # Re-calculate the context vector from the previous step so that we can pass it through a linear layer with this step's input to get a modified version of the input
        context_vector_1, _ = attention(initial_state, memory_1, atten_length_1, mem_enc_padding_mask_1, encoder_features_1,
                                        'attn_1')

        # in decode mode, this is what updates the coverage vector
        for i, inp in enumerate(decoder_inputs):
            if i > 0:
                tf.get_variable_scope().reuse_variables()

            # Merge input and previous attentions into one vector x of the same size as inp
            input_size = inp.get_shape().with_rank(2)[1]
            if input_size.value is None:
                raise ValueError("Could not infer input size from input: %s" % inp.name)
            x = linear([inp, context_vector_1, memory_user], input_size, True)
            #x = linear([inp, context_vector_1], input_size, True)

            # Run the decoder RNN cell. cell_output = decoder state
            cell_output, state = cell(x, state)

            # Run the attention mechanism.
            context_vector_1, attn_dist_1 = attention(state, memory_1, atten_length_1, mem_enc_padding_mask_1,
                                                      encoder_features_1, 'attn_1', reuse=True)

            attn_dists_1.append(attn_dist_1)


            # Concatenate the cell_output (= decoder state) and the context vector, and pass them through a linear layer
            # This is V[s_t, h*_t] + b in the paper
            with tf.variable_scope("AttnOutputProjection"):
                #output = mlp(tf.concat([cell_output, context_vector_1, x, gsm], 1), 3,
                             #cell.output_size, scope="vocab_score")
                output = mlp(tf.concat([cell_output, context_vector_1, memory_user, x], 1), 3,
                             cell.output_size, scope="vocab_score")

            outputs.append(output)

        return outputs, state


def linear(args, output_size, bias, bias_start=0.0, scope=None):
    """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_start: starting value to initialize the bias; 0 by default.
    scope: VariableScope for the created subgraph; defaults to "Linear".

  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
    if args is None or (isinstance(args, (list, tuple)) and not args):
        raise ValueError("`args` must be specified")
    if not isinstance(args, (list, tuple)):
        args = [args]

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        if len(shape) != 2:
            raise ValueError(
                "Linear is expecting 2D arguments: %s" % str(shapes))
        if not shape[1]:
            raise ValueError(
                "Linear expects shape[1] of arguments: %s" % str(shapes))
        else:
            total_arg_size += shape[1]

    # Now the computation.
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [total_arg_size, output_size])
        if len(args) == 1:
            res = tf.matmul(args[0], matrix)
        else:
            res = tf.matmul(tf.concat(axis=1, values=args), matrix)
        if not bias:
            return res
        bias_term = tf.get_variable("Bias", [output_size], initializer=tf.constant_initializer(bias_start))
    return res + bias_term


def mlp(inputs, layers=2, out_dim=1, activation=tf.nn.relu, activate_last=False, scope='mlp'):
    with tf.variable_scope(scope):
        inputs_neurons = inputs.shape.as_list()[-1]
        neurons = inputs_neurons * 2
        output = inputs
        for i in range(1, layers + 1):
            dim = out_dim if i == layers else neurons
            act = None if not activate_last and i == layers else activation
            output = tf.layers.dense(output, dim, activation=act, name=f'layer_{i}')
            output = tf.contrib.layers.maxout(output, inputs_neurons) if i != layers else output
        return output


def normalize(inputs,
              epsilon=1e-8,
              scope="ln",
              reuse=None):
    '''Applies layer normalization.

    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`.
      epsilon: A floating number. A very small number for preventing ZeroDivision Error.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta

    return outputs


def embedding(inputs,
              vocab_size,
              num_units,
              zero_pad=True,
              scale=True,
              scope="embedding",
              reuse=None):
    '''Embeds a given tensor.
    Args:
      inputs: A `Tensor` with type `int32` or `int64` containing the ids
         to be looked up in `lookup table`.
      vocab_size: An int. Vocabulary size.
      num_units: An int. Number of embedding hidden units.
      zero_pad: A boolean. If True, all the values of the fist row (id 0)
        should be constant zeros.
      scale: A boolean. If True. the outputs is multiplied by sqrt num_units.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
    Returns:
      A `Tensor` with one more rank than inputs's. The last dimensionality
        should be `num_units`.

    For example,

    ```
    import tensorflow as tf

    inputs = tf.to_int32(tf.reshape(tf.range(2*3), (2, 3)))
    outputs = embedding(inputs, 6, 2, zero_pad=True)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(outputs)
    >>
    [[[ 0.          0.        ]
      [ 0.09754146  0.67385566]
      [ 0.37864095 -0.35689294]]
     [[-1.01329422 -1.09939694]
      [ 0.7521342   0.38203377]
      [-0.04973143 -0.06210355]]]
    ```

    ```
    import tensorflow as tf

    inputs = tf.to_int32(tf.reshape(tf.range(2*3), (2, 3)))
    outputs = embedding(inputs, 6, 2, zero_pad=False)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(outputs)
    >>
    [[[-0.19172323 -0.39159766]
      [-0.43212751 -0.66207761]
      [ 1.03452027 -0.26704335]]
     [[-0.11634696 -0.35983452]
      [ 0.50208133  0.53509563]
      [ 1.22204471 -0.96587461]]]
    ```
    '''
    with tf.variable_scope(scope, reuse=reuse):
        lookup_table = tf.get_variable('lookup_table',
                                       dtype=tf.float32,
                                       shape=[vocab_size, num_units],
                                       initializer=tf.contrib.layers.xavier_initializer())
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, inputs)

        if scale:
            outputs = outputs * (num_units ** 0.5)

    return outputs


def positional_encoding(inputs,
                        num_units,
                        zero_pad=True,
                        scale=True,
                        scope="positional_encoding",
                        reuse=None):
    '''Sinusoidal Positional_Encoding.
    Args:
      inputs: A 2d Tensor with shape of (N, T).
      num_units: Output dimensionality
      zero_pad: Boolean. If True, all the values of the first row (id = 0) should be constant zero
      scale: Boolean. If True, the output will be multiplied by sqrt num_units(check details from paper)
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
    Returns:
        A 'Tensor' with one more rank than inputs's, with the dimensionality should be 'num_units'
    '''

    N, T = inputs.get_shape().as_list()
    with tf.variable_scope(scope, reuse=reuse):
        position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])

        # First part of the PE function: sin and cos argument
        position_enc = np.array([
            [pos / np.power(10000, 2. * i / num_units) for i in range(num_units)]
            for pos in range(T)])

        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1

        # Convert to a tensor
        lookup_table = tf.convert_to_tensor(position_enc)

        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, position_ind)

        if scale:
            outputs = outputs * num_units ** 0.5

        return outputs


def multihead_attention(queries,
                        keys,
                        num_units=None,
                        num_heads=8,
                        dropout_rate=0,
                        is_training=True,
                        causality=False,
                        scope="multihead_attention",
                        reuse=None):
    '''Applies multihead attention.

    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      causality: Boolean. If true, units that reference the future are masked.
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns
      A 3d tensor with shape of (N, T_q, C)
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]

        # Linear projections
        Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu)  # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)
        V = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        # Key Masking
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # (N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)
            tril = tf.contrib.linalg.LinearOperatorTriL(diag_vals).to_dense()  # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])  # (h*N, T_q, T_k)

            paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Activation
        outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

        # Query Masking
        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))  # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
        outputs *= query_masks  # broadcasting. (N, T_q, C)

        # Dropouts
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

        # Weighted sum
        outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

        # Residual connection
        outputs += queries

        # Normalize
        outputs = normalize(outputs)  # (N, T_q, C)

    return outputs


def feedforward(inputs,
                num_units=(2048, 512),
                scope="multihead_attention",
                reuse=None):
    '''Point-wise feed forward net.

    Args:
      inputs: A 3d tensor with shape of [N, T, C].
      num_units: A list of two integers.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A 3d tensor with the same shape and dtype as inputs
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Inner layer
        params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                  "activation": tf.nn.relu, "use_bias": True}
        outputs = tf.layers.conv1d(**params)

        # Readout layer
        params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                  "activation": None, "use_bias": True}
        outputs = tf.layers.conv1d(**params)

        # Residual connection
        outputs += inputs

        # Normalize
        outputs = normalize(outputs)

    return outputs


def pointer(memory, state, att_size, mask, scope="pointer"):
    with tf.variable_scope(scope):
        u = tf.concat([tf.tile(tf.expand_dims(state, axis=1), [
            1, tf.shape(memory)[1], 1]), memory], axis=2)
        s0 = tf.nn.tanh(tf.layers.dense(u, att_size, use_bias=False, name="s0"))
        s = tf.layers.dense(s0, 1, use_bias=False, name="s")
        s1 = softmax_mask(tf.squeeze(s, [2]), mask)
        a = tf.expand_dims(tf.nn.softmax(s1), axis=2)
        res = tf.reduce_sum(a * memory, axis=1)
        return res, s1


class ptr_net:
    def __init__(self, hidden_dim, dropout_rate=1.0, is_train=None, scope="ptr_net"):
        self.gru = tf.nn.rnn_cell.GRUCell(hidden_dim)
        self.scope = scope
        self.dropout_rate = dropout_rate
        self.is_train = is_train

    def __call__(self, init_state, pos_inp, memory, atten_size, mask):
        assert len(init_state.get_shape().as_list()) == 2
        assert len(pos_inp.get_shape().as_list()) == 2
        with tf.variable_scope(self.scope):
            d_memory = tf.layers.dropout(memory, self.dropout_rate, training=self.is_train)
            c, logits1 = pointer(d_memory, init_state, atten_size, mask)

            _, state = self.gru(tf.concat([c, pos_inp], 2), init_state)
            tf.get_variable_scope().reuse_variables()
            _, logits2 = pointer(d_memory, state, atten_size, mask)
            return logits1, logits2


def summ(memory, hidden_dim, mask, keep_prob=0.5, is_train=None, scope="summ"):
    with tf.variable_scope(scope):
        d_memory = dropout(memory, keep_prob, is_train)
        s0 = tf.nn.tanh(tf.layers.dense(d_memory, hidden_dim))
        s = tf.layers.dense(s0, 1, use_bias=False)
        s1 = tf.nn.softmax(softmax_mask(tf.squeeze(s, [2]), mask))
        a = tf.expand_dims(s1, axis=2)
        res = tf.reduce_sum(a * memory, axis=1)
        return res  # [B,hidden_dim]
