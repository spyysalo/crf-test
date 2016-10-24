from keras import backend as K

from logging import warn


def scan(fn, sequences=None, outputs_info=None, non_sequences=None,
         n_steps=None, truncate_gradient=-1, go_backwards=False, mode=None,
         name=None, profile=False, allow_gc=None, strict=False,
         unroll=False, batch=False):
    """Call appropriate version of scan() with given arguments."""

    # Arguments for implementations minus ones specific to this function
    args = { k: v for k, v in locals().items() if k != 'unroll' }

    if K.backend() == 'tensorflow' and not unroll:
        # Tensorflow scan() is currently not supported, so unrolling
        # is the only option with this backend.
        warn('tensorflow backend, setting unroll=True')
        unroll = True

    # TODO: support batch=True with theano.scan. This could be done
    # with dimshuffles before and after invokind scan(), which could
    # also allow a uniform treatment of the TF and Theano versions and
    # removing the batch argument from unroll_scan().
    if batch and not unroll:
        warn('batch==True, setting unroll=True')
        unroll = True

    if not unroll:
        import theano
        args = { k: v for k, v in args.items() if k != 'batch' }
        return theano.scan(**args)
    else:
        args = _unroll_scan_arguments(args)
        return unroll_scan(**args)


def unroll_scan(fn, sequences=None, initial_values=None, non_sequences=None,
                n_steps=None, batch=False):
    """Limited reimplementation of theano.scan() by unrolling.

    Based on unroll_scan() from Lasagne.
    """
    sequences = _to_list(sequences)
    initial_values = _to_list(initial_values)
    non_sequences = _to_list(non_sequences)

    sequential_outputs = []
    previous = initial_values
    for i in range(n_steps):
        if not batch:
            args = [s[i] for s in sequences]
        else:
            args = [s[:, i] for s in sequences]
        args += previous + non_sequences
        outputs = _to_list(fn(*args))
        sequential_outputs.append(outputs)
        previous = outputs

    # Output formatting. sequential_ouputs is now a list of lists, the
    # outer containing an item for each of the steps (n_steps in
    # total) and each of the inner containing the outputs of the step
    # function fn, i.e.
    #
    # [ [ step_1_out_1 step_1_out_2 ... step_1_out_o ]
    #   [ step_2_out_1 step_2_out_2 ... step_2_out_o ]
    #   ...
    #   [ step_n_out_1 step_n_out_2 ... step_n_out_o ] ]
    #
    # these must be reorganized into the theano.scan() order
    #
    # [ [ step_1_out_1 step_2_out_1 ... step_n_out_1 ]
    #   [ step_1_out_2 step_2_out_2 ... step_n_out_2 ]
    #   ....
    #   [ step_1_out_n step_2_out_2 ... step_n_out_o ] ]
    # 
    # i.e from (n_steps, n_outputs) to (n_outputs, n_steps). Also,
    # the various step values for each output should be packed into
    # a tensor (instead of a list), giving [ out_1_steps, out_2_steps,
    # ... out_o_steps ].
    #
    # Then, if run in batch mode, each of the output tensors will have
    # shape (n_steps, batch_size, ...), which should be permuted into
    # (batch_size, n_steps, ...).
    #
    # Finally, following the model of theano.scan(), if there is only
    # a single output, return the corresponding tensor t instead of
    # a list [t] with a single elements, and if there are no outputs,
    # return None instead of an empty list.
   
    # Reorganize and pack
    output_sequences = []
    n_outputs = len(sequential_outputs[0])
    for o in range(n_outputs):
        outs = [s[o] for s in sequential_outputs]
        output_sequences.append(K.pack(outs))

    # Permute if batchwise
    if batch:
        for o, s in enumerate(output_sequences):
            dim_indices = range(K.ndim(s))    # [0, 1, ...]
            pattern = [1, 0] + dim_indices[2:]    # [1, 0, ...]
            output_sequences[o] = K.permute_dimensions(s, pattern)

    # Remove list wrapping
    if len(output_sequences) == 0:
        output = None
    elif len(output_sequences) == 1:
        output = output_sequences[0]
    else:
        output = output_sequences

    return output, None    # None for updates dummy


def _unroll_scan_arguments(args):
    """Prepare scan() arguments for unroll_scan()."""

    # Check that scan() arguments not supported by unroll_scan() are not set
    def clear_arg(a, v):
        if a in args and args[a] != v:
            raise ValueError('{}={} not supported by unroll_scan()'.format(
                    a, args.get(a)))
        args.pop(a, None)
    for n in ['mode', 'name', 'allow_gc']:
        clear_arg(n, None)
    for f in ['go_backwards', 'profile', 'allow_gc', 'strict']:
        clear_arg(f, False)
    clear_arg('truncate_gradient', -1)

    if not isinstance(args.get('n_steps'), int):
        raise ValueError('unroll_scan requires n_steps (integer)')

    # Other differences
    if isinstance(args.get('outputs_info'), dict):
        raise ValueError('dict outputs_info not supported by unroll_scan()')
    args['initial_values'] = args.pop('outputs_info', None)    # rename
    return args


def _to_list(a):
    """Return a as list for unroll_scan."""
    if isinstance(a, list):
        return a
    elif isinstance(a, tuple):
        return list(a)
    elif a is None:
        return []
    else:
        return [a]
