#!/bin/bash

# Test forward.py, viterbi.py and loss.py by comparing against
# Lample's implementations on test data.

rm -f {correct,theano,tensorflow}-out.txt    # Remove left over

for script in forward viterbi loss; do
    # Run Lample's version for reference
    python lample-tagger/${script}.py 2>&1 > correct-out.txt \
	| egrep -v 'your Theano flags .* specify .*march=X.* flags|It is better to let Theano.* find it automatically'

    # Run with Theano and TF backends and compare
    for be in theano tensorflow; do
	KERAS_BACKEND=$be python ${script}.py 2>&1 > ${be}-out.txt \
	    | egrep -v 'your Theano flags .* specify .*march=X.* flags|It is better to let Theano.* find it automatically|Using (Theano|TensorFlow) backend|setting unroll=True'
	d=$(diff correct-out.txt ${be}-out.txt)
	if [ "$d" = "" ]; then
	    echo "OK: $be $script"
	else
	    echo "ERROR: $script output differs for $be:"
	    echo "$d"
	fi
    done
done

rm -f {correct,theano,tensorflow}-out.txt    # Remove temps
