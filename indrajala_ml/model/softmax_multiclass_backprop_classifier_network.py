from __future__ import annotations

from indrajala_ml.model.multiclass_backprop_classifier_network import MultiClassBackpropClassifierNetwork
from indrajala_ml.model.softmax_output_layer import SoftmaxOutputLayer


class SoftmaxMultiClassBackpropClassifierNetwork(MultiClassBackpropClassifierNetwork):
    """
    A softmax + cross-entropy sibling of MultiClassBackpropClassifierNetwork, aligned with the
    canonical multi-class backprop treatment (e.g. Nielsen's "Neural Networks and Deep
    Learning") rather than that class's one-vs-rest approach (independent per-node sigmoid +
    quadratic loss against a one-hot target - see its own docstring). Two differences that
    actually matter for a mutually-exclusive classification target like digit recognition:

    - predict_probabilities() here genuinely sums to 1.0 (a real probability distribution over
      class_count mutually exclusive outcomes), unlike the one-vs-rest sibling's independent
      sigmoids, which don't sum to anything in particular.
    - cross-entropy loss doesn't suffer quadratic loss's "learning slowdown" pathology, where a
      confidently-wrong saturated sigmoid neuron (activation near 0 or 1, but on the wrong side)
      produces a tiny gradient exactly when the error is largest - softmax+cross-entropy's delta
      (activation - target) stays proportional to the actual error regardless of saturation.

    Structurally, this is the *entire* difference: swapping in SoftmaxOutputLayer for the output
    layer (see BackpropNetworkBase.output_layer_cls) is enough, because softmax's cross-node
    coupling only affects the forward pass - once activations are computed, the softmax +
    cross-entropy delta (see SoftmaxOutputNode.compute_output_delta) is exactly as
    per-node-independent as the one-vs-rest sibling's own delta. Every other method
    (learn/_backward/randomize/randomized/snapshot/restore/save/load) is inherited completely
    unchanged from MultiClassBackpropClassifierNetwork and stays correct as-is.

    Deliberately not a replacement for MultiClassBackpropClassifierNetwork - that class remains
    in active use (existing demos, existing saved models) and is itself a legitimate, simpler
    reference point to contrast this one against, the same way this codebase keeps
    EnsembleBackpropClassifierNetwork's differently-motivated one-vs-rest design alongside both.
    """

    output_layer_cls = SoftmaxOutputLayer
