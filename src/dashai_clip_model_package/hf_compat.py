def extract_pooled_embedding(output):
    """Unwrap the projected embedding tensor from a CLIP-family feature call.

    Transformers >=5 returns a ``BaseModelOutputWithPooling`` whose
    ``pooler_output`` holds the projected embedding; older releases (and the
    test fakes) return the tensor directly. Shared by every CLIP-family
    zero-shot classifier in this package (CLIP, SigLIP, ...).
    """
    return output.pooler_output if hasattr(output, "pooler_output") else output
