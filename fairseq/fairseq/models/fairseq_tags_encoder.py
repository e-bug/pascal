"""
Created by:         Emanuele Bugliarello (@e-bug)
Date created:       9/4/2019
Date last modified: 9/4/2019
"""

import torch.nn as nn


class FairseqTagsEncoder(nn.Module):
    """Base class for encoders."""

    def __init__(self, dictionary, tags_dictionary):
        super().__init__()
        self.dictionary = dictionary
        self.tags_dictionary = tags_dictionary

    def forward(self, src_tokens, src_lengths, src_tags):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (LongTensor): lengths of each source sentence of shape
                `(batch)`
            src_tags (LongTensor): tags in the source language of shape
                `(batch, src_len)`
        """
        raise NotImplementedError

    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to `new_order`.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            `encoder_out` rearranged according to `new_order`
        """
        raise NotImplementedError

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return 1e6  # an arbitrary large number

    def upgrade_state_dict(self, state_dict):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        return state_dict
