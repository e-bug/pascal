"""
Created by:         Emanuele Bugliarello (@e-bug)
Date created:       9/4/2019
Date last modified: 9/4/2019
"""

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import options
from fairseq import utils

from fairseq.modules import (
    AdaptiveInput, AdaptiveSoftmax, CharacterTokenEmbedder, LearnedPositionalEmbedding, MultiheadPascal,
    SinusoidalPositionalEmbedding
)

from . import (
    FairseqIncrementalDecoder, FairseqTagsEncoder, FairseqLanguageModel, FairseqTagsModel, register_model,
    register_model_architecture,
)

from .transformer import (
    TransformerDecoder, TransformerEncoderLayer, Embedding, LayerNorm, Linear, PositionalEmbedding, 
    base_architecture, transformer_iwslt_de_en, transformer_wmt_en_de, 
    transformer_vaswani_wmt_en_de_big, transformer_vaswani_wmt_en_fr_big, transformer_wmt_en_de_big, transformer_wmt_en_de_big_t2t
)


@register_model('pascal_transformer')
class PascalTransformerModel(FairseqTagsModel):
    """
    Transformer+Pascal model from `Bugliarello and Okazaki (2019)
    <https://arxiv.org/abs/1909.03149>`.

    Args:
        encoder (TransformerTagsEncoder): the encoder
        decoder (TransformerDecoder): the decoder

    The Transformer model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.transformer_parser
        :prog:
    """

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--relu-dropout', type=float, metavar='D',
                            help='dropout probability after ReLU in FFN')
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--encoder-pascal-heads', default=None, type=int, nargs='+',
                            help='list of parent-scaled self-attention heads per layer')
        parser.add_argument('--pascal-weight-fn', default='normal', type=str, choices=['normal', 'uniform'],
                            help='weight function for parent tokens in Pascal')
        parser.add_argument('--pascal-weight-param', default=1, type=float,
                            help='parameter of Pascal weight function (std or window size)')
        parser.add_argument('--parent-ignoring', default=0.0, type=float,
                            help='dropout probability for dependencies in Pascal heads')
        parser.add_argument('--encoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the encoder')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads')
        parser.add_argument('--decoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the decoder')
        parser.add_argument('--decoder-normalize-before', action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--no-token-positional-embeddings', default=False, action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--share-all-embeddings', action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion'),
        parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
                            help='sets adaptive softmax dropout for the tail projections')
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = 1024
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = 1024

        src_dict, tgt_dict, src_tags_dict = task.source_dictionary, task.target_dictionary, task.source_tags_dictionary

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError('--share-all-embeddings requires a joined dictionary')
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    '--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim')
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path):
                raise ValueError('--share-all-embeddings not compatible with --decoder-embed-path')
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = build_embedding(
                tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )

        encoder = PascalTransformerEncoder(args, src_dict, encoder_embed_tokens, src_tags_dict)
        decoder = TransformerDecoder(args, tgt_dict, decoder_embed_tokens)
        return PascalTransformerModel(encoder, decoder)


class PascalTransformerEncoder(FairseqTagsEncoder):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
        left_pad (bool, optional): whether the input is left-padded. Default:
            ``True``
    """
    
    def __init__(self, args, dictionary, embed_tokens, tags_dictionary, left_pad=True):
        super().__init__(dictionary, tags_dictionary)
        self.dropout = args.dropout

        self.map_dictionary = dict()
        for k, v in tags_dictionary.indices.items():
            try:
                self.map_dictionary[v] = float(k)
            except:
                pass

        if args.encoder_pascal_heads is None:
            self.encoder_pascal_heads = [0] * args.encoder_layers
        else:
            self.encoder_pascal_heads = args.encoder_pascal_heads
        assert len(self.encoder_pascal_heads) == args.encoder_layers

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.max_source_positions

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = PositionalEmbedding(
            args.max_source_positions, embed_dim, self.padding_idx,
            left_pad=left_pad,
            learned=args.encoder_learned_pos,
        ) if not args.no_token_positional_embeddings else None

        self.layers = nn.ModuleList([])
        self.layers.extend([
            PascalTransformerEncoderLayer(args, h) if h != 0 else TransformerEncoderLayer(args)
            for h in self.encoder_pascal_heads
        ])
        self.register_buffer('version', torch.Tensor([2]))
        self.normalize = args.encoder_normalize_before
        if self.normalize:
            self.layer_norm = LayerNorm(embed_dim)

    def forward(self, src_tokens, src_lengths, src_tags):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            src_tags (LongTensor): tags in the source language of shape
                `(batch, src_len)`

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
        """
        # embed tokens and positions and tags
        x = self.embed_scale * self.embed_tokens(src_tokens)
        if self.embed_positions is not None:
            x += self.embed_positions(src_tokens)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        # retrieve parent positions from embedding indices, else eos
        maxlen = src_tags.size(1) - 1
        parents = torch.cuda.FloatTensor(np.vectorize(lambda e: self.map_dictionary.get(e, maxlen))(src_tags))

        # encoder layers
        for i, h in enumerate(self.encoder_pascal_heads):
            if h != 0:
                x = self.layers[i](x, encoder_padding_mask, parents)
            else:
                x = self.layers[i](x, encoder_padding_mask)

        if self.normalize:
            x = self.layer_norm(x)

        return {
            'encoder_out': x,  # T x B x C
            'encoder_padding_mask': encoder_padding_mask,  # B x T
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if encoder_out['encoder_out'] is not None:
            encoder_out['encoder_out'] = \
                encoder_out['encoder_out'].index_select(1, new_order)
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = \
                encoder_out['encoder_padding_mask'].index_select(0, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions())

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = '{}.embed_positions.weights'.format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict['{}.embed_positions._float_tensor'.format(name)] = torch.FloatTensor(1)
        version_key = '{}.version'.format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])
        return state_dict


class PascalTransformerEncoderLayer(nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args, num_pascal_heads):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        self.num_pascal_heads = num_pascal_heads
        self.self_attn = MultiheadPascal(
            self.embed_dim, args.encoder_attention_heads,
            dropout=args.attention_dropout, parent_ignoring=args.parent_ignoring,
            num_pascal_heads=self.num_pascal_heads,
            weight_fn=args.pascal_weight_fn, weight_param=args.pascal_weight_param,
        )
        self.dropout = args.dropout
        self.relu_dropout = args.relu_dropout
        self.normalize_before = args.encoder_normalize_before
        self.fc1 = Linear(self.embed_dim, args.encoder_ffn_embed_dim)
        self.fc2 = Linear(args.encoder_ffn_embed_dim, self.embed_dim)
        self.layer_norms = nn.ModuleList([LayerNorm(self.embed_dim) for i in range(2)])

    def forward(self, x, encoder_padding_mask, parents):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.

        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        residual = x
        x = self.maybe_layer_norm(0, x, before=True)
        x, _ = self.self_attn(query=x, key=x, value=x, parents=parents, key_padding_mask=encoder_padding_mask)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(0, x, after=True)

        residual = x
        x = self.maybe_layer_norm(1, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(1, x, after=True)
        return x

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x


@register_model_architecture('pascal_transformer', 'pascal_transformer')
def pascal_base_architecture(args):
    base_architecture(args)


@register_model_architecture('pascal_transformer', 'pascal_transformer_iwslt_de_en')
def pascal_transformer_iwslt_de_en(args):
    transformer_iwslt_de_en(args)


@register_model_architecture('pascal_transformer', 'pascal_transformer_wmt_en_de')
def pascal_transformer_wmt_en_de(args):
    transformer_wmt_en_de(args)


# parameters used in the "Attention Is All You Need" paper (Vaswani, et al, 2017)
@register_model_architecture('pascal_transformer', 'pascal_transformer_vaswani_wmt_en_de_big')
def pascal_transformer_vaswani_wmt_en_de_big(args):
    transformer_vaswani_wmt_en_de_big(args)


@register_model_architecture('pascal_transformer', 'pascal_transformer_vaswani_wmt_en_fr_big')
def pascal_transformer_vaswani_wmt_en_fr_big(args):
    transformer_vaswani_wmt_en_fr_big(args)


@register_model_architecture('pascal_transformer', 'pascal_transformer_wmt_en_de_big')
def pascal_transformer_wmt_en_de_big(args):
    transformer_wmt_en_de_big(args)


# default parameters used in tensor2tensor implementation
@register_model_architecture('pascal_transformer', 'pascal_transformer_wmt_en_de_big_t2t')
def pascal_transformer_wmt_en_de_big_t2t(args):
    transformer_wmt_en_de_big_t2t(args)

