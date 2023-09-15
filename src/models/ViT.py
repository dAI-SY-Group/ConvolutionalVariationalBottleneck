import math

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from einops import repeat


from src.models.modules import MLPBlock, MaskDropout, VBModule, prepare_VB_config


def build_VisionTransformer(architecture, config):
    assert config is not None
    if architecture == 'VisionTransformer':
        model = VisionTransformer(config.model.transformer_block_mlp, config.data.shape, config.model.embedding_patch_size, config.num_classes, config.model.attention_heads, config.model.hidden_size, config.model.attention_dropout_rate, config.model.embedding_dropout_rate, config.model.transformer_layers, config.model.bias)
    elif architecture == 'VBVisionTransformer':       
        model = VBVisionTransformer(mlp_config=config.model.transformer_block_mlp, data_shape=config.data.shape, patch_size=config.model.embedding_patch_size, num_classes=config.num_classes, num_attention_heads=config.model.attention_heads, hidden_size=config.model.hidden_size, attention_dropout_rate=config.model.attention_dropout_rate, embedding_dropout_rate=config.model.embedding_dropout_rate, num_transformer_layers=config.model.transformer_layers, bias=config.model.bias, VB_config=prepare_VB_config(config.model.VB))
    else:
        raise ValueError(f'The model architecture {architecture} is not implemented yet..') 
    return model


class VisionTransformer(nn.Module): 
    def __init__(self, mlp_config, data_shape=(3,32,32), patch_size=8, num_classes=10, num_attention_heads=8, hidden_size=64, attention_dropout_rate=0.1, embedding_dropout_rate=0, num_transformer_layers=4, bias=True, *args, **kwargs):
        super().__init__()
        bias = True if bias is None else bias
        self.embedding = ImagePatchEmbedding(data_shape, patch_size=patch_size, hidden_size=hidden_size, dropout_rate=embedding_dropout_rate, bias=bias)

        self.transformer_layers = nn.ModuleList()
        self.final_layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)

        for _ in range(num_transformer_layers):
            self.transformer_layers.append(TransformerBlock(mlp_config, num_attention_heads, hidden_size, attention_dropout_rate, bias))

        self.classifier = MLPBlock(hidden_size, num_classes, use_bias=bias)

    def forward(self, x):
        x = self.embedding(x)
        for transformer_block in self.transformer_layers:
            x = transformer_block(x)
        x = self.final_layer_norm(x)
        x = self.classifier(x[:, 0]) #give cls_token to classifier
        return x
    
class VBVisionTransformer(VBModule, VisionTransformer):
    def __init__(self, mlp_config, data_shape=(3,32,32), patch_size=8, num_classes=10, num_attention_heads=8, hidden_size=64, attention_dropout_rate=0.1, embedding_dropout_rate=0, num_transformer_layers=4, bias=True, VB_config=None, *args, **kwargs):
        super().__init__(mlp_config=mlp_config, data_shape=data_shape, patch_size=patch_size, num_classes=num_classes, num_attention_heads=num_attention_heads, hidden_size=hidden_size, attention_dropout_rate=attention_dropout_rate, embedding_dropout_rate=embedding_dropout_rate, num_transformer_layers=num_transformer_layers, bias=bias, **kwargs)
        self.input_VB = VB_config.VB_class(data_shape, VB_config.K, VB_config.beta, VB_config=VB_config) if -1 in VB_config.positions else None
        
        self.hw = int(math.sqrt(hidden_size))
        assert self.hw ** 2 == hidden_size, f'hidden_size must be a square number for VB, but is {hidden_size}'
        VB_shape = (self.embedding.num_patches+1, self.hw, self.hw)
        self.use_token = False

        for i in range(num_transformer_layers):
            if i in VB_config.positions:
                self.VB_list.append(VB_config.VB_class(VB_shape, VB_config.K, VB_config.beta, VB_config=VB_config))
            else:
                self.VB_list.append(None)


    def forward(self, x):
        if self.input_VB is not None:
            x = self.input_VB(x)
        x = self.embedding(x)
        for i, transformer_block in enumerate(self.transformer_layers):
            x = transformer_block(x)
            #if it is the not the last transformer block
            if i < len(self.transformer_layers)-1:
                if self.VB_list[i] is not None:
                    x = x.reshape(*x.shape[:2], self.hw, self.hw)
                    x = self.VB_list[i](x)
                    x = x.reshape(*x.shape[:2], self.hw**2)
            

        x = self.final_layer_norm(x)
        if self.VB_list[i] is not None:
            if self.use_token:
                x = self.VB_list[i](x[:, 0]) #give cls_token through VB
            else:
                x = x.reshape(*x.shape[:2], self.hw, self.hw)
                x = self.VB_list[i](x)
                x = x.reshape(*x.shape[:2], self.hw**2)
                x = x[:, 0]
        else:
            x = x[:, 0]

        x = self.classifier(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, mlp_config, num_attention_heads=8, hidden_size=64, attention_dropout_rate=0, bias=True):
        super().__init__()
        self.hidden_size = hidden_size

        self.attention_norm = nn.LayerNorm(self.hidden_size, eps=1e-6)
        self.attention = Attention(num_attention_heads, hidden_size, attention_dropout_rate, bias)

        self.layer_norm = nn.LayerNorm(self.hidden_size, eps=1e-6)
        self.mlpb1 = MLPBlock(self.hidden_size, mlp_config.width, mlp_config.regularization_layer, mlp_config.activation_function, mlp_config.bias)
        self.mlpb2 = MLPBlock(mlp_config.width, self.hidden_size, mlp_config.regularization_layer, mlp_config.activation_function, mlp_config.bias)
        

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x = self.attention(x)
        x = x + h #skip connection 1

        h = x
        x = self.layer_norm(x)
        x = self.mlpb1(x)
        x = self.mlpb2(x)
        x = x + h #skip connection 2
        return x


class Attention(nn.Module):
    def __init__(self, num_attention_heads=8, hidden_size=64, attention_dropout_rate=0, bias=True):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.attention_dropout_rate = attention_dropout_rate
        self.attention_head_size = int(self.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(self.hidden_size, self.all_head_size, False)
        self.key = nn.Linear(self.hidden_size, self.all_head_size, False)
        self.value = nn.Linear(self.hidden_size, self.all_head_size, False)

        self.out = nn.Linear(self.hidden_size, self.hidden_size, bias=bias)
        self.proj_dropout = MaskDropout(self.attention_dropout_rate)
        self.attn_dropout = MaskDropout(self.attention_dropout_rate)

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output
        

class ImagePatchEmbedding(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, data_shape, patch_size=8, hidden_size=64, dropout_rate=0, bias=True):
        super().__init__()
        in_channels, image_height, image_width = data_shape[0], data_shape[1], data_shape[2]
        patch_height, patch_width = patch_size, patch_size

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        self.num_patches = (image_height // patch_height) * (image_width // patch_width)

        patch_dim = in_channels * patch_height * patch_width

        self.rearrange = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width)
        self.transform = nn.Linear(patch_dim, hidden_size, bias=bias)

        self.cls_token = ClsToken(hidden_size)

        self.position_encoding = PositionalEncoding(self.num_patches, hidden_size)

        if dropout_rate > 0:
            self.dropout = MaskDropout(dropout_rate)

    def forward(self, x):
        x = self.rearrange(x)
        x = self.transform(x)

        cls_tokens = self.cls_token(x)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.position_encoding(x)

        if hasattr(self, 'dropout'):
            x = self.dropout(x)
        return x

class ClsToken(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.token = nn.Parameter(torch.randn(1, 1, hidden_size))

    def forward(self, x):
        b = x.shape[0]
        return repeat(self.token, '() n d -> b n d', b=b)
    
class PositionalEncoding(nn.Module):
    def __init__(self, num_patches, hidden_size):
        super().__init__()
        self.embedding = nn.Parameter(torch.randn(1, num_patches+1, hidden_size))

    def forward(self, x):
        return x + self.embedding[:, :x.shape[1], :]