import numpy as np
import torch.nn as nn
import torch
from einops import rearrange, repeat
import config as cf
from torch import einsum

class ViT_Manual():
    def __init__(self, model, N_SUBSPACE, K_CLUSTER):
        self.n_subspace = N_SUBSPACE
        self.k_cluster = K_CLUSTER
        self.patch_rearrange = model.to_patch_embedding[0]
        self.patch_linear = self.get_param(model.to_patch_embedding[1])
        self.cls_token = model.cls_token
        self.pos_embedding = model.pos_embedding
        #self.layernorm_1 = self.get_param(model.transformer.layers[0][0].norm)
        # layer 1
        self.layernorm_1_attn = model.transformer.layers[0][0].norm
        self.to_qkv_wight_1 = self.get_param(model.transformer.layers[0][0].fn.to_qkv)[0]
        self.to_out_wight_1 = self.get_param(model.transformer.layers[0][0].fn.to_out[0])
        self.layernorm_1_ffn = model.transformer.layers[0][1].norm
        self.ffn_weight_1 = self.get_param(model.transformer.layers[0][1].fn.net)

        # layer2
        self.layernorm_2_attn = model.transformer.layers[1][0].norm
        self.to_qkv_wight_2 = self.get_param(model.transformer.layers[1][0].fn.to_qkv)[0]
        self.to_out_wight_2 = self.get_param(model.transformer.layers[1][0].fn.to_out[0])
        self.layernorm_2_ffn = model.transformer.layers[1][1].norm
        self.ffn_weight_2 = self.get_param(model.transformer.layers[1][1].fn.net)

        self.layernorm3_mlp_head=model.mlp_head[0]
        self.mlp_head_weight = self.get_param(model.mlp_head[1])

        self.softmax = nn.Softmax(dim = -1)
        self.gelu = nn.GELU()
        self.sigmoid = nn.Sigmoid()

        self.amm_estimators = []
        self.amm_est_queue=[]

        self.mm_res = []

    def get_param(self, model_layer, type="parameters"):
        if type == "parameters":
            return [param.detach().numpy() for param in model_layer.parameters()]
        elif type == "buffer":  # extract BN mean and var, remove track_num in buffer
            return [param.detach().numpy() for param in model_layer.buffers() if param.numel() > 1]
        else:
            raise ValueError("Invalid type in model layer to get parameters")

    def vector_matrix_multiply(self, vector, weight_t, bias=0, type='exact'):
        if type == 'exact':
            res = np.dot(vector, weight_t) + bias
            self.mm_res.append(res)
        return res

    def matrix_matrix_multiply(self, mat_1, mat_2, type='exact'):
        if type == 'exact':
            res = np.matmul(mat_1, mat_2)
            self.mm_res.append(res)
        return res


    def norm_attention_exact(self,x_input,norm_layer, to_qkv_wight, to_out_wight):
        b, n, d, h = *x_input.shape, cf.heads
        scale = d**(-0.5)
        if not torch.is_tensor(x_input):
            x_input = torch.tensor(x_input)
        x_norm = norm_layer(x_input)
        x_vectors = x_norm.reshape(-1, 16).detach().numpy()
        # MM: linear of 3 inputs,np.dot()
        qkv = self.vector_matrix_multiply(x_vectors, to_qkv_wight.transpose()).reshape(b, n, -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=cf.heads), torch.tensor(qkv).chunk(3, dim=-1))  # [647, 2, 11, 16] each

        # MM1: q*k, use np.matmul(), different from np.dot,both matrix has batch dimension
        q_reshaped, k_reshaped = q.detach().numpy().reshape(-1, n, d), k.detach().numpy().reshape(-1, n, d)
        dots = self.matrix_matrix_multiply(q_reshaped, np.transpose(k_reshaped, axes=(0, 2, 1))).reshape(-1, h, n, n)*scale
        attn = self.softmax(torch.tensor(dots))

        # MM2: qk * v, matmul()
        attn_reshaped, v_reshaped = attn.detach().numpy().reshape(-1, n, n), v.detach().numpy().reshape(-1, n, d)
        out = self.matrix_matrix_multiply(attn_reshaped, v_reshaped).reshape(-1, h, n, d)
        out = rearrange(out, 'b h n d -> b n (h d)') # ([1294, 11, 16])*([1294, 16, 11]), (b*h,n,d)

        # MM3: output_linear, np.dot()
        out = self.vector_matrix_multiply(out, to_out_wight[0].transpose()) + to_out_wight[1]
        attn_out = out+x_input.detach().numpy()
        return attn_out

    def norm_ffn_exact(self,x_input,norm_layer, ffn_weights, act_fn=nn.GELU()):
        x = norm_layer(torch.tensor(x_input)).detach().numpy()
        #MM1
        x=self.vector_matrix_multiply(x, ffn_weights[0].transpose())+ffn_weights[1]
        x=act_fn(torch.tensor(x)).detach().numpy()
        #MM2
        x = self.vector_matrix_multiply(x, ffn_weights[2].transpose())+ ffn_weights[3]
        ffn_out = x+x_input
        return ffn_out

    def norm_mlp_head_exact(self,x_input, norm_layer, mlp_weights):
        if not torch.is_tensor(x_input):
            x_input = torch.tensor(x_input)
        x = norm_layer(x_input).detach().numpy()
        x= self.vector_matrix_multiply(x, mlp_weights[0].transpose()) + mlp_weights[1]
        return x


    def forward_exact(self,input_data):
        x = self.patch_rearrange(input_data)

        # MM1: to_patch input linear
        x = self.vector_matrix_multiply(x, self.patch_linear[0].transpose()) + self.patch_linear[1]

        x=torch.tensor(x)
        b, n, d = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)  # torch.Size([647, 11, 16])

        x_attn_input = x + self.pos_embedding[:, :(n + 1)]  # torch.Size([647, 11, 16])

        attn_out_1 = self.norm_attention_exact(x_attn_input,self.layernorm_1_attn, self.to_qkv_wight_1, self.to_out_wight_1)
        ffn_out_1 = self.norm_ffn_exact(attn_out_1, self.layernorm_1_ffn, self.ffn_weight_1)

        attn_out_2 = self.norm_attention_exact(ffn_out_1,self.layernorm_2_attn, self.to_qkv_wight_2, self.to_out_wight_2)
        ffn_out_2 = self.norm_ffn_exact(attn_out_2, self.layernorm_2_ffn, self.ffn_weight_2)

        x = ffn_out_2[:, 0]

        #x = self.mlp_head(torch.tensor(x))

        x = self.norm_mlp_head_exact(x,self.layernorm3_mlp_head,self.mlp_head_weight)

        output = self.sigmoid(torch.tensor(x))
        layer_res = [x_attn_input, attn_out_1, ffn_out_1, attn_out_2, ffn_out_2,output]
        mm_res = self.mm_res.copy()
        self.mm_res.clear()
        return layer_res, mm_res


    def layer_norm_manual(self,x,weight,bias):
        # layernorm is element-wise operation, not dot product(MM)
        # don't exactly match pytorch and cannot optimize by table, so just use pytorch
        mean = np.mean(x, axis=(0, 2), keepdims=True)
        variance = np.var(x, axis=(0, 2), keepdims=True)
        x_normalized = (x - mean) / np.sqrt(variance + 1e-9)
        # Apply weight and bias
        return x_normalized * weight + bias
