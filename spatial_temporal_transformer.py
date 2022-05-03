import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.checkpoint import checkpoint
import matplotlib.pyplot as plt
import pdb
import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "4"


def get_sinusoid_encoding_table(n_position, d_hid):
  # 计算sin-cos位置编码 ST-transformer使用的是（1，n_position, 1, d_hid）的编码
  # TODO 也就是说所有的关节位置编码是相同的，不做区分,可以考虑在这里使用可学习空间编码
  # d_hid表示位置编码向量的长度
  # n_position表示序列的最大长度

  def get_position_angle_vec(position):
    # 计算了三角函数的w角速度，给定时间步的每一个分量包含一个值, //表示除法向下取整
    return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

  sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
  sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
  sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
  # 增加维度从(N,d)变为(1,N,1,d)
  return torch.FloatTensor(sinusoid_table).unsqueeze(0).unsqueeze(2)


class TemporalEncoding(nn.Module):
  # 时间编码，希望可以指示序列内不同的时刻（1，n_position, 1, d_hid）
  def __init__(self, n_position, d_model):
    super(TemporalEncoding, self).__init__()
    self.n_position = n_position
    self.temporal_embed = nn.Embedding(n_position, d_model)
    self.norm = nn.BatchNorm1d(d_model, affine=False, track_running_stats=False)

  def forward(self):
    x = torch.arange(self.n_position).long().cuda()
    # Embedding自带转换one-hot，因此这里不用显式调用F.one_hot了
    temporal_encoding = self.temporal_embed(x)
    temporal_encoding = self.norm(temporal_encoding)
    return temporal_encoding.unsqueeze(0).unsqueeze(2)


def drop_path(x, drop_prob=None, training=False):
  if drop_prob == 0. or training is False:
    return x
  keep_prob = 1 - drop_prob
  shape = (x.shape[0],) + (1,) * (x.ndim - 1)
  # 维度是(N,1,1) 相当于有的样本可能会被删除，也就是说一个batch内不同的样本走的路径不同哦
  random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
  random_tensor.div_(keep_prob)
  return x * random_tensor


class DropPath(nn.Module):
  # 正则化模块DropPath，forward时样本会以drop_prob概率跳过当前层
  def __init__(self, drop_prob=0.):
    super(DropPath, self).__init__()
    self.drop_prob = drop_prob

  def forward(self, x, training):
    return drop_path(x, self.drop_prob, training)


class FFN(nn.Module):
  def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.):
    # in_features：输入维度
    # hidden_features：隐藏层维度，默认为None，和输入宽度相同，原论文中给出先升维后恢复
    # out_features：输出维度，默认为None，即输出和输入宽度相同
    # act_layer：激活函数默认RELU
    # drop：dropout概率
    super().__init__()
    out_features = out_features or in_features
    hidden_features = hidden_features or in_features
    self.fc = nn.Sequential(
      nn.Linear(in_features, hidden_features),
      act_layer(),
      nn.Linear(hidden_features, out_features),
      nn.Dropout(drop)
      )

  def forward(self, x):
    x = self.fc(x)
    return x


class Attention(nn.Module):
  def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
      proj_drop=0.):
    # dim：attention的宽度，也就是input和output都是这个宽度
    # num_heads：head的数量
    # qkv_bias：qv是否引入偏置项
    # qk_scale：qk点乘后是否除以根号下dim进行缩放
    # attn_drop：attention的softmax后进行dropout的比例
    # proj_drop：attention后的线性层的dropout比例
    super().__init__()
    assert dim % num_heads == 0, 'dim should be divisible by num_heads'
    self.num_heads = num_heads
    # 每个head的宽度原来是用attention的宽度除以head的数量吗
    head_dim = dim // num_heads
    # 进行缩放的系数可以提前指定或者用单个head的维度进行计算(若qk_scale为None)
    self.scale = qk_scale or head_dim ** -0.5
    # 将input通过线性映射到QKV(三者形状相同)和多头使用一个矩阵解决
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    if qkv_bias:
      self.q_bias = nn.Parameter(torch.zeros(dim))
      self.v_bias = nn.Parameter(torch.zeros(dim))
    else:
      self.q_bias = None
      self.v_bias = None

    # Dropout用在哪里的呢？——softmax得分之后
    self.attn_drop = nn.Dropout(attn_drop)
    # 定义concat后输出的线性映射和该层的dropout
    self.proj = nn.Linear(dim, dim)
    self.proj_drop = nn.Dropout(proj_drop)


  def forward(self, x, mask=None):
    # batch_num, seq_len, #joints, #embeddings
    # 反正就是后两维做attn，不一定是上面的这个顺序
    B, T, N, C = x.shape
    qkv_bias = None
    # 若q_bias不是None，将qkv对应的bias拼在一起，其中k的是不会梯度更新的0，v也是之前定义的偏置，这样可以让k没有bias
    if self.q_bias is not None:
      qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
    # 定义一个线性映射qkv，返回值便是qkv拼在一起的矩阵
    qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    # 其实如果都没有bias或者都有bias的话直接用下面的就可以了，不用自己指定
    # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    # 将拼起来的qkv调整形状，-1指代的就是每个head的宽度，然后调整维度的顺序至(3,B,T,#head,N,head_dim)
    qkv = qkv.reshape(B, T, N, 3, self.num_heads, -1).permute(3, 0, 1, 4, 2, 5)
    q, k, v = qkv[0], qkv[1], qkv[2]
    # 引入缩放项
    q = q * self.scale
    # 将k的后两维交换，返回(B,#head,N,N),这个乘法消去了head_dim,相当于C->#head*head_dim后
    # 每个head内N个head_dim向量计算彼此间的相关性程度，然后剩下N,N的矩阵
    # 猜测@就是matmu?保证前面高维不变，只对最后两维进行矩阵乘法
    attn = (q @ k.transpose(-2, -1))
    # 这里可以添加mask
    if mask is not None:
      attn = attn + mask.expand(B, T, self.num_heads, -1, -1).to(attn.device)
    # 利用softmax将注意力集中在相关性高的向量上
    attn = attn.softmax(dim=-1)
    # 两两之间的得分有可能会被drop掉
    attn = self.attn_drop(attn)
    # 做完矩阵乘法返回维度是(B,T,#head,N,head_dim)，将中间两维调换位置再把后两维拉平(相当于整合多个head)，得到C的新表示
    x = (attn @ v).transpose(2, 3).reshape(B, T, N, -1)
    # 通过线性映射将#head*head_dim维变化为原始的C维
    x = self.proj(x)
    # 最后每个时间步得到的embedding依然会drop掉一部分
    x = self.proj_drop(x)

    return x


class Temporal_Attention(nn.Module):
  def __init__(self, dim, dot_attn=True, last_append=False):
    super().__init__()
    self.dot_attn = dot_attn
    self.last_append = last_append
    if dot_attn:
      self.q = nn.Linear(dim, dim, bias=False)
      self.k = nn.Linear(dim, dim, bias=False)
    else:
      self.tanh = nn.Tanh()
      self.linear = nn.Linear(dim, dim, bias=True)
      self.u = nn.Parameter(torch.ones(dim))

  def forward(self, x):
    # x shape: (B, T, N, C)
    B, T, N, C = x.shape
    x_last = x[:, -1, :, :].clone()
    if self.dot_attn is True:
      q = self.q(x_last.view(B, -1)).unsqueeze(1)
      k = self.k(x.view(B, T, -1))
      attn = (q @ k.transpose(-2, -1))
      attn = attn.softmax(dim=-1)
      x = (attn @ k).view(B, 1, N, C)
    else:
      tmp_x = x.view(B, T, -1).clone()
      u = self.u.to(tmp_x.device)
      attn = self.tanh(self.linear()) @ u
      attn = attn.softmax(dim=-1)
      x = (attn.unsqueeze(1) @ tmp_x).view(B, 1, N, C)
    if self.last_append:
      x = torch.cat((x, x_last), dim=1)

    return x


class Block(nn.Module):
  def __init__(self, dim, num_heads, ffn_ratio=1., qkv_bias=False, qk_scale=None,
    drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
    norm_layer=nn.LayerNorm):
    # dim：attention的宽度，也就是input和output都是这个宽度
    # num_heads：head的数量
    # ffn_ratio：FFN中隐层的维度扩张倍数
    # qkv_bias：qv是否引入偏置项
    # qk_scale：qk点乘后是否除以根号下dim进行缩放
    # drop：attention中最后一个线性层和FFN最后一个线性层的dropout概率，目前二者相同
    # attn_drop：attention的softmax后进行dropout的比例
    # act_layer：激活函数的选择，默认GELU
    # norm_layer：默认LayerNorm
    super().__init__()
    self.dim = dim
    # TODO 这里layernorm的方法选择也应该注意一下，给一个参数是最后一维内做norm，也就是每个时间步独立
    self.norm1 = norm_layer(dim)
    # 对attention进行初始化
    self.attn = Attention(
      dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
      attn_drop=attn_drop, proj_drop=drop)

    # temporal-attention
    self.temp_attn = Attention(
      dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
      attn_drop=attn_drop, proj_drop=drop)

    # TODO 非attention前的LN选择加上所有时间步一起标准化
    self.norm2 = norm_layer(dim)

    # 确定FFN隐层维度并初始化
    ffn_hidden_dim = int(dim * ffn_ratio)
    self.ffn = FFN(in_features=dim, hidden_features=ffn_hidden_dim, act_layer=act_layer, drop=drop)

    # DropPath
    self.drop_path = DropPath(drop_path)

  def forward(self, x, mask=None):
    x = x + self.drop_path(0.5 * self.attn(self.norm1(x)) + 0.5 * 
      self.temp_attn(self.norm1(x).transpose(1,2), mask).transpose(1,2), self.training)
    # x = x + self.drop_path(self.attn(self.norm1(x)), self.training)
    x = x + self.drop_path(self.ffn(self.norm2(x)), self.training)
    return x


class SpatialTemporalTransformer(nn.Module):
  def __init__(self,
         input_dim=None,
         output_dim=None,						# encoder的输出长度，应和target维度相同，但也有可能单独作为encoder使用,此时为None
         n_position=480,						# 序列长度
         n_prediction=24, 
         n_joint=12,							# 关节数量
         embed_dim=16, 							# transformer_block宽度
         depth=4,								# transformer_block数量
         num_heads=4, 							# attention的head数量
         ffn_ratio=1.,							# FFN层的隐层宽度扩张倍数
         qkv_bias=False, 						# qkv是否引入偏置项
         qk_scale=None, 						# qk点乘后是否除以根号下dim进行缩放
         drop_rate=0., 							# 虽然预测的时候不需要，但是考虑到有可能fine-tune
         attn_drop_rate=0.,						# 保留这两个参数，预测时可以用model.eval关闭
         drop_path_rate=0.,						# stochastic depth
         norm_layer=nn.LayerNorm, 				# norm的方法，输出和输入维度相同
         use_learnable_pos_emb=False,			# 是否使用可学习的位置编码
         use_temporal_pos_emb=False,			# 是否时间编码
         temp_attn=False,						# 是否使用时间attention
         last_append=False):					# 是否拼接最后一个时间步的隐藏层
    super().__init__()

    self.output_dim = output_dim
    self.time_step = n_position + n_prediction
    # 卷积embedding
    #self.conv_embed = nn.Sequential(
    #  nn.Conv1d(in_channels=input_dim, out_channels=embed_dim, kernel_size=3, padding=1),
    #  nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=3, padding=1)
    #  )
    # 线性embedding
    self.linear_embed = nn.Linear(input_dim, embed_dim, bias=True)

    # 标准sin-cos编码
    self.pos_embed = nn.Parameter(get_sinusoid_encoding_table(self.time_step, embed_dim), requires_grad=False)
    # 是否使用额外的时间编码
    if use_temporal_pos_emb:
      self.tem_embed = TemporalEncoding(self.time_step, embed_dim)
    else:
      self.tem_embed = None
    # 是否使用可以学习的位置编码
    if use_learnable_pos_emb:
      self.spa_embed = nn.Parameter(torch.zeros(1, 1, n_joint, embed_dim))
    else:
      self.spa_embed = None

    # 定义MSA+FFN的block
    self.blocks = nn.ModuleList([
      Block(
        dim=embed_dim, num_heads=num_heads, ffn_ratio=ffn_ratio, qkv_bias=qkv_bias,
        qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate,
        norm_layer=norm_layer)
      for i in range(depth)])

    self.mask = self._generate_square_subsequent_mask(n_prediction, n_position)

    # 定义LN层和输出层(若输出维度和transformer decoder宽度相同那么输出层就是一个恒等变换)
    self.norm = norm_layer(embed_dim)
    if temp_attn:
      self.head_attention = Temporal_Attention(n_joint*embed_dim, dot_attn=True, last_append=last_append)
      if last_append:
        self.head_linear = nn.Linear(2, 1, bias=False)
      else:
        self.head_linear = nn.Identity()
    else:
      self.head_attention = None
      # self.head_linear = nn.Linear(n_position, 1, bias=False)
    self.head_norm = nn.LayerNorm(embed_dim)
    self.head_out = nn.Linear(embed_dim, output_dim)

    # 参数初始化
    self.apply(self.init_weights)


  def _generate_square_subsequent_mask(self, sz, left_pad=0):
    if left_pad>0:
      mask = (torch.triu(torch.ones(left_pad+sz, left_pad+sz)) == 1).transpose(0, 1)
      mask[:, :left_pad] = 1
    else:
      mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    
    mask = (
      mask.float()
      .masked_fill(mask == 0, float("-inf"))
      .masked_fill(mask == 1, float(0.0))
    )
    return mask


  def init_weights(self, m):
    # 参数初始化
    if isinstance(m, nn.Linear):
      nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
      if isinstance(m, nn.Linear) and m.bias is not None:
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv1d):
      nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
    elif isinstance(m, nn.LayerNorm):
      pass


  def forward(self, x, y, max_len=24, teacher_forcing_ratio=0.0):
    # x:(B, n_position, n_joint*input_dim)
    x = x.to(torch.float32)
    y = y.to(torch.float32)
    B, T, C = x.shape
    B, t, C = y.shape
    x = x.reshape(B, T, -1, 3)
    y = y.reshape(B, t, -1, 3)
    B, T, N, K = x.shape
    out = torch.zeros((B, max_len, C), device=x.device)
    '''
    for i in range(max_len):
      if i > 0:
        if self.training:
          x = torch.cat((x_ori[:, i:, :, :], y[:, :i, :, :]), dim=1)
        else:
          x = torch.cat((x_ori[:, i:, :, :], out[:, :i, :].view(B, i, -1, 3)), dim=1)
      x = self.linear_embed(x)
      # x:(B, n_position, n_joint, embed_dim)

      # 通过expand函数扩展维度
      x = x + self.pos_embed.expand(B, -1, N, -1).type_as(x).to(x.device).clone()
      
      if self.tem_embed is not None:
        x = x + self.tem_embed().expand(B, -1, N, -1).type_as(x).to(x.device).clone()
      
      if self.spa_embed is not None:
        x = x + self.spa_embed.expand(B, T, -1, -1).type_as(x).to(x.device).clone()
        
      # 通过transformer blocks
      for blk in self.blocks:
        if self.training:
          x = checkpoint(blk, x, self.mask.type_as(x).to(x.device))
        else:
          x = blk(x, self.mask.type_as(x).to(x.device))
      # 从Encoder出来先经过一次LayerNorm
      x = self.norm(x)
      if self.head_attention is not None:
        x = self.head_attention(x)
      x = self.head_linear(x.transpose(1,3)).transpose(1,3)
      x = self.head_out(self.head_norm(x))
      out[:, i, :] = x.view(B, 1, -1).squeeze(1)
    '''
    if self.training:
      # x = torch.cat((x, y[:, :-1, :, :]), dim=1)
      x = torch.cat((x, y), dim=1)
      x = self.linear_embed(x)
      # x = self.conv_embed(x.permute(0,2,3,1).reshape(-1,K,T+t)).reshape(B,N,-1,T+t).permute(0,3,1,2)
      # x = checkpoint(self.linear_embed, x)
      x = x + self.pos_embed.expand(B, -1, N, -1).type_as(x).to(x.device).clone()
      if self.tem_embed is not None:
        x = x + self.tem_embed().expand(B, -1, N, -1).type_as(x).to(x.device).clone()
      if self.spa_embed is not None:
        x = x + self.spa_embed.expand(B, T+t, -1, -1).type_as(x).to(x.device).clone()
      for blk in self.blocks:
        x = checkpoint(blk, x, self.mask.type_as(x).to(x.device))
      x = self.norm(x)
      if self.head_attention is not None:
        x = self.head_attention(x)
      # x = self.head_out(self.head_norm(x))
      x = checkpoint(self.head_out, self.head_norm(x))
      out = x[:, -t-1:-1, :, :].reshape(B, t, -1)
    else:
      x_ori = x.clone()
      x_pad = torch.zeros(B, max_len, N, K).type_as(x).to(x.device)
      #if y.shape[1] == 1:
      #  x_pad[:, 0, :, :] = y.squeeze(1)
      for i in range(max_len):
        x = torch.cat((x_ori, x_pad), dim=1)
        x = self.linear_embed(x)
        # x = self.conv_embed(x.permute(0,2,3,1).reshape(-1,K,T+max_len)).reshape(B,N,-1,T+max_len).permute(0,3,1,2)
        x = x + self.pos_embed.expand(B, -1, N, -1).type_as(x).to(x.device).clone()
        if self.tem_embed is not None:
          x = x + self.tem_embed().expand(B, -1, N, -1).type_as(x).to(x.device).clone()
        if self.spa_embed is not None:
          x = x + self.spa_embed.expand(B, T+max_len, -1, -1).type_as(x).to(x.device).clone()
        for blk in self.blocks:
          x = blk(x, self.mask.type_as(x).to(x.device))
        #if x.shape[1] != 143:
        #  self.norm = nn.LayerNorm([x.shape[1], x.shape[2], x.shape[3]], elementwise_affine=False)
        x = self.norm(x)
        if self.head_attention is not None:
          x = self.head_attention(x)
        x = self.head_out(self.head_norm(x))
        x_pad[:, i, :, :] = x[:, T+i-1, :, :]
      out = x_pad.reshape(B, max_len, -1)
    # output shape: (B, t, C)
    return out


if __name__ == '__main__':

  model = SpatialTemporalTransformer(
    input_dim=3,
    output_dim=3,
    n_position=120,
    n_joint=24,
    embed_dim=4,
    depth=2,
    num_heads=2,
    ffn_ratio=1,
    qkv_bias=False,
    qk_scale=None,
    drop_rate=0.,
    attn_drop_rate=0.,
    drop_path_rate=0.,
    norm_layer=nn.LayerNorm,
    use_learnable_pos_emb=False,
    use_temporal_pos_emb=False,
    temp_attn=False,
    last_append=False)

  # x: shape (N, seq_len, n_joint, n_feature)
  x = torch.rand(32, 120, 72)
  xx = torch.rand(32, 4, 72)
  # y: shape (N, 1, n_joint, n_feature)
  y = model(x, xx, 4)
  loss = nn.functional.mse_loss(xx, y)
  loss.backward()
  print(y.shape, loss)
