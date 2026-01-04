import torch
import triton 

import triton.language as tl 
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
import math 

def ground_truth(q, k, v):
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        o = F.scaled_dot_product_attention(q, k, v)
    return o.transpose(1, 2)


@triton.jit
def _attn_fwd(
        Q, 
        K,
        V,
        softmax_scale,
        L, 
        O,
        stride_q_batch_size,
        stride_q_seq_len,
        stride_q_num_heads,
        stride_q_dim,
        stride_k_batch_size,
        stride_k_seq_len,
        stride_k_num_heads,
        stride_k_dim,
        stride_v_batch_size,
        stride_v_seq_len,
        stride_v_num_heads,
        stride_v_dim,
        stride_l_batch_size,
        stride_l_seq_len,
        stride_l_num_heads,
        BATCH_SIZE,
        SEQ_LEN: tl.constexpr,
        NUM_HEADS: tl.constexpr,
        DIM: tl.constexpr,                  
        BLOCK_SIZE_Q: tl.constexpr,
        BLOCK_SIZE_KV: tl.constexpr,
    ):
        q_index = tl.program_id(0)
        batch_index = tl.program_id(1) // NUM_HEADS 
        head_index =  tl.program_id(1) % NUM_HEADS 

        qkv_offset = batch_index * stride_q_batch_size + head_index * stride_q_num_heads
        l_offset = batch_index * stride_l_batch_size + head_index * stride_l_num_heads

        q_block_ptr = tl.make_block_ptr(
            base = Q + qkv_offset,
            shape = (SEQ_LEN, DIM),
            strides = (stride_q_seq_len, stride_q_dim),
            offsets = (q_index * BLOCK_SIZE_Q, 0), # upper left corder of current tile, now (q_index * BLOCK_SIZE_Q, 0)
            block_shape = (BLOCK_SIZE_Q, DIM),
            order = (1, 0),
        )

        k_block_ptr = tl.make_block_ptr(
            base = K + qkv_offset,
            shape = (DIM, SEQ_LEN),
            strides = (stride_k_dim, stride_k_seq_len),
            offsets = (0, 0),  # upper left corder of current tile, now (0, 0)
            block_shape = (DIM, BLOCK_SIZE_KV),
            order = (0, 1),
        ) # This block has been transposed.

        v_block_ptr = tl.make_block_ptr(
            base = V + qkv_offset,
            shape = (SEQ_LEN, DIM),
            strides = (stride_v_seq_len, stride_v_dim),
            offsets = (0, 0), # upper left corder of current tile, now (0, 0)
            block_shape = (BLOCK_SIZE_KV, DIM),
            order = (1, 0),
        )

        o_block_ptr = tl.make_block_ptr(
            base = O + qkv_offset,
            shape = (SEQ_LEN, DIM),
            strides = (stride_q_seq_len, stride_q_dim),
            offsets = (q_index * BLOCK_SIZE_Q, 0), # upper left corder of current tile
            block_shape = (BLOCK_SIZE_Q, DIM),
            order = (1, 0),
        )

        l_block_ptr = tl.make_block_ptr(
            base = L + l_offset,
            shape = (SEQ_LEN,),
            strides = (stride_l_seq_len,),
            offsets = (q_index * BLOCK_SIZE_Q,),
            block_shape = (BLOCK_SIZE_Q,),
            order = (0,),
        )
        
        o_i = tl.zeros([BLOCK_SIZE_Q, DIM], dtype=tl.float32)

        m_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) - float("inf")

        l_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) 

        q_block = tl.load(q_block_ptr)
        for start_kv in range(0, SEQ_LEN, BLOCK_SIZE_KV):
            start_kv = tl.multiple_of(start_kv, BLOCK_SIZE_KV)

            k_block = tl.load(k_block_ptr)
            qk_block = tl.dot(q_block, k_block)

            m_ij = tl.maximum(m_i, tl.max(qk_block, 1) * softmax_scale)
            qk_block = qk_block * softmax_scale - m_ij[:, None]
            p_block = tl.exp(qk_block)
            v_block = tl.load(v_block_ptr)

            correction_factor = tl.exp(m_i - m_ij)
            l_i *= correction_factor
            l_i += tl.sum(p_block, axis=1)
            o_i *= correction_factor[:, None]
            o_i += tl.dot(p_block.to(tl.bfloat16), v_block) 
            
            m_i = m_ij
            v_block_ptr = tl.advance(v_block_ptr, [BLOCK_SIZE_KV, 0])
            k_block_ptr = tl.advance(k_block_ptr, [0, BLOCK_SIZE_KV])

        o_i = o_i / l_i[:, None]
        l_i = m_i + tl.log(l_i)
        tl.store(o_block_ptr, o_i.to(tl.bfloat16))
        tl.store(l_block_ptr, l_i)

@triton.jit 
def _attn_bwd_preprocess(
    o,
    do,
    D,
    stride_q_batch, 
    stride_q_seq_len,
    stride_q_head,
    stride_q_dim,
    stride_D_batch,
    stride_D_seq_len,
    stride_D_head,
    SEQ_LEN,
    BLOCK_SIZE_Q: tl.constexpr,
    DIM: tl.constexpr,
):
    o_index = tl.program_id(0)
    batch_index = tl.program_id(1) 
    head_index = tl.program_id(2) 
    o_offset = batch_index * stride_q_batch + head_index * stride_q_head
    D_offset = batch_index * stride_D_batch + head_index * stride_D_head

    o_block_ptr = tl.make_block_ptr(
        base = o + o_offset,
        shape = (SEQ_LEN, DIM),
        strides = (stride_q_seq_len, stride_q_dim),
        offsets = (o_index * BLOCK_SIZE_Q, 0), # upper left corder of current tile, now (o_index * BLOCK_SIZE_Q, 0)
        block_shape = (BLOCK_SIZE_Q, DIM),
        order = (1, 0),
    )
    do_block_ptr = tl.make_block_ptr(
        base = do + o_offset,
        shape = (SEQ_LEN, DIM),
        strides = (stride_q_seq_len, stride_q_dim),
        offsets = (o_index * BLOCK_SIZE_Q, 0), # upper left corder of current tile, now (o_index * BLOCK_SIZE_Q, 0)
        block_shape = (BLOCK_SIZE_Q, DIM),
        order = (1, 0),
    )    
    D_block_ptr = tl.make_block_ptr(
        base = D + D_offset,
        shape = (SEQ_LEN, ),
        strides = (stride_D_seq_len,),
        offsets = (o_index * BLOCK_SIZE_Q,), # upper left corder of current tile, now (o_index * BLOCK_SIZE_Q, 0)
        block_shape = (BLOCK_SIZE_Q,),
        order = (0,),
    )    
    
    o_block = tl.load(o_block_ptr).to(tl.float32)
    do_block = tl.load(do_block_ptr).to(tl.float32)
    D_block = tl.sum(o_block * do_block, axis=1)
    tl.store(D_block_ptr, D_block)


@triton.jit
def _attn_bwd_dk_dv(
    Q, 
    K, 
    V, 
    softmax_scale,
    dO,
    dQ,
    dK,
    dV,
    lse,
    D,
    stride_q_batch, 
    stride_q_seq_len,
    stride_q_head,
    stride_q_dim,
    stride_lse_batch, 
    stride_lse_seq_len,
    stride_lse_head,
    SEQ_LEN,
    NUM_HEADS,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    DIM: tl.constexpr,
):
    index_kv = tl.program_id(0)
    index_batch = tl.program_id(1)
    index_head = tl.program_id(2)
    qkv_offset = index_batch * stride_q_batch + index_head * stride_q_head 
    l_offset = index_batch * stride_lse_batch + index_head * stride_lse_head

    q_block_ptr = tl.make_block_ptr(
        base = Q + qkv_offset,
        shape = (SEQ_LEN, DIM),
        strides = (stride_q_seq_len, stride_q_dim),
        offsets = (0, 0), # upper left corder of current tile, now (q_index * BLOCK_SIZE_Q, 0)
        block_shape = (BLOCK_SIZE_Q, DIM),
        order = (1, 0),
    )

    k_block_ptr = tl.make_block_ptr(
        base = K + qkv_offset,
        shape = (SEQ_LEN, DIM),
        strides = (stride_q_seq_len, stride_q_dim),
        offsets = (index_kv * BLOCK_SIZE_KV, 0),  # upper left corder of current tile, now (0, 0)
        block_shape = (BLOCK_SIZE_KV, DIM),
        order = (1, 0),
    ) 

    v_block_ptr = tl.make_block_ptr(
        base = V + qkv_offset,
        shape = (SEQ_LEN, DIM),
        strides = (stride_q_seq_len, stride_q_dim),
        offsets = (index_kv * BLOCK_SIZE_KV, 0), # upper left corder of current tile, now (0, 0)
        block_shape = (BLOCK_SIZE_KV, DIM),
        order = (1, 0),
    )

    do_block_ptr = tl.make_block_ptr(
        base = dO + qkv_offset,
        shape = (SEQ_LEN, DIM),
        strides = (stride_q_seq_len, stride_q_dim),
        offsets = (0, 0), # upper left corder of current tile
        block_shape = (BLOCK_SIZE_Q, DIM),
        order = (1, 0),
    )    

    dq_block_ptr = tl.make_block_ptr(
        base = dQ + qkv_offset,
        shape = (SEQ_LEN, DIM),
        strides = (stride_q_seq_len, stride_q_dim),
        offsets = (0, 0), # upper left corder of current tile
        block_shape = (BLOCK_SIZE_Q, DIM),
        order = (1, 0),
    )    

    dk_block_ptr = tl.make_block_ptr(
        base = dK + qkv_offset,
        shape = (SEQ_LEN, DIM),
        strides = (stride_q_seq_len, stride_q_dim),
        offsets = (index_kv * BLOCK_SIZE_KV, 0), # upper left corder of current tile
        block_shape = (BLOCK_SIZE_KV, DIM),
        order = (1, 0),
    )

    dv_block_ptr = tl.make_block_ptr(
        base = dV + qkv_offset,
        shape = (SEQ_LEN, DIM),
        strides = (stride_q_seq_len, stride_q_dim),
        offsets = (index_kv * BLOCK_SIZE_KV, 0), # upper left corder of current tile
        block_shape = (BLOCK_SIZE_KV, DIM),
        order = (1, 0),
    )

    l_block_ptr = tl.make_block_ptr(
        base = lse + l_offset,
        shape = (SEQ_LEN,),
        strides = (stride_lse_seq_len,),
        offsets = (0,),
        block_shape = (BLOCK_SIZE_Q,),
        order = (0,),
    )

    D_block_ptr = tl.make_block_ptr(
        base = D + l_offset,
        shape = (SEQ_LEN,),
        strides = (stride_lse_seq_len,),
        offsets = (0,),
        block_shape = (BLOCK_SIZE_Q,),
        order = (0,),
    )    

    k_block = tl.load(k_block_ptr)
    v_block = tl.load(v_block_ptr)

    dk_block = tl.zeros([BLOCK_SIZE_KV, DIM], dtype=tl.float32)
    dv_block = tl.zeros([BLOCK_SIZE_KV, DIM], dtype=tl.float32)

    for i in range(SEQ_LEN // BLOCK_SIZE_Q):
        q_block = tl.load(q_block_ptr)
        do_block = tl.load(do_block_ptr)
        l_block = tl.load(l_block_ptr)
        D_block = tl.load(D_block_ptr)

        s_block = tl.dot(q_block, tl.trans(k_block)) * softmax_scale

        p_block = tl.exp(s_block - l_block[:, None]).to(tl.bfloat16)
        dv_block += tl.dot(tl.trans(p_block), do_block)
        dp_block = tl.dot(do_block, tl.trans(v_block))
        ds_block = p_block * (dp_block - D_block[:, None]).to(tl.bfloat16)
        dk_block += softmax_scale * tl.dot(tl.trans(ds_block), q_block)

        q_block_ptr = tl.advance(q_block_ptr, [BLOCK_SIZE_Q, 0])
        do_block_ptr = tl.advance(do_block_ptr, [BLOCK_SIZE_Q, 0])
        l_block_ptr = tl.advance(l_block_ptr, [BLOCK_SIZE_Q])
        D_block_ptr = tl.advance(D_block_ptr, [BLOCK_SIZE_Q])
    
    tl.store(dk_block_ptr, dk_block.to(tl.bfloat16))
    tl.store(dv_block_ptr, dv_block.to(tl.bfloat16))

@triton.jit
def _attn_bwd_dq(
    Q, 
    K, 
    V, 
    softmax_scale,
    dO,
    dQ,
    dK,
    dV,
    lse,
    D,
    stride_q_batch, 
    stride_q_seq_len,
    stride_q_head,
    stride_q_dim,
    stride_lse_batch, 
    stride_lse_seq_len,
    stride_lse_head,
    SEQ_LEN,
    NUM_HEADS,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    DIM: tl.constexpr,
):
    index_q = tl.program_id(0)
    index_batch = tl.program_id(1)
    index_head = tl.program_id(2)
    qkv_offset = index_batch * stride_q_batch + index_head * stride_q_head
    l_offset = index_batch * stride_lse_batch + index_head * stride_lse_head

    q_block_ptr = tl.make_block_ptr(
        base = Q + qkv_offset,
        shape = (SEQ_LEN, DIM),
        strides = (stride_q_seq_len, stride_q_dim),
        offsets = (index_q * BLOCK_SIZE_Q, 0), # upper left corder of current tile, now (q_index * BLOCK_SIZE_Q, 0)
        block_shape = (BLOCK_SIZE_Q, DIM),
        order = (1, 0),
    )

    k_block_ptr = tl.make_block_ptr(
        base = K + qkv_offset,
        shape = (SEQ_LEN, DIM),
        strides = (stride_q_seq_len, stride_q_dim),
        offsets = (0, 0),  # upper left corder of current tile, now (0, 0)
        block_shape = (BLOCK_SIZE_KV, DIM),
        order = (1, 0),
    ) 

    v_block_ptr = tl.make_block_ptr(
        base = V + qkv_offset,
        shape = (SEQ_LEN, DIM),
        strides = (stride_q_seq_len, stride_q_dim),
        offsets = (0, 0), # upper left corder of current tile, now (0, 0)
        block_shape = (BLOCK_SIZE_KV, DIM),
        order = (1, 0),
    )

    do_block_ptr = tl.make_block_ptr(
        base = dO + qkv_offset,
        shape = (SEQ_LEN, DIM),
        strides = (stride_q_seq_len, stride_q_dim),
        offsets = (index_q * BLOCK_SIZE_Q, 0), # upper left corder of current tile
        block_shape = (BLOCK_SIZE_Q, DIM),
        order = (1, 0),
    )    

    dq_block_ptr = tl.make_block_ptr(
        base = dQ + qkv_offset,
        shape = (SEQ_LEN, DIM),
        strides = (stride_q_seq_len, stride_q_dim),
        offsets = (index_q * BLOCK_SIZE_Q, 0), # upper left corder of current tile
        block_shape = (BLOCK_SIZE_Q, DIM),
        order = (1, 0),
    )    

    dk_block_ptr = tl.make_block_ptr(
        base = dK + qkv_offset,
        shape = (SEQ_LEN, DIM),
        strides = (stride_q_seq_len, stride_q_dim),
        offsets = (0, 0), # upper left corder of current tile
        block_shape = (BLOCK_SIZE_Q, DIM),
        order = (1, 0),
    )    

    dv_block_ptr = tl.make_block_ptr(
        base = dV + qkv_offset,
        shape = (SEQ_LEN, DIM),
        strides = (stride_q_seq_len, stride_q_dim),
        offsets = (0, 0), # upper left corder of current tile
        block_shape = (BLOCK_SIZE_Q, DIM),
        order = (1, 0),
    )   

    l_block_ptr = tl.make_block_ptr(
        base = lse + l_offset,
        shape = (SEQ_LEN,),
        strides = (stride_lse_seq_len,),
        offsets = (index_q * BLOCK_SIZE_Q,),
        block_shape = (BLOCK_SIZE_Q,),
        order = (0,),
    )

    D_block_ptr = tl.make_block_ptr(
        base = D + l_offset,
        shape = (SEQ_LEN,),
        strides = (stride_lse_seq_len,),
        offsets = (index_q * BLOCK_SIZE_Q,),
        block_shape = (BLOCK_SIZE_Q,),
        order = (0,),
    )    

    q_block = tl.load(q_block_ptr).to(tl.float32)
    do_block = tl.load(do_block_ptr).to(tl.float32)
    l_block = tl.load(l_block_ptr)
    D_block = tl.load(D_block_ptr)
    dq_block = tl.zeros([BLOCK_SIZE_Q, DIM], dtype=tl.float32)

    for i in range(SEQ_LEN // BLOCK_SIZE_KV):

        k_block = tl.load(k_block_ptr).to(tl.float32)
        v_block = tl.load(v_block_ptr).to(tl.float32)
        s_block = tl.dot(q_block, tl.trans(k_block)) * softmax_scale

        p_block = tl.exp(s_block - l_block[:, None])
        dp_block = tl.dot(do_block, tl.trans(v_block))
        ds_block = p_block * (dp_block - D_block[:, None])
        dq_block += softmax_scale * tl.dot(ds_block, k_block)

        k_block_ptr = tl.advance(k_block_ptr, [BLOCK_SIZE_KV, 0])
        v_block_ptr = tl.advance(v_block_ptr, [BLOCK_SIZE_KV, 0])
    
    tl.store(dq_block_ptr, dq_block.to(tl.bfloat16))


class FlashAttention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v):
        assert q.ndim == k.ndim == v.ndim == 4
        batch_size, seq_len, num_heads, dim = q.shape
        softmax_scale = 1 / math.sqrt(dim)
        o = torch.empty_like(q)
        BLOCK_SIZE_Q = 128
        BLOCK_SIZE_KV = 128
        grid = (triton.cdiv(seq_len, BLOCK_SIZE_Q), batch_size * num_heads)
        lse = torch.empty((batch_size, seq_len, num_heads), device=q.device, dtype=torch.float32)

        _attn_fwd[grid](
            Q=q, 
            K=k,
            V=v,
            softmax_scale=softmax_scale,
            L=lse, 
            O=o,
            stride_q_batch_size=q.stride(0),
            stride_q_seq_len=q.stride(1),
            stride_q_num_heads=q.stride(2),
            stride_q_dim=q.stride(3),
            stride_k_batch_size=k.stride(0),
            stride_k_seq_len=k.stride(1),
            stride_k_num_heads=k.stride(2),
            stride_k_dim=k.stride(3),
            stride_v_batch_size=v.stride(0),
            stride_v_seq_len=v.stride(1),
            stride_v_num_heads=v.stride(2),
            stride_v_dim=v.stride(3),
            stride_l_batch_size=lse.stride(0),
            stride_l_seq_len=lse.stride(1),
            stride_l_num_heads=lse.stride(2),
            BATCH_SIZE=batch_size,
            SEQ_LEN=seq_len,
            NUM_HEADS=num_heads,
            DIM=dim,    
            BLOCK_SIZE_Q=BLOCK_SIZE_Q,
            BLOCK_SIZE_KV=BLOCK_SIZE_KV,              
        )
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.softmax_scale = softmax_scale 
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors 
        softmax_scale = ctx.softmax_scale
        assert do.is_contiguous()
        assert q.stride() == k.stride() == v.stride() == do.stride()
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)

        BATCH_SIZE, SEQ_LEN, NUM_HEADS, DIM = q.shape 
        BLOCK_SIZE_MICRO, BLOCK_SIZE_MACRO = 32, 128
        preprocess_grid = (SEQ_LEN // BLOCK_SIZE_MACRO, BATCH_SIZE, NUM_HEADS)
        D = torch.empty_like(lse)

        _attn_bwd_preprocess[preprocess_grid](
            o=o, 
            do=do, 
            D=D,
            stride_q_batch=q.stride(0), 
            stride_q_seq_len=q.stride(1), 
            stride_q_head=q.stride(2), 
            stride_q_dim=q.stride(3), 
            stride_D_batch=D.stride(0),
            stride_D_seq_len=D.stride(1),
            stride_D_head=D.stride(2), 
            SEQ_LEN=SEQ_LEN,
            BLOCK_SIZE_Q=BLOCK_SIZE_MACRO,
            DIM=dim,
        )

        grid = (SEQ_LEN // BLOCK_SIZE_MACRO, BATCH_SIZE, NUM_HEADS)

        _attn_bwd_dk_dv[grid](
            Q=q, 
            K=k, 
            V=v, 
            softmax_scale=softmax_scale,
            dO=do,
            dQ=dq,
            dK=dk,
            dV=dv,
            lse=lse,
            D=D,
            stride_q_batch=q.stride(0), 
            stride_q_seq_len=q.stride(1),
            stride_q_head=q.stride(2),
            stride_q_dim=q.stride(3),
            stride_lse_batch=lse.stride(0), 
            stride_lse_seq_len=lse.stride(1),
            stride_lse_head=lse.stride(2),
            SEQ_LEN=SEQ_LEN,
            NUM_HEADS=NUM_HEADS,
            BLOCK_SIZE_Q=BLOCK_SIZE_MICRO,
            BLOCK_SIZE_KV=BLOCK_SIZE_MACRO,
            DIM=DIM,
        )

        grid = (SEQ_LEN // BLOCK_SIZE_MICRO, BATCH_SIZE, NUM_HEADS)
        _attn_bwd_dq[grid](
            Q=q, 
            K=k, 
            V=v, 
            softmax_scale=softmax_scale,
            dO=do,
            dQ=dq,
            dK=dk,
            dV=dv,
            lse=lse,
            D=D,
            stride_q_batch=q.stride(0), 
            stride_q_seq_len=q.stride(1),
            stride_q_head=q.stride(2),
            stride_q_dim=q.stride(3),
            stride_lse_batch=lse.stride(0), 
            stride_lse_seq_len=lse.stride(1),
            stride_lse_head=lse.stride(2),
            SEQ_LEN=SEQ_LEN,
            NUM_HEADS=NUM_HEADS,
            BLOCK_SIZE_Q=BLOCK_SIZE_MICRO,
            BLOCK_SIZE_KV=BLOCK_SIZE_MACRO,
            DIM=DIM,
        )
        return dq, dk, dv



if __name__ == "__main__":

    batch_size, num_heads, seq_len, dim = 4, 32, 32000, 128
    q = torch.randn((batch_size, seq_len, num_heads, dim), dtype=torch.bfloat16, device="cuda").requires_grad_(True)
    k = torch.randn((batch_size, seq_len, num_heads, dim), dtype=torch.bfloat16, device="cuda").requires_grad_(True)
    v = torch.randn((batch_size, seq_len, num_heads, dim), dtype=torch.bfloat16, device="cuda").requires_grad_(True)

    o = FlashAttention.apply(q, k, v)
    do = torch.rand_like(o)
    o.backward(do)


    q_ = q.detach().clone().requires_grad_(True)
    k_ = k.detach().clone().requires_grad_(True)
    v_ = v.detach().clone().requires_grad_(True)
    o_ = ground_truth(q_, k_, v_)
    o_.backward(do)

    relative_error = torch.norm(o - o_, p='fro') / torch.norm(o_, p='fro')
    print(relative_error)

    relative_error = torch.norm(q_.grad - q.grad, p='fro') / torch.norm(q_.grad, p='fro')
    print(relative_error)
    relative_error = torch.norm(k_.grad - k.grad, p='fro') / torch.norm(k_.grad, p='fro')
    print(relative_error)
    relative_error = torch.norm(v_.grad - v.grad, p='fro') / torch.norm(v_.grad, p='fro')
    print(relative_error)











