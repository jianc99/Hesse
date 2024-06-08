import torch

def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device
):
    """
    Make causal mask used for bi-directional self-attention.
    Copied from Huggingface
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)
    return mask

class BatchTree:
    def __init__(self, device :str = 'cpu', max_length = 512, dtype = torch.float16, batch_size = 1) -> None:
        self.tokens = torch.zeros(batch_size, max_length, device=device).long()
        self.Successors :list[list[int]] = []
        self.num_nodes = torch.zeros(batch_size,device=device).long()
        self.device = device
        self.max_length = max_length
        self.dtype = dtype
        self.batch_size = batch_size


    def initialize(self, active_mark):
        self.position_ids = torch.zeros(self.batch_size,self.max_length).long().to(self.device)
        self.active_mark = active_mark

    def set_prefix(self, prefix: torch.LongTensor):
        self.tokens[:,:prefix.size(1)] = prefix.to(self.device)
        self.position_ids[:,:prefix.size(1)] = torch.arange(prefix.size(1)).repeat(self.batch_size,1)
        
        self.num_nodes += prefix.size(1)
        self.full_attn_mask = _make_causal_mask((1, self.max_length),dtype=self.dtype, device=self.device)

        
        

    def collective_expand_position(self, expand_tokens :torch.LongTensor):
        self.tokens = torch.cat([self.tokens, expand_tokens], dim=-1)
        

    def verbose(self):
        print(self.tokens)
        print(self.Successors)






        



