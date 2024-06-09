import torch
from torch.nn.functional import softmax
from .Tree import BatchTree
import time
from Hesse.Engine.llm_pipe import LLMEngine
from .utils import get_sampling_logits, ChildrenAccept
class BatchSTree(BatchTree):
    def __init__(self, 
                 draft_model_engine: LLMEngine,
                 target_model_engine: LLMEngine,
                 prefix :torch.LongTensor,
                 temperature :float = 0.6,
                 top_p: float = 0.9,
                 max_length = 256,
                 device :str = 'cpu',
                 max_target_seq = 256,
                 vocab_size = 32000,
                 batch_size = 1,
                 grow_map = None,
                 sampling_callables = None,
                 sample_gather_indices = None,
                 ) -> None:
        super().__init__(device=device, max_length=max_length, batch_size=batch_size)
        assert self.max_length == draft_model_engine.max_length
        self.max_target_seq = max_target_seq
        self.draft_model_engine = draft_model_engine
        self.target_model_engine = target_model_engine
        self.temperature = temperature
        self.top_p = top_p
        self.grow_map = grow_map
        self.sampling_callables = sampling_callables
        self.sample_gather_indices = sample_gather_indices
        self.draft_step = len(self.grow_map["roots"])
        self.vocab_size = vocab_size
        self.grow_map_roots_gpu = []
        for x in self.grow_map["roots"]:
             self.grow_map_roots_gpu.append(torch.Tensor(x).to(self.device).long())
        self.Successors = self.grow_map["Successors"]

        tree_mask :torch.Tensor = self.grow_map["mask"].to(self.device)
        tree_mask = (tree_mask == 0).type(self.dtype)
        tree_mask.masked_fill_(tree_mask > 0, torch.finfo(self.dtype).min)
        self.initialize(None)
        self.set_prefix(prefix=prefix)

        self.tree_size = self.grow_map["size"]
        self.tree_mask = tree_mask

        self.attn_mask = torch.full((self.batch_size, 1, self.tree_size ,self.max_length), torch.tensor(torch.finfo(self.dtype).min, device=device), device=device).to(self.dtype)

        self.depth = self.grow_map["depth"].repeat(self.batch_size,1).to(self.device)

        self.tree_buffer = torch.zeros((self.batch_size, self.tree_size),device=self.device).long()
        self.draft_logits = torch.zeros((self.batch_size, self.tree_size, vocab_size), dtype=torch.float32).to(self.device)

        self.make_inference_para_for_first_itr(prefix.size(1))

        self.draft_model_engine.forward(input_ids=self.tokens[:, :prefix.size(1)], 
                            storage_ids=self.prefill_storage_ids, 
                            position_ids=self.prefill_position_ids,
                            attention_mask=self.prefill_mask)
        
        output = self.target_model_engine.forward(input_ids=self.tokens[:, :prefix.size(1)], 
                            storage_ids=self.prefill_storage_ids, 
                            position_ids=self.prefill_position_ids,
                            attention_mask=self.prefill_mask)
        root_logits = output[:, -1:].clone()
        root_logits = get_sampling_logits(logits=root_logits, top_p=self.top_p, T=self.temperature, replicate=False)
        root_logits = softmax(root_logits / self.temperature, dim=-1)
        root_tokens = root_logits.view(-1, self.vocab_size).multinomial(num_samples=1).view(self.batch_size, 1)
        self.tree_buffer[:, 0] = root_tokens.squeeze(1)
        
        self.prepare_for_next_iter()
    
    @torch.inference_mode()
    def collective_grow_static(self, idx_list, n_branch_list :list[int], benchmark=False, grow_step = None):
        
        if benchmark:
            x1 = 0.0
            x2 = 0.0
        
        total_branch = sum(n_branch_list)
        indices = self.grow_map_roots_gpu[grow_step+1]
        

        # Sample from stored logits
        if benchmark:
                torch.cuda.synchronize()
                t1 = time.time()
        for i in range(self.batch_size):
            new_tokens_set = self.sampling_callables[grow_step](self.draft_logits[i, idx_list])
            self.tree_buffer[i, indices] = new_tokens_set[self.sample_gather_indices[grow_step]]
            
        if benchmark:
                    torch.cuda.synchronize()
                    t2 = time.time()
                    x1 += (t2 - t1)
        
        draft_model_outputs = self.draft_model_engine.forward(
            input_ids = self.tree_buffer[:, indices],
            position_ids = self.position_ids[:, indices],
            attention_mask = self.attn_mask[:,:,indices,:],
            storage_ids=self.storage_ids[indices]
        )
        self.draft_logits[:, indices] = draft_model_outputs[:, -total_branch:]

        if benchmark:
                    torch.cuda.synchronize()
                    t3 = time.time()
                    x2 += (t3 - t2)
        if benchmark:
            return n_branch_list, x1, x2
        return n_branch_list
    
    @torch.inference_mode()
    def accept_step(self, parent_id :int, idx) ->ChildrenAccept:
        logits_id = parent_id
        
        target_token = self.target_token[idx, logits_id]
        children = self.Successors[logits_id]
        if len(children) == 0:
            return -1
        
        for pos in children:
            token = self.tree_buffer[idx, pos]
            if token == target_token:
                return pos
        return -1

        
    @torch.inference_mode()
    def verify(self, benchmark = False):
        # Inference to get the target tokens
        if benchmark:
            torch.cuda.synchronize()
            t1 = time.time()
        target_model_outputs = self.target_model_engine.forward(input_ids = self.tree_buffer, 
                                    position_ids =self.position_ids, attention_mask = self.attn_mask,
                                    storage_ids=self.storage_ids)
        if benchmark:
            torch.cuda.synchronize()
            t2 = time.time()
        self.target_logits :torch.FloatTensor = target_model_outputs[:, -self.tree_size:]
        self.target_logits = get_sampling_logits(logits=self.target_logits, top_p=self.top_p, T=self.temperature, replicate=False)
        self.target_logits = softmax(self.target_logits / self.temperature, dim=-1)
        self.target_token = self.target_logits.view(-1, self.vocab_size).multinomial(num_samples=1).view(self.batch_size, self.tree_size)
# Compare the target token with draft tokens
        terminal = False
        batch_accept_list=[]
        batch_accept_list_kv= []
        for batch_idx in range(self.batch_size):
            accept_list = [0]
            while True:
                parent_id = accept_list[-1]
                pos = self.accept_step(parent_id=parent_id, idx=batch_idx)
                if pos != -1:
                    accept_list.append(pos)
                    if self.tree_buffer[batch_idx, pos] == 0 or self.tree_buffer[batch_idx, pos] == 2:
                        terminal = True
                        break
                else:
                    break
            batch_accept_list.append(accept_list)
            accept_length = len(accept_list)
            batch_accept_list_kv.append([self.max_length-self.tree_size+pos for pos in accept_list])
            self.tokens[batch_idx, self.num_nodes[batch_idx]:self.num_nodes[batch_idx]+accept_length] = self.tree_buffer[batch_idx, accept_list]
            self.num_nodes[batch_idx]+= accept_length
        if benchmark:
            torch.cuda.synchronize()
            t3 = time.time()
# Check Bonus token
        if not terminal:
            for batch_idx in range(self.batch_size):
                accept_list = batch_accept_list[batch_idx]
                bonus_token = self.target_token[batch_idx, accept_list[-1]].reshape(1)
                if (bonus_token == 2) or (bonus_token == 0): 
                    terminal = True
                    break
                self.tree_buffer[batch_idx, 0]= bonus_token
# Gather KV Cache
        if not terminal:
            for batch_idx in range(self.batch_size):
                accept_list = batch_accept_list[batch_idx]
                accept_list_kv = batch_accept_list_kv[batch_idx]
                accept_length = len(accept_list)
                self.draft_model_engine.llm.kv_cache.gather_kv_incremental(accept_list_kv, self.num_nodes[batch_idx]-accept_length, batch_idx)
                self.target_model_engine.llm.kv_cache.gather_kv_incremental(accept_list_kv, self.num_nodes[batch_idx]-accept_length, batch_idx)

        if not terminal:
            if benchmark:
                torch.cuda.synchronize()
                t4 = time.time()
                self.prepare_for_next_iter()
                return self.num_nodes, t2 - t1, t3-t2, t4 - t3, terminal
            self.prepare_for_next_iter()
            return self.num_nodes, terminal
            
        else:
             if benchmark:
                torch.cuda.synchronize()
                t4 = time.time()
                return self.num_nodes, t2 - t1, t3-t2, t4 - t3, terminal
             return self.num_nodes, terminal
    
    
    def verbose(self):
        super().verbose()
    
    
    def construct_grow_map(self, benchmark = False):
        if benchmark:
            sample_time = 0
            compute_time = 0
        for i in range(self.draft_step - 1):
                if benchmark:
                        _, t1, t2 = self.collective_grow_static(self.grow_map_roots_gpu[i], self.grow_map['branches'][i], benchmark=benchmark, grow_step=i)
                        sample_time += t1
                        compute_time += t2   
                else:
                        self.collective_grow_static(self.grow_map_roots_gpu[i], self.grow_map['branches'][i], grow_step=i)
        if benchmark:
            return sample_time, compute_time
        else:
            return None
    
    def prepare_for_next_iter(self):
        if self.num_nodes.max()+ 1 > self.max_target_seq:
              return 
        self.make_inference_para_for_next_itr()
        draft_model_outputs = self.draft_model_engine.forward(input_ids = self.tree_buffer[:,:1], 
                                                    storage_ids=self.storage_ids[:1],
                                                    position_ids=self.position_ids[:, :1],
                                                    attention_mask=self.attn_mask[:,:,:1,:])
        
        self.draft_logits[:, 0] = draft_model_outputs[:, -1]

    def make_inference_para_for_first_itr(self, prefix_len):
        # Prefill
         self.prefill_mask=self.full_attn_mask[:prefix_len][None, None, :, :].repeat(self.batch_size,1,1,1)
         self.prefill_storage_ids = torch.arange(prefix_len).to(self.device)
         self.prefill_position_ids = self.prefill_storage_ids.clone().repeat(self.batch_size, 1)

        # Draft Construct Tree and Target Verify
         self.attn_mask[:, :, :, -self.tree_size:] = self.tree_mask
         for idx in range(self.batch_size):
            self.attn_mask[idx, :, :, :prefix_len] = 0.0
         self.position_ids = (self.grow_map["depth"].to(self.device) + prefix_len).repeat(self.batch_size, 1)
         self.storage_ids = torch.arange(self.max_length-self.tree_size, self.max_length).to(self.device)
    
    def make_inference_para_for_next_itr(self):
         self.position_ids = self.depth + self.num_nodes.view(-1,1)
         for idx in range(self.batch_size):
            self.attn_mask[idx, :, :, :self.num_nodes[idx]] = 0.0


class BatchSTreeTest(BatchTree):
    def __init__(self, 
                 draft_model_engine: LLMEngine,
                 target_model_engine: LLMEngine,
                 prefix :torch.LongTensor,
                 temperature :float = 0.6,
                 top_p: float = 0.9,
                 max_length = 256,
                 device :str = 'cpu',
                 max_target_seq = 256,
                 vocab_size = 32000,
                 batch_size = 1,
                 grow_map = None,
                 sampling_callables = None,
                 sample_gather_indices = None,
                 ) -> None:
        super().__init__(device=device, max_length=max_length, batch_size=batch_size)
        assert self.max_length == draft_model_engine.max_length
        self.max_target_seq = max_target_seq
        self.draft_model_engine = draft_model_engine
        self.target_model_engine = target_model_engine
        self.temperature = temperature
        self.top_p = top_p
        self.grow_map = grow_map
        self.sampling_callables = sampling_callables
        self.sample_gather_indices = sample_gather_indices
        self.draft_step = len(self.grow_map["roots"])
        self.vocab_size = vocab_size
        self.grow_map_roots_gpu = []
        for x in self.grow_map["roots"]:
             self.grow_map_roots_gpu.append(torch.Tensor(x).to(self.device).long())
        self.Successors = self.grow_map["Successors"]

        tree_mask :torch.Tensor = self.grow_map["mask"].to(self.device)
        tree_mask = (tree_mask == 0).type(self.dtype)
        tree_mask.masked_fill_(tree_mask > 0, torch.finfo(self.dtype).min)
        self.initialize(None)
        self.set_prefix(prefix=prefix)

        self.tree_size = self.grow_map["size"]
        self.tree_mask = tree_mask

        self.attn_mask = torch.full((self.batch_size, 1, self.tree_size ,self.max_length), torch.tensor(torch.finfo(self.dtype).min, device=device), device=device).to(self.dtype)

        self.depth = self.grow_map["depth"].repeat(self.batch_size,1).to(self.device)

        self.tree_buffer = torch.zeros((self.batch_size, self.tree_size),device=self.device).long()
        self.draft_logits = torch.zeros((self.batch_size, self.tree_size, vocab_size), dtype=torch.float32).to(self.device)

        self.make_inference_para_for_first_itr(prefix.size(1))

        self.draft_model_engine.forward(input_ids=self.tokens[:, :prefix.size(1)], 
                            storage_ids=self.prefill_storage_ids, 
                            position_ids=self.prefill_position_ids,
                            attention_mask=self.prefill_mask)
        
        output = self.target_model_engine.forward(input_ids=self.tokens[:, :prefix.size(1)], 
                            storage_ids=self.prefill_storage_ids, 
                            position_ids=self.prefill_position_ids,
                            attention_mask=self.prefill_mask)
        root_logits = output[:, -1:].clone()
        root_logits = get_sampling_logits(logits=root_logits, top_p=self.top_p, T=self.temperature, replicate=False)
        root_logits = softmax(root_logits / self.temperature, dim=-1)
        root_tokens = root_logits.view(-1, self.vocab_size).multinomial(num_samples=1).view(self.batch_size, 1)
        self.tree_buffer[:, 0] = root_tokens.squeeze(1)
        
        self.prepare_for_next_iter()
    
    @torch.inference_mode()
    def collective_grow_static(self, idx_list, n_branch_list :list[int], benchmark=False, grow_step = None):
        
        if benchmark:
            x1 = 0.0
            x2 = 0.0
        
        total_branch = sum(n_branch_list)
        indices = self.grow_map_roots_gpu[grow_step+1]
        

        # Sample from stored logits
        if benchmark:
                torch.cuda.synchronize()
                t1 = time.time()
        for i in range(self.batch_size):
            new_tokens_set = self.sampling_callables[grow_step](self.draft_logits[i, idx_list])
            self.tree_buffer[i, indices] = new_tokens_set[self.sample_gather_indices[grow_step]]
            
        if benchmark:
                    torch.cuda.synchronize()
                    t2 = time.time()
                    x1 += (t2 - t1)
        
        draft_model_outputs = self.draft_model_engine.forward(
            input_ids = self.tree_buffer[:, indices],
            position_ids = self.position_ids[:, indices],
            attention_mask = self.attn_mask[:,:,indices,:],
            storage_ids=self.storage_ids[indices]
        )
        self.draft_logits[:, indices] = draft_model_outputs[:, -total_branch:]

        if benchmark:
                    torch.cuda.synchronize()
                    t3 = time.time()
                    x2 += (t3 - t2)
        if benchmark:
            return n_branch_list, x1, x2
        return n_branch_list
    
    @torch.inference_mode()
    def accept_step(self, parent_id :int, idx) ->ChildrenAccept:
        logits_id = parent_id
        
        target_token = self.target_token[idx, logits_id]
        children = self.Successors[logits_id]
        if len(children) == 0:
            return -1
        
        for pos in children:
            token = self.tree_buffer[idx, pos]
            if token == target_token:
                return pos
        return -1

        
    @torch.inference_mode()
    def verify(self, benchmark = False):
        # Inference to get the target tokens
        accept_pos=-1
        if benchmark:
            torch.cuda.synchronize()
            t1 = time.time()
        target_model_outputs = self.target_model_engine.forward(input_ids = self.tree_buffer, 
                                    position_ids =self.position_ids, attention_mask = self.attn_mask,
                                    storage_ids=self.storage_ids)
        if benchmark:
            torch.cuda.synchronize()
            t2 = time.time()
        self.target_logits :torch.FloatTensor = target_model_outputs[:, -self.tree_size:]
        self.target_logits = get_sampling_logits(logits=self.target_logits, top_p=self.top_p, T=self.temperature, replicate=False)
        self.target_logits = softmax(self.target_logits / self.temperature, dim=-1)
        self.target_token = self.target_logits.view(-1, self.vocab_size).multinomial(num_samples=1).view(self.batch_size, self.tree_size)

        if benchmark:
            torch.cuda.synchronize()
            t3 = time.time()
# Compare the target token with draft tokens
        terminal = False
        batch_accept_list=[]
        batch_accept_list_kv= []
        for batch_idx in range(self.batch_size):
            accept_list = [0]
            while True:
                parent_id = accept_list[-1]
                pos = self.accept_step(parent_id=parent_id, idx=batch_idx)
                if pos != -1:
                    accept_list.append(pos)
                    accept_pos = pos
                    if self.tree_buffer[batch_idx, pos] == 0 or self.tree_buffer[batch_idx, pos] == 2:
                        terminal = True
                        break
                else:
                    break
            batch_accept_list.append(accept_list)
            accept_length = len(accept_list)
            batch_accept_list_kv.append([self.max_length-self.tree_size+pos for pos in accept_list])
            self.tokens[batch_idx, self.num_nodes[batch_idx]:self.num_nodes[batch_idx]+accept_length] = self.tree_buffer[batch_idx, accept_list]
            self.num_nodes[batch_idx]+= accept_length
# Check Bonus token
        if not terminal:
            for batch_idx in range(self.batch_size):
                accept_list = batch_accept_list[batch_idx]
                bonus_token = self.target_token[batch_idx, accept_list[-1]].reshape(1)
                if (bonus_token == 2) or (bonus_token == 0): 
                    terminal = True
                    break
                self.tree_buffer[batch_idx, 0]= bonus_token
# Gather KV Cache
        if not terminal:
            for batch_idx in range(self.batch_size):
                accept_list = batch_accept_list[batch_idx]
                accept_list_kv = batch_accept_list_kv[batch_idx]
                accept_length = len(accept_list)
                self.draft_model_engine.llm.kv_cache.gather_kv_incremental(accept_list_kv, self.num_nodes[batch_idx]-accept_length, batch_idx)
                self.target_model_engine.llm.kv_cache.gather_kv_incremental(accept_list_kv, self.num_nodes[batch_idx]-accept_length, batch_idx)

        if not terminal:
            if benchmark:
                torch.cuda.synchronize()
                t4 = time.time()
                self.prepare_for_next_iter()
                return self.num_nodes, t2 - t1, t3-t2, t4 - t3, terminal
            self.prepare_for_next_iter()
            return self.num_nodes, terminal, accept_pos
            
        else:
             if benchmark:
                torch.cuda.synchronize()
                t4 = time.time()
                return self.num_nodes, t2 - t1, t3-t2, t4 - t3, terminal
             return self.num_nodes, terminal, accept_pos
    
    
    def verbose(self):
        super().verbose()
    
    
    def construct_grow_map(self, benchmark = False):
        if benchmark:
            sample_time = 0
            compute_time = 0
        for i in range(self.draft_step - 1):
                if benchmark:
                        _, t1, t2 = self.collective_grow_static(self.grow_map_roots_gpu[i], self.grow_map['branches'][i], benchmark=benchmark, grow_step=i)
                        sample_time += t1
                        compute_time += t2   
                else:
                        self.collective_grow_static(self.grow_map_roots_gpu[i], self.grow_map['branches'][i], grow_step=i)
        if benchmark:
            return sample_time, compute_time
        else:
            return None
    
    def prepare_for_next_iter(self):
        if self.num_nodes.max()+ 1 > self.max_target_seq:
              return 
        self.make_inference_para_for_next_itr()
        draft_model_outputs = self.draft_model_engine.forward(input_ids = self.tree_buffer[:,:1], 
                                                    storage_ids=self.storage_ids[:1],
                                                    position_ids=self.position_ids[:, :1],
                                                    attention_mask=self.attn_mask[:,:,:1,:])
        
        self.draft_logits[:, 0] = draft_model_outputs[:, -1]

    def make_inference_para_for_first_itr(self, prefix_len):
        # Prefill
         self.prefill_mask=self.full_attn_mask[:prefix_len][None, None, :, :].repeat(self.batch_size,1,1,1)
         self.prefill_storage_ids = torch.arange(prefix_len).to(self.device)
         self.prefill_position_ids = self.prefill_storage_ids.clone().repeat(self.batch_size, 1)

        # Draft Construct Tree and Target Verify
         self.attn_mask[:, :, :, -self.tree_size:] = self.tree_mask
         for idx in range(self.batch_size):
            self.attn_mask[idx, :, :, :prefix_len] = 0.0
         self.position_ids = (self.grow_map["depth"].to(self.device) + prefix_len).repeat(self.batch_size, 1)
         self.storage_ids = torch.arange(self.max_length-self.tree_size, self.max_length).to(self.device)
    
    def make_inference_para_for_next_itr(self):
         self.position_ids = self.depth + self.num_nodes.view(-1,1)
         for idx in range(self.batch_size):
            self.attn_mask[idx, :, :, :self.num_nodes[idx]] = 0.0