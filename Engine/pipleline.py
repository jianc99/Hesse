import torch
from Hesse.Engine.llm_pipe import LLMEngine
import torch.distributed as dist

class LLM_Pipeline:
    def __init__(self, 
                model_name: str,
                pp_config,
                cg_list: list,
                type: str = "baseline",
                batch_size :int = 1,
                max_length :int = 256, 
                device :str = 'cuda:0',
                dtype = torch.float16,
                last_stage_rank_0 : int = 0,
                ) -> None:
        if pp_config == None:
            self.pp_config = None
            self.last_stage_rank_0 = last_stage_rank_0
            self.max_length = max_length
            dist.barrier()
            return
        self.pp_config = pp_config
        self.bsz = batch_size
        self.max_length = max_length
        self.is_first_stage = (self.pp_config["current_stage"] == 0)
        self.is_last_stage = (self.pp_config["current_stage"] == self.pp_config["num_stages"] -1)
        self.dtype = dtype
        self.type = type
        self.group_indices = self.pp_config["groups_indices"]
        self.current_stage = self.pp_config["current_stage"]
        self.process_group = self.pp_config["current_group"]
        self.local_rank = dist.get_rank(self.process_group)
        self.num_stages = self.pp_config["num_stages"]
        if self.type == "spec":
            self.global_group = self.pp_config["global_group"]
            
        self.pp_engine = LLMEngine(max_length=max_length, model_name=model_name, device=device, pp_config=pp_config, dtype=dtype, batch_size=batch_size)
        if self.type == "spec":
            dist.barrier(self.global_group)
        else:
            dist.barrier()
        self.hidden_dim = self.pp_engine.llm.hidden_size
        self.pp_engine.initialize_cuda_graph(cg_list)
        dist.barrier()


    def forward(self,input_ids, position_ids, attention_mask, storage_ids):
        # if dist.get_rank()==0:
        #     print(input_ids.size(), dist.get_world_size(self.global_group))
        if self.pp_config == None:
            output = torch.full((input_ids.size(0), input_ids.size(1), 32000), 0, dtype=torch.float32, device=input_ids.device)
            dist.broadcast(output, self.last_stage_rank_0)
            return output
        if self.num_stages == 1:
            # dist.broadcast(input_ids, self.group_indices[self.current_stage][0], self.process_group)
            output = self.pp_engine.inference(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask, storage_ids=storage_ids)
            dist.broadcast(output,self.group_indices[-1][0])
            return output
        hidden_state = torch.full((self.bsz, input_ids.size(1), self.hidden_dim), 0, dtype=self.dtype, device=input_ids.device)
        output = torch.full((self.bsz, input_ids.size(1), 32000), 0, dtype=torch.float32, device=input_ids.device)

        if self.is_first_stage:
            # dist.broadcast(input_ids, self.group_indices[self.current_stage][0], self.process_group)
            hidden_state=self.pp_engine.inference(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask, storage_ids=storage_ids)
            if self.local_rank == 0:
                dist.send(hidden_state, self.group_indices[self.current_stage+1][0])
                # dist.recv(output, self.group_indices[-1][0])
        elif self.is_last_stage:
            if self.local_rank == 0:
                dist.recv(hidden_state,self.group_indices[self.current_stage-1][0])
            dist.broadcast(hidden_state, self.group_indices[self.current_stage][0], self.process_group)
            output=self.pp_engine.inference(input_ids=hidden_state, position_ids=position_ids, attention_mask=attention_mask, storage_ids=storage_ids)
            # if self.local_rank == 0:
            #     dist.send(output,0)
        else:
            if self.local_rank == 0:
                dist.recv(hidden_state,self.group_indices[self.current_stage-1][0])
            dist.broadcast(hidden_state, self.group_indices[self.current_stage][0], self.process_group)
            hidden_state=self.pp_engine.inference(input_ids=hidden_state, position_ids=position_ids, attention_mask=attention_mask, storage_ids=storage_ids)
            if self.local_rank == 0:
                dist.send(hidden_state,self.group_indices[self.current_stage+1][0])
        dist.broadcast(output,self.group_indices[-1][0])
        return output
    
    def gather_kv_incremental(self, indices: list[int], offset:int, batch_idx=None):
        if self.pp_config == None:
            return
        self.pp_engine.llm.kv_cache.gather_kv_incremental(indices, offset, batch_idx)
    
    def clear_kv(self):
        if self.pp_config == None:
            return
        self.pp_engine.llm.kv_cache.clear()
    
    def gather_kv(self, indices: list[int]):
        if self.pp_config == None:
            return
        self.pp_engine.llm.kv_cache.gather_kv(indices)

