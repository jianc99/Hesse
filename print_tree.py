import torch
# torch.set_printoptions(profile="full")

vec = torch.load("demo_tree.pt")
# vec = torch.load("acceptance-rate-vector.pt")
# vec1 = torch.load("btree_acc_1.3b.pt")
# tree = torch.load("demo_tree.pt")
# tree_mask :torch.Tensor = vec["mask"]
# tree_mask = (tree_mask == 0).type(torch.float16)

# tree_mask.masked_fill_(tree_mask > 0, torch.finfo(torch.float16).min)
print(vec)
# print(vec1)