import torch

# model = torch.load("/dataset/f1d6ea5b/gyx-eva/eva2/checkpoints/esc-blender-9999/1/mp_rank_00_model_states.pt", map_location="cpu")

# model = model["module"]

# print(model.keys())
# for i in range(24):
#     print("encoder, 0, {}".format(i), torch.sum(model["encoder.blocks.{}.self_attn.layer_norm.weight".format(i)] < 0))

# for i in range(24):
#     print("encoder, 1, {}".format(i), torch.sum(model["encoder.blocks.{}.self_attn.layer_norm.weight".format(i)] < 0))

# for i in range(24):
#     print("decoder, 0, {}".format(i), torch.sum(model["decoder.blocks.{}.ff.layer_norm.weight".format(i)] < 0))

# for i in range(24):
#     print("decoder, 1, {}".format(i), torch.sum(model["decoder.blocks.{}.ff.layer_norm.weight".format(i)] < 0))


model = torch.load("/dataset/f1d6ea5b/gyx-eva/t5-test/checkpoints/mt5_base/pytorch_model.bin", map_location="cpu")

layer_num = 24

for i in range(layer_num):
    print("encoder, 0, {}".format(i), torch.sum(model["encoder.block.{}.layer.0.layer_norm.weight".format(i)] < 0))

for i in range(layer_num):
    print("encoder, 1, {}".format(i), torch.sum(model["encoder.block.{}.layer.1.layer_norm.weight".format(i)] < 0))

for i in range(layer_num):
    print("decoder, 0, {}".format(i), torch.sum(model["decoder.block.{}.layer.0.layer_norm.weight".format(i)] < 0))

for i in range(layer_num):
    print("decoder, 1, {}".format(i), torch.sum(model["decoder.block.{}.layer.1.layer_norm.weight".format(i)] < 0))