import torch
    # 假设我们有一个训练好的模型model
model_path = '/home/fftai/Wiki-GRx-Gym/legged_gym/model_4000_jit.pt'
model.load_state_dict(torch.load(model_path))
    # 使用torch.jit.script()将模型转换为TorchScript格式
traced_script_module = torch.jit.script(model)
    # 保存TorchScript模型
traced_script_module.save('/home/fftai/Wiki-GRx-Gym/legged_gym/model_scripted.pt')
