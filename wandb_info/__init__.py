import json

def get_wandb_info():
    with open('./wandb_info/wandb.json','r') as f:
        data = json.load(f)
    with open('./wandb_info/wandb.json','w') as f:
        data['exp_num']+=1
        json.dump(data, f, indent=4)
        data['exp_num']-=1
    return data
