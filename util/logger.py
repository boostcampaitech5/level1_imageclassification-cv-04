import wandb
import util
class logger:
    def __init__(self,batch_size,epoch,optim_name,lr,backbone) -> None:
        wandb_config = util.read_json('wandb.json')
        idx = wandb_config['idx']
        print("test_name : "+f'exp{idx}_bs{batch_size}_ep{epoch}_{optim_name}_lr{lr}_{backbone}.dk')
        wandb.init(
            project = f'Image_classification_mask',
            name = f'exp{idx}_bs{batch_size}_ep{epoch}_{optim_name}_lr{lr}_{backbone}.dk',
            entity = 'connect-cv-04'
        )
        wandb_config['idx']+=1
        util.write_json('wandb.json',wandb_config)
    
    def write(self,data):
        print(data)
        wandb.log(data)