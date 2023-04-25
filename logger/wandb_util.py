import wandb


def wandb_init(args,config):
    print('Initialize WandB ...')
    wandb.init(name = f'{args.wandb_exp_name}_{args.exp_num}_bs{args.batch_size}_ep{args.epochs}_{args.loss}_lr{args.learning_rate}_{args.load_model}.{args.user_name}',
                project = args.wandb_project_name,
                entity = args.wandb_entity,
                config = config)