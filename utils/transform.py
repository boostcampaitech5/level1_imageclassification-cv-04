import torchvision
from torchvision import transforms
import json

def get_transform(args):
    print(f'Transform\t>>\t{args.aug_list}')
    with open('./transform_list.json', 'r') as f:
        transform_json = f.read()
    tf_list = json.loads(transform_json)
    
   
    transform_dict = {'resize':'Resize', "totensor":'ToTensor','centercrop' : "CenterCrop",'colorjitter' : "ColorJitter",
                      'randomhorizontalflip':'RandomHorizontalFlip','randomrotation' : "RandomRotation",'gaussianblur' : "GaussianBlur",
                      'randomaffine': "RandomAffine",'normalize':'Normalize'}

    transform_list = []
    config = {}
    for key in args.aug_list:
        transform_list.append(getattr(transforms,transform_dict[key])(**tf_list[key]))
        config[key] = tf_list[key]
    
    transform = transforms.Compose(transform_list)

    return transform, config
    

class CustomTransform():
    def __init__(self):
        super(CustomTransform, self).__init__()
        pass


    def __call__(self, tensor):
        pass


    def __repr__(self):
        pass


if __name__ == '__main__':
    args_dict = {'transform_path' : './transform_list.json',
                 'transform_list' : ['resize', 'totensor', 'normalize', 'centercrop',
                                     'colorjitter', 'randomhorizontalflip', 'randomrotation', 
                                     'gaussianblur', 'randomaffine']}
    
    from collections import namedtuple
    Args = namedtuple('Args', args_dict.keys())
    args = Args(**args_dict)

    transform, config = get_transform(args)
    print(transform)
    print(config)