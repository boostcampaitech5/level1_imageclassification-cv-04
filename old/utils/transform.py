import torchvision
from torchvision import transforms
import json

def get_transform(args):
    with open(args.transform_path, 'r') as f:
        transform_json = f.read()
    tf_list = json.loads(transform_json)

    statistic = {'imagenet' : {'mean' : (0.485, 0.456, 0.406),
                               'std' : (0.229, 0.224, 0.225)},
                 'mask' : {'mean' : (0.5601, 0.5241, 0.5014),
                           'std' : (0.2331, 0.2430, 0.2456)}}
    
    transform_dict = {'resize' : transforms.Resize((tf_list['resize']['img_height'], 
                                                    tf_list['resize']['img_width'])),
                      'totensor' : transforms.ToTensor(),
                      'normalize' : transforms.Normalize(mean=statistic[tf_list['normalize']['normalize_statistic']]['mean'],
                                                         std=statistic[tf_list['normalize']['normalize_statistic']]['std']),
                      'centercrop' : transforms.CenterCrop((tf_list['centercrop']['img_height'], 
                                                            tf_list['centercrop']['img_width'])),
                      'colorjitter' : transforms.ColorJitter(tf_list['colorjitter']['brightness'], 
                                                             tf_list['colorjitter']['contrast'], 
                                                             tf_list['colorjitter']['saturation'], 
                                                             tf_list['colorjitter']['hue']),
                      'randomhorizontalflip' : transforms.RandomHorizontalFlip(tf_list['randomhorizontalflip']['flip_prob']),
                      'randomrotation' : transforms.RandomRotation(tf_list['randomrotation']['degrees']),
                      'gaussianblur' : transforms.GaussianBlur(tf_list['gaussianblur']['kernel_size'],
                                                               (tf_list['gaussianblur']['sigma_min'], tf_list['gaussianblur']['sigma_max'])),
                      'randomaffine': transforms.RandomAffine(degrees=tf_list['randomaffine']['degrees'],
                                                              shear=tuple(tf_list['randomaffine']['shear']),
                                                              translate=tuple(tf_list['randomaffine']['translate']))}
    
    list_ = []
    config = {}
    for key in args.transform_list:
        list_.append(transform_dict[key])
        config[key] = tf_list[key]
    
    transform = transforms.Compose(list_)

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