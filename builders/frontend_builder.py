import tensorflow as tf
import tf_slim as slim
from frontends import resnet_v2
from frontends import mobilenet_v2
from frontends import inception_v4
import os 


def build_frontend(inputs, frontend, is_training=True, pretrained_dir="models"):
    if frontend == 'ResNet50':
        frontend_scope='resnet_v2_50'
        check_scope_fn = resnet_v2.resnet_arg_scope
        network_model = resnet_v2.resnet_v2_50
        opts={'is_training': is_training, 'scope': frontend_scope}

    elif frontend == 'ResNet101':
        frontend_scope='resnet_v2_101'
        check_scope_fn = resnet_v2.resnet_arg_scope
        network_model = resnet_v2.resnet_v2_101
        opts={'is_training': is_training, 'scope': frontend_scope}

    elif frontend == 'ResNet152':
        frontend_scope='resnet_v2_152'
        check_scope_fn = resnet_v2.resnet_arg_scope
        network_model = resnet_v2.resnet_v2_152
        opts={'is_training': is_training, 'scope': frontend_scope}

    elif frontend == 'MobileNetV2':
        frontend_scope='mobilenet_v2'
        check_scope_fn = mobilenet_v2.training_scope
        network_model = mobilenet_v2.mobilenet
        opts={'is_training': is_training, 'scope': frontend_scope, 'base_only':True}

    elif frontend == 'InceptionV4':
        frontend_scope='inception_v4'
        check_scope_fn = inception_v4.inception_v4_arg_scope
        network_model = inception_v4.inception_v4
        opts={'is_training': is_training, 'scope': frontend_scope}

    else:
        raise ValueError("Unsupported fronetnd model '%s'. This function only supports ResNet50, ResNet101, ResNet152, and MobileNetV2" % (frontend))

    with slim.arg_scope(check_scope_fn()):
        logits, end_points = network_model( inputs, **opts)

        init_fn = slim.assign_from_checkpoint_fn(
                    model_path=os.path.join(pretrained_dir, frontend_scope+'.ckpt'),
                    var_list=slim.get_model_variables(frontend_scope),
                    ignore_missing_vars=True)

    return logits, end_points, frontend_scope, init_fn 
