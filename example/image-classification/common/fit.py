# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

""" example train fit utility """
import logging
import os
import time
import re
import math
import mxnet as mx
import horovod.mxnet as hvd

def get_epoch_size(args, kv):
    num_workers = hvd.size() if 'horovod' in args.kv_store else kv.num_workers
    return math.ceil(int(args.num_examples / num_workers) / args.batch_size)

def _get_gpu(gpus):
    idx = hvd.local_rank()
    gpu = gpus.split(",")[idx]
    return gpu

def _get_lr_scheduler(args, kv):
    if 'lr_factor' not in args or args.lr_factor >= 1:
        return (args.lr, None)
    epoch_size = get_epoch_size(args, kv)
    begin_epoch = args.load_epoch if args.load_epoch else 0
    if 'pow' in args.lr_step_epochs:
        lr = args.lr
        max_up = args.num_epochs * epoch_size
        pwr = float(re.sub('pow[- ]*', '', args.lr_step_epochs))
        poly_sched = mx.lr_scheduler.PolyScheduler(max_up, lr, pwr)
        return (lr, poly_sched)
    step_epochs = [int(l) for l in args.lr_step_epochs.split(',')]
    lr = args.lr
    for s in step_epochs:
        if begin_epoch >= s:
            lr *= args.lr_factor
    if lr != args.lr:
        logging.info('Adjust learning rate to %e for epoch %d',
                     lr, begin_epoch)

    steps = [epoch_size * (x - begin_epoch)
             for x in step_epochs if x - begin_epoch > 0]
    if steps:
        if kv:
            num_workers = kv.num_workers
        else:
            num_workers = hvd.size() if 'horovod' in args.kv_store else 1
        epoch_size = math.ceil(int(args.num_examples/num_workers)/args.batch_size)
        return (lr, mx.lr_scheduler.MultiFactorScheduler(step=steps, factor=args.lr_factor,
                                                         base_lr=args.lr, warmup_steps=epoch_size * args.warmup_epochs,
                                                         warmup_mode=args.warmup_strategy))
    else:
        return (lr, None)

def _load_model(args, rank=0):
    if 'load_epoch' not in args or args.load_epoch is None:
        return (None, None, None)
    assert args.model_prefix is not None
    model_prefix = args.model_prefix
    if rank > 0 and os.path.exists("%s-%d-symbol.json" % (model_prefix, rank)):
        model_prefix += "-%d" % (rank)
    sym, arg_params, aux_params = mx.model.load_checkpoint(
        model_prefix, args.load_epoch)
    logging.info('Loaded model %s_%04d.params', model_prefix, args.load_epoch)
    return (sym, arg_params, aux_params)


def _save_model(args, rank=0):
    if args.model_prefix is None:
        return None
    return mx.callback.do_checkpoint(args.model_prefix if rank == 0 else "%s-%d" % (
        args.model_prefix, rank), period=args.save_period)


def add_fit_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    train = parser.add_argument_group('Training', 'model training')
    train.add_argument('--network', type=str,
                       help='the neural network to use')
    train.add_argument('--num-layers', type=int,
                       help='number of layers in the neural network, \
                             required by some networks such as resnet')
    train.add_argument('--gpus', type=str,
                       help='list of gpus to run, e.g. 0 or 0,2,5. empty means using cpu')
    train.add_argument('--kv-store', type=str, default='device',
                       help='key-value store type')
    train.add_argument('--num-epochs', type=int, default=100,
                       help='max num of epochs')
    train.add_argument('--lr', type=float, default=0.1,
                       help='initial learning rate')
    train.add_argument('--lr-factor', type=float, default=0.1,
                       help='the ratio to reduce lr on each step')
    train.add_argument('--lr-step-epochs', type=str,
                       help='the epochs to reduce the lr, e.g. 30,60')
    train.add_argument('--initializer', type=str, default='default',
                       help='the initializer type')
    train.add_argument('--optimizer', type=str, default='sgd',
                       help='the optimizer type')
    train.add_argument('--mom', type=float, default=0.9,
                       help='momentum for sgd')
    train.add_argument('--wd', type=float, default=0.0001,
                       help='weight decay for sgd')
    train.add_argument('--batch-size', type=int, default=128,
                       help='the batch size')
    train.add_argument('--disp-batches', type=int, default=20,
                       help='show progress for every n batches')
    train.add_argument('--model-prefix', type=str,
                       help='model prefix')
    train.add_argument('--save-period', type=int, default=1, help='params saving period')
    parser.add_argument('--monitor', dest='monitor', type=int, default=0,
                        help='log network parameters every N iters if larger than 0')
    train.add_argument('--load-epoch', type=int,
                       help='load the model on an epoch using the model-load-prefix')
    train.add_argument('--top-k', type=int, default=0,
                       help='report the top-k accuracy. 0 means no report.')
    train.add_argument('--loss', type=str, default='',
                       help='show the cross-entropy or nll loss. ce strands for cross-entropy, nll-loss stands for likelihood loss')
    train.add_argument('--test-io', type=int, default=0,
                       help='1 means test reading speed without training')
    train.add_argument('--dtype', type=str, default='float32',
                       help='precision: float32 or float16')
    train.add_argument('--gc-type', type=str, default='none',
                       help='type of gradient compression to use, \
                             takes `2bit` or `none` for now')
    train.add_argument('--gc-threshold', type=float, default=0.5,
                       help='threshold for 2bit gradient compression')
    # additional parameters for large batch sgd
    train.add_argument('--macrobatch-size', type=int, default=0,
                       help='distributed effective batch size')
    train.add_argument('--warmup-epochs', type=int, default=5,
                       help='the epochs to ramp-up lr to scaled large-batch value')
    train.add_argument('--warmup-strategy', type=str, default='linear',
                       help='the ramping-up strategy for large batch sgd')
    train.add_argument('--logging-dir', type=str, default='logs')
    train.add_argument('--log', type=str, default='')
    train.add_argument('--bn-gamma-init0', action='store_true')
    train.add_argument('--epoch-size',type=int, default=0,
                       help='set number of batches in an epoch. useful for debugging')
    #train.add_argument('--tensorboard', type=str, default='',
    #                   help='log parameters to visualize in tensorboard every epoch. takes name to specify as tensorboard run. Empty means tensorboard logging is disabled')
    train.add_argument('--profile-worker-suffix', type=str, default='',
                       help='profile workers actions into this file. During distributed training\
                             filename saved will be rank1_ followed by this suffix')
    train.add_argument('--profile-server-suffix', type=str, default='',
                       help='profile server actions into a file with name like rank1_ followed by this suffix \
                             during distributed training')
    train.add_argument('--use-imagenet-data-augmentation', type=int, default=0,
                       help='enable data augmentation of ImageNet data, default disabled')
    return train


def fit(args, network, data_loader, **kwargs):
    """
    train a model
    args : argparse returns
    network : the symbol definition of the nerual network
    data_loader : function that returns the train and val data iterators
    """
    # select gpu for horovod process
    if 'horovod' in args.kv_store:
        args.gpus = _get_gpu(args.gpus)

    # kvstore
    if 'horovod' in args.kv_store:
        kv = None
        rank = hvd.rank()
        num_workers = hvd.size()
    else:
        kv = mx.kvstore.create(args.kv_store)
        rank = kv.rank
        num_workers = kv.num_workers
        if args.gc_type != 'none':
            kv.set_gradient_compression({'type': args.gc_type,
                                         'threshold': args.gc_threshold})
    if args.profile_server_suffix:
        mx.profiler.set_config(filename=args.profile_server_suffix, profile_all=True, profile_process='server')
        mx.profiler.set_state(state='run', profile_process='server')

    if args.profile_worker_suffix:
        if num_workers > 1:
            filename = 'rank' + str(rank) + '_' + args.profile_worker_suffix
        else:
            filename = args.profile_worker_suffix
        mx.profiler.set_config(filename=filename, profile_all=True, profile_process='worker')
        mx.profiler.set_state(state='run', profile_process='worker')

    # logging
    head = '%(asctime)-15s Node[' + str(rank) + '] %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)
    logging.info('start with arguments %s', args)

    epoch_size = get_epoch_size(args, kv)

    # data iterators
    (train, val) = data_loader(args, kv)
    if 'dist' in args.kv_store and not 'async' in args.kv_store:
        logging.info('Resizing training data to %d batches per machine', epoch_size)
        # resize train iter to ensure each machine has same number of batches per epoch
        # if not, dist_sync can hang at the end with one machine waiting for other machines
        if not args.use_dali:
            train = mx.io.ResizeIter(train, epoch_size)

    if args.test_io:
        tic = time.time()
        for i, batch in enumerate(train):
            if isinstance(batch, list):
                for b in batch:
                    for j in b.data:
                        j.wait_to_read()
            else:
                for j in batch.data:
                    j.wait_to_read()
            if (i + 1) % args.disp_batches == 0:
                logging.info('Batch [%d]\tSpeed: %.2f samples/sec', i,
                             args.disp_batches * args.batch_size / (time.time() - tic))
                tic = time.time()
        return

    # load model
    if 'arg_params' in kwargs and 'aux_params' in kwargs:
        arg_params = kwargs['arg_params']
        aux_params = kwargs['aux_params']
    else:
        sym, arg_params, aux_params = _load_model(args, rank)
        if sym is not None:
            assert sym.tojson() == network.tojson()

    # save model
    checkpoint = _save_model(args, rank)
    epoch_end_callbacks = []
    if checkpoint:
        epoch_end_callbacks.append(checkpoint)
    #if args.tensorboard:
    #    lm = mx.contrib.tensorboard.LogMetricsCallback("./logs/"+args.tensorboard)
    #    epoch_end_callbacks.append(lm.node_histogram_visualization()

    # devices for training
    devs = mx.cpu() if args.gpus is None or args.gpus == "" else [
        mx.gpu(int(i)) for i in args.gpus.split(',')]

    # learning rate
    lr, lr_scheduler = _get_lr_scheduler(args, kv)

    # create model
    model = mx.mod.Module(
        context=devs,
        symbol=network
    )

    optimizer_params = {
        'learning_rate': lr,
        'wd': args.wd,
        'lr_scheduler': lr_scheduler,
        'multi_precision': True}

    # Only a limited number of optimizers have 'momentum' property
    has_momentum = {'sgd', 'dcasgd', 'nag', 'signum', 'lbsgd'}
    if args.optimizer in has_momentum:
        optimizer_params['momentum'] = args.mom


    monitor = mx.mon.Monitor(
        args.monitor, pattern=".*") if args.monitor > 0 else None

    # A limited number of optimizers have a warmup period
    has_warmup = {'lbsgd', 'lbnag'}
    if args.optimizer in has_warmup:
        if 'dist' in args.kv_store:
            nworkers = kv.num_workers
        else:
            nworkers = 1
        epoch_size = args.num_examples / args.batch_size / nworkers

        if epoch_size < 1:
            epoch_size = 1
        macrobatch_size = args.macrobatch_size
        if macrobatch_size < args.batch_size * nworkers:
            macrobatch_size = args.batch_size * nworkers
        #batch_scale = round(float(macrobatch_size) / args.batch_size / nworkers +0.4999)
        batch_scale = math.ceil(
            float(macrobatch_size) / args.batch_size / nworkers)
        optimizer_params['updates_per_epoch'] = epoch_size
        optimizer_params['begin_epoch'] = args.load_epoch if args.load_epoch else 0
        optimizer_params['batch_scale'] = batch_scale
        optimizer_params['warmup_strategy'] = args.warmup_strategy
        optimizer_params['warmup_epochs'] = args.warmup_epochs
        optimizer_params['num_epochs'] = args.num_epochs

    if 'horovod' in args.kv_store:
        optimizer_params['rescale_grad'] = 1.0 / args.batch_size
        opt = mx.optimizer.create(args.optimizer, sym=sym, **optimizer_params)

        # Horovod: wrap optimizer with DistributedOptimizer
        opt = hvd.DistributedOptimizer(opt)
    else:
        opt = args.optimizer

    if args.initializer == 'default':
        if args.network == 'alexnet':
            # AlexNet will not converge using Xavier
            initializer = mx.init.Normal()
            # VGG will not trend to converge using Xavier-Gaussian
        elif args.network and 'vgg' in args.network:
            initializer = mx.init.Xavier()
        else:
            initializer = mx.init.Xavier(
                rnd_type='gaussian', factor_type="in", magnitude=2)
    # initializer   = mx.init.Xavier(factor_type="in", magnitude=2.34),
    elif args.initializer == 'xavier':
        initializer = mx.init.Xavier()
    elif args.initializer == 'msra':
        initializer = mx.init.MSRAPrelu()
    elif args.initializer == 'orthogonal':
        initializer = mx.init.Orthogonal()
    elif args.initializer == 'normal':
        initializer = mx.init.Normal()
    elif args.initializer == 'uniform':
        initializer = mx.init.Uniform()
    elif args.initializer == 'one':
        initializer = mx.init.One()
    elif args.initializer == 'zero':
        initializer = mx.init.Zero()

    if 'horovod' in args.kv_store:
        if 'load_epoch' not in args or args.load_epoch is None:
            model.bind(data_shapes=train.provide_data,
                     label_shapes=train.provide_label)
            model.init_params(initializer, arg_params=arg_params, aux_params=aux_params)
            # Horovod: fetch and broadcast parameters
            (arg_params, aux_params) = model.get_params()
            if arg_params is not None:
                hvd.broadcast_parameters(arg_params, root_rank=0)
            if aux_params is not None:
                hvd.broadcast_parameters(aux_params, root_rank=0)
            model.set_params(arg_params=arg_params, aux_params=aux_params)


    # evaluation metrices
    eval_metrics = ['accuracy']
    if args.top_k > 0:
        eval_metrics.append(mx.metric.create(
            'top_k_accuracy', top_k=args.top_k))

    supported_loss = ['ce', 'nll_loss']
    if len(args.loss) > 0:
        # ce or nll loss is only applicable to softmax output
        loss_type_list = args.loss.split(',')
        if 'softmax_output' in network.list_outputs():
            for loss_type in loss_type_list:
                loss_type = loss_type.strip()
                if loss_type == 'nll':
                    loss_type = 'nll_loss'
                if loss_type not in supported_loss:
                    logging.warning(loss_type + ' is not an valid loss type, only cross-entropy or ' \
                                    'negative likelihood loss is supported!')
                else:
                    eval_metrics.append(mx.metric.create(loss_type))
        else:
            logging.warning("The output is not softmax_output, loss argument will be skipped!")

    # callbacks that run after each batch
    batch_end_callbacks = []
    if 'horovod' in args.kv_store:
        # if using horovod, only report on rank 0 with global batch size
        if rank == 0:
            batch_end_callbacks.append(mx.callback.Speedometer(
                num_workers*args.batch_size, args.disp_batches))
    else:
        batch_end_callbacks.append(mx.callback.Speedometer(
            args.batch_size, args.disp_batches))

    if 'batch_end_callback' in kwargs:
        cbs = kwargs['batch_end_callback']
        batch_end_callbacks += cbs if isinstance(cbs, list) else [cbs]

    # run
    model.fit(train,
              begin_epoch=args.load_epoch if args.load_epoch else 0,
              num_epoch=args.num_epochs,
              eval_data=val,
              eval_metric=eval_metrics,
              kvstore=kv,
              optimizer=opt,
              optimizer_params=optimizer_params,
              initializer=None if 'horovod' in args.kv_store else initializer,
              arg_params=arg_params,
              aux_params=aux_params,
              batch_end_callback=batch_end_callbacks,
              epoch_end_callback=epoch_end_callbacks, #checkpoint if args.use_dali else ,,
              allow_missing=True,
              monitor=monitor)

    # When using horovod, ensure all ops scheduled by the engine complete before exiting
    if 'horovod' in args.kv_store:
        mx.ndarray.waitall()

    if args.profile_server_suffix:
        mx.profiler.set_state(state='run', profile_process='server')
    if args.profile_worker_suffix:
        mx.profiler.set_state(state='run', profile_process='worker')
