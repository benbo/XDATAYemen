from __future__ import print_function

import itertools
import os
import sys
import tempfile

from google.protobuf import text_format as proto_text
import h5py
import numpy as np

# Make sure that caffe is on the python path:
CAFFE_ROOT = '/home/scratch/{}/caffe/'.format(os.environ['USER'])
sys.path.insert(0, CAFFE_ROOT + 'python')
import caffe
from caffe.proto import caffe_pb2

c = lambda s: os.path.join(CAFFE_ROOT, s)
CAFFENET_PROTO = c('models/bvlc_reference_caffenet/deploy.prototxt')
CAFFENET_MODEL = c(
    'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel')
MEAN_FILE = c('python/caffe/imagenet/ilsvrc_2012_mean.npy')
MEAN = np.load(MEAN_FILE).mean(axis=(1, 2))


def load_net(prototxt=CAFFENET_PROTO, model=CAFFENET_MODEL, layers=['pool5'],
             data_mean=MEAN, batch_size=100):
    param = caffe_pb2.NetParameter()
    with open(prototxt, 'r') as f:
        proto_text.Parse(f.read(), param)

    param.layer[0].input_param.shape[0].dim[0] = batch_size

    # assume that everything after the last layer is garbage
    last_layer = max(i for i, l in enumerate(param.layer) if l.name in layers)
    out_size = sum
    del param.layer[last_layer + 1:]

    with tempfile.NamedTemporaryFile() as f:
        f.write(proto_text.MessageToString(param))
        f.flush()
        net = caffe.Net(f.name, model, caffe.TEST)

    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_mean('data', data_mean) # mean pixel
    transformer.set_raw_scale('data', 255)  # model works on [0,255], not [0,1]
    transformer.set_channel_swap('data', (2,1,0))  # model has channels in BGR

    return net, transformer


def preprocess_images(images, transformer):
    for image in images:
        if isinstance(image, (str, bytes)):
            image = caffe.io.load_image(image)
        yield transformer.preprocess('data', image)


def group(things, batch_size):
    out = []
    for x in things:
        out.append(x)
        if len(out) == batch_size:
            yield out
            out = []

    if out:
        yield out


def get_features(images, layers=['pool5'], batch_size=100, **kwargs):
    net, transformer = load_net(layers=layers, batch_size=batch_size, **kwargs)
    image_stream = preprocess_images(images, transformer)
    for batch in group(image_stream, batch_size):
        batch = np.asarray(batch)
        if len(batch) != batch_size:  # last group
            net.blobs['data'].reshape(*batch.shape)
        net.blobs['data'].data[...] = batch
        net.forward()
        all_out = np.hstack([
            net.blobs[l].data.reshape(batch.shape[0], -1) for l in layers])
        for row in all_out:
            yield row


def main():
    import argparse
    try:
        import progressbar as pb
    except ImportError:
        print("Run `pip install progressbar2`, or take this line out",
              file=sys.stderr)
        raise

    parser = argparse.ArgumentParser()
    parser.add_argument('images_file')
    parser.add_argument('out_h5')
    parser.add_argument('out_dset')
    parser.add_argument('--layers', '-l', nargs='+', default=['pool5'])
    parser.add_argument('--batch-size', type=int, default=500)
    g = parser.add_mutually_exclusive_group()
    g.add_argument('--cpu', action='store_true')
    g.add_argument('--gpu', type=int, default=None)
    args = parser.parse_args()

    if args.cpu:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        if args.gpu is not None:
            caffe.set_device(args.gpu)

    with open(args.images_file) as f:
        images = f.read().splitlines()

    with h5py.File(args.out_h5) as f:
        if args.out_dset in f:
            parser.error(
                '{}: {} already exists'.format(args.out_h5, args.out_dset))

        feats = iter(get_features(
            images, args.layers, batch_size=args.batch_size))
        n = len(images)

        # peek at the first result, to check dimensionality
        first = next(feats)
        dim = first.size
        feats = itertools.chain([first], feats)

        dset = f.create_dataset(args.out_dset, (n, dim), chunks=(1, dim))
        for i, feat in enumerate(pb.ProgressBar(max_value=n)(feats)):
            dset[i] = feat
            if i % 100000 == 0:
                f.flush()

if __name__ == '__main__':
    main()
