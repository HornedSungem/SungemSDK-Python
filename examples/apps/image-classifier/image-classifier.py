#!/usr/bin/python3

import os
import numpy
import ntpath
import argparse
import skimage.io
import skimage.transform

import sys
sys.path.append('../../../hsapi')
from core import *

NUM_PREDICTIONS      = 2
ARGS                 = None


def open_hs_device():

    devices = EnumerateDevices()
    if len(devices) == 0:
        print( "No devices found" )
        quit()

    device = Device( devices[0] )
    device.OpenDevice()

    return device

def load_graph( device ):
    with open( ARGS.graph, mode='rb' ) as f:
        blob = f.read()

    graph = device.AllocateGraph( blob )
    return graph

def pre_process_image( img_draw ):
    img = skimage.transform.resize( img_draw, ARGS.dim, preserve_range=True )

    if( ARGS.colormode == "bgr" ):
        img = img[:, :, ::-1]

    img = img.astype( numpy.float16 )
    img = ( img - numpy.float16( ARGS.mean ) ) * ARGS.scale

    return img

def infer_image( graph, img ):

    graph.LoadTensor( img, 'user object' )
    output, userobj = graph.GetResult()

    graph.LoadTensor( img, 'user object' )
    output, userobj = graph.GetResult()

    order = output.argsort()[::-1][:NUM_PREDICTIONS]
    inference_time = graph.GetGraphOption( GraphOption.TIME_TAKEN )

    print( "\n==============================================================" )
    print( "Top predictions for", ntpath.basename( ARGS.image ) )
    print( "Execution time: " + str( numpy.sum( inference_time ) ) + "ms" )
    print( "--------------------------------------------------------------" )
    for i in range( 0, NUM_PREDICTIONS ):
        print( "%3.1f%%\t" % (100.0 * output[ order[i] ] )
               + labels[ order[i] ] )
    print( "==============================================================" )

    if 'DISPLAY' in os.environ:
        skimage.io.imshow( ARGS.image )
        skimage.io.show()


def close_hs_device( device, graph ):
    graph.DeallocateGraph()
    device.CloseDevice()


def main():
    device = open_hs_device()
    graph = load_graph( device )

    img_draw = skimage.io.imread( ARGS.image )
    img = pre_process_image( img_draw )
    infer_image( graph, img )

    close_hs_device( device, graph )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                         description="Image classifier using \
                         Intel® Movidius™ Neural Compute Stick." )

    parser.add_argument( '-g', '--graph', type=str,
                         default='../../graphs/graph_googlenet',
                         help="Absolute path to the neural network graph file." )

    parser.add_argument( '-i', '--image', type=str,
                         default='../../misc/cat.jpg',
                         help="Absolute path to the image that needs to be inferred." )

    parser.add_argument( '-l', '--labels', type=str,
                         default='../../misc/synset_words.txt',
                         help="Absolute path to labels file." )

    parser.add_argument( '-M', '--mean', type=float,
                         nargs='+',
                         default=[104.00698793, 116.66876762, 122.67891434],
                         help="',' delimited floating point values for image mean." )

    parser.add_argument( '-S', '--scale', type=float,
                         default=1,
                         help="Absolute path to labels file." )

    parser.add_argument( '-D', '--dim', type=int,
                         nargs='+',
                         default=[224, 224],
                         help="Image dimensions. ex. -D 224 224" )

    parser.add_argument( '-c', '--colormode', type=str,
                         default="bgr",
                         help="RGB vs BGR color sequence. This is network dependent." )

    ARGS = parser.parse_args()

    labels =[ line.rstrip('\n') for line in
              open( ARGS.labels ) if line != 'classes\n']

    main()
