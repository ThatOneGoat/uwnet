#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <float.h>
#include "uwnet.h"


// Run a maxpool layer on input
// layer l: pointer to layer to run
// matrix in: input to layer
// returns: the result of running the layer
matrix forward_maxpool_layer(layer l, matrix in)
{
    // Saving our input
    // Probably don't change this
    free_matrix(*l.x);
    *l.x = copy_matrix(in);

    int outw = (l.width-1)/l.stride + 1;
    int outh = (l.height-1)/l.stride + 1;
    matrix out = make_matrix(in.rows, outw*outh*l.channels);
    // TODO: 6.1 - iterate over the input and fill in the output with max values
    for (int i = 0; i < out.rows; i++) {
        for (int channel = 0; channel < l.channels; channel++) {
            for (int j = 0; j < outw * outh; j++) {
                int centerX = (j % outw) * l.stride;
                int centerY = (j / outw) * l.stride;
                int topLeftX = centerX - (l.size - 1) / 2;
                int topLeftY = centerY - (l.size - 1) / 2;
                int assignedMax = 0;
                float max;
                for (int imageY = topLeftY; imageY < topLeftY + l.size; imageY++) {
                    for (int imageX = topLeftX; imageX < topLeftX + l.size; imageX++) {
                        if (imageX >= 0 && imageX < l.width && imageY >= 0 && imageY < l.height) {
                            float val = in.data[i * in.cols + l.width*(l.height*channel + imageY) + imageX];
                            if (!assignedMax || max < val) {
                                assignedMax = 1;
                                max = val;
                            }
                        }
                    }
                }
                out.data[i * out.cols + channel * outw*outh + j] = max;
            }
        }
    }

    return out;
}

// Run a maxpool layer backward
// layer l: layer to run
// matrix dy: error term for the previous layer
matrix backward_maxpool_layer(layer l, matrix dy)
{
    matrix in    = *l.x;
    matrix dx = make_matrix(dy.rows, l.width*l.height*l.channels);

    int outw = (l.width-1)/l.stride + 1;
    int outh = (l.height-1)/l.stride + 1;
    // TODO: 6.2 - find the max values in the input again and fill in the
    // corresponding delta with the delta from the output. This should be
    // similar to the forward method in structure.
    for (int i = 0; i < dx.rows; i++) {
        for (int channel = 0; channel < l.channels; channel++) {
            for (int j = 0; j < outw * outh; j++) {
                int centerX = (j % outw) * l.stride;
                int centerY = (j / outw) * l.stride;
                int topLeftX = centerX - (l.size - 1) / 2;
                int topLeftY = centerY - (l.size - 1) / 2;
                int assignedMax = 0;
                float max;
                int maxIndex = -1;
                for (int imageY = topLeftY; imageY < topLeftY + l.size; imageY++) {
                    for (int imageX = topLeftX; imageX < topLeftX + l.size; imageX++) {
                        if (imageX >= 0 && imageX < l.width && imageY >= 0 && imageY < l.height) {
                            int index = i * in.cols + l.width*(l.height*channel + imageY) + imageX;
                            float val = in.data[index];
                            if (!assignedMax || max < val) {
                                assignedMax = 1;
                                max = val;
                                maxIndex = index;
                            }
                        }
                    }
                }
                dx.data[maxIndex] += dy.data[i * dy.cols + channel * outw*outh + j];
            }
        }
    }


    return dx;
}

// Update maxpool layer
// Leave this blank since maxpool layers have no update
void update_maxpool_layer(layer l, float rate, float momentum, float decay){}

// Make a new maxpool layer
// int w: width of input image
// int h: height of input image
// int c: number of channels
// int size: size of maxpool filter to apply
// int stride: stride of operation
layer make_maxpool_layer(int w, int h, int c, int size, int stride)
{
    layer l = {0};
    l.width = w;
    l.height = h;
    l.channels = c;
    l.size = size;
    l.stride = stride;
    l.x = calloc(1, sizeof(matrix));
    l.forward  = forward_maxpool_layer;
    l.backward = backward_maxpool_layer;
    l.update   = update_maxpool_layer;
    return l;
}

