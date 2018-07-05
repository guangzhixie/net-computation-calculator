

# input_shape: Format:(channels, rows,cols)
# conv_filter: Format: (num_filters, channels, rows, cols)
def get_for_conv(input_shape=(3, 224, 224), conv_filter=(64, 3, 7, 7), stride=0, padding=0):
    n = conv_filter[1] * conv_filter[2] * conv_filter[3]  # vector_length
    flops_per_instance = n + (n-1)  # general defination for number of flops (n: multiplications and n-1: additions)

    output_row_num = ((input_shape[1] - conv_filter[2] + 2*padding) / stride) + 1
    output_col_num = ((input_shape[2] - conv_filter[3] + 2*padding) / stride) + 1

    flops_per_filter = output_row_num * output_col_num * flops_per_instance
    total_flops_per_layer = flops_per_filter * conv_filter[0]    # multiply with number of filters

    return total_flops_per_layer, output_row_num, output_col_num


# input_shape: Format:(channels, rows,cols)
def get_for_relu(input_shape):
    n = input_shape[0]*input_shape[1]*input_shape[2]
    return n+(n-1)


# input_shape: Format:(channels, rows,cols)
def get_for_scale(input_shape):
    n = input_shape[0] * input_shape[1] * input_shape[2]
    return n + (n - 1)


# input_shape: Format:(channels, rows,cols)
def get_for_batchnorm(input_shape):
    n = input_shape[0] * input_shape[1] * input_shape[2]
    return n + (n - 1)


# input_shape: Format:(channels, rows,cols)
def get_for_eltwise(input_shape):
    n = input_shape[0] * input_shape[1] * input_shape[2]
    return n + (n - 1)



# input_shape: Format:(channels, rows,cols)
def get_for_fc(input_shape, output_num):
    n = input_shape[0]*input_shape[1]*input_shape[2]*output_num
    return n + (n - 1)


# conv + scale + batchnorm + relu
# input_shape: Format:(channels, rows,cols)
# conv_filter: Format: (num_filters, channels, rows, cols)
def get_for_conv_layer(input_shape=(3, 224, 224), conv_filter=(64, 3, 7, 7), stride=0, padding=0, include_relu=True):
    flops, output_row_num, output_col_num = get_for_conv(input_shape, conv_filter, stride, padding)

    flops += get_for_scale((conv_filter[0], output_row_num, output_col_num))

    flops += get_for_batchnorm((conv_filter[0], output_row_num, output_col_num))

    if include_relu:
        flops += get_for_relu((conv_filter[0], output_row_num, output_col_num))

    return flops


# input_shape: Format:(channels, rows,cols)
def get_for_eltwise_and_relu(input_shape):
    return get_for_eltwise(input_shape) + get_for_relu(input_shape)