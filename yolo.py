# The calculation refers to the model here: http://dgschwend.github.io/netscope/#/preset/YOLO

import calculator

total_flops = 0


########################################################################layer1
#conv
total_flops += calculator.get_for_conv_and_relu(input_shape=(3, 448, 448), conv_filter=(64, 3, 7, 7), stride=2, padding=3)

#pool
pool1, _, _ = calculator.get_for_conv(input_shape=(64, 224, 224), conv_filter=(1, 64, 2, 2), stride=2)
total_flops += pool1


########################################################################layer2
#conv
total_flops += calculator.get_for_conv_and_relu(input_shape=(64, 112, 112), conv_filter=(192, 64, 3, 3), stride=1, padding=1)

#pool
pool1, _, _ = calculator.get_for_conv(input_shape=(192, 112, 112), conv_filter=(1, 192, 2, 2), stride=2)
total_flops += pool1


########################################################################layer3
#conv
total_flops += calculator.get_for_conv_and_relu(input_shape=(192, 56, 56), conv_filter=(128, 192, 1, 1), stride=1)


########################################################################layer4
#conv
total_flops += calculator.get_for_conv_and_relu(input_shape=(128, 56, 56), conv_filter=(256, 128, 3, 3), stride=1, padding=1)


########################################################################layer5
#conv
total_flops += calculator.get_for_conv_and_relu(input_shape=(256, 56, 56), conv_filter=(256, 256, 1, 1), stride=1)


########################################################################layer6
#conv
total_flops += calculator.get_for_conv_and_relu(input_shape=(256, 56, 56), conv_filter=(512, 256, 3, 3), stride=1, padding=1)

#pool
pool1, _, _ = calculator.get_for_conv(input_shape=(512, 56, 56), conv_filter=(1, 512, 2, 2), stride=2)
total_flops += pool1


########################################################################layer7
#conv
total_flops += calculator.get_for_conv_and_relu(input_shape=(512, 28, 28), conv_filter=(256, 512, 1, 1), stride=1)


########################################################################layer8
#conv
total_flops += calculator.get_for_conv_and_relu(input_shape=(256, 28, 28), conv_filter=(512, 256, 3, 3), stride=1, padding=1)


########################################################################layer9
#conv
total_flops += calculator.get_for_conv_and_relu(input_shape=(512, 28, 28), conv_filter=(256, 512, 1, 1), stride=1)


########################################################################layer10
#conv
total_flops += calculator.get_for_conv_and_relu(input_shape=(256, 28, 28), conv_filter=(512, 256, 3, 3), stride=1, padding=1)


########################################################################layer11
#conv
total_flops += calculator.get_for_conv_and_relu(input_shape=(512, 28, 28), conv_filter=(256, 512, 1, 1), stride=1)


########################################################################layer12
#conv
total_flops += calculator.get_for_conv_and_relu(input_shape=(256, 28, 28), conv_filter=(512, 256, 3, 3), stride=1, padding=1)


########################################################################layer13
#conv
total_flops += calculator.get_for_conv_and_relu(input_shape=(512, 28, 28), conv_filter=(256, 512, 1, 1), stride=1)


########################################################################layer14
#conv
total_flops += calculator.get_for_conv_and_relu(input_shape=(256, 28, 28), conv_filter=(512, 256, 3, 3), stride=1, padding=1)


########################################################################layer15
#conv
total_flops += calculator.get_for_conv_and_relu(input_shape=(512, 28, 28), conv_filter=(512, 512, 1, 1), stride=1)


########################################################################layer16
#conv
total_flops += calculator.get_for_conv_and_relu(input_shape=(512, 28, 28), conv_filter=(1024, 512, 3, 3), stride=1, padding=1)

#pool
pool1, _, _ = calculator.get_for_conv(input_shape=(1024, 28, 28), conv_filter=(1, 1024, 2, 2), stride=2)
total_flops += pool1


########################################################################layer17
#conv
total_flops += calculator.get_for_conv_and_relu(input_shape=(1024, 14, 14), conv_filter=(512, 1024, 1, 1), stride=1)


########################################################################layer18
#conv
total_flops += calculator.get_for_conv_and_relu(input_shape=(512, 14, 14), conv_filter=(1024, 512, 3, 3), stride=1, padding=1)


########################################################################layer19
#conv
total_flops += calculator.get_for_conv_and_relu(input_shape=(1024, 14, 14), conv_filter=(512, 1024, 1, 1), stride=1)


########################################################################layer20
#conv
total_flops += calculator.get_for_conv_and_relu(input_shape=(512, 14, 14), conv_filter=(1024, 512, 3, 3), stride=1, padding=1)


########################################################################layer21
#conv
total_flops += calculator.get_for_conv_and_relu(input_shape=(1024, 14, 14), conv_filter=(1024, 1024, 3, 3), stride=1, padding=1)


########################################################################layer22
#conv
total_flops += calculator.get_for_conv_and_relu(input_shape=(1024, 14, 14), conv_filter=(1024, 1024, 3, 3), stride=2, padding=1)


########################################################################layer23
#conv
total_flops += calculator.get_for_conv_and_relu(input_shape=(1024, 7, 7), conv_filter=(1024, 1024, 3, 3), stride=1, padding=1)


########################################################################layer24
#conv
total_flops += calculator.get_for_conv_and_relu(input_shape=(1024, 7, 7), conv_filter=(1024, 1024, 3, 3), stride=1, padding=1)


########################################################################layer25
#fc
total_flops += calculator.get_for_fc(input_shape=(1024, 7, 7), output_num=4096)
#relu
total_flops += calculator.get_for_relu(input_shape=(4096, 1, 1))


########################################################################layer26
#fc
total_flops += calculator.get_for_fc(input_shape=(4096, 1, 1), output_num=1470)


########################################################################
print('YOLO: %.3f GFLOPS.' % (total_flops/1000000000.0))





