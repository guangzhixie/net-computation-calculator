# The calculation refers to the model here: http://dgschwend.github.io/netscope/#/preset/resnet-50

import calculator

total_flops = 0

#conv1
total_flops += calculator.get_for_conv_layer(input_shape=(3, 224, 224), conv_filter=(64, 3, 7, 7), stride=2, padding=3)

#pool1
pool1, _, _ = calculator.get_for_conv(input_shape=(64, 112, 112), conv_filter=(1, 64, 3, 3), stride=2)
total_flops += pool1


########################################################################
#conv
total_flops += calculator.get_for_conv_layer(input_shape=(64, 56, 56), conv_filter=(1, 64, 1, 1), stride=1)

#conv
total_flops += calculator.get_for_conv_layer(input_shape=(64, 56, 56), conv_filter=(1, 64, 3, 3), stride=1, padding=1)

#conv
total_flops += calculator.get_for_conv_layer(input_shape=(64, 56, 56), conv_filter=(256, 64, 1, 1), stride=1, include_relu=False)

#conv
total_flops += calculator.get_for_conv_layer(input_shape=(64, 56, 56), conv_filter=(256, 64, 1, 1), stride=1, include_relu=False)

#concat
total_flops += calculator.get_for_eltwise_and_relu(input_shape=(256, 56, 56))


########################################################################
#conv
total_flops += calculator.get_for_conv_layer(input_shape=(256, 56, 56), conv_filter=(64, 256, 1, 1), stride=1)

#conv
total_flops += calculator.get_for_conv_layer(input_shape=(64, 56, 56), conv_filter=(64, 64, 3, 3), stride=1, padding=1)

#conv
total_flops += calculator.get_for_conv_layer(input_shape=(64, 56, 56), conv_filter=(256, 64, 1, 1), stride=1, include_relu=False)

#concat
total_flops += calculator.get_for_eltwise_and_relu(input_shape=(256, 56, 56))



########################################################################
#conv
total_flops += calculator.get_for_conv_layer(input_shape=(256, 56, 56), conv_filter=(64, 256, 1, 1), stride=1)

#conv
total_flops += calculator.get_for_conv_layer(input_shape=(64, 56, 56), conv_filter=(64, 64, 3, 3), stride=1, padding=1)

#conv
total_flops += calculator.get_for_conv_layer(input_shape=(64, 56, 56), conv_filter=(256, 64, 1, 1), stride=1, include_relu=False)

#concat
total_flops += calculator.get_for_eltwise_and_relu(input_shape=(256, 56, 56))



########################################################################
#conv
total_flops += calculator.get_for_conv_layer(input_shape=(256, 56, 56), conv_filter=(128, 256, 1, 1), stride=2)

#conv
total_flops += calculator.get_for_conv_layer(input_shape=(128, 28, 28), conv_filter=(128, 128, 3, 3), stride=1, padding=1)

#conv
total_flops += calculator.get_for_conv_layer(input_shape=(128, 28, 28), conv_filter=(512, 128, 1, 1), stride=1, include_relu=False)

#conv
total_flops += calculator.get_for_conv_layer(input_shape=(256, 56, 56), conv_filter=(512, 256, 1, 1), stride=2, include_relu=False)

#concat
total_flops += calculator.get_for_eltwise_and_relu(input_shape=(512, 28, 28))



########################################################################
#conv
total_flops += calculator.get_for_conv_layer(input_shape=(512, 28, 28), conv_filter=(128, 512, 1, 1), stride=1)

#conv
total_flops += calculator.get_for_conv_layer(input_shape=(128, 28, 28), conv_filter=(128, 128, 3, 3), stride=1, padding=1)

#conv
total_flops += calculator.get_for_conv_layer(input_shape=(128, 28, 28), conv_filter=(512, 128, 1, 1), stride=1, include_relu=False)

#concat
total_flops += calculator.get_for_eltwise_and_relu(input_shape=(512, 28, 28))



########################################################################
#conv
total_flops += calculator.get_for_conv_layer(input_shape=(512, 28, 28), conv_filter=(128, 512, 1, 1), stride=1)

#conv
total_flops += calculator.get_for_conv_layer(input_shape=(128, 28, 28), conv_filter=(128, 128, 3, 3), stride=1, padding=1)

#conv
total_flops += calculator.get_for_conv_layer(input_shape=(128, 28, 28), conv_filter=(512, 128, 1, 1), stride=1, include_relu=False)

#concat
total_flops += calculator.get_for_eltwise_and_relu(input_shape=(512, 28, 28))



########################################################################
#conv
total_flops += calculator.get_for_conv_layer(input_shape=(512, 28, 28), conv_filter=(128, 512, 1, 1), stride=1)

#conv
total_flops += calculator.get_for_conv_layer(input_shape=(128, 28, 28), conv_filter=(128, 128, 3, 3), stride=1, padding=1)

#conv
total_flops += calculator.get_for_conv_layer(input_shape=(128, 28, 28), conv_filter=(512, 128, 1, 1), stride=1, include_relu=False)

#concat
total_flops += calculator.get_for_eltwise_and_relu(input_shape=(512, 28, 28))



########################################################################
#conv
total_flops += calculator.get_for_conv_layer(input_shape=(512, 28, 28), conv_filter=(256, 512, 1, 1), stride=2)

#conv
total_flops += calculator.get_for_conv_layer(input_shape=(256, 14, 14), conv_filter=(256, 256, 3, 3), stride=1, padding=1)

#conv
total_flops += calculator.get_for_conv_layer(input_shape=(256, 14, 14), conv_filter=(1024, 256, 1, 1), stride=1, include_relu=False)

#conv
total_flops += calculator.get_for_conv_layer(input_shape=(512, 28, 28), conv_filter=(1024, 512, 1, 1), stride=2, include_relu=False)

#concat
total_flops += calculator.get_for_eltwise_and_relu(input_shape=(1024, 14, 14))



########################################################################
#conv
total_flops += calculator.get_for_conv_layer(input_shape=(1024, 14, 14), conv_filter=(256, 1024, 1, 1), stride=1)

#conv
total_flops += calculator.get_for_conv_layer(input_shape=(256, 14, 14), conv_filter=(256, 256, 3, 3), stride=1, padding=1)

#conv
total_flops += calculator.get_for_conv_layer(input_shape=(256, 14, 14), conv_filter=(1024, 256, 1, 1), stride=1, include_relu=False)

#concat
total_flops += calculator.get_for_eltwise_and_relu(input_shape=(1024, 14, 14))



########################################################################
#conv
total_flops += calculator.get_for_conv_layer(input_shape=(1024, 14, 14), conv_filter=(256, 1024, 1, 1), stride=1)

#conv
total_flops += calculator.get_for_conv_layer(input_shape=(256, 14, 14), conv_filter=(256, 256, 3, 3), stride=1, padding=1)

#conv
total_flops += calculator.get_for_conv_layer(input_shape=(256, 14, 14), conv_filter=(1024, 256, 1, 1), stride=1, include_relu=False)

#concat
total_flops += calculator.get_for_eltwise_and_relu(input_shape=(1024, 14, 14))



########################################################################
#conv
total_flops += calculator.get_for_conv_layer(input_shape=(1024, 14, 14), conv_filter=(256, 1024, 1, 1), stride=1)

#conv
total_flops += calculator.get_for_conv_layer(input_shape=(256, 14, 14), conv_filter=(256, 256, 3, 3), stride=1, padding=1)

#conv
total_flops += calculator.get_for_conv_layer(input_shape=(256, 14, 14), conv_filter=(1024, 256, 1, 1), stride=1, include_relu=False)

#concat
total_flops += calculator.get_for_eltwise_and_relu(input_shape=(1024, 14, 14))



########################################################################
#conv
total_flops += calculator.get_for_conv_layer(input_shape=(1024, 14, 14), conv_filter=(256, 1024, 1, 1), stride=1)

#conv
total_flops += calculator.get_for_conv_layer(input_shape=(256, 14, 14), conv_filter=(256, 256, 3, 3), stride=1, padding=1)

#conv
total_flops += calculator.get_for_conv_layer(input_shape=(256, 14, 14), conv_filter=(1024, 256, 1, 1), stride=1, include_relu=False)

#concat
total_flops += calculator.get_for_eltwise_and_relu(input_shape=(1024, 14, 14))



########################################################################
#conv
total_flops += calculator.get_for_conv_layer(input_shape=(1024, 14, 14), conv_filter=(256, 1024, 1, 1), stride=1)

#conv
total_flops += calculator.get_for_conv_layer(input_shape=(256, 14, 14), conv_filter=(256, 256, 3, 3), stride=1, padding=1)

#conv
total_flops += calculator.get_for_conv_layer(input_shape=(256, 14, 14), conv_filter=(1024, 256, 1, 1), stride=1, include_relu=False)

#concat
total_flops += calculator.get_for_eltwise_and_relu(input_shape=(1024, 14, 14))



########################################################################
#conv
total_flops += calculator.get_for_conv_layer(input_shape=(1024, 14, 14), conv_filter=(512, 1024, 1, 1), stride=2)

#conv
total_flops += calculator.get_for_conv_layer(input_shape=(512, 7, 7), conv_filter=(512, 512, 3, 3), stride=1, padding=1)

#conv
total_flops += calculator.get_for_conv_layer(input_shape=(512, 7, 7), conv_filter=(2048, 512, 1, 1), stride=1, include_relu=False)

#conv
total_flops += calculator.get_for_conv_layer(input_shape=(1024, 14, 14), conv_filter=(2048, 1024, 1, 1), stride=2, include_relu=False)

#concat
total_flops += calculator.get_for_eltwise_and_relu(input_shape=(2048, 7, 7))



########################################################################
#conv
total_flops += calculator.get_for_conv_layer(input_shape=(2048, 7, 7), conv_filter=(512, 2048, 1, 1), stride=1)

#conv
total_flops += calculator.get_for_conv_layer(input_shape=(512, 7, 7), conv_filter=(512, 512, 3, 3), stride=1, padding=1)

#conv
total_flops += calculator.get_for_conv_layer(input_shape=(512, 7, 7), conv_filter=(2048, 512, 1, 1), stride=1, include_relu=False)

#concat
total_flops += calculator.get_for_eltwise_and_relu(input_shape=(2048, 7, 7))



########################################################################
#conv
total_flops += calculator.get_for_conv_layer(input_shape=(2048, 7, 7), conv_filter=(512, 2048, 1, 1), stride=1)

#conv
total_flops += calculator.get_for_conv_layer(input_shape=(512, 7, 7), conv_filter=(512, 512, 3, 3), stride=1, padding=1)

#conv
total_flops += calculator.get_for_conv_layer(input_shape=(512, 7, 7), conv_filter=(2048, 512, 1, 1), stride=1, include_relu=False)

#concat
total_flops += calculator.get_for_eltwise_and_relu(input_shape=(2048, 7, 7))



########################################################################
#pool
pool1, _, _ = calculator.get_for_conv(input_shape=(2048, 7, 7), conv_filter=(1, 2048, 7, 7), stride=1)
total_flops += pool1

#fc
total_flops += calculator.get_for_fc(input_shape=(2048, 1, 1), output_num=1000)



########################################################################
print('ResNet50: %.3f GFLOPS.' % (total_flops/1000000000.0))




