#include <float.h>
#include "cl_device.h"
#include "NeuralNetwork/Parser/cfg_darknet/cfg_reader.h"
#include "NeuralNetwork/Parser/cfg_darknet/cfg_parser.h"
#include "NeuralNetwork/Parser/weight_darknet/weight_reader.h"
#include "NeuralNetwork/CPU/nn_cpu_network.h"


int main(){
    cl_platforms* platforms = cl_initDevices();
    cfg_reader* config = read_cfg_malloc("/media/human/2A520E98520E68C1/labs/Comparison of the images/OpenCL/yolov1-tiny.cfg");
    cfg_darknet_network_info* net_info = nn_darknet_network_parser_malloc(config);
    weight_darknet_network_info* net_weight = weights_reader_malloc(net_info, "/media/human/2A520E98520E68C1/Jupyter/Database/WIDER/Face-Detection-master/backup/yolov1-tiny_100.weights");
    nn_cpu_network* cpu_network = nn_cpu_network_malloc(net_info, net_weight);

    nn_cpu_forward_network(cpu_network);

    nn_cpu_network_free(cpu_network);
    weights_reader_free(net_weight);
    read_cfg_free(config);
    nn_darknet_network_parser_free(net_info);
    cl_freeDevices(&platforms);
    return 0;
}