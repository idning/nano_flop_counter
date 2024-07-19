# nano_flop_counter

A very simple flops counter output fqn style stats

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/gist/idning/94815a257f99be88269ecb333ab1c88d/nanoflopcounter.ipynb)



    from torchvision import models as torchvision_models
    resnet18 = torchvision_models.resnet18()
    a = torch.randn((1, 3, 224, 224), requires_grad=True)
    with NanoFlopCounter(resnet18) as mode:
        resnet18(a)
    print(tabulate(mode.report(), headers='keys', tablefmt='plain', intfmt=",")) 

       module                           flops      params
     0  ~                        3,628,146,688  11,689,512
     1  ~.conv1                    236,027,904       9,408
     2  ~.layer1                   924,844,032     147,968
     3  ~.layer1.0                 462,422,016      73,984
     4  ~.layer1.0.conv1           231,211,008      36,864
     5  ~.layer1.0.conv2           231,211,008      36,864
     6  ~.layer1.1                 462,422,016      73,984
     7  ~.layer1.1.conv1           231,211,008      36,864
     8  ~.layer1.1.conv2           231,211,008      36,864
     9  ~.layer2                   822,083,584     525,568
    10  ~.layer2.0                 359,661,568     230,144
    11  ~.layer2.0.conv1           115,605,504      73,728
    12  ~.layer2.0.conv2           231,211,008     147,456
    13  ~.layer2.0.downsample       12,845,056       8,448
    14  ~.layer2.0.downsample.0     12,845,056       8,192
    15  ~.layer2.1                 462,422,016     295,424
    16  ~.layer2.1.conv1           231,211,008     147,456
    17  ~.layer2.1.conv2           231,211,008     147,456
    18  ~.layer3                   822,083,584   2,099,712
    19  ~.layer3.0                 359,661,568     919,040
    20  ~.layer3.0.conv1           115,605,504     294,912
    21  ~.layer3.0.conv2           231,211,008     589,824
    22  ~.layer3.0.downsample       12,845,056      33,280
    23  ~.layer3.0.downsample.0     12,845,056      32,768
    24  ~.layer3.1                 462,422,016   1,180,672
    25  ~.layer3.1.conv1           231,211,008     589,824
    26  ~.layer3.1.conv2           231,211,008     589,824
    27  ~.layer4                   822,083,584   8,393,728
    28  ~.layer4.0                 359,661,568   3,673,088
    29  ~.layer4.0.conv1           115,605,504   1,179,648
    30  ~.layer4.0.conv2           231,211,008   2,359,296
    31  ~.layer4.0.downsample       12,845,056     132,096
    32  ~.layer4.0.downsample.0     12,845,056     131,072
    33  ~.layer4.1                 462,422,016   4,720,640
    34  ~.layer4.1.conv1           231,211,008   2,359,296
    35  ~.layer4.1.conv2           231,211,008   2,359,296
    36  ~.fc                         1,024,000     513,000


== Original torch.utils.flop_counter.FlopCounterMode

    from torch.utils.flop_counter import FlopCounterMode

    with FlopCounterMode(display=False) as fc:
        resnet18(a)
    print(fc.get_table())

    Module                    FLOP    % Total
    -------------------  ---------  ---------
    Global               3628.147M    100.00%
     - aten.convolution  3627.123M     99.97%
     - aten.addmm           1.024M      0.03%

